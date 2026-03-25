from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm.auto import tqdm
except ImportError as exc:
    raise SystemExit("This benchmark requires torch in the active environment.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.arc_sampler_fqi_benchmark import (
    ArcFeatureEncoder,
    action_masks,
    action_vocabulary,
    aggregate,
    build_q_encoder,
    build_training_batch,
    build_training_prefix_cache,
    evaluate_policy,
    generate_sampler_episodes,
    plot_summary,
    save_csv,
    slice_training_batch,
    summarize_episode_timings,
)
from experiments.arc_sampler_structured_verifier import VerifierFitConfig


@dataclass(frozen=True)
class TrainConfig:
    gamma: float = 1.0
    fqi_iterations: int = 6
    optimizer_epochs: int = 6
    batch_size: int = 256
    target_batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    projection_dim: int = 192
    grad_clip: float = 5.0
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.0


@dataclass
class TensorBatch:
    features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_features: torch.Tensor
    next_masks: torch.Tensor
    dones: torch.Tensor


def progress_enabled(enabled: bool) -> bool:
    return enabled and sys.stderr.isatty()


def auto_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but MPS is not available.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda, but CUDA is not available.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cpu":
        cpu_count = max(1, os.cpu_count() or 1)
        # Keep CPU thread count modest to avoid oversubscription on dense MLP batches.
        torch.set_num_threads(min(cpu_count, 8))
    if device.type == "mps":
        # Fallback avoids hard failures if an op is not supported on Metal.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_actions: int,
        *,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, num_actions)
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.to(dtype=torch.float32)
        if len(self.backbone) > 0:
            x = self.backbone(x)
        return self.head(x)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(features))

    def greedy_action(self, features: object, *, allowed_mask: object | None = None) -> int:
        self.eval()
        with torch.inference_mode():
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            values = torch.sigmoid(self.forward(feature_tensor)).squeeze(0)
            if allowed_mask is None:
                return int(torch.argmax(values).item())
            allowed = torch.as_tensor(allowed_mask, dtype=torch.bool, device=self.device)
            if bool(torch.all(~allowed)):
                return int(torch.argmax(values).item())
            masked = values.masked_fill(~allowed, float("-inf"))
            return int(torch.argmax(masked).item())

    def clone(self) -> "MLPQNetwork":
        clone = MLPQNetwork(
            self.feature_dim,
            self.num_actions,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)
        clone.load_state_dict(self.state_dict())
        return clone


def make_tensor_batch(batch: dict[str, object], *, device: torch.device) -> TensorBatch:
    return TensorBatch(
        features=torch.from_numpy(batch["features"]).to(device=device, dtype=torch.float32),
        actions=torch.from_numpy(batch["actions"]).to(device=device, dtype=torch.long),
        rewards=torch.from_numpy(batch["rewards"]).to(device=device, dtype=torch.float32),
        next_features=torch.from_numpy(batch["next_features"]).to(device=device, dtype=torch.float32),
        next_masks=torch.from_numpy(batch["next_masks"]).to(device=device, dtype=torch.bool),
        dones=torch.from_numpy(batch["dones"]).to(device=device, dtype=torch.float32),
    )


def predict_masked_max(
    model: MLPQNetwork,
    features: torch.Tensor,
    masks: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            feature_batch = features[start : start + batch_size]
            mask_batch = masks[start : start + batch_size]
            predicted = model.predict(feature_batch)
            predicted = predicted.masked_fill(~mask_batch, float("-inf"))
            max_values = predicted.max(dim=1).values
            max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
            values.append(max_values)
    return torch.cat(values, dim=0)


def train_fqi(
    batch: TensorBatch,
    *,
    num_actions: int,
    train_config: TrainConfig,
    loss_name: str,
    seed: int,
    device: torch.device,
    progress: bool = False,
    progress_desc: str | None = None,
) -> MLPQNetwork:
    torch.manual_seed(seed)
    model = MLPQNetwork(
        batch.features.shape[1],
        num_actions,
        hidden_dims=train_config.hidden_dims,
        dropout=train_config.dropout,
    ).to(device)
    gamma = float(train_config.gamma)
    iterations = range(train_config.fqi_iterations)
    if progress:
        iterations = tqdm(
            iterations,
            total=train_config.fqi_iterations,
            desc=progress_desc or f"train {loss_name}",
            leave=False,
            dynamic_ncols=True,
            disable=not progress_enabled(progress),
        )
    for _ in iterations:
        next_values = predict_masked_max(
            model,
            batch.next_features,
            batch.next_masks,
            batch_size=train_config.target_batch_size,
        )
        targets = torch.clamp(batch.rewards + gamma * (1.0 - batch.dones) * next_values, 0.0, 1.0)
        updated = model.clone()
        optimizer = torch.optim.AdamW(
            updated.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        updated.train()
        for _epoch in range(train_config.optimizer_epochs):
            permutation = torch.randperm(batch.features.shape[0], device=device)
            for start in range(0, batch.features.shape[0], train_config.batch_size):
                index = permutation[start : start + train_config.batch_size]
                feature_batch = batch.features.index_select(0, index)
                action_batch = batch.actions.index_select(0, index)
                target_batch = targets.index_select(0, index)

                logits = updated.forward(feature_batch)
                chosen_logits = logits.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                if loss_name == "log":
                    loss = F.binary_cross_entropy_with_logits(chosen_logits, target_batch)
                elif loss_name == "sq":
                    loss = F.mse_loss(torch.sigmoid(chosen_logits), target_batch)
                else:
                    raise ValueError(loss_name)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(updated.parameters(), train_config.grad_clip)
                optimizer.step()
        model = updated
    return model


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = auto_device(args.device)
    configure_runtime(device)
    train_config = TrainConfig(
        gamma=args.gamma,
        fqi_iterations=args.fqi_iterations,
        optimizer_epochs=args.optimizer_epochs,
        batch_size=args.batch_size,
        target_batch_size=args.target_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
        grad_clip=args.grad_clip,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    verifier_config = VerifierFitConfig(
        model_name=args.verifier_model,
        ridge_lambda=args.verifier_ridge_lambda,
        random_feature_dim=args.verifier_random_feature_dim,
        seed=args.seed + 913,
    )

    raw_rows: list[dict[str, object]] = []
    episode_timing_rows: list[dict[str, object]] = []
    seed_iterator = range(args.seeds)
    if args.progress:
        seed_iterator = tqdm(
            seed_iterator,
            total=args.seeds,
            desc="seed sweeps",
            dynamic_ncols=True,
            disable=not progress_enabled(args.progress),
        )

    for trial in seed_iterator:
        train_seed_start = args.seed + trial * 10_000
        eval_seed_start = train_seed_start + max(args.dataset_sizes) + 1_000
        max_train_episodes = max(args.dataset_sizes)

        train_episodes = generate_sampler_episodes(
            count=max_train_episodes,
            seed_start=train_seed_start,
            include_train_trajectories=True,
            timing_rows=episode_timing_rows,
            timing_split="train",
            trial_index=trial + 1,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} train episodes",
        )
        eval_episodes = generate_sampler_episodes(
            count=args.eval_episodes,
            seed_start=eval_seed_start,
            include_train_trajectories=False,
            timing_rows=episode_timing_rows,
            timing_split="eval",
            trial_index=trial + 1,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} eval episodes",
        )

        vocabulary = action_vocabulary(train_episodes + eval_episodes)
        action_to_index = {token: idx for idx, token in enumerate(vocabulary)}
        mask_lookup = action_masks(train_episodes + eval_episodes, action_to_index=action_to_index)
        base_encoder: ArcFeatureEncoder | None = None
        full_batch: dict[str, object] | None = None
        episode_transition_ends = None
        if args.state_encoder != "structured_reconstructed":
            base_encoder = build_q_encoder(
                state_encoder=args.state_encoder,
                projection_dim=args.projection_dim,
                seed=train_seed_start + 31,
                action_tokens=vocabulary,
                train_episodes=train_episodes,
                action_to_index=action_to_index,
                verifier_config=VerifierFitConfig(
                    model_name=verifier_config.model_name,
                    ridge_lambda=verifier_config.ridge_lambda,
                    random_feature_dim=verifier_config.random_feature_dim,
                    seed=train_seed_start + 913,
                ),
            )
            full_batch, episode_transition_ends = build_training_prefix_cache(
                train_episodes,
                encoder=base_encoder,
                action_to_index=action_to_index,
                mask_lookup=mask_lookup,
            )
        eval_trajectories = [episode.test_trajectory for episode in eval_episodes]

        fit_bar = None
        if args.progress:
            fit_bar = tqdm(
                total=len(args.dataset_sizes) * 2,
                desc=f"seed {trial + 1} fits",
                leave=False,
                dynamic_ncols=True,
                disable=not progress_enabled(args.progress),
            )

        for dataset_episodes in args.dataset_sizes:
            subset_train_episodes = train_episodes[:dataset_episodes]
            if args.state_encoder == "structured_reconstructed":
                encoder = build_q_encoder(
                    state_encoder=args.state_encoder,
                    projection_dim=args.projection_dim,
                    seed=train_seed_start + 31,
                    action_tokens=vocabulary,
                    train_episodes=subset_train_episodes,
                    action_to_index=action_to_index,
                    verifier_config=VerifierFitConfig(
                        model_name=verifier_config.model_name,
                        ridge_lambda=verifier_config.ridge_lambda,
                        random_feature_dim=verifier_config.random_feature_dim,
                        seed=train_seed_start + dataset_episodes + 913,
                    ),
                )
                batch = build_training_batch(
                    subset_train_episodes,
                    encoder=encoder,
                    action_to_index=action_to_index,
                    mask_lookup=mask_lookup,
                )
            else:
                assert base_encoder is not None
                assert full_batch is not None
                assert episode_transition_ends is not None
                encoder = base_encoder
                batch = slice_training_batch(full_batch, int(episode_transition_ends[dataset_episodes - 1]))
            tensor_batch = make_tensor_batch(batch, device=device)
            for loss_name in ("log", "sq"):
                model = train_fqi(
                    tensor_batch,
                    num_actions=len(vocabulary),
                    train_config=train_config,
                    loss_name=loss_name,
                    seed=train_seed_start + dataset_episodes + (0 if loss_name == "log" else 50_000),
                    device=device,
                    progress=args.progress,
                    progress_desc=f"seed {trial + 1} {dataset_episodes} {loss_name}",
                )
                metrics = evaluate_policy(
                    eval_trajectories,
                    model=model,
                    encoder=encoder,
                    action_to_index=action_to_index,
                    mask_lookup=mask_lookup,
                )
                raw_rows.append(
                    {
                        "seed": train_seed_start,
                        "dataset_episodes": dataset_episodes,
                        "loss": loss_name,
                        "state_encoder": args.state_encoder,
                        "verifier_model": args.verifier_model if args.state_encoder == "structured_reconstructed" else "n/a",
                        "action_vocab_size": len(vocabulary),
                        "model_hidden_dims": "-".join(str(dim) for dim in train_config.hidden_dims),
                        "device": device.type,
                        **metrics,
                    }
                )
                if fit_bar is not None:
                    fit_bar.update(1)
        if fit_bar is not None:
            fit_bar.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate(raw_rows)
    timing_summary_rows = summarize_episode_timings(episode_timing_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    save_csv(episode_timing_rows, output_dir / "episode_timings.csv")
    save_csv(timing_summary_rows, output_dir / "episode_timing_summary.csv")
    plot_summary(summary_rows, output_dir / "comparison.png")
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "dataset_sizes": args.dataset_sizes,
                "seeds": args.seeds,
                "eval_episodes": args.eval_episodes,
                "state_encoder": args.state_encoder,
                "verifier_model": args.verifier_model,
                "train_config": {
                    **train_config.__dict__,
                    "hidden_dims": list(train_config.hidden_dims),
                },
                "device": device.type,
                "summary_rows": summary_rows,
                "episode_timing_summary_rows": timing_summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows, "episode_timing_summary_rows": timing_summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Higher-capacity ARC sampler FQI benchmark with a Torch MLP Q head.")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fqi-iterations", type=int, default=6)
    parser.add_argument("--optimizer-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--verifier-model",
        choices=("linear", "random_features"),
        default="random_features",
        help="Verifier sidecar used only when --state-encoder=structured_reconstructed.",
    )
    parser.add_argument(
        "--verifier-ridge-lambda",
        type=float,
        default=1e-3,
        help="Ridge regularization used when fitting the structured verifier sidecar.",
    )
    parser.add_argument(
        "--verifier-random-feature-dim",
        type=int,
        default=128,
        help="Hidden width for the random-feature verifier when --verifier-model=random_features.",
    )
    parser.add_argument(
        "--state-encoder",
        choices=("flat_grid", "structured_oracle", "structured_reconstructed"),
        default="flat_grid",
        help=(
            "Feature encoder for ARC states. "
            "'structured_oracle' reads workspace_state plus verifier labels from the trajectory records; "
            "'structured_reconstructed' fits a lightweight verifier over workspace-only structured features "
            "and feeds its predictions back into the same encoder surface."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/arc_sampler_fqi_capacity")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
