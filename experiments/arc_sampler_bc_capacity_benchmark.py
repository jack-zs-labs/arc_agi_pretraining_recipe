from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

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

from experiments.arc_sampler_bc_benchmark import build_bc_prefix_cache, slice_training_batch
from experiments.arc_sampler_fqi_benchmark import (
    ArcTraceFeatureEncoder,
    action_masks,
    action_vocabulary,
    evaluate_policy,
    generate_sampler_episodes,
    save_csv,
    summarize_episode_timings,
)
from experiments.arc_sampler_fqi_capacity_benchmark import auto_device, configure_runtime, progress_enabled


LOGIT_FLOOR = -1e9


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    projection_dim: int = 192
    grad_clip: float = 5.0
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.0


class MLPPolicyNetwork(nn.Module):
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

    def greedy_action(self, features: object, *, allowed_mask: object | None = None) -> int:
        self.eval()
        with torch.inference_mode():
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.forward(feature_tensor).squeeze(0)
            if allowed_mask is None:
                return int(torch.argmax(logits).item())
            allowed = torch.as_tensor(allowed_mask, dtype=torch.bool, device=self.device)
            if bool(torch.all(~allowed)):
                return int(torch.argmax(logits).item())
            masked_logits = logits.masked_fill(~allowed, LOGIT_FLOOR)
            return int(torch.argmax(masked_logits).item())


@dataclass
class TensorBatch:
    features: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor


def make_tensor_batch(batch: dict[str, object], *, device: torch.device) -> TensorBatch:
    return TensorBatch(
        features=torch.from_numpy(batch["features"]).to(device=device, dtype=torch.float32),
        actions=torch.from_numpy(batch["actions"]).to(device=device, dtype=torch.long),
        masks=torch.from_numpy(batch["masks"]).to(device=device, dtype=torch.bool),
    )


def train_behavior_cloning(
    batch: TensorBatch,
    *,
    num_actions: int,
    train_config: TrainConfig,
    seed: int,
    device: torch.device,
    progress: bool = False,
    progress_desc: str | None = None,
) -> MLPPolicyNetwork:
    torch.manual_seed(seed)
    model = MLPPolicyNetwork(
        batch.features.shape[1],
        num_actions,
        hidden_dims=train_config.hidden_dims,
        dropout=train_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    epochs = range(train_config.epochs)
    if progress:
        epochs = tqdm(
            epochs,
            total=train_config.epochs,
            desc=progress_desc or "train bc-mlp",
            leave=False,
            dynamic_ncols=True,
            disable=not progress_enabled(progress),
        )

    for _ in epochs:
        permutation = torch.randperm(batch.features.shape[0], device=device)
        model.train()
        for start in range(0, batch.features.shape[0], train_config.batch_size):
            index = permutation[start : start + train_config.batch_size]
            feature_batch = batch.features.index_select(0, index)
            action_batch = batch.actions.index_select(0, index)
            mask_batch = batch.masks.index_select(0, index)

            logits = model.forward(feature_batch)
            masked_logits = logits.masked_fill(~mask_batch, LOGIT_FLOOR)
            loss = F.cross_entropy(masked_logits, action_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
    return model


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dataset_episodes"])].append(row)

    metric_names = [
        "mean_normalized_return",
        "mean_prefix_accuracy",
        "exact_trace_success",
        "exact_answer_success",
        "final_grid_exact_match",
        "mean_matched_steps",
        "open_loop_action_accuracy",
    ]

    summary_rows: list[dict[str, object]] = []
    for dataset_episodes, group in sorted(grouped.items()):
        summary_row: dict[str, object] = {
            "dataset_episodes": dataset_episodes,
            "method": "bc_mlp",
        }
        for metric_name in metric_names:
            values = np.asarray([float(item[metric_name]) for item in group], dtype=np.float64)
            summary_row[metric_name] = float(values.mean())
            summary_row[f"stderr_{metric_name}"] = float(values.std(ddof=0) / np.sqrt(len(values)))
        summary_rows.append(summary_row)
    return summary_rows


def plot_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        return
    dataset_sizes = [int(row["dataset_episodes"]) for row in summary_rows]
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    metric_specs = [
        ("mean_normalized_return", "Normalized Return"),
        ("exact_trace_success", "Exact Trace Success"),
        ("exact_answer_success", "Exact Answer (Abstract)"),
        ("final_grid_exact_match", "Final Grid Exact"),
        ("open_loop_action_accuracy", "Open-Loop Accuracy"),
    ]

    for axis, (metric_name, title) in zip(axes, metric_specs, strict=True):
        y = [float(row[metric_name]) for row in summary_rows]
        yerr = [float(row[f"stderr_{metric_name}"]) for row in summary_rows]
        axis.errorbar(
            dataset_sizes,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color="#7c3aed",
            label="BC-MLP",
        )
        axis.set_title(title)
        axis.set_xlabel("Training Episodes")
        axis.set_ylabel(title)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25, linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = auto_device(args.device)
    configure_runtime(device)
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
        grad_clip=args.grad_clip,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
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
        encoder = ArcTraceFeatureEncoder(
            projection_dim=args.projection_dim,
            seed=train_seed_start + 31,
            action_tokens=vocabulary,
        )
        action_to_index = {token: idx for idx, token in enumerate(vocabulary)}
        mask_lookup = action_masks(train_episodes + eval_episodes, action_to_index=action_to_index)
        full_batch, episode_transition_ends = build_bc_prefix_cache(
            train_episodes,
            encoder=encoder,
            action_to_index=action_to_index,
            mask_lookup=mask_lookup,
        )
        eval_trajectories = [episode.test_trajectory for episode in eval_episodes]

        dataset_iterator = args.dataset_sizes
        if args.progress:
            dataset_iterator = tqdm(
                args.dataset_sizes,
                total=len(args.dataset_sizes),
                desc=f"seed {trial + 1} fits",
                leave=False,
                dynamic_ncols=True,
                disable=not progress_enabled(args.progress),
            )

        for dataset_episodes in dataset_iterator:
            batch = slice_training_batch(full_batch, int(episode_transition_ends[dataset_episodes - 1]))
            tensor_batch = make_tensor_batch(batch, device=device)
            model = train_behavior_cloning(
                tensor_batch,
                num_actions=len(vocabulary),
                train_config=train_config,
                seed=train_seed_start + dataset_episodes,
                device=device,
                progress=args.progress,
                progress_desc=f"seed {trial + 1} {dataset_episodes} bc-mlp",
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
                    "method": "bc_mlp",
                    "action_vocab_size": len(vocabulary),
                    "model_hidden_dims": "-".join(str(dim) for dim in train_config.hidden_dims),
                    "device": device.type,
                    **metrics,
                }
            )

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
    parser = argparse.ArgumentParser(description="Higher-capacity masked behavior cloning on ARC sampler traces.")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/arc_sampler_bc_capacity")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
