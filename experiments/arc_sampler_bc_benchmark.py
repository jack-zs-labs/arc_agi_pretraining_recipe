from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.arc_sampler_fqi_benchmark import (
    ArcFeatureEncoder,
    ArcStructuredStateEncoder,
    ArcStructuredWorkspaceEncoder,
    FLOAT_DTYPE,
    SamplerEpisode,
    action_masks,
    action_vocabulary,
    build_data_only_payload,
    build_data_only_summary_row,
    build_feature_encoder,
    canonical_action_label,
    evaluate_policy,
    generate_sampler_episodes,
    save_csv,
    summarize_episode_timings,
)
from experiments.arc_sampler_structured_verifier import (
    VerifierFitConfig,
    build_verifier_adapter,
    build_verifier_training_data,
)


LOGIT_FLOOR = FLOAT_DTYPE(-1e9)
GRAD_CLIP = FLOAT_DTYPE(8.0)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    projection_dim: int = 192
    verifier_ridge_lambda: float = 1e-3


class MaskedSoftmaxLinearPolicy:
    def __init__(self, feature_dim: int, num_actions: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(loc=0.0, scale=0.02, size=(num_actions, feature_dim)).astype(
            FLOAT_DTYPE,
            copy=False,
        )
        self.num_actions = num_actions

    def greedy_action(self, features: np.ndarray, *, allowed_mask: np.ndarray | None = None) -> int:
        logits = (features @ self.weights.T).astype(FLOAT_DTYPE, copy=False)
        if allowed_mask is None or np.all(~allowed_mask):
            return int(np.argmax(logits))
        masked_logits = np.where(allowed_mask, logits, LOGIT_FLOOR)
        return int(np.argmax(masked_logits))

    def fit(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        num_samples = features.shape[0]
        if num_samples == 0:
            return

        batch_size = max(1, min(batch_size, num_samples))
        learning_rate = FLOAT_DTYPE(learning_rate)
        weight_decay = FLOAT_DTYPE(weight_decay)
        rng = np.random.default_rng(seed)
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1 = FLOAT_DTYPE(0.9)
        beta2 = FLOAT_DTYPE(0.999)
        eps = FLOAT_DTYPE(1e-8)
        step_count = 0

        for _ in range(epochs):
            order = rng.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_indices = order[start : start + batch_size]
                batch_features = features[batch_indices]
                batch_actions = actions[batch_indices]
                batch_masks = masks[batch_indices]

                logits = (batch_features @ self.weights.T).astype(FLOAT_DTYPE, copy=False)
                masked_logits = np.where(batch_masks, logits, LOGIT_FLOOR)
                shifted_logits = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
                exp_logits = np.exp(shifted_logits).astype(FLOAT_DTYPE, copy=False)
                exp_logits *= batch_masks
                normalizers = np.clip(exp_logits.sum(axis=1, keepdims=True), FLOAT_DTYPE(1e-8), None)
                probabilities = exp_logits / normalizers

                probabilities[np.arange(len(batch_actions)), batch_actions] -= FLOAT_DTYPE(1.0)
                gradient = (probabilities.T @ batch_features) / FLOAT_DTYPE(len(batch_actions))
                gradient += weight_decay * self.weights
                np.clip(gradient, -GRAD_CLIP, GRAD_CLIP, out=gradient)

                step_count += 1
                m = beta1 * m + (FLOAT_DTYPE(1.0) - beta1) * gradient
                v = beta2 * v + (FLOAT_DTYPE(1.0) - beta2) * (gradient * gradient)
                m_hat = m / (FLOAT_DTYPE(1.0) - beta1**step_count)
                v_hat = v / (FLOAT_DTYPE(1.0) - beta2**step_count)
                self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def build_bc_prefix_cache(
    episodes: list[SamplerEpisode],
    *,
    encoder: ArcFeatureEncoder,
    action_to_index: dict[str, int],
    mask_lookup: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = []
    actions = []
    masks = []
    episode_transition_ends = []
    transition_count = 0

    for episode in episodes:
        for trajectory in episode.train_trajectories:
            previous_action_index = encoder.start_action_index
            for step_index, step in enumerate(trajectory.steps):
                features.append(encoder.encode(trajectory, step_index, previous_action_index))
                actions.append(action_to_index[canonical_action_label(step)])
                masks.append(mask_lookup[(trajectory.family, step_index)])
                previous_action_index = actions[-1]
                transition_count += 1
        episode_transition_ends.append(transition_count)

    return {
        "features": np.asarray(features, dtype=FLOAT_DTYPE),
        "actions": np.asarray(actions, dtype=np.int64),
        "masks": np.asarray(masks, dtype=bool),
    }, np.asarray(episode_transition_ends, dtype=np.int64)


def build_policy_encoder(
    *,
    state_encoder: str,
    projection_dim: int,
    seed: int,
    action_tokens: tuple[str, ...],
    train_episodes: list[SamplerEpisode],
    action_to_index: dict[str, int],
    verifier_config: VerifierFitConfig,
) -> ArcFeatureEncoder:
    if state_encoder == "structured_reconstructed":
        workspace_encoder = build_feature_encoder(
            encoder_name="structured_workspace",
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
        )
        if not isinstance(workspace_encoder, ArcStructuredWorkspaceEncoder):
            raise TypeError("structured_workspace must build an ArcStructuredWorkspaceEncoder")
        verifier_features, verifier_target_array = build_verifier_training_data(
            train_episodes,
            workspace_encoder=workspace_encoder,
            action_to_index=action_to_index,
            action_labeler=canonical_action_label,
        )
        verifier_adapter = build_verifier_adapter(
            workspace_encoder=workspace_encoder,
            features=verifier_features,
            targets=verifier_target_array,
            config=VerifierFitConfig(
                model_name=verifier_config.model_name,
                ridge_lambda=verifier_config.ridge_lambda,
                random_feature_dim=verifier_config.random_feature_dim,
                seed=verifier_config.seed,
            ),
        )
        return ArcStructuredStateEncoder(
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
            verifier_provider=verifier_adapter,
        )
    return build_feature_encoder(
        encoder_name=state_encoder,
        projection_dim=projection_dim,
        seed=seed,
        action_tokens=action_tokens,
    )


def slice_training_batch(batch: dict[str, np.ndarray], transition_count: int) -> dict[str, np.ndarray]:
    return {name: values[:transition_count] for name, values in batch.items()}


def train_behavior_cloning(
    batch: dict[str, np.ndarray],
    *,
    num_actions: int,
    train_config: TrainConfig,
    seed: int,
    progress: bool = False,
    progress_desc: str | None = None,
) -> MaskedSoftmaxLinearPolicy:
    model = MaskedSoftmaxLinearPolicy(batch["features"].shape[1], num_actions=num_actions, seed=seed)
    epochs = range(train_config.epochs)
    if progress:
        epochs = tqdm(
            epochs,
            total=train_config.epochs,
            desc=progress_desc or "train bc",
            leave=False,
            dynamic_ncols=True,
        )
    for epoch in epochs:
        model.fit(
            batch["features"],
            batch["actions"],
            batch["masks"],
            epochs=1,
            batch_size=train_config.batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            seed=seed + epoch,
        )
    return model


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["dataset_episodes"]),
                str(row.get("state_encoder", "flat_grid")),
                str(row.get("verifier_model", "n/a")),
            )
        ].append(row)

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
    for (dataset_episodes, state_encoder, verifier_model), group in sorted(grouped.items()):
        summary_row: dict[str, object] = {
            "dataset_episodes": dataset_episodes,
            "method": "bc",
            "state_encoder": state_encoder,
            "verifier_model": verifier_model,
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
    state_encoder = str(summary_rows[0].get("state_encoder", "flat_grid"))
    verifier_model = str(summary_rows[0].get("verifier_model", "n/a"))
    if state_encoder == "structured_oracle":
        color = "#b45309"
        label = "BC (structured-oracle)"
    elif state_encoder == "structured_reconstructed":
        color = "#0f766e"
        label = f"BC (structured-reconstructed/{verifier_model})"
    else:
        color = "#2563eb"
        label = "BC"
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
            color=color,
            label=label,
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
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
        verifier_ridge_lambda=args.verifier_ridge_lambda,
    )
    verifier_config = VerifierFitConfig(
        model_name=args.verifier_model,
        ridge_lambda=args.verifier_ridge_lambda,
        random_feature_dim=args.verifier_random_feature_dim,
        seed=args.seed + 913,
    )

    raw_rows: list[dict[str, object]] = []
    data_only_rows: list[dict[str, object]] = []
    episode_timing_rows: list[dict[str, object]] = []
    seed_iterator = range(args.seeds)
    if args.progress:
        seed_iterator = tqdm(seed_iterator, total=args.seeds, desc="seed sweeps", dynamic_ncols=True)

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
        if args.data_only:
            data_only_rows.append(
                build_data_only_summary_row(
                    seed=train_seed_start,
                    dataset_sizes=args.dataset_sizes,
                    train_episodes=train_episodes,
                    eval_episodes=eval_episodes,
                    vocabulary=vocabulary,
                    mask_lookup=mask_lookup,
                )
            )
            continue
        base_encoder: ArcFeatureEncoder | None = None
        full_batch: dict[str, np.ndarray] | None = None
        episode_transition_ends: np.ndarray | None = None
        if args.state_encoder != "structured_reconstructed":
            base_encoder = build_policy_encoder(
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
            full_batch, episode_transition_ends = build_bc_prefix_cache(
                train_episodes,
                encoder=base_encoder,
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
            )
        for dataset_episodes in dataset_iterator:
            subset_train_episodes = train_episodes[:dataset_episodes]
            if args.state_encoder == "structured_reconstructed":
                encoder = build_policy_encoder(
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
                batch, _ = build_bc_prefix_cache(
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
            model = train_behavior_cloning(
                batch,
                num_actions=len(vocabulary),
                train_config=train_config,
                seed=train_seed_start + dataset_episodes,
                progress=args.progress,
                progress_desc=f"seed {trial + 1} {dataset_episodes} bc",
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
                    "method": "bc",
                    "state_encoder": args.state_encoder,
                    "verifier_model": args.verifier_model if args.state_encoder == "structured_reconstructed" else "n/a",
                    "action_vocab_size": len(vocabulary),
                    **metrics,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_summary_rows = summarize_episode_timings(episode_timing_rows)
    save_csv(episode_timing_rows, output_dir / "episode_timings.csv")
    save_csv(timing_summary_rows, output_dir / "episode_timing_summary.csv")
    if args.data_only:
        save_csv(data_only_rows, output_dir / "data_only_summary.csv")
        payload = build_data_only_payload(
            args=args,
            data_only_rows=data_only_rows,
            timing_summary_rows=timing_summary_rows,
        )
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    summary_rows = aggregate(raw_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    plot_summary(summary_rows, output_dir / "comparison.png")
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "dataset_sizes": args.dataset_sizes,
                "seeds": args.seeds,
                "eval_episodes": args.eval_episodes,
                "state_encoder": args.state_encoder,
                "verifier_model": args.verifier_model,
                "train_config": train_config.__dict__,
                "summary_rows": summary_rows,
                "episode_timing_summary_rows": timing_summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows, "episode_timing_summary_rows": timing_summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark masked behavior cloning on ARC sampler trajectories.")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--verifier-ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--verifier-model",
        choices=("linear", "random_features"),
        default="random_features",
        help="Model used to reconstruct verifier signals from workspace-only structured features.",
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
    parser.add_argument("--output-dir", default="results/arc_sampler_bc")
    parser.add_argument("--data-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    if args.data_only:
        print(json.dumps(payload["data_only_summary_rows"], indent=2))
    else:
        print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
