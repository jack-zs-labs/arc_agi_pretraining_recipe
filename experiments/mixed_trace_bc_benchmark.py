from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.mixed_reasoning_dataset import (
    ReasoningTextExample,
    build_arc_reasoning_examples,
    build_core_reasoning_examples,
    build_gsm8k_reasoning_examples,
    build_mmlu_pro_reasoning_examples,
    build_mmlu_reasoning_examples,
    build_mmlu_redux_reasoning_examples,
    split_examples,
)
from experiments.arc_sampler_fqi_benchmark import hashed_bag, save_csv


FLOAT_DTYPE = np.float32
LOGIT_FLOOR = FLOAT_DTYPE(-1e9)
TOKEN_RE = re.compile(r"[A-Za-z0-9_./%:-]+")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    feature_dim: int = 4096


@dataclass(frozen=True)
class ActionTextExample:
    benchmark: str
    trajectory_id: str
    step_index: int
    trace_step: str
    candidate_bucket: str
    input_text: str
    target_action: str


class HashedTextEncoder:
    def __init__(self, *, feature_dim: int):
        self.output_dim = feature_dim

    def encode(self, text: str) -> np.ndarray:
        tokens: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "=" not in stripped:
                tokens.extend(f"raw:{token.lower()}" for token in TOKEN_RE.findall(stripped))
                continue
            key, value = stripped.split("=", 1)
            tokens.append(f"field:{key}")
            if key == "state_tokens":
                tokens.extend(f"state:{token}" for token in value.split())
                continue
            if key == "state_scalars":
                for index, scalar in enumerate(value.split(",")):
                    tokens.append(f"scalar:{index}:{scalar}")
                continue
            tokens.extend(f"{key}:{token.lower()}" for token in TOKEN_RE.findall(value))
        return hashed_bag(tokens, self.output_dim).astype(FLOAT_DTYPE, copy=False)


class MaskedSoftmaxLinearClassifier:
    def __init__(self, feature_dim: int, num_actions: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=0.0,
            scale=0.02,
            size=(num_actions, feature_dim),
        ).astype(FLOAT_DTYPE, copy=False)
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
        progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        num_examples = features.shape[0]
        if num_examples == 0:
            return
        rng = np.random.default_rng(seed)
        indices = np.arange(num_examples)
        epoch_iterator: Iterable[int] = range(epochs)
        if progress:
            epoch_iterator = tqdm(
                epoch_iterator,
                total=epochs,
                desc=progress_desc or "train mixed bc",
                leave=False,
                dynamic_ncols=True,
            )
        for _ in epoch_iterator:
            rng.shuffle(indices)
            for start in range(0, num_examples, batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_features = features[batch_indices]
                batch_actions = actions[batch_indices]
                batch_masks = masks[batch_indices]

                logits = (batch_features @ self.weights.T).astype(FLOAT_DTYPE, copy=False)
                valid_rows = np.any(batch_masks, axis=1)
                effective_masks = batch_masks.copy()
                effective_masks[~valid_rows] = True
                masked_logits = np.where(effective_masks, logits, LOGIT_FLOOR)
                masked_logits -= masked_logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(masked_logits).astype(FLOAT_DTYPE, copy=False)
                exp_logits *= effective_masks.astype(FLOAT_DTYPE, copy=False)
                probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-8, None)
                probs[np.arange(len(batch_actions)), batch_actions] -= FLOAT_DTYPE(1.0)
                probs /= FLOAT_DTYPE(len(batch_actions))

                grad = probs.T @ batch_features
                grad += FLOAT_DTYPE(weight_decay) * self.weights
                self.weights -= FLOAT_DTYPE(learning_rate) * grad.astype(FLOAT_DTYPE, copy=False)


def parse_record_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def action_example_from_reasoning_text(example: ReasoningTextExample) -> ActionTextExample | None:
    fields = parse_record_fields(example.text)
    target_action = fields.get("target_action")
    if target_action is None or fields.get("record_type") != "decision_action":
        return None
    input_lines: list[str] = []
    for raw_line in example.text.splitlines():
        line = raw_line.strip()
        if line.startswith("target_action=") or line.startswith("target_stop=") or line.startswith("target_answer="):
            continue
        if line:
            input_lines.append(line)
    return ActionTextExample(
        benchmark=example.benchmark,
        trajectory_id=example.trajectory_id,
        step_index=example.step_index,
        trace_step=example.trace_step,
        candidate_bucket=fields.get("candidate_bucket", f"benchmark:{example.benchmark}|step:{example.trace_step}"),
        input_text="\n".join(input_lines) + "\n",
        target_action=target_action,
    )


def convert_action_examples(examples: Sequence[ReasoningTextExample]) -> tuple[tuple[ActionTextExample, ...], dict[str, int]]:
    converted: list[ActionTextExample] = []
    skipped_by_benchmark: dict[str, int] = defaultdict(int)
    for example in examples:
        converted_example = action_example_from_reasoning_text(example)
        if converted_example is None:
            skipped_by_benchmark[example.benchmark] += 1
            continue
        converted.append(converted_example)
    return tuple(converted), dict(sorted(skipped_by_benchmark.items()))


def build_dataset_bundle(args: argparse.Namespace, *, seed_offset: int) -> dict[str, object]:
    arc_train = build_arc_reasoning_examples(
        num_episodes=args.arc_train_episodes,
        seed_start=args.seed + seed_offset,
        include_verifier_targets=args.include_verifier_targets,
    )
    arc_val = build_arc_reasoning_examples(
        num_episodes=args.arc_val_episodes,
        seed_start=args.seed + seed_offset + 100_000,
        include_verifier_targets=args.include_verifier_targets,
    )

    gsm8k_all = build_gsm8k_reasoning_examples(
        data_dir=args.gsm8k_data_dir,
        max_rows=args.gsm8k_max_rows,
        include_verifier_targets=False,
    )
    gsm8k_train, gsm8k_val = split_examples(gsm8k_all, validation_fraction=args.validation_fraction)

    mmlu_all = build_mmlu_reasoning_examples(
        data_dir=args.mmlu_data_dir,
        max_rows=args.mmlu_max_rows,
        include_verifier_targets=False,
    )
    mmlu_train, mmlu_val = split_examples(mmlu_all, validation_fraction=args.validation_fraction)

    mmlu_pro_eval: tuple[ReasoningTextExample, ...] = ()
    if args.mmlu_pro_max_rows > 0:
        mmlu_pro_eval = build_mmlu_pro_reasoning_examples(
            max_rows=args.mmlu_pro_max_rows,
            include_verifier_targets=False,
        )

    mmlu_redux_eval: tuple[ReasoningTextExample, ...] = ()
    if args.mmlu_redux_max_rows > 0:
        mmlu_redux_eval = build_mmlu_redux_reasoning_examples(
            max_rows=args.mmlu_redux_max_rows,
            include_verifier_targets=False,
            label_mode=args.mmlu_redux_label_mode,
        )

    core_eval: tuple[ReasoningTextExample, ...] = ()
    if args.core_max_rows > 0:
        core_eval = build_core_reasoning_examples(
            data_dir=args.core_data_dir,
            max_rows=args.core_max_rows,
        )

    train_raw = {
        "arc": arc_train,
        "gsm8k": gsm8k_train,
        "mmlu": mmlu_train,
    }
    eval_raw = {
        "arc": arc_val,
        "gsm8k": gsm8k_val,
        "mmlu": mmlu_val,
        "mmlu_pro": mmlu_pro_eval,
        "mmlu_redux": mmlu_redux_eval,
        "core": core_eval,
    }

    train_examples: dict[str, tuple[ActionTextExample, ...]] = {}
    eval_examples: dict[str, tuple[ActionTextExample, ...]] = {}
    skipped_non_action: dict[str, int] = {}

    for benchmark, examples in train_raw.items():
        converted, skipped = convert_action_examples(examples)
        train_examples[benchmark] = converted
        skipped_non_action.update({f"train_{benchmark}": count for benchmark, count in skipped.items()})
    for benchmark, examples in eval_raw.items():
        converted, skipped = convert_action_examples(examples)
        eval_examples[benchmark] = converted
        skipped_non_action.update({f"eval_{benchmark}": count for benchmark, count in skipped.items()})

    return {
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "skipped_non_action": skipped_non_action,
    }


def flatten_examples(grouped: dict[str, tuple[ActionTextExample, ...]]) -> list[ActionTextExample]:
    merged: list[ActionTextExample] = []
    for examples in grouped.values():
        merged.extend(examples)
    return merged


def action_vocabulary(examples: Sequence[ActionTextExample]) -> tuple[str, ...]:
    return tuple(sorted({example.target_action for example in examples}))


def candidate_masks(
    examples: Sequence[ActionTextExample],
    *,
    action_to_index: dict[str, int],
) -> dict[str, np.ndarray]:
    bucket_to_actions: dict[str, set[int]] = defaultdict(set)
    for example in examples:
        bucket_to_actions[example.candidate_bucket].add(action_to_index[example.target_action])
    masks: dict[str, np.ndarray] = {}
    for bucket, indices in bucket_to_actions.items():
        mask = np.zeros(len(action_to_index), dtype=bool)
        for index in indices:
            mask[index] = True
        masks[bucket] = mask
    return masks


def build_training_batch(
    examples: Sequence[ActionTextExample],
    *,
    encoder: HashedTextEncoder,
    action_to_index: dict[str, int],
    bucket_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    features = np.stack([encoder.encode(example.input_text) for example in examples], axis=0).astype(FLOAT_DTYPE, copy=False)
    actions = np.asarray([action_to_index[example.target_action] for example in examples], dtype=np.int64)
    masks = np.stack(
        [
            bucket_masks.get(example.candidate_bucket, np.ones(len(action_to_index), dtype=bool))
            for example in examples
        ],
        axis=0,
    )
    return {
        "features": features,
        "actions": actions,
        "masks": masks,
    }


def evaluate_examples(
    examples_by_benchmark: dict[str, tuple[ActionTextExample, ...]],
    *,
    model: MaskedSoftmaxLinearClassifier,
    encoder: HashedTextEncoder,
    action_to_index: dict[str, int],
    bucket_masks: dict[str, np.ndarray],
    train_action_set: set[str],
) -> dict[str, float]:
    benchmark_action_accuracy: dict[str, float] = {}
    benchmark_trace_success: dict[str, float] = {}
    benchmark_seen_action_rate: dict[str, float] = {}
    total_correct = 0
    total_examples = 0
    total_trace_success = 0
    total_trace_count = 0
    total_seen = 0

    for benchmark, examples in sorted(examples_by_benchmark.items()):
        if not examples:
            benchmark_action_accuracy[benchmark] = float("nan")
            benchmark_trace_success[benchmark] = float("nan")
            benchmark_seen_action_rate[benchmark] = float("nan")
            continue
        example_correct: list[bool] = []
        trajectory_correct: dict[str, list[bool]] = defaultdict(list)
        seen_count = 0
        for example in examples:
            features = encoder.encode(example.input_text)
            mask = bucket_masks.get(example.candidate_bucket)
            predicted_index = model.greedy_action(features, allowed_mask=mask)
            predicted_action = next(
                action for action, index in action_to_index.items() if index == predicted_index
            )
            is_correct = predicted_action == example.target_action
            example_correct.append(is_correct)
            trajectory_correct[example.trajectory_id].append(is_correct)
            seen_count += int(example.target_action in train_action_set)
        action_accuracy = float(np.mean(np.asarray(example_correct, dtype=np.float64)))
        trace_success = float(
            np.mean(
                np.asarray([all(flags) for flags in trajectory_correct.values()], dtype=np.float64)
            )
        )
        benchmark_action_accuracy[benchmark] = action_accuracy
        benchmark_trace_success[benchmark] = trace_success
        benchmark_seen_action_rate[benchmark] = float(seen_count / len(examples))
        total_correct += sum(int(flag) for flag in example_correct)
        total_examples += len(examples)
        total_trace_success += sum(int(all(flags)) for flags in trajectory_correct.values())
        total_trace_count += len(trajectory_correct)
        total_seen += seen_count

    valid_action = [value for value in benchmark_action_accuracy.values() if value == value]
    valid_trace = [value for value in benchmark_trace_success.values() if value == value]
    metrics: dict[str, float] = {
        "overall_action_accuracy": float(total_correct / max(total_examples, 1)),
        "overall_exact_trace_success": float(total_trace_success / max(total_trace_count, 1)),
        "overall_seen_action_rate": float(total_seen / max(total_examples, 1)),
        "macro_action_accuracy": float(np.mean(np.asarray(valid_action, dtype=np.float64))) if valid_action else float("nan"),
        "macro_exact_trace_success": float(np.mean(np.asarray(valid_trace, dtype=np.float64))) if valid_trace else float("nan"),
        "eval_example_count": float(total_examples),
        "eval_trajectory_count": float(total_trace_count),
    }
    for benchmark, value in benchmark_action_accuracy.items():
        metrics[f"action_accuracy_{benchmark}"] = value
    for benchmark, value in benchmark_trace_success.items():
        metrics[f"trace_success_{benchmark}"] = value
    for benchmark, value in benchmark_seen_action_rate.items():
        metrics[f"seen_action_rate_{benchmark}"] = value
    return metrics


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    metric_names = [key for key in rows[0] if key not in {"seed"}]
    summary: dict[str, object] = {}
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            summary[metric_name] = float("nan")
            summary[f"stderr_{metric_name}"] = float("nan")
            continue
        summary[metric_name] = float(valid.mean())
        summary[f"stderr_{metric_name}"] = float(valid.std(ddof=0) / np.sqrt(len(valid)))
    return [summary]


def plot_summary(summary_row: dict[str, object], output_path: Path) -> None:
    action_metrics = {
        key.removeprefix("action_accuracy_"): float(value)
        for key, value in summary_row.items()
        if key.startswith("action_accuracy_") and float(value) == float(value)
    }
    trace_metrics = {
        key.removeprefix("trace_success_"): float(value)
        for key, value in summary_row.items()
        if key.startswith("trace_success_") and float(value) == float(value)
    }
    benchmarks = sorted(set(action_metrics) | set(trace_metrics))
    if not benchmarks:
        return
    x = np.arange(len(benchmarks))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x - width / 2, [action_metrics.get(name, 0.0) for name in benchmarks], width=width, color="#0f766e")
    axes[0].bar(x + width / 2, [trace_metrics.get(name, 0.0) for name in benchmarks], width=width, color="#b45309")
    axes[0].set_title("Per-Benchmark Accuracy")
    axes[0].set_xticks(x, benchmarks, rotation=25, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(["Action Accuracy", "Exact Trace Success"], frameon=False)
    axes[0].grid(alpha=0.2, axis="y")

    overall_names = ["overall_action_accuracy", "overall_exact_trace_success", "macro_action_accuracy", "macro_exact_trace_success"]
    overall_values = [float(summary_row.get(name, float("nan"))) for name in overall_names]
    axes[1].bar(
        np.arange(len(overall_names)),
        [0.0 if value != value else value for value in overall_values],
        color=["#0f766e", "#b45309", "#155e75", "#92400e"],
    )
    axes[1].set_title("Overall Summary")
    axes[1].set_xticks(
        np.arange(len(overall_names)),
        ["overall action", "overall trace", "macro action", "macro trace"],
        rotation=20,
        ha="right",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_data_only_row(
    *,
    seed: int,
    train_examples: dict[str, tuple[ActionTextExample, ...]],
    eval_examples: dict[str, tuple[ActionTextExample, ...]],
    vocabulary: tuple[str, ...],
    bucket_masks: dict[str, np.ndarray],
    skipped_non_action: dict[str, int],
) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": seed,
        "train_example_count": sum(len(examples) for examples in train_examples.values()),
        "eval_example_count": sum(len(examples) for examples in eval_examples.values()),
        "action_vocab_size": len(vocabulary),
        "candidate_bucket_count": len(bucket_masks),
        "skipped_non_action_json": json.dumps(skipped_non_action, sort_keys=True),
    }
    for benchmark, examples in sorted(train_examples.items()):
        row[f"train_examples_{benchmark}"] = len(examples)
    for benchmark, examples in sorted(eval_examples.items()):
        row[f"eval_examples_{benchmark}"] = len(examples)
    return row


def build_data_only_payload(
    *,
    args: argparse.Namespace,
    data_only_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "mode": "data_only",
        "seeds": args.seeds,
        "data_only_summary_rows": data_only_rows,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        feature_dim=args.feature_dim,
    )
    raw_rows: list[dict[str, object]] = []
    data_only_rows: list[dict[str, object]] = []
    seed_iterator = range(args.seeds)
    if args.progress:
        seed_iterator = tqdm(seed_iterator, total=args.seeds, desc="seed sweeps", dynamic_ncols=True)

    for trial in seed_iterator:
        bundle = build_dataset_bundle(args, seed_offset=trial * 10_000)
        train_examples = bundle["train_examples"]
        eval_examples = bundle["eval_examples"]
        skipped_non_action = bundle["skipped_non_action"]
        merged_train = flatten_examples(train_examples)
        merged_eval = flatten_examples(eval_examples)
        vocabulary = action_vocabulary(merged_train + merged_eval)
        action_to_index = {token: idx for idx, token in enumerate(vocabulary)}
        bucket_masks = candidate_masks(merged_train + merged_eval, action_to_index=action_to_index)

        if args.data_only:
            data_only_rows.append(
                build_data_only_row(
                    seed=args.seed + trial * 10_000,
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    vocabulary=vocabulary,
                    bucket_masks=bucket_masks,
                    skipped_non_action=skipped_non_action,
                )
            )
            continue

        encoder = HashedTextEncoder(feature_dim=train_config.feature_dim)
        batch = build_training_batch(
            merged_train,
            encoder=encoder,
            action_to_index=action_to_index,
            bucket_masks=bucket_masks,
        )
        model = MaskedSoftmaxLinearClassifier(
            feature_dim=train_config.feature_dim,
            num_actions=len(vocabulary),
            seed=args.seed + trial * 10_000,
        )
        model.fit(
            batch["features"],
            batch["actions"],
            batch["masks"],
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            seed=args.seed + trial * 10_000 + 17,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} mixed bc",
        )
        metrics = evaluate_examples(
            eval_examples,
            model=model,
            encoder=encoder,
            action_to_index=action_to_index,
            bucket_masks=bucket_masks,
            train_action_set={example.target_action for example in merged_train},
        )
        raw_rows.append(
            {
                "seed": args.seed + trial * 10_000,
                "train_example_count": len(merged_train),
                "eval_example_count": len(merged_eval),
                "action_vocab_size": len(vocabulary),
                "candidate_bucket_count": len(bucket_masks),
                **metrics,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.data_only:
        save_csv(data_only_rows, output_dir / "data_only_summary.csv")
        payload = build_data_only_payload(args=args, data_only_rows=data_only_rows)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    summary_rows = aggregate(raw_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    if summary_rows:
        plot_summary(summary_rows[0], output_dir / "comparison.png")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": args.seeds,
                "train_config": train_config.__dict__,
                "summary_rows": summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed trace behavior-cloning benchmark over ARC, GSM8K, and MMLU-style reasoning traces.")
    parser.add_argument("--arc-train-episodes", type=int, default=16)
    parser.add_argument("--arc-val-episodes", type=int, default=4)
    parser.add_argument("--gsm8k-max-rows", type=int, default=48)
    parser.add_argument("--mmlu-max-rows", type=int, default=48)
    parser.add_argument("--mmlu-pro-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-label-mode", choices=("original", "corrected_single"), default="corrected_single")
    parser.add_argument("--core-max-rows", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gsm8k-data-dir", type=str, default="arc_trajectory_sampler/data/gsm8k")
    parser.add_argument("--mmlu-data-dir", type=str, default="arc_trajectory_sampler/data/mmlu")
    parser.add_argument("--core-data-dir", type=str, default="arc_trajectory_sampler/data/core")
    parser.add_argument("--include-verifier-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="results/mixed_trace_bc")
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
