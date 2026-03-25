from __future__ import annotations

import argparse
from collections import defaultdict
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm
except ImportError as exc:
    raise SystemExit("This benchmark requires torch in the active environment.") from exc

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
    build_olympiad_math_reasoning_examples,
    split_examples,
)
from experiments.arc_sampler_fqi_benchmark import save_csv
from experiments.arc_sampler_fqi_capacity_benchmark import auto_device, configure_runtime, progress_enabled
from experiments.mixed_trace_bc_benchmark import (
    ActionTextExample,
    aggregate,
    build_data_only_payload,
    convert_action_examples,
    flatten_examples,
    plot_summary,
)
from models import AttentionBackendConfig, BenchmarkLatentPolicy, DecoderModelConfig


EOS_TOKEN_ID = 256
BYTE_VOCAB_SIZE = 257
LOGIT_FLOOR = -1e9
OUTPUT_HEAD_ALIASES = {
    "mmlu_pro": "mmlu",
    "mmlu_redux": "mmlu",
}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    max_seq_len: int = 384
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 6
    num_kv_heads: int = 2
    intermediate_size: int = 512
    latent_kv_dim: int = 48
    attention_preset: str = "mla_default"
    attention_dropout: float = 0.0


@dataclass(frozen=True)
class TokenizedActionExample:
    benchmark: str
    output_head: str
    trajectory_id: str
    token_ids: tuple[int, ...]
    target_action: str
    target_index: int
    candidate_mask: tuple[bool, ...]


@dataclass
class CollatedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    benchmark_ids: torch.Tensor
    output_heads: list[str]
    targets: list[int]
    candidate_masks: list[tuple[bool, ...]]


def encode_text(text: str, *, max_seq_len: int) -> tuple[int, ...]:
    token_ids = list(text.encode("utf-8", errors="strict"))
    token_ids.append(EOS_TOKEN_ID)
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[-max_seq_len:]
    return tuple(int(token_id) for token_id in token_ids)


def output_head_for_benchmark(benchmark: str) -> str:
    return OUTPUT_HEAD_ALIASES.get(benchmark, benchmark)


def build_arc_action_examples(
    *,
    num_episodes: int,
    seed_start: int,
    include_verifier_targets: bool,
) -> tuple[ActionTextExample, ...]:
    raw_examples = build_arc_reasoning_examples(
        num_episodes=num_episodes,
        seed_start=seed_start,
        include_verifier_targets=include_verifier_targets,
    )
    converted, _ = convert_action_examples(raw_examples)
    return converted


def build_static_action_bundle(args: argparse.Namespace) -> dict[str, object]:
    train_raw: dict[str, tuple[ReasoningTextExample, ...]] = {}
    eval_raw: dict[str, tuple[ReasoningTextExample, ...]] = {}

    gsm8k_all = build_gsm8k_reasoning_examples(
        data_dir=args.gsm8k_data_dir,
        max_rows=args.gsm8k_max_rows,
        include_verifier_targets=False,
    )
    gsm8k_train, gsm8k_val = split_examples(gsm8k_all, validation_fraction=args.validation_fraction)
    train_raw["gsm8k"] = gsm8k_train
    eval_raw["gsm8k"] = gsm8k_val

    mmlu_all = build_mmlu_reasoning_examples(
        data_dir=args.mmlu_data_dir,
        max_rows=args.mmlu_max_rows,
        include_verifier_targets=False,
    )
    mmlu_train, mmlu_val = split_examples(mmlu_all, validation_fraction=args.validation_fraction)
    train_raw["mmlu"] = mmlu_train
    eval_raw["mmlu"] = mmlu_val

    if args.olympiad_math_max_rows > 0:
        olympiad_all = build_olympiad_math_reasoning_examples(
            configs=args.olympiad_math_configs,
            max_rows=args.olympiad_math_max_rows,
            include_verifier_targets=False,
        )
        olympiad_train, olympiad_val = split_examples(olympiad_all, validation_fraction=args.validation_fraction)
        train_raw["olympiad_math"] = olympiad_train
        eval_raw["olympiad_math"] = olympiad_val

    if args.mmlu_pro_max_rows > 0:
        eval_raw["mmlu_pro"] = build_mmlu_pro_reasoning_examples(
            max_rows=args.mmlu_pro_max_rows,
            include_verifier_targets=False,
        )

    if args.mmlu_redux_max_rows > 0:
        eval_raw["mmlu_redux"] = build_mmlu_redux_reasoning_examples(
            max_rows=args.mmlu_redux_max_rows,
            include_verifier_targets=False,
            label_mode=args.mmlu_redux_label_mode,
        )

    if args.core_max_rows > 0:
        eval_raw["core"] = build_core_reasoning_examples(
            data_dir=args.core_data_dir,
            max_rows=args.core_max_rows,
        )

    train_examples: dict[str, tuple[ActionTextExample, ...]] = {}
    eval_examples: dict[str, tuple[ActionTextExample, ...]] = {}
    skipped_non_action: dict[str, int] = {}

    for benchmark, examples in train_raw.items():
        converted, skipped = convert_action_examples(examples)
        train_examples[benchmark] = converted
        skipped_non_action.update({f"train_{name}": count for name, count in skipped.items()})
    for benchmark, examples in eval_raw.items():
        converted, skipped = convert_action_examples(examples)
        eval_examples[benchmark] = converted
        skipped_non_action.update({f"eval_{name}": count for name, count in skipped.items()})

    return {
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "skipped_non_action": skipped_non_action,
    }


def merge_action_examples(
    *,
    arc_train: tuple[ActionTextExample, ...],
    arc_eval: tuple[ActionTextExample, ...],
    static_bundle: dict[str, object],
) -> tuple[dict[str, tuple[ActionTextExample, ...]], dict[str, tuple[ActionTextExample, ...]], dict[str, int]]:
    train_examples = dict(static_bundle["train_examples"])
    eval_examples = dict(static_bundle["eval_examples"])
    skipped_non_action = dict(static_bundle["skipped_non_action"])
    train_examples["arc"] = arc_train
    eval_examples["arc"] = arc_eval
    return train_examples, eval_examples, skipped_non_action


def build_output_vocabularies(
    examples: Sequence[ActionTextExample],
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, int]]]:
    actions_by_head: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        actions_by_head[output_head_for_benchmark(example.benchmark)].add(example.target_action)
    vocabularies = {
        head_name: tuple(sorted(actions))
        for head_name, actions in sorted(actions_by_head.items())
    }
    action_to_index = {
        head_name: {action: index for index, action in enumerate(vocabulary)}
        for head_name, vocabulary in vocabularies.items()
    }
    return vocabularies, action_to_index


def build_candidate_masks(
    examples: Sequence[ActionTextExample],
    *,
    action_to_index: dict[str, dict[str, int]],
) -> dict[str, np.ndarray]:
    bucket_to_indices: dict[str, set[int]] = defaultdict(set)
    for example in examples:
        head_name = output_head_for_benchmark(example.benchmark)
        bucket_key = f"{head_name}|{example.candidate_bucket}"
        bucket_to_indices[bucket_key].add(action_to_index[head_name][example.target_action])
    masks: dict[str, np.ndarray] = {}
    for bucket_key, indices in bucket_to_indices.items():
        head_name, _ = bucket_key.split("|", 1)
        mask = np.zeros(len(action_to_index[head_name]), dtype=bool)
        for index in indices:
            mask[index] = True
        masks[bucket_key] = mask
    return masks


def tokenize_examples(
    examples: Sequence[ActionTextExample],
    *,
    max_seq_len: int,
    action_to_index: dict[str, dict[str, int]],
    candidate_masks: dict[str, np.ndarray],
) -> tuple[TokenizedActionExample, ...]:
    tokenized: list[TokenizedActionExample] = []
    for example in examples:
        head_name = output_head_for_benchmark(example.benchmark)
        bucket_key = f"{head_name}|{example.candidate_bucket}"
        mask = candidate_masks.get(bucket_key)
        if mask is None:
            mask = np.ones(len(action_to_index[head_name]), dtype=bool)
        tokenized.append(
            TokenizedActionExample(
                benchmark=example.benchmark,
                output_head=head_name,
                trajectory_id=example.trajectory_id,
                token_ids=encode_text(example.input_text, max_seq_len=max_seq_len),
                target_action=example.target_action,
                target_index=action_to_index[head_name][example.target_action],
                candidate_mask=tuple(bool(flag) for flag in mask.tolist()),
            )
        )
    return tuple(tokenized)


def build_data_only_row(
    *,
    seed: int,
    train_examples: dict[str, tuple[ActionTextExample, ...]],
    eval_examples: dict[str, tuple[ActionTextExample, ...]],
    output_vocabularies: dict[str, tuple[str, ...]],
    candidate_masks: dict[str, np.ndarray],
    skipped_non_action: dict[str, int],
) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": seed,
        "train_example_count": sum(len(examples) for examples in train_examples.values()),
        "eval_example_count": sum(len(examples) for examples in eval_examples.values()),
        "benchmark_count": len({*train_examples.keys(), *eval_examples.keys()}),
        "output_head_count": len(output_vocabularies),
        "total_action_count": sum(len(vocabulary) for vocabulary in output_vocabularies.values()),
        "candidate_bucket_count": len(candidate_masks),
        "output_head_sizes_json": json.dumps(
            {head_name: len(vocabulary) for head_name, vocabulary in sorted(output_vocabularies.items())},
            sort_keys=True,
        ),
        "skipped_non_action_json": json.dumps(skipped_non_action, sort_keys=True),
    }
    for benchmark, examples in sorted(train_examples.items()):
        row[f"train_examples_{benchmark}"] = len(examples)
    for benchmark, examples in sorted(eval_examples.items()):
        row[f"eval_examples_{benchmark}"] = len(examples)
    return row


def build_decoder_config(train_config: TrainConfig) -> DecoderModelConfig:
    attention = AttentionBackendConfig.from_preset(
        train_config.attention_preset,
        latent_kv_dim=train_config.latent_kv_dim,
        dropout=train_config.attention_dropout,
    )
    return DecoderModelConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        max_position_embeddings=train_config.max_seq_len,
        hidden_size=train_config.hidden_size,
        num_hidden_layers=train_config.num_layers,
        num_attention_heads=train_config.num_heads,
        num_key_value_heads=train_config.num_kv_heads,
        intermediate_size=train_config.intermediate_size,
        attention=attention,
    )


def collate_examples(
    examples: Sequence[TokenizedActionExample],
    *,
    benchmark_to_id: dict[str, int],
    device: torch.device,
) -> CollatedBatch:
    batch_size = len(examples)
    max_len = max(len(example.token_ids) for example in examples)
    input_ids = torch.full((batch_size, max_len), EOS_TOKEN_ID, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    benchmark_ids = torch.empty(batch_size, dtype=torch.long, device=device)
    output_heads: list[str] = []
    targets: list[int] = []
    candidate_masks: list[tuple[bool, ...]] = []

    for index, example in enumerate(examples):
        token_ids = torch.as_tensor(example.token_ids, dtype=torch.long, device=device)
        input_ids[index, : token_ids.numel()] = token_ids
        attention_mask[index, : token_ids.numel()] = True
        benchmark_ids[index] = benchmark_to_id[example.benchmark]
        output_heads.append(example.output_head)
        targets.append(example.target_index)
        candidate_masks.append(example.candidate_mask)

    return CollatedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        benchmark_ids=benchmark_ids,
        output_heads=output_heads,
        targets=targets,
        candidate_masks=candidate_masks,
    )


def grouped_loss(
    *,
    model: BenchmarkLatentPolicy,
    pooled_states: torch.Tensor,
    batch: CollatedBatch,
) -> torch.Tensor:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, output_head in enumerate(batch.output_heads):
        grouped_indices[output_head].append(index)

    total_loss = pooled_states.new_zeros(())
    total_count = 0
    for output_head, indices in grouped_indices.items():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=pooled_states.device)
        logits = model.head_logits(pooled_states.index_select(0, index_tensor), output_head=output_head)
        targets = torch.as_tensor([batch.targets[index] for index in indices], dtype=torch.long, device=pooled_states.device)
        candidate_masks = torch.as_tensor(
            np.asarray([batch.candidate_masks[index] for index in indices], dtype=np.bool_),
            dtype=torch.bool,
            device=pooled_states.device,
        )
        valid_rows = torch.any(candidate_masks, dim=1)
        if not bool(torch.all(valid_rows)):
            candidate_masks = candidate_masks.clone()
            candidate_masks[~valid_rows] = True
        masked_logits = logits.masked_fill(~candidate_masks, LOGIT_FLOOR)
        total_loss = total_loss + F.cross_entropy(masked_logits, targets, reduction="sum")
        total_count += len(indices)
    return total_loss / max(total_count, 1)


def train_behavior_cloning(
    examples: Sequence[TokenizedActionExample],
    *,
    benchmark_names: Sequence[str],
    output_vocabularies: dict[str, tuple[str, ...]],
    train_config: TrainConfig,
    seed: int,
    device: torch.device,
    progress: bool = False,
    progress_desc: str | None = None,
) -> BenchmarkLatentPolicy:
    torch.manual_seed(seed)
    decoder_config = build_decoder_config(train_config)
    model = BenchmarkLatentPolicy(
        decoder_config,
        benchmark_names=benchmark_names,
        output_head_sizes={head_name: len(vocabulary) for head_name, vocabulary in output_vocabularies.items()},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    benchmark_to_id = model.benchmark_to_id
    epochs = range(train_config.epochs)
    if progress:
        epochs = tqdm(
            epochs,
            total=train_config.epochs,
            desc=progress_desc or "train mixed latent bc",
            leave=False,
            dynamic_ncols=True,
            disable=not progress_enabled(progress),
        )

    for _ in epochs:
        permutation = torch.randperm(len(examples), device=device).tolist()
        model.train()
        for start in range(0, len(examples), train_config.batch_size):
            batch_indices = permutation[start : start + train_config.batch_size]
            batch_examples = [examples[index] for index in batch_indices]
            batch = collate_examples(batch_examples, benchmark_to_id=benchmark_to_id, device=device)
            pooled_states = model.encode(
                batch.input_ids,
                benchmark_ids=batch.benchmark_ids,
                attention_mask=batch.attention_mask,
            )
            loss = grouped_loss(model=model, pooled_states=pooled_states, batch=batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
    return model


def predict_actions(
    *,
    model: BenchmarkLatentPolicy,
    batch: CollatedBatch,
    output_vocabularies: dict[str, tuple[str, ...]],
) -> list[str]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, output_head in enumerate(batch.output_heads):
        grouped_indices[output_head].append(index)

    pooled_states = model.encode(
        batch.input_ids,
        benchmark_ids=batch.benchmark_ids,
        attention_mask=batch.attention_mask,
    )
    predictions = [""] * len(batch.output_heads)
    for output_head, indices in grouped_indices.items():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=pooled_states.device)
        logits = model.head_logits(pooled_states.index_select(0, index_tensor), output_head=output_head)
        candidate_masks = torch.as_tensor(
            np.asarray([batch.candidate_masks[index] for index in indices], dtype=np.bool_),
            dtype=torch.bool,
            device=pooled_states.device,
        )
        valid_rows = torch.any(candidate_masks, dim=1)
        if not bool(torch.all(valid_rows)):
            candidate_masks = candidate_masks.clone()
            candidate_masks[~valid_rows] = True
        masked_logits = logits.masked_fill(~candidate_masks, LOGIT_FLOOR)
        predicted_indices = torch.argmax(masked_logits, dim=-1).tolist()
        vocabulary = output_vocabularies[output_head]
        for local_index, predicted_index in enumerate(predicted_indices):
            predictions[indices[local_index]] = vocabulary[int(predicted_index)]
    return predictions


@torch.inference_mode()
def evaluate_examples(
    examples_by_benchmark: dict[str, tuple[TokenizedActionExample, ...]],
    *,
    model: BenchmarkLatentPolicy,
    output_vocabularies: dict[str, tuple[str, ...]],
    train_action_set: set[tuple[str, str]],
    eval_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    benchmark_to_id = model.benchmark_to_id
    benchmark_action_accuracy: dict[str, float] = {}
    benchmark_trace_success: dict[str, float] = {}
    benchmark_seen_action_rate: dict[str, float] = {}
    total_correct = 0
    total_examples = 0
    total_trace_success = 0
    total_trace_count = 0
    total_seen = 0

    model.eval()
    for benchmark, examples in sorted(examples_by_benchmark.items()):
        if not examples:
            benchmark_action_accuracy[benchmark] = float("nan")
            benchmark_trace_success[benchmark] = float("nan")
            benchmark_seen_action_rate[benchmark] = float("nan")
            continue

        correct_flags: list[bool] = []
        trajectory_correct: dict[str, list[bool]] = defaultdict(list)
        seen_count = 0

        for start in range(0, len(examples), eval_batch_size):
            batch_examples = examples[start : start + eval_batch_size]
            batch = collate_examples(batch_examples, benchmark_to_id=benchmark_to_id, device=device)
            predicted_actions = predict_actions(
                model=model,
                batch=batch,
                output_vocabularies=output_vocabularies,
            )
            for example, predicted_action in zip(batch_examples, predicted_actions, strict=True):
                is_correct = predicted_action == example.target_action
                correct_flags.append(is_correct)
                trajectory_correct[example.trajectory_id].append(is_correct)
                seen_count += int((example.output_head, example.target_action) in train_action_set)

        action_accuracy = float(np.mean(np.asarray(correct_flags, dtype=np.float64)))
        trace_success = float(
            np.mean(np.asarray([all(flags) for flags in trajectory_correct.values()], dtype=np.float64))
        )
        benchmark_action_accuracy[benchmark] = action_accuracy
        benchmark_trace_success[benchmark] = trace_success
        benchmark_seen_action_rate[benchmark] = float(seen_count / len(examples))
        total_correct += sum(int(flag) for flag in correct_flags)
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


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = auto_device(args.device)
    configure_runtime(device)
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        latent_kv_dim=args.latent_kv_dim,
        attention_preset=args.attention_preset,
        attention_dropout=args.attention_dropout,
    )

    static_bundle = build_static_action_bundle(args)
    raw_rows: list[dict[str, object]] = []
    data_only_rows: list[dict[str, object]] = []
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
        seed_value = args.seed + trial * 10_000
        arc_train = build_arc_action_examples(
            num_episodes=args.arc_train_episodes,
            seed_start=seed_value,
            include_verifier_targets=args.include_verifier_targets,
        )
        arc_eval = build_arc_action_examples(
            num_episodes=args.arc_val_episodes,
            seed_start=seed_value + 100_000,
            include_verifier_targets=args.include_verifier_targets,
        )
        train_examples, eval_examples, skipped_non_action = merge_action_examples(
            arc_train=arc_train,
            arc_eval=arc_eval,
            static_bundle=static_bundle,
        )
        merged_train = flatten_examples(train_examples)
        merged_eval = flatten_examples(eval_examples)
        output_vocabularies, action_to_index = build_output_vocabularies(merged_train + merged_eval)
        candidate_masks = build_candidate_masks(merged_train + merged_eval, action_to_index=action_to_index)

        if args.data_only:
            data_only_rows.append(
                build_data_only_row(
                    seed=seed_value,
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    output_vocabularies=output_vocabularies,
                    candidate_masks=candidate_masks,
                    skipped_non_action=skipped_non_action,
                )
            )
            continue

        tokenized_train = tokenize_examples(
            merged_train,
            max_seq_len=train_config.max_seq_len,
            action_to_index=action_to_index,
            candidate_masks=candidate_masks,
        )
        tokenized_eval_by_benchmark = {
            benchmark: tokenize_examples(
                examples,
                max_seq_len=train_config.max_seq_len,
                action_to_index=action_to_index,
                candidate_masks=candidate_masks,
            )
            for benchmark, examples in sorted(eval_examples.items())
        }
        benchmark_names = sorted({example.benchmark for example in tokenized_train} | {example.benchmark for examples in tokenized_eval_by_benchmark.values() for example in examples})
        model = train_behavior_cloning(
            tokenized_train,
            benchmark_names=benchmark_names,
            output_vocabularies=output_vocabularies,
            train_config=train_config,
            seed=seed_value,
            device=device,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} latent bc",
        )
        metrics = evaluate_examples(
            tokenized_eval_by_benchmark,
            model=model,
            output_vocabularies=output_vocabularies,
            train_action_set={(example.output_head, example.target_action) for example in tokenized_train},
            eval_batch_size=train_config.eval_batch_size,
            device=device,
        )
        raw_rows.append(
            {
                "seed": seed_value,
                "train_example_count": len(tokenized_train),
                "eval_example_count": sum(len(examples) for examples in tokenized_eval_by_benchmark.values()),
                "benchmark_count": len(benchmark_names),
                "output_head_count": len(output_vocabularies),
                "total_action_count": sum(len(vocabulary) for vocabulary in output_vocabularies.values()),
                "candidate_bucket_count": len(candidate_masks),
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
                "device": device.type,
                "train_config": {
                    **train_config.__dict__,
                    "olympiad_math_configs": list(args.olympiad_math_configs),
                },
                "summary_rows": summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed trace behavior-cloning benchmark using the shared multi-head latent attention stack with per-benchmark output heads."
    )
    parser.add_argument("--arc-train-episodes", type=int, default=16)
    parser.add_argument("--arc-val-episodes", type=int, default=4)
    parser.add_argument("--gsm8k-max-rows", type=int, default=48)
    parser.add_argument("--mmlu-max-rows", type=int, default=48)
    parser.add_argument("--mmlu-pro-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-label-mode", choices=("original", "corrected_single"), default="corrected_single")
    parser.add_argument("--olympiad-math-max-rows", type=int, default=0)
    parser.add_argument(
        "--olympiad-math-configs",
        nargs="+",
        choices=("en-easy", "en-hard", "zh-easy", "zh-hard"),
        default=("en-easy", "en-hard"),
    )
    parser.add_argument("--core-max-rows", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--latent-kv-dim", type=int, default=48)
    parser.add_argument("--attention-preset", choices=("mla_default", "mla_sia_prefill_l1"), default="mla_default")
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--gsm8k-data-dir", type=str, default="arc_trajectory_sampler/data/gsm8k")
    parser.add_argument("--mmlu-data-dir", type=str, default="arc_trajectory_sampler/data/mmlu")
    parser.add_argument("--core-data-dir", type=str, default="arc_trajectory_sampler/data/core")
    parser.add_argument("--include-verifier-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="results/mixed_trace_latent_attention")
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
