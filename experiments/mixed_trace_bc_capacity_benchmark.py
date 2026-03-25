from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

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

from experiments.arc_sampler_fqi_benchmark import save_csv
from experiments.arc_sampler_fqi_capacity_benchmark import auto_device, configure_runtime, progress_enabled
from experiments.mixed_trace_bc_benchmark import (
    HashedTextEncoder,
    action_vocabulary,
    build_data_only_payload,
    build_data_only_row,
    build_dataset_bundle,
    build_training_batch,
    candidate_masks,
    evaluate_examples,
    flatten_examples,
    plot_summary,
)


LOGIT_FLOOR = -1e9


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    feature_dim: int = 4096
    grad_clip: float = 5.0
    hidden_dims: tuple[int, ...] = (1024, 512)
    dropout: float = 0.0


class MLPTextPolicyNetwork(nn.Module):
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
) -> MLPTextPolicyNetwork:
    torch.manual_seed(seed)
    model = MLPTextPolicyNetwork(
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
            desc=progress_desc or "train mixed bc-mlp",
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
            valid_rows = torch.any(mask_batch, dim=1)
            effective_mask = mask_batch.clone()
            effective_mask[~valid_rows] = True
            masked_logits = logits.masked_fill(~effective_mask, LOGIT_FLOOR)
            loss = F.cross_entropy(masked_logits, action_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
    return model


def aggregate_numeric(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    summary: dict[str, object] = {}
    for metric_name in rows[0]:
        if metric_name in {"seed", "device", "model_hidden_dims"}:
            continue
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            summary[metric_name] = float("nan")
            summary[f"stderr_{metric_name}"] = float("nan")
            continue
        summary[metric_name] = float(valid.mean())
        summary[f"stderr_{metric_name}"] = float(valid.std(ddof=0) / max(len(valid), 1) ** 0.5)
    return [summary]


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = auto_device(args.device)
    configure_runtime(device)
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        feature_dim=args.feature_dim,
        grad_clip=args.grad_clip,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )

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
                    seed=seed_value,
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
        tensor_batch = make_tensor_batch(batch, device=device)
        model = train_behavior_cloning(
            tensor_batch,
            num_actions=len(vocabulary),
            train_config=train_config,
            seed=seed_value,
            device=device,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} mixed bc-mlp",
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
                "seed": seed_value,
                "train_example_count": len(merged_train),
                "eval_example_count": len(merged_eval),
                "action_vocab_size": len(vocabulary),
                "candidate_bucket_count": len(bucket_masks),
                "model_hidden_dims": "-".join(str(dim) for dim in train_config.hidden_dims),
                "device": device.type,
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

    summary_rows = aggregate_numeric(raw_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    if summary_rows:
        plot_summary(summary_rows[0], output_dir / "comparison.png")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": args.seeds,
                "train_config": {
                    **train_config.__dict__,
                    "hidden_dims": list(train_config.hidden_dims),
                },
                "device": device.type,
                "summary_rows": summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Higher-capacity mixed trace behavior-cloning benchmark over ARC, GSM8K, and MMLU-style reasoning traces."
    )
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 512])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gsm8k-data-dir", type=str, default="arc_trajectory_sampler/data/gsm8k")
    parser.add_argument("--mmlu-data-dir", type=str, default="arc_trajectory_sampler/data/mmlu")
    parser.add_argument("--core-data-dir", type=str, default="arc_trajectory_sampler/data/core")
    parser.add_argument("--include-verifier-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default="results/mixed_trace_bc_capacity")
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
