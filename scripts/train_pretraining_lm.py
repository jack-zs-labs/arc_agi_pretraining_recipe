from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None

from models import AttentionBackendConfig, DecoderLanguageModel, DecoderModelConfig, InferenceBudget, MoEConfig
from training.packed_lm_dataset import PackedSequenceDataset, read_packed_manifest
from training.distributed_sampler import DeterministicDistributedBatchSampler
from training.prefetch_loader import AsyncPackedBatchLoader
from training.runtime import (
    autocast_context,
    destroy_distributed,
    distributed_barrier,
    initialize_distributed,
    precision_mode,
    require_torch,
    resolve_runtime_device,
    set_global_seed,
    set_step_seed,
    synchronize_device,
)
from training.sharding import (
    SHARDING_CHOICES,
    build_data_parallel_model,
    is_fsdp_model,
    load_optimizer_state,
    model_state_load_context,
    model_state_save_context,
    optimizer_state_for_save,
    sharding_context,
    unwrap_sharded_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure autoregressive LM from packed token shards.")
    parser.add_argument("--packed-manifest", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-rank micro-batch size.")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=0, help="Run validation every N optimizer steps; 0 means final-only.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Cap validation batches; 0 means full validation split.")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--progress-output", type=str, default="")
    parser.add_argument("--csv-output", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--sharding-strategy", choices=SHARDING_CHOICES, default="ddp")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--architecture", choices=("dense", "moe"), default="dense")
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--latent-kv-dim", type=int, default=48)
    parser.add_argument("--attention-preset", choices=("mla_default", "mla_sia_prefill_l1"), default="mla_default")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--router-jitter-noise", type=float, default=0.0)
    parser.add_argument("--moe-auxiliary-loss-weight", type=float, default=1e-2)
    parser.add_argument("--hf-upload-mode", choices=("disabled", "best_effort", "required"), default="")
    parser.add_argument("--hf-upload-repo-id", type=str, default="")
    parser.add_argument("--hf-upload-repo-type", choices=("model", "dataset", "space"), default="")
    parser.add_argument("--hf-upload-token", type=str, default="")
    parser.add_argument("--hf-upload-path-prefix", type=str, default="")
    parser.add_argument("--hf-upload-private", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def write_json(path: str | Path | None, payload: dict[str, object]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path | None, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_hf_upload_config(args: argparse.Namespace) -> dict[str, object] | None:
    mode = str(args.hf_upload_mode or os.environ.get("HF_UPLOAD_MODE", "")).strip().lower() or "disabled"
    repo_id = str(args.hf_upload_repo_id or os.environ.get("HF_UPLOAD_REPO_ID", "")).strip()
    if mode == "disabled" or not repo_id:
        return None
    repo_type = str(args.hf_upload_repo_type or os.environ.get("HF_UPLOAD_REPO_TYPE", "model")).strip() or "model"
    token = str(
        args.hf_upload_token
        or os.environ.get("HF_UPLOAD_TOKEN", "")
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    ).strip()
    private: bool | None = args.hf_upload_private
    if private is None:
        private = _env_bool("HF_UPLOAD_PRIVATE", False)
    path_prefix = str(args.hf_upload_path_prefix or os.environ.get("HF_UPLOAD_PATH_PREFIX", "")).strip().strip("/")
    if not path_prefix:
        for candidate in (args.checkpoint_dir, args.output, args.progress_output):
            if candidate:
                path_prefix = Path(str(candidate)).resolve().parent.name
                break
    if not path_prefix:
        path_prefix = f"run_seed_{int(args.seed)}"
    return {
        "mode": mode,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "token": token,
        "private": private,
        "path_prefix": path_prefix,
    }


def _require_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Hugging Face checkpoint upload was enabled, but `huggingface_hub` is not installed. "
            "Reinstall the repo requirements before launching the paid run."
        ) from exc
    return HfApi


def _hf_join(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _hf_upload_bytes(*, api, payload: bytes, path_in_repo: str, repo_id: str, repo_type: str, commit_message: str) -> None:
    api.upload_file(
        path_or_fileobj=io.BytesIO(payload),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )


def upload_checkpoint_artifacts(
    *,
    config: dict[str, object] | None,
    checkpoint_path: Path | None,
    checkpoint_dir: Path | None,
    step_index: int,
    progress_output: str,
) -> None:
    if config is None or checkpoint_path is None:
        return
    HfApi = _require_hf_api()
    token = str(config["token"]).strip() or None
    api = HfApi(token=token)
    repo_id = str(config["repo_id"])
    repo_type = str(config["repo_type"])
    path_prefix = str(config["path_prefix"])
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=bool(config["private"]), exist_ok=True)

    checkpoint_name = checkpoint_path.name
    checkpoint_repo_path = _hf_join(path_prefix, "checkpoints", checkpoint_name)
    commit_message = f"Upload checkpoint at step {step_index}"
    if checkpoint_path.is_dir():
        api.upload_folder(
            folder_path=str(checkpoint_path),
            path_in_repo=checkpoint_repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
    else:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_repo_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )

    latest_pointer_repo_path = _hf_join(path_prefix, "checkpoints", "latest_step.txt")
    _hf_upload_bytes(
        api=api,
        payload=(checkpoint_name + "\n").encode("utf-8"),
        path_in_repo=latest_pointer_repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Update latest checkpoint pointer at step {step_index}",
    )
    if progress_output:
        progress_path = Path(progress_output)
        if progress_path.exists():
            api.upload_file(
                path_or_fileobj=str(progress_path),
                path_in_repo=_hf_join(path_prefix, progress_path.name),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Upload progress snapshot at step {step_index}",
            )


def upload_final_artifacts(
    *,
    config: dict[str, object] | None,
    output_path: str,
    progress_output: str,
    csv_output: str,
) -> None:
    if config is None:
        return
    HfApi = _require_hf_api()
    token = str(config["token"]).strip() or None
    api = HfApi(token=token)
    repo_id = str(config["repo_id"])
    repo_type = str(config["repo_type"])
    path_prefix = str(config["path_prefix"])
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=bool(config["private"]), exist_ok=True)
    for local_path_str in (output_path, progress_output, csv_output):
        if not local_path_str:
            continue
        local_path = Path(local_path_str)
        if not local_path.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=_hf_join(path_prefix, local_path.name),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload final artifact {local_path.name}",
        )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build_optimizer_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    resolved_total_steps = max(int(total_steps), 1)
    resolved_warmup_steps = max(0, min(int(warmup_steps), resolved_total_steps - 1))
    resolved_min_scale = float(max(0.0, min(min_learning_rate_scale, 1.0)))

    def lr_lambda(step: int) -> float:
        if resolved_warmup_steps > 0 and step < resolved_warmup_steps:
            return float(step + 1) / float(resolved_warmup_steps)
        if resolved_total_steps <= resolved_warmup_steps:
            return 1.0
        decay_progress = (step - resolved_warmup_steps) / float(max(resolved_total_steps - resolved_warmup_steps, 1))
        decay_progress = max(0.0, min(decay_progress, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return resolved_min_scale + ((1.0 - resolved_min_scale) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_model_config(args: argparse.Namespace, *, vocab_size: int, seq_len: int) -> DecoderModelConfig:
    attention = AttentionBackendConfig.from_preset(
        args.attention_preset,
        latent_kv_dim=args.latent_kv_dim,
    )
    moe = MoEConfig()
    if args.architecture == "moe":
        moe = MoEConfig.reference(
            num_experts=args.num_experts,
            experts_per_token=args.experts_per_token,
            router_jitter_noise=args.router_jitter_noise,
            auxiliary_loss_weight=args.moe_auxiliary_loss_weight,
        )
    return DecoderModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max(seq_len + 1, 4096),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        attention=attention,
        moe=moe,
    )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def latest_checkpoint_pointer(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "latest.txt"


def rank_checkpoint_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / f"rank_{rank:05d}.pt"


def resolve_resume_checkpoint_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_dir():
        pointer = latest_checkpoint_pointer(candidate)
        if pointer.exists():
            resolved = pointer.read_text(encoding="utf-8").strip()
            if not resolved:
                raise SystemExit(f"Checkpoint pointer is empty: {pointer}")
            return Path(resolved)
        checkpoints = sorted(candidate.glob("step_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        raise SystemExit(f"No checkpoints found under {candidate}")
    return candidate


def save_checkpoint(
    *,
    checkpoint_dir: Path | None,
    distributed_context,
    step_index: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config: DecoderModelConfig,
    packed_manifest: dict[str, object],
    train_state: dict[str, object],
    args: argparse.Namespace,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = f"step_{step_index:08d}"
    if is_fsdp_model(model):
        checkpoint_path = checkpoint_dir / checkpoint_stem
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = rank_checkpoint_path(checkpoint_path, distributed_context.rank)
    else:
        checkpoint_path = checkpoint_dir / f"{checkpoint_stem}.pt"
        checkpoint_file = checkpoint_path
    with model_state_save_context(model):
        model_state = model.state_dict() if is_fsdp_model(model) else unwrap_sharded_model(model).state_dict()
        optimizer_state = optimizer_state_for_save(model, optimizer)
    torch.save(
        {
            "format_version": 2,
            "checkpoint_format": "fsdp_local" if is_fsdp_model(model) else "full",
            "rank": distributed_context.rank,
            "world_size": distributed_context.world_size,
            "step_index": step_index,
            "seed": args.seed,
            "architecture": args.architecture,
            "sharding_strategy": args.sharding_strategy,
            "activation_checkpointing": bool(args.activation_checkpointing),
            "packed_manifest_path": str(Path(args.packed_manifest).resolve()),
            "packed_manifest": packed_manifest,
            "model_config": asdict(model_config),
            "train_state": train_state,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler.state_dict(),
            "args_snapshot": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "grad_accumulation_steps": args.grad_accumulation_steps,
                "precision": args.precision,
                "learning_rate": args.learning_rate,
            },
        },
        checkpoint_file,
    )
    if distributed_context.is_primary:
        latest_checkpoint_pointer(checkpoint_dir).write_text(str(checkpoint_path.resolve()), encoding="utf-8")
        return checkpoint_path
    return None


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    distributed_context,
    fsdp: bool,
) -> dict[str, object]:
    checkpoint_path = path
    if fsdp and path.is_dir():
        checkpoint_path = rank_checkpoint_path(path, distributed_context.rank)
        if not checkpoint_path.exists():
            raise SystemExit(
                f"Missing FSDP rank-local checkpoint shard for rank {distributed_context.rank}: {checkpoint_path}"
            )
    return torch.load(checkpoint_path, map_location=device)


def write_progress(
    *,
    args: argparse.Namespace,
    step_index: int,
    total_steps: int,
    elapsed_s: float,
    train_loss_mean: float,
    train_aux_loss_mean: float,
    consumed_samples: int,
    latest_val_loss: float | None,
    data_wait_s: float | None = None,
) -> None:
    payload: dict[str, object] = {
        "mode": "packed_pretraining_lm_progress",
        "step": step_index,
        "total_steps": total_steps,
        "elapsed_s": elapsed_s,
        "train_loss_mean": train_loss_mean,
        "train_aux_loss_mean": train_aux_loss_mean,
        "consumed_samples": consumed_samples,
        "updated_at_unix_s": time.time(),
    }
    if latest_val_loss is not None:
        payload["latest_val_loss"] = latest_val_loss
    if data_wait_s is not None:
        payload["data_wait_s"] = data_wait_s
    write_json(args.progress_output, payload)


def evaluate_loss(
    *,
    model: torch.nn.Module,
    dataset: PackedSequenceDataset,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    budget: InferenceBudget,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        total_batches = math.ceil(len(dataset) / batch_size)
        if max_batches > 0:
            total_batches = min(total_batches, max_batches)
        for batch_index in range(total_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(dataset))
            indices = list(range(start, end))
            inputs_np, targets_np = dataset.batch_input_target_pair(indices)
            inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.long)
            targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
            outputs = model(inputs, budget=budget)
            losses.append(float(cross_entropy_loss(outputs.logits, targets).item()))
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    require_torch()
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive.")
    distributed_context = initialize_distributed(args.device)
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None
    hf_upload_config = resolve_hf_upload_config(args)
    packed_manifest = read_packed_manifest(args.packed_manifest)
    train_dataset = PackedSequenceDataset(args.packed_manifest, split="train")
    val_dataset = PackedSequenceDataset(args.packed_manifest, split="val")
    if len(train_dataset) == 0:
        raise SystemExit("Packed train split is empty.")
    device = resolve_runtime_device(args.device, distributed_context)
    precision = precision_mode(args.precision, device)
    sharding = sharding_context(
        args.sharding_strategy,
        activation_checkpointing=args.activation_checkpointing,
    )
    seq_len = int(packed_manifest["seq_len"])
    tokenizer_summary = dict(packed_manifest.get("tokenizer", {}))

    set_global_seed(args.seed)
    model_config = build_model_config(args, vocab_size=int(tokenizer_summary["vocab_size"]), seq_len=seq_len)
    base_model = DecoderLanguageModel(model_config).to(device)
    model: torch.nn.Module = build_data_parallel_model(
        base_model=base_model,
        distributed_context=distributed_context,
        device=device,
        precision=precision,
        sharding=sharding,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_optimizer_scheduler(
        optimizer,
        total_steps=args.steps,
        warmup_steps=args.lr_warmup_steps,
        min_learning_rate_scale=args.min_learning_rate_scale,
    )
    train_budget = InferenceBudget(
        max_prompt_tokens=seq_len,
        max_new_tokens=0,
        active_layers=model_config.num_hidden_layers,
        use_kv_cache=False,
    )
    eval_budget = InferenceBudget(
        max_prompt_tokens=seq_len,
        max_new_tokens=0,
        active_layers=model_config.num_hidden_layers,
        use_kv_cache=False,
    )
    grad_accumulation_steps = max(int(args.grad_accumulation_steps), 1)
    checkpoint_every = max(int(args.checkpoint_every), 1)
    eval_every = max(int(args.eval_every), 0)

    train_sampler = DeterministicDistributedBatchSampler(
        dataset_size=len(train_dataset),
        per_rank_batch_size=args.batch_size,
        world_size=distributed_context.world_size,
        rank=distributed_context.rank,
        seed=args.seed,
        shuffle=True,
    )
    global_samples_per_step = train_sampler.step_sample_count(grad_accumulation_steps=grad_accumulation_steps)

    start_step = 0
    consumed_samples = 0
    elapsed_offset = 0.0
    train_loss_total = 0.0
    train_aux_loss_total = 0.0
    train_tokens = 0
    data_wait_s_total = 0.0
    last_val_loss: float | None = None
    eval_history: list[dict[str, object]] = []

    if args.resume_from:
        resume_checkpoint = resolve_resume_checkpoint_path(args.resume_from)
        resume_payload = load_checkpoint(
            resume_checkpoint,
            device=torch.device("cpu"),
            distributed_context=distributed_context,
            fsdp=is_fsdp_model(model),
        )
        with model_state_load_context(model):
            if is_fsdp_model(model):
                model.load_state_dict(resume_payload["model_state"])
            else:
                unwrap_sharded_model(model).load_state_dict(resume_payload["model_state"])
        load_optimizer_state(model, optimizer, resume_payload["optimizer_state"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        train_state = dict(resume_payload.get("train_state", {}))
        start_step = int(resume_payload.get("step_index", 0))
        consumed_samples = int(train_state.get("consumed_samples", 0))
        elapsed_offset = float(train_state.get("elapsed_s_total", 0.0))
        train_loss_total = float(train_state.get("train_loss_total", 0.0))
        train_aux_loss_total = float(train_state.get("train_aux_loss_total", 0.0))
        train_tokens = int(train_state.get("train_tokens", 0))
        data_wait_s_total = float(train_state.get("data_wait_s_total", 0.0))
        last_val_loss = train_state.get("last_val_loss")
        last_val_loss = float(last_val_loss) if last_val_loss is not None else None
        eval_history = list(train_state.get("eval_history", []))

    start_time = time.perf_counter()
    model.train()
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    total_micro_batches = max(args.steps - start_step, 0) * grad_accumulation_steps
    with AsyncPackedBatchLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        start_consumed_samples=consumed_samples,
        total_batches=total_micro_batches,
        prefetch_batches=args.prefetch_batches,
        pin_memory=pin_memory,
    ) as train_loader:
        train_batch_iterator = iter(train_loader)
        for step_index in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            for micro_step_index in range(grad_accumulation_steps):
                batch_wait_start = time.perf_counter()
                batch = next(train_batch_iterator)
                data_wait_s_total += max(time.perf_counter() - batch_wait_start, 0.0)
                inputs, targets = batch.to(device, non_blocking=pin_memory)
                set_step_seed(
                    seed=args.seed,
                    distributed_context=distributed_context,
                    step_index=step_index,
                    micro_step_index=micro_step_index,
                )
                sync_context = (
                    model.no_sync()
                    if distributed_context.enabled and micro_step_index + 1 < grad_accumulation_steps
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context(device, precision):
                        outputs = model(inputs, budget=train_budget)
                        main_loss = cross_entropy_loss(outputs.logits, targets)
                        aux_loss = outputs.auxiliary_loss if outputs.auxiliary_loss is not None else main_loss.new_zeros(())
                        total_loss = (main_loss + aux_loss) / grad_accumulation_steps
                    total_loss.backward()
                train_loss_total += float(main_loss.detach().item())
                train_aux_loss_total += float(aux_loss.detach().item())
                train_tokens += int(inputs.numel()) * distributed_context.world_size

            if args.max_grad_norm > 0.0:
                if is_fsdp_model(model):
                    model.clip_grad_norm_(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            consumed_samples += global_samples_per_step

            should_eval = (eval_every > 0 and step_index % eval_every == 0) or step_index == args.steps
            if should_eval:
                distributed_barrier(distributed_context)
                synchronize_device(device)
                elapsed_eval = elapsed_offset + max(time.perf_counter() - start_time, 1e-9)
                if distributed_context.is_primary:
                    pass
                val_loss = evaluate_loss(
                    model=model,
                    dataset=val_dataset,
                    batch_size=max(1, args.eval_batch_size),
                    max_batches=max(0, args.max_eval_batches),
                    device=device,
                    budget=eval_budget,
                )
                if distributed_context.is_primary:
                    last_val_loss = val_loss
                    eval_history.append(
                        {
                            "step": step_index,
                            "val_loss": val_loss,
                            "elapsed_s": elapsed_eval,
                        }
                    )
                    write_progress(
                        args=args,
                        step_index=step_index,
                        total_steps=args.steps,
                        elapsed_s=elapsed_eval,
                        train_loss_mean=train_loss_total / max(step_index * grad_accumulation_steps, 1),
                        train_aux_loss_mean=train_aux_loss_total / max(step_index * grad_accumulation_steps, 1),
                        consumed_samples=consumed_samples,
                        latest_val_loss=last_val_loss,
                        data_wait_s=data_wait_s_total,
                    )
                distributed_barrier(distributed_context)
                model.train()

            if step_index % checkpoint_every == 0 or step_index == args.steps:
                elapsed_so_far = elapsed_offset + max(time.perf_counter() - start_time, 1e-9)
                if distributed_context.is_primary:
                    write_progress(
                        args=args,
                        step_index=step_index,
                        total_steps=args.steps,
                        elapsed_s=elapsed_so_far,
                        train_loss_mean=train_loss_total / max(step_index * grad_accumulation_steps, 1),
                        train_aux_loss_mean=train_aux_loss_total / max(step_index * grad_accumulation_steps, 1),
                        consumed_samples=consumed_samples,
                        latest_val_loss=last_val_loss,
                        data_wait_s=data_wait_s_total,
                    )
                saved_checkpoint_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    distributed_context=distributed_context,
                    step_index=step_index,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_config=model_config,
                    packed_manifest=packed_manifest,
                    train_state={
                        "elapsed_s_total": elapsed_so_far,
                        "consumed_samples": consumed_samples,
                        "train_loss_total": train_loss_total,
                        "train_aux_loss_total": train_aux_loss_total,
                        "train_tokens": train_tokens,
                        "data_wait_s_total": data_wait_s_total,
                        "last_val_loss": last_val_loss,
                        "eval_history": eval_history,
                    },
                    args=args,
                )
                distributed_barrier(distributed_context)
                if distributed_context.is_primary and saved_checkpoint_path is not None and hf_upload_config is not None:
                    try:
                        upload_checkpoint_artifacts(
                            config=hf_upload_config,
                            checkpoint_path=saved_checkpoint_path,
                            checkpoint_dir=checkpoint_dir,
                            step_index=step_index,
                            progress_output=args.progress_output,
                        )
                    except Exception as exc:
                        if str(hf_upload_config.get("mode", "disabled")) == "required":
                            raise
                        print(f"[hf-upload] checkpoint upload failed at step {step_index}: {exc}", file=sys.stderr)

    synchronize_device(device)
    distributed_barrier(distributed_context)
    elapsed_total = elapsed_offset + max(time.perf_counter() - start_time, 1e-9)

    if last_val_loss is None:
        last_val_loss_candidate = evaluate_loss(
            model=model,
            dataset=val_dataset,
            batch_size=max(1, args.eval_batch_size),
            max_batches=max(0, args.max_eval_batches),
            device=device,
            budget=eval_budget,
        )
        if distributed_context.is_primary:
            last_val_loss = last_val_loss_candidate
            eval_history.append(
                {
                    "step": args.steps,
                    "val_loss": last_val_loss,
                    "elapsed_s": elapsed_total,
                }
            )
        distributed_barrier(distributed_context)

    if distributed_context.is_primary:
        parameter_count = count_parameters(unwrap_sharded_model(model))
        payload = {
            "mode": "packed_pretraining_lm",
            "packed_manifest": str(Path(args.packed_manifest).resolve()),
            "document_manifest_source_name": packed_manifest.get("document_manifest_source_name"),
            "document_manifest_corpus_name": packed_manifest.get("document_manifest_corpus_name"),
            "document_manifest_corpus_counts": dict(packed_manifest.get("document_manifest_corpus_counts", {})),
            "document_manifest_band_counts": dict(packed_manifest.get("document_manifest_band_counts", {})),
            "document_manifest_holdout_counts": dict(packed_manifest.get("document_manifest_holdout_counts", {})),
            "document_manifest_selection_parameters": dict(packed_manifest.get("document_manifest_selection_parameters", {})),
            "document_manifest_extra_metadata": dict(packed_manifest.get("document_manifest_extra_metadata", {})),
            "architecture": args.architecture,
            "distributed": {
                "enabled": distributed_context.enabled,
                "world_size": distributed_context.world_size,
                "backend": distributed_context.backend,
            },
            "sharding": {
                "strategy": sharding.strategy,
                "activation_checkpointing": sharding.activation_checkpointing,
                "fsdp": sharding.is_fsdp,
                "checkpoint_format": "fsdp_local" if sharding.is_fsdp else "full",
            },
            "device": str(device),
            "precision": precision,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": grad_accumulation_steps,
            "global_batch_size": train_sampler.global_batch_size,
            "global_samples_per_step": global_samples_per_step,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
            "prefetch_batches": args.prefetch_batches,
            "pin_memory": pin_memory,
            "hf_upload": {
                "enabled": hf_upload_config is not None,
                "mode": "" if hf_upload_config is None else hf_upload_config.get("mode", ""),
                "repo_id": "" if hf_upload_config is None else hf_upload_config.get("repo_id", ""),
                "repo_type": "" if hf_upload_config is None else hf_upload_config.get("repo_type", ""),
                "path_prefix": "" if hf_upload_config is None else hf_upload_config.get("path_prefix", ""),
            },
            "seq_len": seq_len,
            "tokenizer": tokenizer_summary,
            "model_config": asdict(model_config),
            "parameter_count": parameter_count,
            "train_sequence_count": len(train_dataset),
            "val_sequence_count": len(val_dataset),
            "train_corpus_document_counts": dict(packed_manifest.get("splits", {}).get("train", {}).get("corpus_document_counts", {})),
            "val_corpus_document_counts": dict(packed_manifest.get("splits", {}).get("val", {}).get("corpus_document_counts", {})),
            "train_band_document_counts": dict(packed_manifest.get("splits", {}).get("train", {}).get("band_document_counts", {})),
            "val_band_document_counts": dict(packed_manifest.get("splits", {}).get("val", {}).get("band_document_counts", {})),
            "train_corpus_document_token_counts": dict(
                packed_manifest.get("splits", {}).get("train", {}).get("corpus_document_token_counts", {})
            ),
            "val_corpus_document_token_counts": dict(
                packed_manifest.get("splits", {}).get("val", {}).get("corpus_document_token_counts", {})
            ),
            "consumed_samples": consumed_samples,
            "train_tokens": train_tokens,
            "data_wait_s_total": data_wait_s_total,
            "elapsed_s": elapsed_total,
            "train_loss_mean": train_loss_total / max(args.steps * grad_accumulation_steps, 1),
            "train_aux_loss_mean": train_aux_loss_total / max(args.steps * grad_accumulation_steps, 1),
            "val_loss": last_val_loss,
            "eval_history": eval_history,
            "throughput": {
                "train_tokens_per_s": train_tokens / max(elapsed_total, 1e-9),
                "optimizer_steps_per_s": args.steps / max(elapsed_total, 1e-9),
                "data_wait_fraction": data_wait_s_total / max(elapsed_total, 1e-9),
            },
        }
        write_json(args.output, payload)
        write_csv(
            args.csv_output,
            [
                {
                    "mode": payload["mode"],
                    "architecture": payload["architecture"],
                    "precision": payload["precision"],
                    "steps": payload["steps"],
                    "batch_size": payload["batch_size"],
                    "grad_accumulation_steps": payload["grad_accumulation_steps"],
                    "global_batch_size": payload["global_batch_size"],
                    "parameter_count": payload["parameter_count"],
                    "train_sequence_count": payload["train_sequence_count"],
                    "val_sequence_count": payload["val_sequence_count"],
                    "consumed_samples": payload["consumed_samples"],
                    "train_tokens": payload["train_tokens"],
                    "elapsed_s": payload["elapsed_s"],
                    "train_loss_mean": payload["train_loss_mean"],
                    "train_aux_loss_mean": payload["train_aux_loss_mean"],
                    "val_loss": payload["val_loss"],
                    "train_tokens_per_s": payload["throughput"]["train_tokens_per_s"],
                }
            ],
        )
        if hf_upload_config is not None:
            try:
                upload_final_artifacts(
                    config=hf_upload_config,
                    output_path=args.output,
                    progress_output=args.progress_output,
                    csv_output=args.csv_output,
                )
            except Exception as exc:
                if str(hf_upload_config.get("mode", "disabled")) == "required":
                    raise
                print(f"[hf-upload] final artifact upload failed: {exc}", file=sys.stderr)
        print(json.dumps(payload, indent=2))

    distributed_barrier(distributed_context)
    destroy_distributed(distributed_context)


if __name__ == "__main__":
    main()
