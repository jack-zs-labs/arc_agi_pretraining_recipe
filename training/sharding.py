from __future__ import annotations

from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import (
        CPUOffload,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel,
        LocalOptimStateDictConfig,
        LocalStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.nn.parallel import DistributedDataParallel
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    FullyShardedDataParallel = None
    DistributedDataParallel = None
    MixedPrecision = None
    StateDictType = None
    FullStateDictConfig = None
    FullOptimStateDictConfig = None
    LocalStateDictConfig = None
    LocalOptimStateDictConfig = None
    ShardingStrategy = None
    CPUOffload = None
    transformer_auto_wrap_policy = None
    checkpoint_wrapper = None
    CheckpointImpl = None
    apply_activation_checkpointing = None

from models.decoder import DecoderBlock
from training.runtime import DistributedContext


SHARDING_CHOICES: tuple[str, ...] = ("ddp", "fsdp_full_shard", "fsdp_shard_grad_op")


@dataclass(frozen=True)
class ShardingContext:
    strategy: str
    activation_checkpointing: bool

    @property
    def is_fsdp(self) -> bool:
        return self.strategy.startswith("fsdp_")


def sharding_context(strategy: str, *, activation_checkpointing: bool) -> ShardingContext:
    normalized = str(strategy).strip().lower()
    if normalized not in SHARDING_CHOICES:
        raise SystemExit(f"Unsupported sharding strategy {strategy!r}; expected one of {SHARDING_CHOICES}.")
    return ShardingContext(
        strategy=normalized,
        activation_checkpointing=bool(activation_checkpointing),
    )


def _require_fsdp() -> None:
    if FullyShardedDataParallel is None or MixedPrecision is None or StateDictType is None:
        raise SystemExit(
            "FSDP requested but torch.distributed.fsdp is unavailable in this environment."
        )


def build_data_parallel_model(
    *,
    base_model: torch.nn.Module,
    distributed_context: DistributedContext,
    device: torch.device,
    precision: str,
    sharding: ShardingContext,
) -> torch.nn.Module:
    if not distributed_context.enabled:
        if sharding.is_fsdp:
            raise SystemExit("FSDP requires a distributed torchrun launch with WORLD_SIZE > 1.")
        return base_model

    if sharding.strategy == "ddp":
        if DistributedDataParallel is None:
            raise SystemExit("DDP requested but torch.nn.parallel.DistributedDataParallel is unavailable.")
        return DistributedDataParallel(
            base_model,
            device_ids=[distributed_context.local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    if device.type != "cuda":
        raise SystemExit("FSDP is only supported for CUDA devices in this trainer.")
    _require_fsdp()

    if sharding.activation_checkpointing:
        def checkpoint_wrapper_fn(module: torch.nn.Module) -> torch.nn.Module:
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

        apply_activation_checkpointing(
            base_model,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=lambda module: isinstance(module, DecoderBlock),
        )

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DecoderBlock},
    )
    mixed_precision_policy = None
    if precision == "bf16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    strategy = (
        ShardingStrategy.FULL_SHARD
        if sharding.strategy == "fsdp_full_shard"
        else ShardingStrategy.SHARD_GRAD_OP
    )
    return FullyShardedDataParallel(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
        cpu_offload=CPUOffload(offload_params=False),
        mixed_precision=mixed_precision_policy,
        sharding_strategy=strategy,
    )


def is_fsdp_model(model: torch.nn.Module) -> bool:
    return FullyShardedDataParallel is not None and isinstance(model, FullyShardedDataParallel)


def model_state_save_context(model: torch.nn.Module):
    if not is_fsdp_model(model):
        return nullcontext()
    return FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.LOCAL_STATE_DICT,
        LocalStateDictConfig(offload_to_cpu=True),
        LocalOptimStateDictConfig(offload_to_cpu=True),
    )


def model_state_load_context(model: torch.nn.Module):
    if not is_fsdp_model(model):
        return nullcontext()
    return FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.LOCAL_STATE_DICT,
        LocalStateDictConfig(offload_to_cpu=True),
        LocalOptimStateDictConfig(offload_to_cpu=True),
    )


def optimizer_state_for_save(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    if not is_fsdp_model(model):
        return optimizer.state_dict()
    return FullyShardedDataParallel.optim_state_dict(model, optimizer)


def load_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, object],
) -> None:
    if not is_fsdp_model(model):
        optimizer.load_state_dict(state_dict)
        return
    state_to_load = FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, state_dict)
    optimizer.load_state_dict(state_to_load)


def unwrap_sharded_model(model: torch.nn.Module) -> torch.nn.Module:
    if is_fsdp_model(model):
        return model.module
    if DistributedDataParallel is not None and isinstance(model, DistributedDataParallel):
        return model.module
    return model
