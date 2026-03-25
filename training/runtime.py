from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    dist = None
    DistributedDataParallel = None


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = ""

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def require_torch() -> None:
    if torch is None:
        raise SystemExit("Torch is required for LM training. Install requirements-models.txt or use .venv_atari.")


def auto_device() -> torch.device:
    require_torch()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(device_name: str) -> torch.device:
    require_torch()
    if device_name == "auto":
        return auto_device()
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise SystemExit("MPS requested but not available.")
        return torch.device("mps")
    return torch.device("cpu")


def initialize_distributed(device_name: str) -> DistributedContext:
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)
    require_torch()
    if dist is None:
        raise SystemExit("Distributed launch requested but torch.distributed is unavailable.")
    rank = int(__import__("os").environ.get("RANK", "0"))
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", str(rank)))
    if device_name in {"auto", "cuda"}:
        if not torch.cuda.is_available():
            raise SystemExit("Distributed CUDA launch requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def resolve_runtime_device(device_name: str, distributed_context: DistributedContext) -> torch.device:
    require_torch()
    if not distributed_context.enabled:
        return resolve_device(device_name)
    if device_name in {"auto", "cuda"}:
        return torch.device("cuda", distributed_context.local_rank)
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        raise SystemExit("Distributed training does not support MPS.")
    return resolve_device(device_name)


def synchronize_device(device: torch.device) -> None:
    require_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def distributed_barrier(distributed_context: DistributedContext) -> None:
    if distributed_context.enabled and dist is not None:
        dist.barrier()


def destroy_distributed(distributed_context: DistributedContext) -> None:
    if distributed_context.enabled and dist is not None and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if DistributedDataParallel is not None and isinstance(model, DistributedDataParallel):
        return model.module
    return model


def precision_mode(requested_precision: str, device: torch.device) -> str:
    if requested_precision == "auto":
        return "bf16" if device.type == "cuda" else "fp32"
    if requested_precision == "bf16" and device.type != "cuda":
        raise SystemExit("BF16 precision is only supported for CUDA devices in this trainer.")
    return requested_precision


def autocast_context(device: torch.device, precision: str):
    if torch is None or precision != "bf16":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def set_step_seed(
    *,
    seed: int,
    distributed_context: DistributedContext,
    step_index: int,
    micro_step_index: int = 0,
) -> int:
    step_seed = (
        int(seed)
        + (distributed_context.rank * 1_000_003)
        + (step_index * 1_003)
        + micro_step_index
    )
    set_global_seed(step_seed)
    return step_seed
