from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Iterator

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from training.distributed_sampler import DeterministicDistributedBatchSampler
from training.packed_lm_dataset import PackedSequenceDataset


@dataclass(frozen=True)
class PackedLMBatch:
    consumed_samples: int
    batch_indices: tuple[int, ...]
    inputs: "torch.Tensor"
    targets: "torch.Tensor"

    def to(self, device: "torch.device", *, non_blocking: bool = False) -> tuple["torch.Tensor", "torch.Tensor"]:
        if self.inputs.device == device and self.targets.device == device:
            return self.inputs, self.targets
        return (
            self.inputs.to(device=device, dtype=torch.long, non_blocking=non_blocking),
            self.targets.to(device=device, dtype=torch.long, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class _WorkerError:
    error: BaseException


class AsyncPackedBatchLoader(Iterator[PackedLMBatch]):
    def __init__(
        self,
        *,
        dataset: PackedSequenceDataset,
        sampler: DeterministicDistributedBatchSampler,
        start_consumed_samples: int,
        total_batches: int,
        prefetch_batches: int = 2,
        pin_memory: bool = False,
    ) -> None:
        if torch is None:
            raise SystemExit("Torch is required for the async packed batch loader.")
        if total_batches < 0:
            raise ValueError("total_batches must be non-negative.")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative.")
        self.dataset = dataset
        self.sampler = sampler
        self.start_consumed_samples = int(start_consumed_samples)
        self.total_batches = int(total_batches)
        self.prefetch_batches = int(prefetch_batches)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._next_batch_index = 0
        self._closed = False
        self._sentinel = object()
        self._queue: Queue[PackedLMBatch | _WorkerError | object] | None = None
        self._worker: Thread | None = None

    def __enter__(self) -> "AsyncPackedBatchLoader":
        if self.prefetch_batches > 0:
            self._queue = Queue(maxsize=max(self.prefetch_batches, 1))
            self._worker = Thread(target=self._worker_main, name="packed-batch-prefetch", daemon=True)
            self._worker.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._closed = True
        self._queue = None
        self._worker = None

    def _prepare_batch(self, batch_index: int) -> PackedLMBatch:
        consumed_samples = self.start_consumed_samples + (batch_index * self.sampler.global_batch_size)
        batch_indices = tuple(self.sampler.batch_indices(consumed_samples))
        self.dataset.prefetch_shards(batch_indices)
        inputs_np, targets_np = self.dataset.batch_input_target_pair(batch_indices)
        inputs = torch.from_numpy(inputs_np)
        targets = torch.from_numpy(targets_np)
        if self.pin_memory:
            inputs = inputs.pin_memory()
            targets = targets.pin_memory()
        return PackedLMBatch(
            consumed_samples=consumed_samples,
            batch_indices=batch_indices,
            inputs=inputs,
            targets=targets,
        )

    def _worker_main(self) -> None:
        assert self._queue is not None
        try:
            for batch_index in range(self.total_batches):
                if self._closed:
                    break
                self._queue.put(self._prepare_batch(batch_index))
            self._queue.put(self._sentinel)
        except BaseException as exc:  # pragma: no cover - exercised indirectly in training
            self._queue.put(_WorkerError(exc))

    def __iter__(self) -> "AsyncPackedBatchLoader":
        return self

    def __next__(self) -> PackedLMBatch:
        if self._next_batch_index >= self.total_batches:
            raise StopIteration
        if self.prefetch_batches <= 0:
            batch = self._prepare_batch(self._next_batch_index)
            self._next_batch_index += 1
            return batch
        if self._queue is None:
            raise RuntimeError("AsyncPackedBatchLoader must be entered before iteration.")
        item = self._queue.get()
        if item is self._sentinel:
            self._next_batch_index = self.total_batches
            raise StopIteration
        if isinstance(item, _WorkerError):
            self._next_batch_index = self.total_batches
            raise item.error
        self._next_batch_index += 1
        return item
