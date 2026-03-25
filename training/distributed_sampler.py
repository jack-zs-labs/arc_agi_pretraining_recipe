from __future__ import annotations

import hashlib
import math


def _stable_int(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _coprime_multiplier(modulus: int, *, seed: int, epoch: int) -> int:
    if modulus <= 1:
        return 1
    candidate = _stable_int("multiplier", seed, epoch) % modulus
    if candidate == 0:
        candidate = 1
    while math.gcd(candidate, modulus) != 1:
        candidate = (candidate + 1) % modulus
        if candidate == 0:
            candidate = 1
    return candidate


def permuted_index(position: int, *, dataset_size: int, seed: int, shuffle: bool) -> int:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    epoch = position // dataset_size
    offset = position % dataset_size
    if not shuffle or dataset_size == 1:
        return offset
    multiplier = _coprime_multiplier(dataset_size, seed=seed, epoch=epoch)
    shift = _stable_int("shift", seed, epoch) % dataset_size
    return (multiplier * offset + shift) % dataset_size


class DeterministicDistributedBatchSampler:
    def __init__(
        self,
        *,
        dataset_size: int,
        per_rank_batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if per_rank_batch_size <= 0:
            raise ValueError("per_rank_batch_size must be positive.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size).")
        self.dataset_size = int(dataset_size)
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

    @property
    def global_batch_size(self) -> int:
        return self.per_rank_batch_size * self.world_size

    def batch_indices(self, consumed_samples: int) -> list[int]:
        if consumed_samples < 0:
            raise ValueError("consumed_samples must be non-negative.")
        global_batch_size = self.global_batch_size
        if consumed_samples % global_batch_size != 0:
            raise ValueError(
                f"consumed_samples={consumed_samples} is not aligned to the global batch size {global_batch_size}."
            )
        batch_start = consumed_samples + (self.rank * self.per_rank_batch_size)
        return [
            permuted_index(
                batch_start + offset,
                dataset_size=self.dataset_size,
                seed=self.seed,
                shuffle=self.shuffle,
            )
            for offset in range(self.per_rank_batch_size)
        ]

    def step_sample_count(self, *, grad_accumulation_steps: int = 1) -> int:
        if grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive.")
        return self.global_batch_size * grad_accumulation_steps
