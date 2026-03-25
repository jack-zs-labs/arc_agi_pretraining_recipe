from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from training.token_packer import SUPPORTED_PACKED_SPLITS, token_dtype_from_name


@dataclass(frozen=True)
class PackedShardView:
    split: str
    shard_index: int
    sequence_count: int
    token_count: int
    start_sequence_index: int
    bin_path: Path
    idx_path: Path


def read_packed_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    return manifest


class PackedSequenceDataset:
    def __init__(self, manifest_path: str | Path, *, split: str) -> None:
        if split not in SUPPORTED_PACKED_SPLITS:
            raise ValueError(f"Unsupported split {split!r}; expected one of {SUPPORTED_PACKED_SPLITS}.")
        manifest = read_packed_manifest(manifest_path)
        split_payload = dict(manifest.get("splits", {}).get(split, {}))
        if not split_payload:
            raise KeyError(f"Packed manifest does not contain split {split!r}.")
        self.manifest_path = Path(str(manifest["manifest_path"]))
        self.split = split
        self.sample_length = int(manifest["sample_length"])
        self.token_dtype = token_dtype_from_name(str(manifest["token_dtype"]))
        self.shards = tuple(
            PackedShardView(
                split=str(payload["split"]),
                shard_index=int(payload["shard_index"]),
                sequence_count=int(payload["sequence_count"]),
                token_count=int(payload["token_count"]),
                start_sequence_index=int(payload["start_sequence_index"]),
                bin_path=Path(str(payload["bin_path"])).resolve(),
                idx_path=Path(str(payload["idx_path"])).resolve(),
            )
            for payload in split_payload.get("shards", [])
        )
        cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += shard.sequence_count
            cumulative.append(total)
        self._cumulative_sequences = tuple(cumulative)
        self._row_maps: dict[int, np.memmap] = {}
        self._total_sequences = total

    def __len__(self) -> int:
        return self._total_sequences

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += self._total_sequences
        if not 0 <= index < self._total_sequences:
            raise IndexError(f"PackedSequenceDataset index {index} out of range for length {self._total_sequences}.")
        return index

    def _locate_shard(self, index: int) -> tuple[int, int]:
        normalized = self._normalize_index(index)
        shard_position = bisect_right(self._cumulative_sequences, normalized)
        shard_start = 0 if shard_position == 0 else self._cumulative_sequences[shard_position - 1]
        return shard_position, normalized - shard_start

    def _open_shard_rows(self, shard_position: int) -> np.memmap:
        if shard_position not in self._row_maps:
            shard = self.shards[shard_position]
            expected_token_count = shard.sequence_count * self.sample_length
            if shard.token_count != expected_token_count:
                raise ValueError(
                    f"Shard {shard.bin_path} advertises {shard.token_count} tokens, expected {expected_token_count} "
                    f"for {shard.sequence_count} fixed-length sequences."
                )
            self._row_maps[shard_position] = np.memmap(
                shard.bin_path,
                dtype=self.token_dtype,
                mode="r",
                shape=(shard.sequence_count, self.sample_length),
            )
        return self._row_maps[shard_position]

    def prefetch_shards(self, indices: Sequence[int]) -> None:
        shard_positions = {
            self._locate_shard(index)[0]
            for index in indices
        }
        for shard_position in shard_positions:
            self._open_shard_rows(shard_position)

    def __getitem__(self, index: int) -> np.ndarray:
        shard_position, shard_local_index = self._locate_shard(index)
        rows = self._open_shard_rows(shard_position)
        return np.asarray(rows[shard_local_index], dtype=np.int64)

    def take(self, indices: Sequence[int]) -> np.ndarray:
        batch = np.empty((len(indices), self.sample_length), dtype=np.int64)
        grouped_indices: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for batch_index, index in enumerate(indices):
            shard_position, shard_local_index = self._locate_shard(index)
            grouped_indices[shard_position].append((batch_index, shard_local_index))
        for shard_position, positions in grouped_indices.items():
            rows = self._open_shard_rows(shard_position)
            batch_positions = np.fromiter((batch_index for batch_index, _ in positions), dtype=np.int64)
            local_indices = np.fromiter((local_index for _, local_index in positions), dtype=np.int64)
            batch[batch_positions] = np.asarray(rows[local_indices], dtype=np.int64)
        return batch

    def input_target_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = self[index]
        return tokens[:-1], tokens[1:]

    def batch_input_target_pair(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.take(indices)
        return tokens[:, :-1], tokens[:, 1:]
