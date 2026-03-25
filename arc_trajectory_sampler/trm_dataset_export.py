from __future__ import annotations

"""Export sampled ARC episodes in TinyRecursiveModels dataset format."""

from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from .stage2_episode_sampler import EpisodeSpec
    from .stage3_grid_executor import Grid, execute_episode
except ImportError:  # pragma: no cover - direct script execution
    from stage2_episode_sampler import EpisodeSpec  # type: ignore
    from stage3_grid_executor import Grid, execute_episode  # type: ignore


ARC_MAX_GRID_SIZE = 30
PUZZLE_ID_SEPARATOR = "|||"
SET_NAME = "all"


@dataclass(frozen=True)
class ExportPuzzle:
    puzzle_id: str
    examples: Tuple[Tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class ExportSummary:
    train_puzzles: int
    test_puzzles: int
    num_puzzle_identifiers: int
    output_dir: str


def grid_fits_trm(grid: Grid) -> bool:
    return len(grid) <= ARC_MAX_GRID_SIZE and (len(grid[0]) if grid else 0) <= ARC_MAX_GRID_SIZE


def grid_to_np(grid: Grid) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("grid must be 2D")
    if arr.shape[0] > ARC_MAX_GRID_SIZE or arr.shape[1] > ARC_MAX_GRID_SIZE:
        raise ValueError(f"grid shape {arr.shape} exceeds ARC max size {ARC_MAX_GRID_SIZE}")
    if not np.all((arr >= 0) & (arr <= 9)):
        raise ValueError("grid values must be in [0, 9]")
    return arr


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0:
        return arr
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    raise ValueError(f"unknown dihedral transform id: {tid}")


def np_grid_to_seq_translational_augment(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    do_translation: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if do_translation:
        pad_r = int(rng.integers(0, ARC_MAX_GRID_SIZE - max(inp.shape[0], out.shape[0]) + 1))
        pad_c = int(rng.integers(0, ARC_MAX_GRID_SIZE - max(inp.shape[1], out.shape[1]) + 1))
    else:
        pad_r = 0
        pad_c = 0

    result = []
    for grid in (inp, out):
        nrow, ncol = grid.shape
        padded = np.pad(
            grid + 2,
            ((pad_r, ARC_MAX_GRID_SIZE - pad_r - nrow), (pad_c, ARC_MAX_GRID_SIZE - pad_c - ncol)),
            constant_values=0,
        )
        eos_row = pad_r + nrow
        eos_col = pad_c + ncol
        if eos_row < ARC_MAX_GRID_SIZE:
            padded[eos_row, pad_c:eos_col] = 1
        if eos_col < ARC_MAX_GRID_SIZE:
            padded[pad_r:eos_row, eos_col] = 1
        result.append(padded.astype(np.uint8).reshape(-1))
    return result[0], result[1]


def grid_hash(grid: np.ndarray) -> str:
    buffer = [dim.to_bytes(1, byteorder="big") for dim in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: ExportPuzzle) -> str:
    hashes = [f"{grid_hash(inp)}|{grid_hash(out)}" for inp, out in puzzle.examples]
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def sample_augmentation_plan(rng: np.random.Generator) -> Tuple[int, np.ndarray]:
    trans_id = int(rng.integers(0, 8))
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), rng.permutation(np.arange(1, 10, dtype=np.uint8))])
    return trans_id, mapping


def augment_puzzle(puzzle: ExportPuzzle, trans_id: int, mapping: np.ndarray) -> ExportPuzzle:
    puzzle_id = f"{puzzle.puzzle_id}{PUZZLE_ID_SEPARATOR}t{trans_id}{PUZZLE_ID_SEPARATOR}{''.join(str(x) for x in mapping)}"

    def map_grid(grid: np.ndarray) -> np.ndarray:
        return dihedral_transform(mapping[grid], trans_id)

    return ExportPuzzle(
        puzzle_id=puzzle_id,
        examples=tuple((map_grid(inp), map_grid(out)) for inp, out in puzzle.examples),
    )


def build_augmented_group(
    base_puzzle: ExportPuzzle,
    *,
    num_aug: int,
    rng: np.random.Generator,
) -> List[ExportPuzzle]:
    group = [base_puzzle]
    if num_aug <= 0:
        return group

    hashes = {puzzle_hash(base_puzzle)}
    max_trials = max(1, num_aug * 5)
    for _ in range(max_trials):
        trans_id, mapping = sample_augmentation_plan(rng)
        candidate = augment_puzzle(base_puzzle, trans_id, mapping)
        candidate_hash = puzzle_hash(candidate)
        if candidate_hash in hashes:
            continue
        hashes.add(candidate_hash)
        group.append(candidate)
        if len(group) >= num_aug + 1:
            break
    return group


def build_augmented_group_pair(
    train_puzzle: ExportPuzzle,
    test_puzzle: ExportPuzzle,
    *,
    num_aug: int,
    rng: np.random.Generator,
) -> Tuple[List[ExportPuzzle], List[ExportPuzzle]]:
    train_group = [train_puzzle]
    test_group = [test_puzzle]
    if num_aug <= 0:
        return train_group, test_group

    hashes = {(puzzle_hash(train_puzzle), puzzle_hash(test_puzzle))}
    max_trials = max(1, num_aug * 5)
    for _ in range(max_trials):
        trans_id, mapping = sample_augmentation_plan(rng)
        train_candidate = augment_puzzle(train_puzzle, trans_id, mapping)
        test_candidate = augment_puzzle(test_puzzle, trans_id, mapping)
        signature = (puzzle_hash(train_candidate), puzzle_hash(test_candidate))
        if signature in hashes:
            continue
        hashes.add(signature)
        train_group.append(train_candidate)
        test_group.append(test_candidate)
        if len(train_group) >= num_aug + 1:
            break
    return train_group, test_group


def episode_to_puzzles(episode: EpisodeSpec, episode_index: int) -> Tuple[ExportPuzzle, ExportPuzzle]:
    executed = execute_episode(episode)
    puzzle_id = f"arc_traj_{episode_index:06d}_{episode.latent.family.value}"

    train_examples = tuple((grid_to_np(item.input_grid), grid_to_np(item.output_grid)) for item in executed.train_examples)
    test_examples = ((grid_to_np(executed.test_example.input_grid), grid_to_np(executed.test_example.output_grid)),)
    return (
        ExportPuzzle(puzzle_id=puzzle_id, examples=train_examples),
        ExportPuzzle(puzzle_id=puzzle_id, examples=test_examples),
    )


def episode_is_trm_compatible(episode: EpisodeSpec) -> bool:
    executed = execute_episode(episode)
    for example in [*executed.train_examples, executed.test_example]:
        if not grid_fits_trm(example.input_grid):
            return False
        if not grid_fits_trm(example.output_grid):
            return False
    return True


def save_split(
    output_dir: Path,
    split_name: str,
    groups: Sequence[Sequence[ExportPuzzle]],
    identifier_map: Dict[str, int],
    *,
    rng: np.random.Generator,
    enable_translation: bool,
) -> Dict[str, int]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[np.ndarray]] = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [np.array(0, dtype=np.int32)],
        "group_indices": [np.array(0, dtype=np.int32)],
    }
    example_id = 0
    puzzle_id = 0

    for group in groups:
        for puzzle in group:
            if not puzzle.examples:
                continue
            no_aug_id = int(rng.integers(0, len(puzzle.examples)))
            for example_index, (inp, out) in enumerate(puzzle.examples):
                encoded_inp, encoded_out = np_grid_to_seq_translational_augment(
                    inp,
                    out,
                    do_translation=enable_translation and example_index != no_aug_id,
                    rng=rng,
                )
                results["inputs"].append(encoded_inp)
                results["labels"].append(encoded_out)
                example_id += 1

            results["puzzle_indices"].append(np.array(example_id, dtype=np.int32))
            results["puzzle_identifiers"].append(np.array(identifier_map[puzzle.puzzle_id], dtype=np.int32))
            puzzle_id += 1
        results["group_indices"].append(np.array(puzzle_id, dtype=np.int32))

    array_payloads = {
        "inputs": np.stack(results["inputs"], axis=0) if results["inputs"] else np.zeros((0, ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE), dtype=np.uint8),
        "labels": np.stack(results["labels"], axis=0) if results["labels"] else np.zeros((0, ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE), dtype=np.uint8),
        "puzzle_identifiers": np.asarray(results["puzzle_identifiers"], dtype=np.int32),
        "puzzle_indices": np.asarray(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.asarray(results["group_indices"], dtype=np.int32),
    }
    for name, array in array_payloads.items():
        np.save(split_dir / f"{SET_NAME}__{name}.npy", array)

    total_puzzles = max(0, len(array_payloads["puzzle_identifiers"]))
    total_examples = int(array_payloads["inputs"].shape[0])
    total_groups = max(0, len(groups))
    metadata = {
        "seq_len": ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
        "vocab_size": 12,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": len(identifier_map) + 1,
        "total_groups": total_groups,
        "mean_puzzle_examples": (total_examples / total_puzzles) if total_puzzles else 0.0,
        "total_puzzles": total_puzzles,
        "sets": [SET_NAME],
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata, indent=2))
    return {
        "examples": total_examples,
        "puzzles": total_puzzles,
        "groups": total_groups,
    }


def write_trm_dataset(
    episodes: Sequence[EpisodeSpec],
    output_dir: str | Path,
    *,
    seed: int = 0,
    include_test: bool = True,
    num_aug: int = 0,
) -> ExportSummary:
    dataset_dir = Path(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    train_groups: List[List[ExportPuzzle]] = []
    test_groups: List[List[ExportPuzzle]] = []
    identifier_map: Dict[str, int] = {}
    next_identifier = 1

    for episode_index, episode in enumerate(episodes):
        train_puzzle, test_puzzle = episode_to_puzzles(episode, episode_index)
        if include_test:
            train_group, test_group = build_augmented_group_pair(train_puzzle, test_puzzle, num_aug=num_aug, rng=rng)
            test_groups.append(test_group)
        else:
            train_group = build_augmented_group(train_puzzle, num_aug=num_aug, rng=rng)
        train_groups.append(train_group)

        for group in [train_group] + ([test_group] if include_test else []):
            for puzzle in group:
                if puzzle.puzzle_id not in identifier_map:
                    identifier_map[puzzle.puzzle_id] = next_identifier
                    next_identifier += 1

    train_stats = save_split(
        dataset_dir,
        "train",
        train_groups,
        identifier_map,
        rng=rng,
        enable_translation=True,
    )
    test_stats = {"puzzles": 0}
    if include_test:
        test_stats = save_split(
            dataset_dir,
            "test",
            test_groups,
            identifier_map,
            rng=rng,
            enable_translation=False,
        )

    identifiers = ["<blank>"] + [None] * len(identifier_map)
    for puzzle_id, identifier in identifier_map.items():
        identifiers[identifier] = puzzle_id
    (dataset_dir / "identifiers.json").write_text(json.dumps(identifiers, indent=2))
    summary = ExportSummary(
        train_puzzles=int(train_stats["puzzles"]),
        test_puzzles=int(test_stats["puzzles"]),
        num_puzzle_identifiers=len(identifier_map) + 1,
        output_dir=str(dataset_dir),
    )
    (dataset_dir / "summary.json").write_text(json.dumps(summary.__dict__, indent=2))
    return summary
