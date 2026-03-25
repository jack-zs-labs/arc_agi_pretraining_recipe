from __future__ import annotations

"""Evaluate raw ARC-AGI-2 grid fit under the current scene/object representation.

This benchmark is intentionally representation-level, not solver-level. It asks:

1. Can a raw ARC grid be segmented into objects under our current vocabulary?
2. If we fit each connected component to one of our supported object primitives,
   how accurately can we reconstruct the original grid?

It does not search over latent families or recover executable programs.
"""

from collections import Counter, deque
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .stage2_episode_sampler import ObjectSpec, SceneSpec, compute_mass
    from .stage3_grid_executor import object_cells, render_scene
except ImportError:  # pragma: no cover - direct script execution
    from stage2_episode_sampler import ObjectSpec, SceneSpec, compute_mass  # type: ignore
    from stage3_grid_executor import object_cells, render_scene  # type: ignore


Grid = Tuple[Tuple[int, ...], ...]
Coord = Tuple[int, int]


@dataclass(frozen=True)
class ComponentFit:
    object_spec: ObjectSpec
    primitive: str
    exact: bool
    iou: float
    component_size: int
    bbox_height: int
    bbox_width: int


@dataclass(frozen=True)
class GridParseResult:
    background_color: int
    border_color: Optional[int]
    rendered_grid: Grid
    scene: SceneSpec
    components: Tuple[ComponentFit, ...]
    exact_grid: bool
    cell_accuracy: float
    exact_component_rate: float
    mean_component_iou: float


def to_grid(rows: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def iter_cells(grid: Grid) -> Iterable[int]:
    for row in grid:
        yield from row


def grid_shape(grid: Grid) -> Tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def grid_accuracy(first: Grid, second: Grid) -> float:
    height = max(len(first), len(second))
    width = max(len(first[0]) if first else 0, len(second[0]) if second else 0)
    if height == 0 or width == 0:
        return 1.0
    matches = 0
    total = height * width
    for row in range(height):
        for col in range(width):
            first_value = first[row][col] if row < len(first) and col < len(first[row]) else 0
            second_value = second[row][col] if row < len(second) and col < len(second[row]) else 0
            if first_value == second_value:
                matches += 1
    return matches / total


def outer_ring_coords(height: int, width: int) -> Tuple[Coord, ...]:
    if height == 0 or width == 0:
        return tuple()
    coords = {(0, col) for col in range(width)}
    coords.update((height - 1, col) for col in range(width))
    coords.update((row, 0) for row in range(height))
    coords.update((row, width - 1) for row in range(height))
    return tuple(sorted(coords))


def candidate_background_colors(grid: Grid) -> Tuple[int, ...]:
    counts = Counter(iter_cells(grid))
    ranked = [color for color, _count in counts.most_common()]
    top_left = grid[0][0] if grid and grid[0] else 0
    ordered = [top_left]
    ordered.extend(ranked[:4])
    return tuple(dict.fromkeys(ordered))


def uniform_border_color(grid: Grid) -> Optional[int]:
    height, width = grid_shape(grid)
    ring = outer_ring_coords(height, width)
    if not ring:
        return None
    colors = {grid[row][col] for row, col in ring}
    if len(colors) != 1:
        return None
    return next(iter(colors))


def connected_components(grid: Grid, background_color: int, border_color: Optional[int]) -> List[Tuple[int, Tuple[Coord, ...]]]:
    height, width = grid_shape(grid)
    blocked = set(outer_ring_coords(height, width)) if border_color is not None else set()
    visited: set[Coord] = set()
    components: List[Tuple[int, Tuple[Coord, ...]]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in visited or (row, col) in blocked:
                continue
            color = grid[row][col]
            if color == background_color:
                continue
            queue: deque[Coord] = deque([(row, col)])
            visited.add((row, col))
            cells: List[Coord] = []
            while queue:
                cur_row, cur_col = queue.popleft()
                cells.append((cur_row, cur_col))
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (cur_row + d_row, cur_col + d_col)
                    if nxt in visited or nxt in blocked:
                        continue
                    if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                        continue
                    if grid[nxt[0]][nxt[1]] != color:
                        continue
                    visited.add(nxt)
                    queue.append(nxt)
            components.append((color, tuple(sorted(cells))))
    return components


def bitmap_attributes(cells: Sequence[Coord], top: int, left: int) -> Dict[str, Any]:
    return {
        "bitmap_cells": tuple(sorted((row - top, col - left) for row, col in cells)),
    }


def make_candidate_object(
    *,
    primitive: str,
    color: int,
    top: int,
    left: int,
    height: int,
    width: int,
    orientation: str = "up",
    holes: int = 0,
    is_container: bool = False,
    attributes: Optional[Dict[str, Any]] = None,
) -> ObjectSpec:
    shape = primitive if primitive != "frame" else "rect"
    filled_cells = None
    if primitive == "bitmap":
        shape = "bitmap"
        bitmap_cells = (attributes or {}).get("bitmap_cells", ())
        filled_cells = len(bitmap_cells)
    return ObjectSpec(
        object_id="component",
        shape=shape,
        color=color,
        top=top,
        left=left,
        height=height,
        width=width,
        mass=compute_mass(shape, height, width, holes, filled_cells=filled_cells),
        orientation=orientation,
        holes=holes,
        is_container=is_container,
        tags=("frame",) if is_container else (),
        attributes=attributes or {},
    )


def primitive_candidates(color: int, top: int, left: int, height: int, width: int) -> List[Tuple[str, ObjectSpec]]:
    candidates: List[Tuple[str, ObjectSpec]] = []
    if height == 1 and width == 1:
        candidates.append(("dot", make_candidate_object(primitive="dot", color=color, top=top, left=left, height=1, width=1)))
    candidates.append(("line", make_candidate_object(primitive="line", color=color, top=top, left=left, height=height, width=width)))
    for orientation in ("up", "right", "down", "left"):
        candidates.append(
            (
                "l_shape",
                make_candidate_object(
                    primitive="l_shape",
                    color=color,
                    top=top,
                    left=left,
                    height=height,
                    width=width,
                    orientation=orientation,
                ),
            )
        )
    candidates.append(("cross", make_candidate_object(primitive="cross", color=color, top=top, left=left, height=height, width=width)))
    if height >= 3 and width >= 3:
        candidates.append(
            (
                "frame",
                make_candidate_object(
                    primitive="frame",
                    color=color,
                    top=top,
                    left=left,
                    height=height,
                    width=width,
                    is_container=True,
                ),
            )
        )
    for holes in range(4):
        candidates.append(
            (
                "rect",
                make_candidate_object(
                    primitive="rect",
                    color=color,
                    top=top,
                    left=left,
                    height=height,
                    width=width,
                    holes=holes,
                ),
            )
        )
    return candidates


def candidate_iou(cells: set[Coord], obj: ObjectSpec) -> float:
    rendered = {(obj.top + d_row, obj.left + d_col) for d_row, d_col in object_cells(obj)}
    if not rendered and not cells:
        return 1.0
    if not rendered or not cells:
        return 0.0
    intersection = len(cells & rendered)
    union = len(cells | rendered)
    return intersection / max(1, union)


def fit_component(component_index: int, color: int, cells: Sequence[Coord]) -> ComponentFit:
    top = min(row for row, _ in cells)
    left = min(col for _, col in cells)
    bottom = max(row for row, _ in cells)
    right = max(col for _, col in cells)
    height = bottom - top + 1
    width = right - left + 1
    cell_set = set(cells)
    best_primitive = "rect"
    best_object = make_candidate_object(primitive="rect", color=color, top=top, left=left, height=height, width=width)
    best_iou = -1.0
    exact_object: Optional[ObjectSpec] = None
    exact_primitive: Optional[str] = None

    preference = {
        "dot": 0,
        "line": 1,
        "l_shape": 2,
        "cross": 3,
        "frame": 4,
        "rect": 5,
    }

    for primitive, candidate in primitive_candidates(color, top, left, height, width):
        iou = candidate_iou(cell_set, candidate)
        if iou == 1.0:
            if exact_object is None or preference[primitive] < preference[exact_primitive or "rect"]:
                exact_object = candidate
                exact_primitive = primitive
            continue
        if iou > best_iou or (iou == best_iou and preference[primitive] < preference[best_primitive]):
            best_iou = iou
            best_object = candidate
            best_primitive = primitive

    if exact_object is not None:
        return ComponentFit(
            object_spec=ObjectSpec(
                object_id=f"obj_{component_index}",
                shape=exact_object.shape,
                color=exact_object.color,
                top=exact_object.top,
                left=exact_object.left,
                height=exact_object.height,
                width=exact_object.width,
                mass=exact_object.mass,
                orientation=exact_object.orientation,
                holes=exact_object.holes,
                is_container=exact_object.is_container,
                tags=exact_object.tags,
                attributes={},
            ),
            primitive=exact_primitive or "rect",
            exact=True,
            iou=1.0,
            component_size=len(cell_set),
            bbox_height=height,
            bbox_width=width,
        )

    bitmap_object = make_candidate_object(
        primitive="bitmap",
        color=color,
        top=top,
        left=left,
        height=height,
        width=width,
        attributes=bitmap_attributes(cells, top, left),
    )
    bitmap_iou = candidate_iou(cell_set, bitmap_object)
    if bitmap_iou == 1.0:
        return ComponentFit(
            object_spec=ObjectSpec(
                object_id=f"obj_{component_index}",
                shape=bitmap_object.shape,
                color=bitmap_object.color,
                top=bitmap_object.top,
                left=bitmap_object.left,
                height=bitmap_object.height,
                width=bitmap_object.width,
                mass=bitmap_object.mass,
                orientation=bitmap_object.orientation,
                holes=bitmap_object.holes,
                is_container=bitmap_object.is_container,
                tags=bitmap_object.tags,
                attributes=dict(bitmap_object.attributes),
            ),
            primitive="bitmap",
            exact=True,
            iou=1.0,
            component_size=len(cell_set),
            bbox_height=height,
            bbox_width=width,
        )

    return ComponentFit(
        object_spec=ObjectSpec(
            object_id=f"obj_{component_index}",
            shape=best_object.shape,
            color=best_object.color,
            top=best_object.top,
            left=best_object.left,
            height=best_object.height,
            width=best_object.width,
            mass=best_object.mass,
            orientation=best_object.orientation,
            holes=best_object.holes,
            is_container=best_object.is_container,
            tags=best_object.tags,
            attributes={},
        ),
        primitive=best_primitive,
        exact=False,
        iou=max(0.0, best_iou),
        component_size=len(cell_set),
        bbox_height=height,
        bbox_width=width,
    )


def parse_grid_once(grid: Grid, *, background_color: int, border_color: Optional[int]) -> GridParseResult:
    height, width = grid_shape(grid)
    components = connected_components(grid, background_color, border_color)
    fitted = tuple(fit_component(index, color, cells) for index, (color, cells) in enumerate(components))
    scene = SceneSpec(
        height=height,
        width=width,
        background_color=background_color,
        border_color=border_color,
        outline_color=None,
        marker_position=None,
        objects=tuple(item.object_spec for item in fitted),
        legend=tuple(),
        attributes={"workspace": "arc_agi2_parse"},
    )
    rendered = render_scene(scene)
    accuracy = grid_accuracy(grid, rendered)
    exact_components = sum(1 for item in fitted if item.exact)
    return GridParseResult(
        background_color=background_color,
        border_color=border_color,
        rendered_grid=rendered,
        scene=scene,
        components=fitted,
        exact_grid=rendered == grid,
        cell_accuracy=accuracy,
        exact_component_rate=(exact_components / len(fitted)) if fitted else 1.0,
        mean_component_iou=(sum(item.iou for item in fitted) / len(fitted)) if fitted else 1.0,
    )


def parse_grid_best(grid: Grid) -> GridParseResult:
    border_candidate = uniform_border_color(grid)
    candidates: List[GridParseResult] = []
    for background_color in candidate_background_colors(grid):
        candidates.append(parse_grid_once(grid, background_color=background_color, border_color=None))
        if border_candidate is not None and border_candidate != background_color:
            candidates.append(parse_grid_once(grid, background_color=background_color, border_color=border_candidate))
    return max(
        candidates,
        key=lambda item: (
            1 if item.exact_grid else 0,
            item.cell_accuracy,
            item.exact_component_rate,
            item.mean_component_iou,
        ),
    )


def load_arc_agi2_grids(dataset_root: Path) -> List[Dict[str, Any]]:
    challenges_path = dataset_root / "data" / "arc-agi_evaluation_challenges.json"
    solutions_path = dataset_root / "data" / "arc-agi_evaluation_solutions.json"
    with challenges_path.open() as handle:
        challenges = json.load(handle)
    with solutions_path.open() as handle:
        solutions = json.load(handle)

    records: List[Dict[str, Any]] = []
    for task_id, task in challenges.items():
        for pair_index, pair in enumerate(task["train"]):
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:train:{pair_index}:input",
                    "role": "train_input",
                    "grid": to_grid(pair["input"]),
                }
            )
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:train:{pair_index}:output",
                    "role": "train_output",
                    "grid": to_grid(pair["output"]),
                }
            )
        for pair_index, pair in enumerate(task["test"]):
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:test:{pair_index}:input",
                    "role": "test_input",
                    "grid": to_grid(pair["input"]),
                }
            )
        for pair_index, output in enumerate(solutions.get(task_id, [])):
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:test:{pair_index}:output",
                    "role": "test_output",
                    "grid": to_grid(output),
                }
            )
    return records


def load_arc_task_records(task_id: str, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if "train" in task or "test" in task:
        for pair_index, pair in enumerate(task.get("train", [])):
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:train:{pair_index}:input",
                    "role": "train_input",
                    "grid": to_grid(pair["input"]),
                }
            )
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:train:{pair_index}:output",
                    "role": "train_output",
                    "grid": to_grid(pair["output"]),
                }
            )
        for pair_index, pair in enumerate(task.get("test", [])):
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:test:{pair_index}:input",
                    "role": "test_input",
                    "grid": to_grid(pair["input"]),
                }
            )
            if "output" in pair:
                records.append(
                    {
                        "task_id": task_id,
                        "grid_id": f"{task_id}:test:{pair_index}:output",
                        "role": "test_output",
                        "grid": to_grid(pair["output"]),
                    }
                )
        return records

    if "examples" in task:
        for pair_index, pair in enumerate(task.get("examples", [])):
            records.append(
                {
                    "task_id": task_id,
                    "grid_id": f"{task_id}:example:{pair_index}:input",
                    "role": "example_input",
                    "grid": to_grid(pair["input"]),
                }
            )
            if "output" in pair:
                records.append(
                    {
                        "task_id": task_id,
                        "grid_id": f"{task_id}:example:{pair_index}:output",
                        "role": "example_output",
                        "grid": to_grid(pair["output"]),
                    }
                )
        return records

    raise ValueError(f"Unsupported ARC task schema for {task_id}: expected train/test or examples keys")


def load_task_directory_grids(dataset_root: Path, *, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    json_paths = sorted(dataset_root.rglob("*.json"))
    if max_tasks is not None:
        json_paths = json_paths[:max_tasks]

    for path in json_paths:
        with path.open() as handle:
            task = json.load(handle)
        task_id = str(path.relative_to(dataset_root).with_suffix(""))
        records.extend(load_arc_task_records(task_id, task))
    return records


def load_dataset_grids(
    dataset_root: Path,
    *,
    dataset_format: str = "auto",
    max_tasks: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if dataset_format in {"auto", "arc_agi2"}:
        challenges_path = dataset_root / "data" / "arc-agi_evaluation_challenges.json"
        solutions_path = dataset_root / "data" / "arc-agi_evaluation_solutions.json"
        if challenges_path.is_file() and solutions_path.is_file():
            return "arc_agi2", load_arc_agi2_grids(dataset_root)
        if dataset_format == "arc_agi2":
            raise FileNotFoundError(
                f"ARC-AGI-2 aggregate files not found under {dataset_root}"
            )

    if dataset_format in {"auto", "task_directory"}:
        json_paths = sorted(dataset_root.rglob("*.json"))
        if json_paths:
            return "task_directory", load_task_directory_grids(dataset_root, max_tasks=max_tasks)
        if dataset_format == "task_directory":
            raise FileNotFoundError(f"No task JSON files found under {dataset_root}")

    raise FileNotFoundError(f"Could not infer a supported dataset format under {dataset_root}")


def evaluate_representation(
    dataset_root: Path,
    *,
    dataset_format: str = "auto",
    max_examples: Optional[int] = None,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_format, records = load_dataset_grids(
        dataset_root,
        dataset_format=dataset_format,
        max_tasks=max_tasks,
    )
    if max_examples is not None:
        records = records[:max_examples]

    role_metrics: Dict[str, Dict[str, float]] = {}
    role_exact_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    exact_component_total = 0
    component_total = 0
    primitive_exact = Counter()
    primitive_approx = Counter()
    background_counter = Counter()
    border_counter = Counter()
    task_exact: Dict[str, bool] = {}
    worst_examples: List[Dict[str, Any]] = []
    mean_cell_accuracy = 0.0
    mean_component_iou = 0.0

    for record in records:
        result = parse_grid_best(record["grid"])
        role = str(record["role"])
        role_counts[role] += 1
        if result.exact_grid:
            role_exact_counts[role] += 1
        background_counter[result.background_color] += 1
        border_counter["none" if result.border_color is None else str(result.border_color)] += 1
        task_exact[record["task_id"]] = task_exact.get(record["task_id"], True) and result.exact_grid
        exact_component_total += sum(1 for component in result.components if component.exact)
        component_total += len(result.components)
        mean_cell_accuracy += result.cell_accuracy
        mean_component_iou += result.mean_component_iou
        for component in result.components:
            if component.exact:
                primitive_exact[component.primitive] += 1
            else:
                primitive_approx[component.primitive] += 1
        worst_examples.append(
            {
                "grid_id": record["grid_id"],
                "task_id": record["task_id"],
                "role": role,
                "cell_accuracy": round(result.cell_accuracy, 6),
                "exact_grid": result.exact_grid,
                "exact_component_rate": round(result.exact_component_rate, 6),
                "mean_component_iou": round(result.mean_component_iou, 6),
                "background_color": result.background_color,
                "border_color": result.border_color,
                "component_count": len(result.components),
            }
        )
        bucket = role_metrics.setdefault(
            role,
            {
                "mean_cell_accuracy": 0.0,
                "mean_component_iou": 0.0,
                "exact_component_total": 0.0,
                "component_total": 0.0,
            },
        )
        bucket["mean_cell_accuracy"] += result.cell_accuracy
        bucket["mean_component_iou"] += result.mean_component_iou
        bucket["exact_component_total"] += sum(1 for component in result.components if component.exact)
        bucket["component_total"] += len(result.components)

    total_examples = len(records)
    summary_roles: Dict[str, Dict[str, Any]] = {}
    for role, metrics in role_metrics.items():
        count = role_counts[role]
        summary_roles[role] = {
            "examples": count,
            "exact_grid_rate": round(role_exact_counts[role] / max(1, count), 6),
            "mean_cell_accuracy": round(metrics["mean_cell_accuracy"] / max(1, count), 6),
            "mean_component_iou": round(metrics["mean_component_iou"] / max(1, count), 6),
            "exact_component_rate": round(metrics["exact_component_total"] / max(1, metrics["component_total"]), 6),
        }

    worst_examples.sort(key=lambda item: (item["cell_accuracy"], item["exact_component_rate"], item["grid_id"]))
    return {
        "dataset_root": str(dataset_root),
        "dataset_format": resolved_format,
        "tasks": len({record["task_id"] for record in records}),
        "examples": total_examples,
        "overall": {
            "exact_grid_rate": round(sum(role_exact_counts.values()) / max(1, total_examples), 6),
            "mean_cell_accuracy": round(mean_cell_accuracy / max(1, total_examples), 6),
            "mean_component_iou": round(mean_component_iou / max(1, total_examples), 6),
            "exact_component_rate": round(exact_component_total / max(1, component_total), 6),
            "task_all_grids_exact_rate": round(sum(1 for value in task_exact.values() if value) / max(1, len(task_exact)), 6),
        },
        "by_role": summary_roles,
        "background_colors": dict(background_counter.most_common()),
        "border_colors": dict(border_counter.most_common()),
        "exact_primitives": dict(primitive_exact.most_common()),
        "approx_primitives": dict(primitive_approx.most_common()),
        "worst_examples": worst_examples[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate raw ARC-style grid fit under the current object representation."
    )
    parser.add_argument(
        "--dataset-root",
        default="/Users/jjmayo/projects/demo_day/arc-agi/arc-agi-2",
        help="Path to the local ARC-AGI-2 repo root or an ARC-style task directory.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "arc_agi2", "task_directory"),
        default="auto",
        help="Dataset loader to use. 'auto' detects ARC-AGI-2 aggregate files or per-task JSON directories.",
    )
    parser.add_argument(
        "--output",
        default="arc_trajectory_sampler/results/arc_agi2_representation_summary.json",
        help="Destination JSON path for the evaluation summary.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of grids to evaluate.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on the number of task JSON files to load for task-directory datasets.",
    )
    args = parser.parse_args()

    summary = evaluate_representation(
        Path(args.dataset_root),
        dataset_format=args.dataset_format,
        max_examples=args.max_examples,
        max_tasks=args.max_tasks,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote representation summary to {output_path}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
