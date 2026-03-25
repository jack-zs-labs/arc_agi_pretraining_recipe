from __future__ import annotations

"""Stage 2 episode construction for ARC-style synthetic tasks.

This module turns a Stage 1 ``TaskLatent`` into a concrete train/test episode
specification using reject sampling. It stays at the object-scene level for now:
the output is a structured blueprint that is ready for later execution and
rendering stages, but it does not yet emit final output grids.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .stage1_latent_sampler import (
        CueKind,
        Family,
        RelationAction,
        RoleVar,
        Selector,
        SelectorKind,
        ShapeClass,
        TaskLatent,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import (  # type: ignore
        CueKind,
        Family,
        RelationAction,
        RoleVar,
        Selector,
        SelectorKind,
        ShapeClass,
        TaskLatent,
    )


CANONICAL_SHAPES = (
    ShapeClass.RECT.value,
    ShapeClass.LINE.value,
    ShapeClass.DOT.value,
    ShapeClass.L_SHAPE.value,
    ShapeClass.CROSS.value,
    ShapeClass.REGULAR.value,
)

ORIENTATIONS = ("up", "right", "down", "left")
MARKER_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
SIZE_BUCKETS = ("small", "medium", "large")
PLACEMENT_SIDES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    shape: str
    color: int
    top: int
    left: int
    height: int
    width: int
    mass: int
    orientation: str = "up"
    holes: int = 0
    is_container: bool = False
    tags: Tuple[str, ...] = ()
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.top, self.left, self.bottom, self.right)


@dataclass(frozen=True)
class LegendEntry:
    object_id: str
    key: Any
    value: Any


@dataclass(frozen=True)
class SceneSpec:
    height: int
    width: int
    background_color: int
    border_color: Optional[int]
    outline_color: Optional[int]
    marker_position: Optional[str]
    objects: Tuple[ObjectSpec, ...]
    legend: Tuple[LegendEntry, ...] = ()
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleSpec:
    example_id: str
    input_scene: SceneSpec
    selected_object_ids: Tuple[str, ...]
    notes: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSpec:
    latent: TaskLatent
    role_bindings: Dict[str, Any]
    episode_metadata: Dict[str, Any]
    train_examples: Tuple[ExampleSpec, ...]
    test_example: ExampleSpec
    shortcut_checks: Tuple[str, ...]
    rejection_counts: Dict[str, int]
    sampling_attempts: int

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


def _encode(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, RoleVar):
        return asdict(obj)
    if isinstance(obj, tuple):
        return [_encode(item) for item in obj]
    if isinstance(obj, list):
        return [_encode(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if is_dataclass(obj):
        return {key: _encode(value) for key, value in asdict(obj).items()}
    return obj


def pick(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


def randint_inclusive(rng: random.Random, low: int, high: int) -> int:
    return rng.randint(low, high)


def choose_weighted(rng: random.Random, items: Sequence[Tuple[float, Any]]) -> Any:
    total = sum(weight for weight, _ in items)
    target = rng.random() * total
    running = 0.0
    for weight, item in items:
        running += weight
        if target <= running:
            return item
    return items[-1][1]


def resolve_value(value: Any, bindings: Dict[str, Any]) -> Any:
    if isinstance(value, RoleVar):
        return bindings[value.name]
    if isinstance(value, tuple):
        return tuple(resolve_value(item, bindings) for item in value)
    if isinstance(value, list):
        return [resolve_value(item, bindings) for item in value]
    if isinstance(value, dict):
        return {key: resolve_value(item, bindings) for key, item in value.items()}
    return value


def collect_role_vars(obj: Any, sink: Optional[Dict[str, RoleVar]] = None) -> Dict[str, RoleVar]:
    if sink is None:
        sink = {}
    if isinstance(obj, RoleVar):
        sink[obj.name] = obj
        return sink
    if isinstance(obj, dict):
        for item in obj.values():
            collect_role_vars(item, sink)
        return sink
    if isinstance(obj, (list, tuple)):
        for item in obj:
            collect_role_vars(item, sink)
        return sink
    if is_dataclass(obj):
        for field_name in obj.__dataclass_fields__:
            collect_role_vars(getattr(obj, field_name), sink)
    return sink


def resolve_role_bindings(latent: TaskLatent, rng: random.Random) -> Dict[str, Any]:
    role_vars = list(collect_role_vars(latent.program).values())
    color_roles = [role for role in role_vars if role.domain == "color"]
    shape_roles = [role for role in role_vars if role.domain == "shape"]
    int_roles = [role for role in role_vars if role.domain == "small_int"]

    bindings: Dict[str, Any] = {}
    available_colors = list(range(1, 10))
    rng.shuffle(available_colors)
    for role, color in zip(color_roles, available_colors, strict=False):
        bindings[role.name] = color

    shape_pool = list(dict.fromkeys(latent.input_schema.shape_bias + CANONICAL_SHAPES))
    rng.shuffle(shape_pool)
    if len(shape_pool) < len(shape_roles):
        shape_pool.extend(pick(rng, CANONICAL_SHAPES) for _ in range(len(shape_roles) - len(shape_pool)))
    for role, shape in zip(shape_roles, shape_pool, strict=False):
        bindings[role.name] = shape

    for role in int_roles:
        if role.name == "N_holes":
            bindings[role.name] = pick(rng, (1, 2))
        elif role.name == "N_repeat":
            bindings[role.name] = pick(rng, (2, 3))
        else:
            bindings[role.name] = pick(rng, (1, 2, 3))

    return bindings


def choose_grid(latent: TaskLatent, rng: random.Random) -> Tuple[int, int]:
    return (
        randint_inclusive(rng, *latent.input_schema.grid_h_range),
        randint_inclusive(rng, *latent.input_schema.grid_w_range),
    )


def choose_shape(latent: TaskLatent, rng: random.Random, shape_vocab: Optional[Sequence[str]] = None) -> str:
    allowed_shapes = tuple(shape_vocab) if shape_vocab else tuple(dict.fromkeys(latent.input_schema.shape_bias + CANONICAL_SHAPES))
    weighted = [(2.0, shape) for shape in allowed_shapes if shape in latent.input_schema.shape_bias]
    weighted.extend((0.6, shape) for shape in allowed_shapes if shape not in latent.input_schema.shape_bias)
    return choose_weighted(rng, weighted)


def choose_other_shape(
    latent: TaskLatent,
    rng: random.Random,
    forbidden: Sequence[str],
    shape_vocab: Optional[Sequence[str]] = None,
) -> str:
    shape_pool = list(shape_vocab) if shape_vocab else list(dict.fromkeys(latent.input_schema.shape_bias + CANONICAL_SHAPES))
    choices = [shape for shape in shape_pool if shape not in set(forbidden)]
    return pick(rng, choices or shape_pool)


def episode_visual_vocab(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    rng: random.Random,
    *,
    color_count: int = 3,
    shape_count: int = 3,
) -> Dict[str, Tuple[Any, ...]]:
    color_pool = list(range(1, latent.input_schema.max_distinct_colors + 1))
    bound_colors = [value for value in bindings.values() if isinstance(value, int) and value in color_pool]
    ordered_colors = list(dict.fromkeys(bound_colors))
    remaining_colors = [color for color in color_pool if color not in ordered_colors]
    rng.shuffle(remaining_colors)
    ordered_colors.extend(remaining_colors)

    shape_pool = list(dict.fromkeys(latent.input_schema.shape_bias + CANONICAL_SHAPES))
    bound_shapes = [value for value in bindings.values() if isinstance(value, str) and value in shape_pool]
    ordered_shapes = list(dict.fromkeys(bound_shapes))
    remaining_shapes = [shape for shape in shape_pool if shape not in ordered_shapes]
    rng.shuffle(remaining_shapes)
    ordered_shapes.extend(remaining_shapes)

    return {
        "color_vocab": tuple(ordered_colors[: max(1, min(color_count, len(ordered_colors)))]),
        "shape_vocab": tuple(ordered_shapes[: max(1, min(shape_count, len(ordered_shapes)))]),
    }


def restricted_vocab(
    vocab: Optional[Sequence[Any]],
    allowed: Optional[Sequence[Any]],
) -> Optional[Tuple[Any, ...]]:
    if allowed is None:
        return tuple(vocab) if vocab is not None else None
    allowed_values = tuple(allowed)
    if vocab is None:
        return allowed_values
    filtered = tuple(item for item in vocab if item in set(allowed_values))
    return filtered or tuple(vocab)


def choose_dims(shape: str, rng: random.Random) -> Tuple[int, int]:
    if shape == ShapeClass.DOT.value:
        return (1, 1)
    if shape == ShapeClass.LINE.value:
        if rng.random() < 0.5:
            return (1, randint_inclusive(rng, 2, 5))
        return (randint_inclusive(rng, 2, 5), 1)
    if shape == ShapeClass.L_SHAPE.value:
        arm = randint_inclusive(rng, 2, 4)
        return (arm, arm)
    if shape == ShapeClass.CROSS.value:
        arm = pick(rng, (3, 5))
        return (arm, arm)
    return (randint_inclusive(rng, 2, 4), randint_inclusive(rng, 2, 4))


def choose_compact_dims(shape: str, rng: random.Random) -> Tuple[int, int]:
    if shape == ShapeClass.DOT.value:
        return (1, 1)
    if shape == ShapeClass.LINE.value:
        return (1, pick(rng, (2, 3)))
    if shape == ShapeClass.L_SHAPE.value:
        return (2, 2)
    if shape == ShapeClass.CROSS.value:
        return (3, 3)
    return (pick(rng, (1, 2)), pick(rng, (2, 3)))


def shape_dim_candidates(shape: str, *, compact: bool = False) -> Tuple[Tuple[int, int], ...]:
    if shape == ShapeClass.DOT.value:
        return ((1, 1),)
    if shape == ShapeClass.LINE.value:
        compact_dims = ((1, 2), (2, 1), (1, 3), (3, 1))
        return compact_dims if compact else compact_dims + ((1, 4), (4, 1), (1, 5), (5, 1))
    if shape == ShapeClass.L_SHAPE.value:
        return ((2, 2), (3, 3)) if compact else ((2, 2), (3, 3), (4, 4))
    if shape == ShapeClass.CROSS.value:
        return ((3, 3),) if compact else ((3, 3), (5, 5))
    compact_dims = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2))
    return compact_dims if compact else compact_dims + ((3, 3), (3, 4), (4, 3), (4, 4))


def choose_preferred_shape(shape_vocab: Optional[Sequence[str]], preferred: Sequence[str], rng: random.Random) -> str:
    allowed = tuple(shape_vocab) if shape_vocab else tuple(CANONICAL_SHAPES)
    choices = [shape for shape in preferred if shape in allowed]
    return pick(rng, choices or allowed)


def compute_mass(
    shape: str,
    height: int,
    width: int,
    holes: int = 0,
    *,
    filled_cells: Optional[int] = None,
) -> int:
    if shape == "bitmap":
        if filled_cells is not None:
            return max(1, filled_cells)
        return max(1, height * width - holes)
    if shape == ShapeClass.DOT.value:
        return 1
    if shape == ShapeClass.LINE.value:
        return max(height, width)
    if shape == ShapeClass.L_SHAPE.value:
        return height + width - 1
    if shape == ShapeClass.CROSS.value:
        return height + width - 1
    return max(1, height * width - holes)


def bboxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def all_bbox_positions(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> List[Tuple[int, int]]:
    row_min = 0
    col_min = 0
    row_max = grid_h - box_h
    col_max = grid_w - box_w
    if region is not None:
        row_min = max(row_min, region[0])
        col_min = max(col_min, region[1])
        row_max = min(row_max, region[2] - box_h + 1)
        col_max = min(col_max, region[3] - box_w + 1)
    if row_max < row_min or col_max < col_min:
        return []
    return [(top, left) for top in range(row_min, row_max + 1) for left in range(col_min, col_max + 1)]


def place_bbox(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    occupied: Sequence[Tuple[int, int, int, int]],
    region: Optional[Tuple[int, int, int, int]] = None,
    tries: int = 64,
) -> Tuple[int, int]:
    positions = all_bbox_positions(grid_h, grid_w, box_h, box_w, region)
    if not positions:
        raise ValueError("object does not fit in requested region")

    for _ in range(tries):
        top, left = positions[rng.randrange(len(positions))]
        bbox = (top, left, top + box_h - 1, left + box_w - 1)
        if all(not bboxes_overlap(bbox, existing) for existing in occupied):
            return (top, left)
    valid_positions = []
    for top, left in positions:
        bbox = (top, left, top + box_h - 1, left + box_w - 1)
        if all(not bboxes_overlap(bbox, existing) for existing in occupied):
            valid_positions.append((top, left))
    if valid_positions:
        return pick(rng, valid_positions)
    raise ValueError("failed to place non-overlapping object")


def place_with_dim_options(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    occupied: Sequence[Tuple[int, int, int, int]],
    dim_options: Sequence[Tuple[int, int]],
    *,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    seen: set[Tuple[int, int]] = set()
    for box_h, box_w in dim_options:
        dims = (box_h, box_w)
        if dims in seen:
            continue
        seen.add(dims)
        try:
            top, left = place_bbox(rng, grid_h, grid_w, box_h, box_w, occupied, region=region)
            return top, left, box_h, box_w
        except ValueError:
            continue
    raise ValueError("failed to place non-overlapping object")


def intersect_regions(
    first: Optional[Tuple[int, int, int, int]],
    second: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if first is None:
        return second
    if second is None:
        return first
    region = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    if region[2] < region[0] or region[3] < region[1]:
        raise ValueError("placement regions do not intersect")
    return region


def region_capacity(region: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
    if region is None:
        return None
    return (region[2] - region[0] + 1, region[3] - region[1] + 1)


def make_object(
    object_id: str,
    shape: str,
    color: int,
    top: int,
    left: int,
    height: int,
    width: int,
    *,
    orientation: str = "up",
    holes: int = 0,
    is_container: bool = False,
    tags: Sequence[str] = (),
    attributes: Optional[Dict[str, Any]] = None,
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        shape=shape,
        color=color,
        top=top,
        left=left,
        height=height,
        width=width,
        mass=compute_mass(shape, height, width, holes),
        orientation=orientation,
        holes=holes,
        is_container=is_container,
        tags=tuple(tags),
        attributes=attributes or {},
    )


def collect_colors(scene: SceneSpec) -> set[int]:
    colors = {scene.background_color}
    if scene.border_color is not None:
        colors.add(scene.border_color)
    if scene.outline_color is not None:
        colors.add(scene.outline_color)
    for obj in scene.objects:
        colors.add(obj.color)
    return colors


def collect_shapes(scene: SceneSpec) -> set[str]:
    return {obj.shape for obj in scene.objects}


def choose_free_color(
    used: Iterable[int],
    rng: random.Random,
    palette: Optional[Sequence[int]] = None,
) -> int:
    used_set = set(used)
    palette_values = tuple(palette) if palette else tuple(range(1, 10))
    free = [color for color in palette_values if color not in used_set]
    if free:
        return pick(rng, free)
    if palette is not None:
        return pick(rng, palette_values)
    fallback = [color for color in range(1, 10) if color not in used_set]
    return pick(rng, fallback or palette_values)


def object_size_bucket(obj: ObjectSpec) -> str:
    if obj.mass <= 1:
        return "small"
    if obj.mass <= 3:
        return "medium"
    return "large"


def selector_matches(selector: Selector, obj: ObjectSpec, scene: SceneSpec, bindings: Dict[str, Any]) -> bool:
    if "frame" in obj.tags and selector.kind not in (SelectorKind.INSIDE_FRAME, SelectorKind.OUTSIDE_FRAME):
        return False
    if selector.kind == SelectorKind.ALL_OBJECTS:
        return True
    if selector.kind == SelectorKind.BY_COLOR:
        target_color = resolve_value(selector.args[0], bindings)
        return obj.color == target_color
    if selector.kind == SelectorKind.BY_SHAPE:
        target_shape = resolve_value(selector.args[0], bindings)
        return obj.shape == target_shape
    if selector.kind == SelectorKind.BY_HOLES:
        target_holes = resolve_value(selector.args[0], bindings)
        return obj.holes == target_holes
    if selector.kind == SelectorKind.INSIDE_FRAME:
        return "inside_frame" in obj.tags
    if selector.kind == SelectorKind.OUTSIDE_FRAME:
        return "outside_frame" in obj.tags
    return False


def select_objects(selector: Selector, scene: SceneSpec, bindings: Dict[str, Any]) -> Tuple[str, ...]:
    candidate_objects = [obj for obj in scene.objects if not (obj.is_container or "frame" in obj.tags)]
    if not candidate_objects:
        candidate_objects = list(scene.objects)
    if selector.kind == SelectorKind.ARGMAX_SIZE:
        winner = max(candidate_objects, key=lambda obj: (obj.mass, -int(obj.object_id.split("_")[-1])))
        if sum(obj.mass == winner.mass for obj in candidate_objects) != 1:
            return ()
        return (winner.object_id,)
    if selector.kind == SelectorKind.ARGMIN_SIZE:
        winner = min(candidate_objects, key=lambda obj: (obj.mass, obj.object_id))
        if sum(obj.mass == winner.mass for obj in candidate_objects) != 1:
            return ()
        return (winner.object_id,)
    selected = [obj.object_id for obj in scene.objects if selector_matches(selector, obj, scene, bindings)]
    return tuple(selected)


def add_distractors(
    latent: TaskLatent,
    scene_objects: List[ObjectSpec],
    occupied: List[Tuple[int, int, int, int]],
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    target_count: int,
    forbidden_colors: Sequence[int] = (),
    forbidden_shapes: Sequence[str] = (),
    region: Optional[Tuple[int, int, int, int]] = None,
    blocked_bboxes: Sequence[Tuple[int, int, int, int]] = (),
    color_vocab: Optional[Sequence[int]] = None,
    shape_vocab: Optional[Sequence[str]] = None,
    max_mass: Optional[int] = None,
    min_mass: Optional[int] = None,
    compact_shapes: bool = False,
    forbidden_shape_color_pairs: Sequence[Tuple[str, int]] = (),
) -> None:
    next_index = len(scene_objects)
    used_colors = {obj.color for obj in scene_objects}
    forbidden_pairs = set(forbidden_shape_color_pairs)
    palette_values = tuple(color_vocab) if color_vocab else tuple(range(1, 10))
    while len(scene_objects) < target_count:
        placed = False
        for _ in range(48):
            shape = choose_shape(latent, rng, shape_vocab)
            if shape in forbidden_shapes:
                shape = choose_other_shape(latent, rng, forbidden_shapes, shape_vocab)
            color = choose_free_color(used_colors | set(forbidden_colors), rng, color_vocab)
            if (shape, color) in forbidden_pairs:
                continue
            box_h, box_w = choose_compact_dims(shape, rng) if compact_shapes else choose_dims(shape, rng)
            mass = compute_mass(shape, box_h, box_w)
            if max_mass is not None and mass >= max_mass:
                continue
            if min_mass is not None and mass <= min_mass:
                continue
            try:
                top, left = place_bbox(
                    rng,
                    grid_h,
                    grid_w,
                    box_h,
                    box_w,
                    list(occupied) + list(blocked_bboxes),
                    region=region,
                )
            except ValueError:
                continue
            distractor = make_object(
                object_id=f"obj_{next_index}",
                shape=shape,
                color=color,
                top=top,
                left=left,
                height=box_h,
                width=box_w,
                orientation=pick(rng, ORIENTATIONS),
                tags=("distractor",),
            )
            scene_objects.append(distractor)
            occupied.append(distractor.bbox)
            used_colors.add(color)
            next_index += 1
            placed = True
            break
        if not placed:
            color_candidates = [
                color for color in palette_values if color not in used_colors and color not in set(forbidden_colors)
            ]
            color_candidates.extend(
                color
                for color in palette_values
                if color not in set(forbidden_colors) and color not in color_candidates
            )
            shape_candidates = tuple(shape_vocab) if shape_vocab else tuple(dict.fromkeys(latent.input_schema.shape_bias + CANONICAL_SHAPES))
            for shape in shape_candidates:
                if shape in forbidden_shapes:
                    continue
                dim_candidates = shape_dim_candidates(shape, compact=compact_shapes)
                for color in color_candidates:
                    if (shape, color) in forbidden_pairs:
                        continue
                    dim_options = []
                    for box_h, box_w in dim_candidates:
                        mass = compute_mass(shape, box_h, box_w)
                        if max_mass is not None and mass >= max_mass:
                            continue
                        if min_mass is not None and mass <= min_mass:
                            continue
                        dim_options.append((box_h, box_w))
                    if not dim_options:
                        continue
                    try:
                        top, left, box_h, box_w = place_with_dim_options(
                            rng,
                            grid_h,
                            grid_w,
                            list(occupied) + list(blocked_bboxes),
                            dim_options,
                            region=region,
                        )
                    except ValueError:
                        continue
                    distractor = make_object(
                        object_id=f"obj_{next_index}",
                        shape=shape,
                        color=color,
                        top=top,
                        left=left,
                        height=box_h,
                        width=box_w,
                        orientation=pick(rng, ORIENTATIONS),
                        tags=("distractor",),
                    )
                    scene_objects.append(distractor)
                    occupied.append(distractor.bbox)
                    used_colors.add(color)
                    next_index += 1
                    placed = True
                    break
                if placed:
                    break
        if not placed:
            raise ValueError("failed to place non-overlapping object")


def make_frame(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    color: int,
    occupied: List[Tuple[int, int, int, int]],
    *,
    size_mode: str = "large",
) -> ObjectSpec:
    if size_mode == "tiny":
        frame_h = randint_inclusive(rng, 3, max(3, max(3, grid_h // 4)))
        frame_w = randint_inclusive(rng, 3, max(3, max(3, grid_w // 4)))
    elif size_mode == "small":
        frame_h = randint_inclusive(rng, 3, max(3, max(4, grid_h // 3)))
        frame_w = randint_inclusive(rng, 3, max(3, max(4, grid_w // 3)))
    else:
        frame_h = randint_inclusive(rng, max(4, grid_h // 2), max(4, grid_h - 1))
        frame_w = randint_inclusive(rng, max(4, grid_w // 2), max(4, grid_w - 1))
    top, left = place_bbox(rng, grid_h, grid_w, frame_h, frame_w, occupied)
    frame = make_object(
        object_id="obj_0",
        shape=ShapeClass.RECT.value,
        color=color,
        top=top,
        left=left,
        height=frame_h,
        width=frame_w,
        holes=1,
        is_container=True,
        tags=("frame",),
    )
    return frame


def choose_branch_schedule(rng: random.Random, num_train: int) -> Tuple[str, ...]:
    train_schedule = ["then", "else"]
    while len(train_schedule) < num_train:
        train_schedule.append(pick(rng, ("then", "else")))
    rng.shuffle(train_schedule)
    return tuple(train_schedule + [pick(rng, ("then", "else"))])


def sample_unary_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    rng: random.Random,
    example_id: str,
    episode_metadata: Optional[Dict[str, Any]] = None,
    allowed_colors: Optional[Sequence[int]] = None,
    allowed_shapes: Optional[Sequence[str]] = None,
) -> ExampleSpec:
    selector, action = latent.program.args
    episode_metadata = episode_metadata or {}
    color_vocab = restricted_vocab(episode_metadata.get("color_vocab"), allowed_colors)
    shape_vocab = restricted_vocab(episode_metadata.get("shape_vocab"), allowed_shapes)
    min_objects, max_objects = latent.input_schema.num_objects_range
    capped_max_objects = max_objects
    if selector.kind == SelectorKind.INSIDE_FRAME:
        capped_max_objects = min(capped_max_objects, 3)
    if selector.kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE):
        capped_max_objects = min(capped_max_objects, 3)
    if action.name == "translate":
        capped_max_objects = min(capped_max_objects, 4)
    if selector.kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE):
        target_total = 2
    else:
        target_total = max(2, randint_inclusive(rng, min_objects, max(min_objects, capped_max_objects)))
    grid_h_low, grid_h_high = latent.input_schema.grid_h_range
    grid_w_low, grid_w_high = latent.input_schema.grid_w_range
    grid_load_bias = 4 + target_total // 2
    if selector.kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE):
        grid_load_bias += 2
    if action.name == "translate":
        grid_load_bias += 1
    grid_h = randint_inclusive(rng, min(grid_h_high, max(grid_h_low, grid_load_bias)), grid_h_high)
    grid_w = randint_inclusive(rng, min(grid_w_high, max(grid_w_low, grid_load_bias)), grid_w_high)
    occupied: List[Tuple[int, int, int, int]] = []
    objects: List[ObjectSpec] = []
    notes: List[str] = []

    base_region: Optional[Tuple[int, int, int, int]] = None
    blocked_bboxes: List[Tuple[int, int, int, int]] = []
    if selector.kind == SelectorKind.INSIDE_FRAME:
        frame_color = bindings.get("C_frame", choose_free_color([], rng, color_vocab))
        frame = make_frame(rng, grid_h, grid_w, frame_color, occupied, size_mode="small")
        objects.append(frame)
        base_region = (frame.top + 1, frame.left + 1, frame.bottom - 1, frame.right - 1)
        blocked_bboxes.append(frame.bbox)
        notes.append("frame present for inside-frame selection")

    transform_name = action.name
    target_shape = choose_shape(latent, rng, shape_vocab)
    target_color = choose_free_color({obj.color for obj in objects}, rng, color_vocab)
    target_holes = 0
    selector_signature = selector.kind.value

    if selector.kind == SelectorKind.BY_COLOR:
        target_color = resolve_value(selector.args[0], bindings)
    elif selector.kind == SelectorKind.BY_SHAPE:
        target_shape = resolve_value(selector.args[0], bindings)
    elif selector.kind == SelectorKind.BY_HOLES:
        target_holes = resolve_value(selector.args[0], bindings)
        if target_holes < 1:
            target_holes = 1
    elif selector.kind == SelectorKind.ARGMIN_SIZE:
        target_shape = choose_preferred_shape(
            shape_vocab,
            (ShapeClass.DOT.value, ShapeClass.LINE.value, ShapeClass.RECT.value, ShapeClass.REGULAR.value),
            rng,
        )
    elif selector.kind == SelectorKind.ARGMAX_SIZE:
        target_shape = choose_preferred_shape(
            shape_vocab,
            (ShapeClass.RECT.value, ShapeClass.REGULAR.value, ShapeClass.L_SHAPE.value),
            rng,
        )
    if transform_name == "fill_holes":
        target_holes = max(target_holes, 1)

    placement_region = base_region
    if selector.kind == SelectorKind.ARGMIN_SIZE:
        box_h, box_w = choose_compact_dims(target_shape, rng)
    else:
        box_h, box_w = choose_dims(target_shape, rng)
    capacity = region_capacity(placement_region)
    if capacity is not None:
        box_h = min(box_h, capacity[0])
        box_w = min(box_w, capacity[1])
    if transform_name == "scale_up":
        factor = resolve_value(action.params["factor"], bindings)
        box_h = min(box_h, max(1, grid_h // factor))
        box_w = min(box_w, max(1, grid_w // factor))
        if capacity is not None:
            box_h = min(box_h, max(1, capacity[0] // factor))
            box_w = min(box_w, max(1, capacity[1] // factor))
    if transform_name == "translate":
        direction = resolve_value(action.params["direction"], bindings)
        distance = resolve_value(action.params["distance"], bindings)
        if capacity is not None:
            if direction in ("up", "down", "diag_up_right"):
                distance = min(distance, max(0, capacity[0] - box_h))
            else:
                distance = min(distance, max(0, capacity[1] - box_w))
        else:
            if direction in ("up", "down", "diag_up_right"):
                distance = min(distance, max(0, grid_h - box_h))
            else:
                distance = min(distance, max(0, grid_w - box_w))
        if direction in ("down", "diag_up_right"):
            placement_region = intersect_regions(
                placement_region,
                (0, 0, grid_h - distance - box_h, grid_w - box_w),
            )
        elif direction == "up":
            placement_region = intersect_regions(
                placement_region,
                (distance, 0, grid_h - box_h, grid_w - box_w),
            )
        elif direction == "right":
            placement_region = intersect_regions(
                placement_region,
                (0, 0, grid_h - box_h, grid_w - distance - box_w),
            )
        elif direction == "left":
            placement_region = intersect_regions(
                placement_region,
                (0, distance, grid_h - box_h, grid_w - box_w),
            )
        capacity = region_capacity(placement_region)
        if capacity is not None:
            box_h = min(box_h, capacity[0])
            box_w = min(box_w, capacity[1])

    target_top, target_left = place_bbox(
        rng,
        grid_h,
        grid_w,
        box_h,
        box_w,
        occupied,
        region=placement_region,
    )
    target_tags = ["selected"]
    if selector.kind == SelectorKind.INSIDE_FRAME:
        target_tags.append("inside_frame")
    target = make_object(
        object_id=f"obj_{len(objects)}",
        shape=target_shape,
        color=target_color,
        top=target_top,
        left=target_left,
        height=box_h,
        width=box_w,
        orientation=pick(rng, ORIENTATIONS),
        holes=target_holes,
        tags=target_tags,
        attributes={"selector_signature": selector_signature},
    )
    objects.append(target)
    occupied.append(target.bbox)

    forbidden_colors = (target_color,) if selector.kind == SelectorKind.BY_COLOR else ()
    forbidden_shapes = (target_shape,) if selector.kind == SelectorKind.BY_SHAPE else ()
    distractor_max_mass = target.mass if selector.kind == SelectorKind.ARGMAX_SIZE else None
    distractor_min_mass = target.mass if selector.kind == SelectorKind.ARGMIN_SIZE else None
    add_distractors(
        latent,
        objects,
        occupied,
        rng,
        grid_h,
        grid_w,
        target_total,
        forbidden_colors=forbidden_colors,
        forbidden_shapes=forbidden_shapes,
        blocked_bboxes=blocked_bboxes,
        color_vocab=color_vocab,
        shape_vocab=shape_vocab,
        max_mass=distractor_max_mass,
        min_mass=distractor_min_mass,
        compact_shapes=selector.kind == SelectorKind.ARGMAX_SIZE,
    )

    scene = SceneSpec(
        height=grid_h,
        width=grid_w,
        background_color=0,
        border_color=None,
        outline_color=None,
        marker_position=None,
        objects=tuple(objects),
        attributes={"family": latent.family.value},
    )
    selected = select_objects(selector, scene, bindings)
    if not selected:
        raise ValueError("unary selector did not pick any object")
    return ExampleSpec(
        example_id=example_id,
        input_scene=scene,
        selected_object_ids=selected,
        notes=tuple(notes),
        metadata={
            "diversity_token": f"{selector.kind.value}:{transform_name}:{grid_h}x{grid_w}:{target.shape}:{target.color}:{target.mass}",
            "selector_kind": selector.kind.value,
            "transform": transform_name,
        },
    )


def relational_pair_positions(
    action_name: str,
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    params: Dict[str, Any],
) -> Tuple[Tuple[int, int], Tuple[int, int], Dict[str, Any]]:
    if action_name == RelationAction.MOVE_UNTIL_CONTACT.value:
        direction = params["direction"]
        gap = randint_inclusive(rng, 1, 3)
        if direction == "up":
            col = randint_inclusive(rng, 0, max(0, grid_w - max(src_w, dst_w)))
            dst_top = randint_inclusive(rng, 0, max(0, grid_h - dst_h - src_h - gap - 1))
            src_top = dst_top + dst_h + gap
            return (src_top, col), (dst_top, col), {"gap": gap, "direction": direction}
        if direction == "down":
            col = randint_inclusive(rng, 0, max(0, grid_w - max(src_w, dst_w)))
            src_top = randint_inclusive(rng, 0, max(0, grid_h - src_h - dst_h - gap - 1))
            dst_top = src_top + src_h + gap
            return (src_top, col), (dst_top, col), {"gap": gap, "direction": direction}
        if direction == "left":
            row = randint_inclusive(rng, 0, max(0, grid_h - max(src_h, dst_h)))
            dst_left = randint_inclusive(rng, 0, max(0, grid_w - dst_w - src_w - gap - 1))
            src_left = dst_left + dst_w + gap
            return (row, src_left), (row, dst_left), {"gap": gap, "direction": direction}
        row = randint_inclusive(rng, 0, max(0, grid_h - max(src_h, dst_h)))
        src_left = randint_inclusive(rng, 0, max(0, grid_w - src_w - dst_w - gap - 1))
        dst_left = src_left + src_w + gap
        return (row, src_left), (row, dst_left), {"gap": gap, "direction": direction}

    src_top = randint_inclusive(rng, 0, max(0, grid_h - src_h))
    src_left = randint_inclusive(rng, 0, max(0, grid_w - src_w))
    dst_top = randint_inclusive(rng, 0, max(0, grid_h - dst_h))
    dst_left = randint_inclusive(rng, 0, max(0, grid_w - dst_w))
    if action_name == RelationAction.ALIGN_TO.value:
        axis = params["axis"]
        gap = randint_inclusive(rng, 1, 2)
        if axis == "x" and grid_h >= src_h + dst_h + gap:
            shared_left = randint_inclusive(rng, 0, max(0, grid_w - max(src_w, dst_w)))
            src_left = shared_left
            dst_left = shared_left
            src_top = randint_inclusive(rng, 0, max(0, grid_h - src_h - dst_h - gap))
            dst_top = src_top + src_h + gap
        elif axis == "y" and grid_w >= src_w + dst_w + gap:
            shared_top = randint_inclusive(rng, 0, max(0, grid_h - max(src_h, dst_h)))
            src_top = shared_top
            dst_top = shared_top
            src_left = randint_inclusive(rng, 0, max(0, grid_w - src_w - dst_w - gap))
            dst_left = src_left + src_w + gap
        elif axis == "both" and grid_h >= src_h + dst_h + gap and grid_w >= src_w + dst_w + gap:
            src_top = randint_inclusive(rng, 0, max(0, grid_h - src_h - dst_h - gap))
            src_left = randint_inclusive(rng, 0, max(0, grid_w - src_w - dst_w - gap))
            dst_top = src_top + src_h + gap
            dst_left = src_left + src_w + gap
        else:
            if grid_h >= src_h + dst_h + gap:
                shared_left = randint_inclusive(rng, 0, max(0, grid_w - max(src_w, dst_w)))
                src_left = shared_left
                dst_left = shared_left
                src_top = randint_inclusive(rng, 0, max(0, grid_h - src_h - dst_h - gap))
                dst_top = src_top + src_h + gap
            elif grid_w >= src_w + dst_w + gap:
                shared_top = randint_inclusive(rng, 0, max(0, grid_h - max(src_h, dst_h)))
                src_top = shared_top
                dst_top = shared_top
                src_left = randint_inclusive(rng, 0, max(0, grid_w - src_w - dst_w - gap))
                dst_left = src_left + src_w + gap
        return (src_top, src_left), (dst_top, dst_left), {"axis": axis}
    if action_name == RelationAction.STACK_NEXT_TO.value:
        return (src_top, src_left), (dst_top, dst_left), {"side": params["side"]}
    if action_name == RelationAction.DRAW_CONNECTING_LINE.value:
        return (src_top, src_left), (dst_top, dst_left), {"line_type": params["line_type"]}
    return (src_top, src_left), (dst_top, dst_left), {"fit_mode": params.get("fit_mode", "center")}


def relational_pair_fallback_positions(
    action_name: str,
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    params: Dict[str, Any],
    blocked_bboxes: Sequence[Tuple[int, int, int, int]],
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Dict[str, Any]]]:
    def valid_pair(src_pos: Tuple[int, int], dst_pos: Tuple[int, int]) -> bool:
        src_bbox = (src_pos[0], src_pos[1], src_pos[0] + src_h - 1, src_pos[1] + src_w - 1)
        dst_bbox = (dst_pos[0], dst_pos[1], dst_pos[0] + dst_h - 1, dst_pos[1] + dst_w - 1)
        if bboxes_overlap(src_bbox, dst_bbox):
            return False
        return not any(bboxes_overlap(src_bbox, blocked) or bboxes_overlap(dst_bbox, blocked) for blocked in blocked_bboxes)

    if action_name == RelationAction.MOVE_UNTIL_CONTACT.value:
        direction = params["direction"]
        gaps = [1, 2, 3]
        rng.shuffle(gaps)
        candidates: List[Tuple[Tuple[int, int], Tuple[int, int], Dict[str, Any]]] = []
        for gap in gaps:
            if direction == "up":
                for col in range(max(0, grid_w - max(src_w, dst_w)) + 1):
                    for dst_top in range(max(0, grid_h - dst_h - src_h - gap - 1) + 1):
                        src_top = dst_top + dst_h + gap
                        candidates.append(((src_top, col), (dst_top, col), {"gap": gap, "direction": direction}))
            elif direction == "down":
                for col in range(max(0, grid_w - max(src_w, dst_w)) + 1):
                    for src_top in range(max(0, grid_h - src_h - dst_h - gap - 1) + 1):
                        dst_top = src_top + src_h + gap
                        candidates.append(((src_top, col), (dst_top, col), {"gap": gap, "direction": direction}))
            elif direction == "left":
                for row in range(max(0, grid_h - max(src_h, dst_h)) + 1):
                    for dst_left in range(max(0, grid_w - dst_w - src_w - gap - 1) + 1):
                        src_left = dst_left + dst_w + gap
                        candidates.append(((row, src_left), (row, dst_left), {"gap": gap, "direction": direction}))
            else:
                for row in range(max(0, grid_h - max(src_h, dst_h)) + 1):
                    for src_left in range(max(0, grid_w - src_w - dst_w - gap - 1) + 1):
                        dst_left = src_left + src_w + gap
                        candidates.append(((row, src_left), (row, dst_left), {"gap": gap, "direction": direction}))
        rng.shuffle(candidates)
        for src_pos, dst_pos, relation_meta in candidates:
            if valid_pair(src_pos, dst_pos):
                return src_pos, dst_pos, relation_meta
        return None

    if action_name == RelationAction.ALIGN_TO.value:
        axis = params["axis"]
        gaps = [1, 2]
        rng.shuffle(gaps)
        candidates: List[Tuple[Tuple[int, int], Tuple[int, int], Dict[str, Any]]] = []
        for gap in gaps:
            if axis == "x" or (axis == "both" and grid_h >= src_h + dst_h + gap):
                for shared_left in range(max(0, grid_w - max(src_w, dst_w)) + 1):
                    for src_top in range(max(0, grid_h - src_h - dst_h - gap) + 1):
                        dst_top = src_top + src_h + gap
                        candidates.append(((src_top, shared_left), (dst_top, shared_left), {"axis": axis}))
            if axis == "y" or (axis == "both" and grid_w >= src_w + dst_w + gap):
                for shared_top in range(max(0, grid_h - max(src_h, dst_h)) + 1):
                    for src_left in range(max(0, grid_w - src_w - dst_w - gap) + 1):
                        dst_left = src_left + src_w + gap
                        candidates.append(((shared_top, src_left), (shared_top, dst_left), {"axis": axis}))
            if axis == "both" and grid_h >= src_h + dst_h + gap and grid_w >= src_w + dst_w + gap:
                for src_top in range(max(0, grid_h - src_h - dst_h - gap) + 1):
                    for src_left in range(max(0, grid_w - src_w - dst_w - gap) + 1):
                        dst_top = src_top + src_h + gap
                        dst_left = src_left + src_w + gap
                        candidates.append(((src_top, src_left), (dst_top, dst_left), {"axis": axis}))
        rng.shuffle(candidates)
        for src_pos, dst_pos, relation_meta in candidates:
            if valid_pair(src_pos, dst_pos):
                return src_pos, dst_pos, relation_meta
        return None

    src_positions = all_bbox_positions(grid_h, grid_w, src_h, src_w)
    dst_positions = all_bbox_positions(grid_h, grid_w, dst_h, dst_w)
    rng.shuffle(src_positions)
    rng.shuffle(dst_positions)
    if action_name == RelationAction.STACK_NEXT_TO.value:
        relation_meta = {"side": params["side"]}
    elif action_name == RelationAction.DRAW_CONNECTING_LINE.value:
        relation_meta = {"line_type": params["line_type"]}
    else:
        relation_meta = {"fit_mode": params.get("fit_mode", "center")}
    for src_pos in src_positions:
        for dst_pos in dst_positions:
            if valid_pair(src_pos, dst_pos):
                return src_pos, dst_pos, relation_meta
    return None


def sample_relational_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    rng: random.Random,
    example_id: str,
    episode_metadata: Optional[Dict[str, Any]] = None,
    allowed_colors: Optional[Sequence[int]] = None,
    allowed_shapes: Optional[Sequence[str]] = None,
) -> ExampleSpec:
    src_selector, dst_selector, action = latent.program.args
    episode_metadata = episode_metadata or {}
    color_vocab = restricted_vocab(episode_metadata.get("color_vocab"), allowed_colors)
    shape_vocab = restricted_vocab(episode_metadata.get("shape_vocab"), allowed_shapes)
    min_objects, max_objects = latent.input_schema.num_objects_range
    capped_max_objects = max_objects
    heavy_relation = dst_selector.kind == SelectorKind.OUTSIDE_FRAME or action.name in (
        RelationAction.PLACE_IN_CONTAINER.value,
        RelationAction.DRAW_CONNECTING_LINE.value,
    )
    mixed_selector_relation = {src_selector.kind, dst_selector.kind} == {SelectorKind.BY_COLOR, SelectorKind.BY_SHAPE}
    if heavy_relation:
        capped_max_objects = min(capped_max_objects, 4)
    if mixed_selector_relation:
        capped_max_objects = min(capped_max_objects, 3)
    target_total = max(3, randint_inclusive(rng, min_objects, max(min_objects, capped_max_objects)))
    grid_h_low, grid_h_high = latent.input_schema.grid_h_range
    grid_w_low, grid_w_high = latent.input_schema.grid_w_range
    grid_load_bias = 6 + target_total // 2
    if dst_selector.kind == SelectorKind.OUTSIDE_FRAME:
        grid_load_bias += 2
    if action.name == RelationAction.PLACE_IN_CONTAINER.value:
        grid_load_bias += 2
    if action.name == RelationAction.ALIGN_TO.value and action.params.get("axis") == "both":
        grid_load_bias += 1
    grid_h = randint_inclusive(rng, min(grid_h_high, max(grid_h_low, grid_load_bias)), grid_h_high)
    grid_w = randint_inclusive(rng, min(grid_w_high, max(grid_w_low, grid_load_bias)), grid_w_high)
    occupied: List[Tuple[int, int, int, int]] = []
    objects: List[ObjectSpec] = []
    notes: List[str] = []
    blocked_bboxes: List[Tuple[int, int, int, int]] = []

    if dst_selector.kind == SelectorKind.OUTSIDE_FRAME:
        frame_color = bindings.get("C_frame", choose_free_color([], rng, color_vocab))
        frame = make_frame(rng, grid_h, grid_w, frame_color, occupied, size_mode="tiny")
        objects.append(frame)
        blocked_bboxes.append(frame.bbox)
        notes.append("frame present to support outside-frame target selection")

    src_shape = choose_shape(latent, rng, shape_vocab)
    dst_shape = choose_shape(latent, rng, shape_vocab)
    src_color = choose_free_color([], rng, color_vocab)
    dst_color = choose_free_color([src_color], rng, color_vocab)

    if src_selector.kind == SelectorKind.BY_COLOR:
        src_color = resolve_value(src_selector.args[0], bindings)
    elif src_selector.kind == SelectorKind.BY_SHAPE:
        src_shape = resolve_value(src_selector.args[0], bindings)
    if dst_selector.kind == SelectorKind.BY_COLOR:
        dst_color = resolve_value(dst_selector.args[0], bindings)
    elif dst_selector.kind == SelectorKind.BY_SHAPE:
        dst_shape = resolve_value(dst_selector.args[0], bindings)

    if src_selector.kind != SelectorKind.BY_COLOR and dst_selector.kind == SelectorKind.BY_COLOR and src_color == dst_color:
        src_color = choose_free_color((dst_color,), rng, color_vocab)
    if dst_selector.kind != SelectorKind.BY_COLOR and src_selector.kind == SelectorKind.BY_COLOR and dst_color == src_color:
        dst_color = choose_free_color((src_color,), rng, color_vocab)
    if src_selector.kind != SelectorKind.BY_SHAPE and dst_selector.kind == SelectorKind.BY_SHAPE and src_shape == dst_shape:
        src_shape = choose_other_shape(latent, rng, (dst_shape,), shape_vocab)
    if dst_selector.kind != SelectorKind.BY_SHAPE and src_selector.kind == SelectorKind.BY_SHAPE and dst_shape == src_shape:
        dst_shape = choose_other_shape(latent, rng, (src_shape,), shape_vocab)
    if src_selector.kind == SelectorKind.ARGMAX_SIZE:
        preferred = [shape for shape in (ShapeClass.RECT.value, ShapeClass.REGULAR.value) if shape in (shape_vocab or ())]
        src_shape = pick(rng, preferred or (ShapeClass.RECT.value, ShapeClass.REGULAR.value))
        if dst_selector.kind == SelectorKind.BY_SHAPE and src_shape == dst_shape:
            src_shape = choose_other_shape(latent, rng, (dst_shape,), shape_vocab)

    forbidden_shape_color_pairs: List[Tuple[str, int]] = []
    if src_selector.kind == SelectorKind.BY_SHAPE and dst_selector.kind == SelectorKind.BY_COLOR:
        forbidden_shape_color_pairs.append((src_shape, dst_color))
    if src_selector.kind == SelectorKind.BY_COLOR and dst_selector.kind == SelectorKind.BY_SHAPE:
        forbidden_shape_color_pairs.append((dst_shape, src_color))

    src_h, src_w = choose_dims(src_shape, rng)
    dst_h, dst_w = choose_dims(dst_shape, rng)
    action_params = resolve_value(action.params, bindings)

    if src_selector.kind == SelectorKind.ARGMAX_SIZE:
        src_h = max(src_h, dst_h + 1, 3)
        src_w = max(src_w, dst_w + 1, 3)
    if action.name == RelationAction.PLACE_IN_CONTAINER.value:
        dst_h = max(dst_h, src_h + 2)
        dst_w = max(dst_w, src_w + 2)

    relation_meta: Dict[str, Any] = {}
    src_pos: Optional[Tuple[int, int]] = None
    dst_pos: Optional[Tuple[int, int]] = None
    for _ in range(32):
        candidate_src, candidate_dst, relation_meta = relational_pair_positions(
            action.name,
            rng,
            grid_h,
            grid_w,
            src_h,
            src_w,
            dst_h,
            dst_w,
            action_params,
        )
        src_bbox = (
            candidate_src[0],
            candidate_src[1],
            candidate_src[0] + src_h - 1,
            candidate_src[1] + src_w - 1,
        )
        dst_bbox = (
            candidate_dst[0],
            candidate_dst[1],
            candidate_dst[0] + dst_h - 1,
            candidate_dst[1] + dst_w - 1,
        )
        if bboxes_overlap(src_bbox, dst_bbox):
            continue
        if any(bboxes_overlap(src_bbox, blocked) or bboxes_overlap(dst_bbox, blocked) for blocked in blocked_bboxes):
            continue
        src_pos = candidate_src
        dst_pos = candidate_dst
        break
    if src_pos is None or dst_pos is None:
        fallback = relational_pair_fallback_positions(
            action.name,
            rng,
            grid_h,
            grid_w,
            src_h,
            src_w,
            dst_h,
            dst_w,
            action_params,
            blocked_bboxes,
        )
        if fallback is not None:
            src_pos, dst_pos, relation_meta = fallback
    if src_pos is None or dst_pos is None:
        raise ValueError("failed to place relational pair")

    dst_tags = ["target"]
    if action.name == RelationAction.PLACE_IN_CONTAINER.value:
        dst_tags.append("container")
        notes.append("container target for placement relation")
    src_object_id = f"obj_{len(objects)}"
    dst_object_id = f"obj_{len(objects) + 1}"
    src = make_object(
        object_id=src_object_id,
        shape=src_shape,
        color=src_color,
        top=src_pos[0],
        left=src_pos[1],
        height=src_h,
        width=src_w,
        orientation=pick(rng, ORIENTATIONS),
        tags=("source",),
    )
    dst = make_object(
        object_id=dst_object_id,
        shape=dst_shape,
        color=dst_color,
        top=dst_pos[0],
        left=dst_pos[1],
        height=dst_h,
        width=dst_w,
        orientation=pick(rng, ORIENTATIONS),
        is_container=action.name == RelationAction.PLACE_IN_CONTAINER.value,
        tags=dst_tags,
    )
    objects.extend([src, dst])
    occupied.extend([src.bbox, dst.bbox])

    add_distractors(
        latent,
        objects,
        occupied,
        rng,
        grid_h,
        grid_w,
        target_total,
        forbidden_colors=(src.color, dst.color),
        blocked_bboxes=blocked_bboxes,
        color_vocab=color_vocab,
        shape_vocab=shape_vocab,
        max_mass=src.mass if src_selector.kind == SelectorKind.ARGMAX_SIZE else None,
        forbidden_shape_color_pairs=forbidden_shape_color_pairs,
    )

    if dst_selector.kind == SelectorKind.OUTSIDE_FRAME:
        objects = [
            make_object(
                object_id=obj.object_id,
                shape=obj.shape,
                color=obj.color,
                top=obj.top,
                left=obj.left,
                height=obj.height,
                width=obj.width,
                orientation=obj.orientation,
                holes=obj.holes,
                is_container=obj.is_container,
                tags=obj.tags + (("outside_frame",) if obj.object_id == dst.object_id else ()),
                attributes=obj.attributes,
            )
            if obj.object_id == dst.object_id
            else obj
            for obj in objects
        ]

    scene = SceneSpec(
        height=grid_h,
        width=grid_w,
        background_color=0,
        border_color=None,
        outline_color=None,
        marker_position=None,
        objects=tuple(objects),
        attributes={"family": latent.family.value},
    )
    selected_source = select_objects(src_selector, scene, bindings)
    selected_target = select_objects(dst_selector, scene, bindings)
    if set(selected_source) & set(selected_target) and src_selector.kind == SelectorKind.ARGMAX_SIZE:
        src_index = next((index for index, obj in enumerate(objects) if obj.object_id == src_object_id), None)
        if src_index is not None:
            src_obj = objects[src_index]
            repaired_scene: Optional[SceneSpec] = None
            repaired_source: Tuple[str, ...] = ()
            repaired_target: Tuple[str, ...] = ()
            if dst_selector.kind == SelectorKind.BY_SHAPE:
                target_shape = resolve_value(dst_selector.args[0], bindings)
                candidate_shapes = [
                    shape
                    for shape in (ShapeClass.RECT.value, ShapeClass.REGULAR.value)
                    if shape != target_shape and shape in (shape_vocab or (ShapeClass.RECT.value, ShapeClass.REGULAR.value))
                ]
                for alt_shape in candidate_shapes:
                    alt_src = make_object(
                        object_id=src_obj.object_id,
                        shape=alt_shape,
                        color=src_obj.color,
                        top=src_obj.top,
                        left=src_obj.left,
                        height=src_obj.height,
                        width=src_obj.width,
                        orientation=src_obj.orientation,
                        holes=src_obj.holes,
                        is_container=src_obj.is_container,
                        tags=src_obj.tags,
                        attributes=src_obj.attributes,
                    )
                    repaired_objects = list(objects)
                    repaired_objects[src_index] = alt_src
                    candidate_scene = SceneSpec(
                        height=scene.height,
                        width=scene.width,
                        background_color=scene.background_color,
                        border_color=scene.border_color,
                        outline_color=scene.outline_color,
                        marker_position=scene.marker_position,
                        objects=tuple(repaired_objects),
                        attributes=scene.attributes,
                    )
                    candidate_source = select_objects(src_selector, candidate_scene, bindings)
                    candidate_target = select_objects(dst_selector, candidate_scene, bindings)
                    if candidate_source and candidate_target and not (set(candidate_source) & set(candidate_target)):
                        repaired_scene = candidate_scene
                        repaired_source = candidate_source
                        repaired_target = candidate_target
                        break
            elif dst_selector.kind == SelectorKind.BY_COLOR:
                target_color = resolve_value(dst_selector.args[0], bindings)
                palette_values = tuple(color_vocab) if color_vocab else tuple(range(1, 10))
                candidate_colors = [color for color in palette_values if color != target_color and color != src_obj.color]
                for alt_color in candidate_colors:
                    alt_src = make_object(
                        object_id=src_obj.object_id,
                        shape=src_obj.shape,
                        color=alt_color,
                        top=src_obj.top,
                        left=src_obj.left,
                        height=src_obj.height,
                        width=src_obj.width,
                        orientation=src_obj.orientation,
                        holes=src_obj.holes,
                        is_container=src_obj.is_container,
                        tags=src_obj.tags,
                        attributes=src_obj.attributes,
                    )
                    repaired_objects = list(objects)
                    repaired_objects[src_index] = alt_src
                    candidate_scene = SceneSpec(
                        height=scene.height,
                        width=scene.width,
                        background_color=scene.background_color,
                        border_color=scene.border_color,
                        outline_color=scene.outline_color,
                        marker_position=scene.marker_position,
                        objects=tuple(repaired_objects),
                        attributes=scene.attributes,
                    )
                    candidate_source = select_objects(src_selector, candidate_scene, bindings)
                    candidate_target = select_objects(dst_selector, candidate_scene, bindings)
                    if candidate_source and candidate_target and not (set(candidate_source) & set(candidate_target)):
                        repaired_scene = candidate_scene
                        repaired_source = candidate_source
                        repaired_target = candidate_target
                        break
            if repaired_scene is not None:
                scene = repaired_scene
                selected_source = repaired_source
                selected_target = repaired_target
    if not selected_source or not selected_target:
        raise ValueError("relational selectors not satisfiable")
    if set(selected_source) & set(selected_target):
        raise ValueError("relational selectors chose the same object")
    return ExampleSpec(
        example_id=example_id,
        input_scene=scene,
        selected_object_ids=selected_source + selected_target,
        notes=tuple(notes),
        metadata={
            "diversity_token": f"{action.name}:{grid_h}x{grid_w}:{src.shape}:{dst.shape}:{relation_meta}",
            "action": action.name,
            "relation_meta": relation_meta,
            "source_ids": selected_source,
            "target_ids": selected_target,
        },
    )


def count_vocab(group_by: str, latent: TaskLatent, bindings: Dict[str, Any], rng: random.Random) -> Tuple[Any, ...]:
    if group_by == "shape":
        base = list(dict.fromkeys(latent.input_schema.shape_bias + (ShapeClass.REGULAR.value,)))
        rng.shuffle(base)
        return tuple(base[:3])
    if group_by == "color":
        colors = list(range(1, latent.input_schema.max_distinct_colors + 1))
        rng.shuffle(colors)
        return tuple(colors[:3])
    if group_by == "size":
        return SIZE_BUCKETS
    return (0, 1, resolve_value(RoleVar("N_holes", "small_int"), {**bindings, "N_holes": bindings.get("N_holes", 2)}))


def count_group_key(group_by: str, obj: ObjectSpec) -> Any:
    if group_by == "shape":
        return obj.shape
    if group_by == "color":
        return obj.color
    if group_by == "size":
        return object_size_bucket(obj)
    return obj.holes


def choose_size_band_dims(shape: str, band: str, rng: random.Random) -> Tuple[int, int]:
    return pick(rng, size_band_dim_candidates(shape, band))


def size_band_dim_candidates(shape: str, band: str) -> Tuple[Tuple[int, int], ...]:
    if shape == ShapeClass.DOT.value:
        return ((1, 1),)
    if shape == ShapeClass.LINE.value:
        if band == "small":
            return ((1, 2), (2, 1))
        if band == "medium":
            return ((1, 3), (3, 1), (1, 4), (4, 1))
        return ((1, 5), (5, 1))
    if shape == ShapeClass.L_SHAPE.value:
        if band == "small":
            return ((2, 2),)
        if band == "medium":
            return ((3, 3),)
        return ((4, 4),)
    if shape == ShapeClass.CROSS.value:
        if band == "small":
            return ((3, 3),)
        return ((5, 5),)
    if band == "small":
        return ((1, 1), (1, 2), (2, 1))
    if band == "medium":
        return ((1, 2), (2, 2), (2, 3), (3, 2))
    return ((3, 3), (3, 4), (4, 3), (4, 4))


def choose_group_size_bands(
    rng: random.Random,
    reducer_kind: SelectorKind,
    groups: Sequence[Any],
    forced_winner: Any,
) -> Dict[Any, str]:
    others = [group for group in groups if group != forced_winner]
    rng.shuffle(others)
    if reducer_kind == SelectorKind.ARGMAX_SIZE:
        return {
            forced_winner: "large",
            others[0]: "medium",
            others[1]: "small",
        }
    if reducer_kind == SelectorKind.ARGMIN_SIZE:
        return {
            forced_winner: "small",
            others[0]: "medium",
            others[1]: "large",
        }
    return {group: pick(rng, ("small", "medium", "large")) for group in groups}


def choose_unique_counts(rng: random.Random, reducer_kind: SelectorKind, groups: Sequence[Any]) -> Tuple[Dict[Any, int], Any]:
    return {group: pick(rng, (1, 2, 3)) for group in groups}, pick(rng, groups)


def choose_count_winner_schedule(rng: random.Random, vocab: Sequence[Any], num_train: int) -> Tuple[Any, ...]:
    train_winners = list(vocab[: min(len(vocab), num_train)])
    while len(train_winners) < num_train:
        train_winners.append(pick(rng, vocab))
    rng.shuffle(train_winners)
    return tuple(train_winners + [pick(rng, vocab)])


def choose_group_counts(
    rng: random.Random,
    reducer_kind: SelectorKind,
    groups: Sequence[Any],
    forced_winner: Any,
) -> Dict[Any, int]:
    others = [group for group in groups if group != forced_winner]
    rng.shuffle(others)
    if reducer_kind == SelectorKind.ARGMAX_COUNT:
        return {
            forced_winner: 3,
            others[0]: 2,
            others[1]: 1,
        }
    if reducer_kind == SelectorKind.ARGMIN_COUNT:
        return {
            forced_winner: 1,
            others[0]: 2,
            others[1]: 3,
        }
    return {group: pick(rng, (1, 2, 3)) for group in groups}


def sample_count_select_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    rng: random.Random,
    example_id: str,
    episode_metadata: Dict[str, Any],
    forced_winner: Any,
    allowed_colors: Optional[Sequence[int]] = None,
    allowed_shapes: Optional[Sequence[str]] = None,
) -> ExampleSpec:
    group_by, reducer_selector, post_action = latent.program.args
    occupied: List[Tuple[int, int, int, int]] = []
    objects: List[ObjectSpec] = []
    notes: List[str] = [f"grouped by {group_by}"]
    reducer_kind = reducer_selector.kind

    group_keys = list(episode_metadata["group_vocab"][:3])
    counts = choose_group_counts(rng, reducer_kind, group_keys, forced_winner)
    total_objects = sum(counts.values())
    grid_h_low, grid_h_high = latent.input_schema.grid_h_range
    grid_w_low, grid_w_high = latent.input_schema.grid_w_range
    grid_bias = 6 + total_objects // 2
    if reducer_kind in (SelectorKind.ARGMAX_COUNT, SelectorKind.ARGMIN_COUNT) and group_by != "size":
        grid_bias += 1
    min_grid_h = min(grid_h_high, max(grid_h_low, grid_bias))
    min_grid_w = min(grid_w_high, max(grid_w_low, grid_bias))
    grid_h = randint_inclusive(rng, min_grid_h, grid_h_high)
    grid_w = randint_inclusive(rng, min_grid_w, grid_w_high)
    shape_vocab = list(restricted_vocab(episode_metadata["shape_vocab"], allowed_shapes) or episode_metadata["shape_vocab"])
    color_vocab = list(restricted_vocab(episode_metadata["color_vocab"], allowed_colors) or episode_metadata["color_vocab"])
    size_bands = (
        choose_group_size_bands(rng, reducer_kind, group_keys, forced_winner)
        if reducer_kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE) and group_by != "size"
        else {}
    )

    object_index = 0
    winner_key = forced_winner
    winner_mass = -1 if reducer_kind == SelectorKind.ARGMAX_SIZE else 10**9
    for group_key in group_keys:
        group_count = counts[group_key]
        for _ in range(group_count):
            if group_by == "shape":
                shape = group_key
                color = pick(rng, color_vocab)
                holes = 0
            elif group_by == "color":
                shape = pick(rng, shape_vocab)
                color = int(group_key)
                holes = 0
            elif group_by == "size":
                bucket = group_key
                shape = pick(rng, (ShapeClass.RECT.value, ShapeClass.REGULAR.value))
                color = pick(rng, color_vocab)
                if bucket == "small":
                    dims = (1, 1)
                elif bucket == "medium":
                    dims = (1, 2)
                else:
                    dims = (2, 2)
                holes = 0
            else:
                shape = pick(rng, shape_vocab)
                color = pick(rng, color_vocab)
                holes = int(group_key)
            if group_by == "size":
                dim_options = (dims,)
            elif reducer_kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE):
                preferred_dims = choose_size_band_dims(shape, size_bands[group_key], rng)
                dim_options = (preferred_dims,) + tuple(
                    dims for dims in size_band_dim_candidates(shape, size_bands[group_key]) if dims != preferred_dims
                )
            else:
                preferred_dims = choose_dims(shape, rng)
                dim_options = (preferred_dims,) + tuple(dims for dims in shape_dim_candidates(shape) if dims != preferred_dims)
            top, left, box_h, box_w = place_with_dim_options(rng, grid_h, grid_w, occupied, dim_options)
            obj = make_object(
                object_id=f"obj_{object_index}",
                shape=shape,
                color=color,
                top=top,
                left=left,
                height=box_h,
                width=box_w,
                orientation=pick(rng, ORIENTATIONS),
                holes=holes,
                attributes={"group_key": group_key},
            )
            if reducer_kind == SelectorKind.ARGMAX_SIZE and obj.mass > winner_mass:
                winner_mass = obj.mass
                winner_key = group_key
            elif reducer_kind == SelectorKind.ARGMIN_SIZE and obj.mass < winner_mass:
                winner_mass = obj.mass
                winner_key = group_key
            objects.append(obj)
            occupied.append(obj.bbox)
            object_index += 1

    scene = SceneSpec(
        height=grid_h,
        width=grid_w,
        background_color=0,
        border_color=None,
        outline_color=None,
        marker_position=None,
        objects=tuple(objects),
        attributes={"family": latent.family.value},
    )
    group_summary: Dict[str, Dict[str, int]] = {}
    for obj in scene.objects:
        group_key = count_group_key(group_by, obj)
        key_str = str(group_key)
        group_summary.setdefault(key_str, {"count": 0, "max_mass": 0, "min_mass": 10**9})
        group_summary[key_str]["count"] += 1
        group_summary[key_str]["max_mass"] = max(group_summary[key_str]["max_mass"], obj.mass)
        group_summary[key_str]["min_mass"] = min(group_summary[key_str]["min_mass"], obj.mass)

    if reducer_kind == SelectorKind.ARGMAX_COUNT:
        winner_key = max(group_summary.items(), key=lambda item: item[1]["count"])[0]
    elif reducer_kind == SelectorKind.ARGMIN_COUNT:
        winner_key = min(group_summary.items(), key=lambda item: item[1]["count"])[0]
    elif reducer_kind == SelectorKind.ARGMAX_SIZE:
        winner_key = max(group_summary.items(), key=lambda item: item[1]["max_mass"])[0]
    else:
        winner_key = min(group_summary.items(), key=lambda item: item[1]["min_mass"])[0]

    winner_ids = tuple(
        obj.object_id for obj in scene.objects if str(count_group_key(group_by, obj)) == str(winner_key)
    )
    return ExampleSpec(
        example_id=example_id,
        input_scene=scene,
        selected_object_ids=winner_ids,
        notes=tuple(notes),
        metadata={
            "diversity_token": f"{group_by}:{reducer_kind.value}:{winner_key}:{grid_h}x{grid_w}:{len(scene.objects)}",
            "group_by": group_by,
            "winner_key": winner_key,
            "reducer": reducer_kind.value,
            "post_action": post_action.name,
            "group_summary": group_summary,
        },
    )


def contextual_episode_metadata(cue: CueKind, bindings: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    if cue == CueKind.OUTLINE_COLOR:
        return {"cue_trigger": bindings.get("C_outline", choose_free_color([], rng))}
    if cue == CueKind.BORDER_COLOR:
        return {"cue_trigger": bindings.get("C_frame", choose_free_color([], rng))}
    if cue == CueKind.MARKER_POSITION:
        return {"cue_trigger": pick(rng, MARKER_POSITIONS)}
    if cue == CueKind.PARITY_OF_COUNT:
        return {"cue_trigger": pick(rng, ("even", "odd"))}
    return {"cue_trigger": bindings.get("N_holes", 1)}


def sample_contextual_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    episode_metadata: Dict[str, Any],
    rng: random.Random,
    example_id: str,
    branch: str,
    allowed_colors: Optional[Sequence[int]] = None,
    allowed_shapes: Optional[Sequence[str]] = None,
) -> ExampleSpec:
    cue, selector, then_action, else_action = latent.program.args
    min_objects, max_objects = latent.input_schema.num_objects_range
    capped_max_objects = max_objects
    stack_heavy = then_action.name.startswith("stack_") or else_action.name.startswith("stack_")
    if selector.kind == SelectorKind.ALL_OBJECTS:
        capped_max_objects = min(capped_max_objects, 5)
    if stack_heavy:
        capped_max_objects = min(capped_max_objects, 5)
    base_count = max(2, randint_inclusive(rng, min_objects, max(min_objects, capped_max_objects)))
    grid_h_low, grid_h_high = latent.input_schema.grid_h_range
    grid_w_low, grid_w_high = latent.input_schema.grid_w_range
    grid_load_bias = 5 + base_count // 2
    if selector.kind == SelectorKind.ALL_OBJECTS:
        grid_load_bias += 1
    if stack_heavy:
        grid_load_bias += 1
    grid_h = randint_inclusive(rng, min(grid_h_high, max(grid_h_low, grid_load_bias)), grid_h_high)
    grid_w = randint_inclusive(rng, min(grid_w_high, max(grid_w_low, grid_load_bias)), grid_w_high)
    occupied: List[Tuple[int, int, int, int]] = []
    objects: List[ObjectSpec] = []
    notes = [f"branch={branch}"]
    color_vocab = restricted_vocab(episode_metadata.get("color_vocab"), allowed_colors)
    shape_vocab = restricted_vocab(episode_metadata.get("shape_vocab"), allowed_shapes)

    selected_shape = bindings.get("S_group", choose_shape(latent, rng, shape_vocab))
    selected_color = bindings.get("C_src", choose_free_color([], rng, color_vocab))

    if cue == CueKind.PARITY_OF_COUNT:
        target_parity = episode_metadata["cue_trigger"]
        needs_even = (branch == "then" and target_parity == "even") or (branch == "else" and target_parity == "odd")
        if needs_even and base_count % 2 == 1:
            base_count += 1
        elif not needs_even and base_count % 2 == 0:
            base_count += 1

    dims_fn = choose_compact_dims if (base_count >= 5 or selector.kind == SelectorKind.ALL_OBJECTS or stack_heavy) else choose_dims

    for index in range(base_count):
        shape = choose_shape(latent, rng, shape_vocab)
        color = choose_free_color([obj.color for obj in objects], rng, color_vocab)
        if selector.kind == SelectorKind.BY_SHAPE and index == 0:
            shape = selected_shape
        if selector.kind == SelectorKind.BY_COLOR and index == 0:
            color = selected_color
        box_h, box_w = dims_fn(shape, rng)
        top, left = place_bbox(rng, grid_h, grid_w, box_h, box_w, occupied)
        obj = make_object(
            object_id=f"obj_{index}",
            shape=shape,
            color=color,
            top=top,
            left=left,
            height=box_h,
            width=box_w,
            orientation=pick(rng, ORIENTATIONS),
            holes=episode_metadata["cue_trigger"] if cue == CueKind.HOLE_COUNT and index == 0 and branch == "then" else 0,
        )
        objects.append(obj)
        occupied.append(obj.bbox)

    border_color = None
    outline_color = None
    marker_position = None
    cue_value: Any
    if cue == CueKind.OUTLINE_COLOR:
        cue_value = (
            episode_metadata["cue_trigger"]
            if branch == "then"
            else choose_free_color([episode_metadata["cue_trigger"]], rng, color_vocab)
        )
        outline_color = cue_value
    elif cue == CueKind.BORDER_COLOR:
        cue_value = (
            episode_metadata["cue_trigger"]
            if branch == "then"
            else choose_free_color([episode_metadata["cue_trigger"]], rng, color_vocab)
        )
        border_color = cue_value
    elif cue == CueKind.MARKER_POSITION:
        cue_value = episode_metadata["cue_trigger"] if branch == "then" else pick(
            rng, [pos for pos in MARKER_POSITIONS if pos != episode_metadata["cue_trigger"]]
        )
        marker_position = cue_value
    elif cue == CueKind.PARITY_OF_COUNT:
        cue_value = "even" if len(objects) % 2 == 0 else "odd"
    else:
        cue_value = episode_metadata["cue_trigger"] if branch == "then" else episode_metadata["cue_trigger"] + 1
        objects[0] = make_object(
            object_id=objects[0].object_id,
            shape=objects[0].shape,
            color=objects[0].color,
            top=objects[0].top,
            left=objects[0].left,
            height=objects[0].height,
            width=objects[0].width,
            orientation=objects[0].orientation,
            holes=int(cue_value),
            tags=objects[0].tags,
            attributes=objects[0].attributes,
        )

    scene = SceneSpec(
        height=grid_h,
        width=grid_w,
        background_color=0,
        border_color=border_color,
        outline_color=outline_color,
        marker_position=marker_position,
        objects=tuple(objects),
        attributes={"family": latent.family.value, "cue_value": cue_value},
    )
    selected = select_objects(selector, scene, bindings)
    if not selected:
        raise ValueError("contextual selector did not pick any object")
    return ExampleSpec(
        example_id=example_id,
        input_scene=scene,
        selected_object_ids=selected,
        notes=tuple(notes),
        metadata={
            "diversity_token": f"{cue.value}:{branch}:{cue_value}:{grid_h}x{grid_w}:{len(objects)}",
            "branch": branch,
            "cue_kind": cue.value,
            "cue_value": cue_value,
            "then_action": then_action.name,
            "else_action": else_action.name,
        },
    )


def symbol_map_key_vocab(latent: TaskLatent, key_type: str) -> Tuple[Any, ...]:
    if key_type == "hole_count":
        return (0, 1, 2)
    if key_type == "shape_class":
        return tuple(dict.fromkeys(latent.input_schema.shape_bias + CANONICAL_SHAPES))[:3]
    return SIZE_BUCKETS


def symbol_map_value_vocab(value_type: str) -> Tuple[Any, ...]:
    if value_type == "color":
        return (1, 2, 3)
    if value_type == "orientation":
        return ORIENTATIONS[:3]
    return PLACEMENT_SIDES[:3]


def choose_symbol_map_mapping(
    latent: TaskLatent,
    key_type: str,
    value_type: str,
    rng: random.Random,
) -> Dict[Any, Any]:
    key_vocab = list(symbol_map_key_vocab(latent, key_type))
    value_vocab = list(symbol_map_value_vocab(value_type))
    rng.shuffle(value_vocab)
    return dict(zip(key_vocab, value_vocab, strict=False))


def symbol_map_key_spec(key_type: str, key: Any, rng: random.Random) -> Tuple[str, int, int, int]:
    if key_type == "hole_count":
        return (ShapeClass.RECT.value, 2, 2, int(key))
    if key_type == "shape_class":
        shape = str(key)
        box_h, box_w = choose_compact_dims(shape, rng)
        return (shape, box_h, box_w, 0)
    if key == "small":
        return (ShapeClass.RECT.value, 1, 1, 0)
    if key == "medium":
        return (ShapeClass.RECT.value, 1, 2, 0)
    return (ShapeClass.RECT.value, 2, 2, 0)


def sample_symbol_map_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    rng: random.Random,
    example_id: str,
    episode_metadata: Dict[str, Any],
    allowed_colors: Optional[Sequence[int]] = None,
    allowed_shapes: Optional[Sequence[str]] = None,
    allowed_legend_keys: Optional[Sequence[Any]] = None,
) -> ExampleSpec:
    key_type, value_type, apply_mode = latent.program.args
    occupied: List[Tuple[int, int, int, int]] = []
    objects: List[ObjectSpec] = []
    legend: List[LegendEntry] = []
    notes = [f"legend maps {key_type} to {value_type}"]
    mapping = episode_metadata["symbol_mapping"]
    color_vocab = restricted_vocab(episode_metadata.get("color_vocab"), allowed_colors)
    shape_vocab = restricted_vocab(episode_metadata.get("shape_vocab"), allowed_shapes)
    key_vocab = tuple(mapping.keys())
    legend_keys = list(allowed_legend_keys or key_vocab)
    min_legend_count = 1 if len(legend_keys) == 1 else 2
    legend_count = randint_inclusive(rng, min_legend_count, len(legend_keys))
    rng.shuffle(legend_keys)
    legend_keys = legend_keys[:legend_count]
    target_keys = list(legend_keys)
    rng.shuffle(target_keys)
    min_target_count = 1 if len(target_keys) == 1 else 2
    target_count = randint_inclusive(rng, min_target_count, len(target_keys))
    total_objects = legend_count + target_count
    grid_h_low, grid_h_high = latent.input_schema.grid_h_range
    grid_w_low, grid_w_high = latent.input_schema.grid_w_range
    grid_bias = 5 + total_objects // 2
    if key_type == "hole_count":
        grid_bias += 1
    grid_h = randint_inclusive(rng, min(grid_h_high, max(grid_h_low, grid_bias)), grid_h_high)
    grid_w = randint_inclusive(rng, min(grid_w_high, max(grid_w_low, grid_bias)), grid_w_high)
    legend_region = (0, 0, max(1, grid_h // 3), grid_w - 1)
    target_region = (max(2, grid_h // 3), 0, grid_h - 1, grid_w - 1)

    object_index = 0
    for key in legend_keys:
        value = mapping[key]
        shape, box_h, box_w, holes = symbol_map_key_spec(key_type, key, rng)
        top, left = place_bbox(rng, grid_h, grid_w, box_h, box_w, occupied, region=legend_region)
        color = value if value_type == "color" else choose_free_color([obj.color for obj in objects], rng, color_vocab)
        legend_object = make_object(
            object_id=f"obj_{object_index}",
            shape=shape,
            color=color,
            top=top,
            left=left,
            height=box_h,
            width=box_w,
            orientation=value if value_type == "orientation" else "up",
            holes=holes,
            tags=("legend",),
            attributes={"legend_key": key, "legend_value": value},
        )
        objects.append(legend_object)
        occupied.append(legend_object.bbox)
        legend.append(LegendEntry(object_id=legend_object.object_id, key=key, value=value))
        object_index += 1

    for key in target_keys[:target_count]:
        if key_type == "shape_class":
            shape = str(key)
            holes = 0
            box_h, box_w = choose_compact_dims(shape, rng)
        elif key_type == "size_bucket":
            shape = choose_shape(latent, rng, shape_vocab)
            holes = 0
            _, box_h, box_w, _ = symbol_map_key_spec(key_type, key, rng)
        else:
            shape = choose_shape(latent, rng, shape_vocab)
            holes = int(key) if key_type == "hole_count" else 0
            box_h, box_w = choose_compact_dims(shape, rng) if key_type == "hole_count" else choose_dims(shape, rng)
        top, left = place_bbox(rng, grid_h, grid_w, box_h, box_w, occupied, region=target_region)
        target_object = make_object(
            object_id=f"obj_{object_index}",
            shape=shape,
            color=choose_free_color([obj.color for obj in objects], rng, color_vocab),
            top=top,
            left=left,
            height=box_h,
            width=box_w,
            orientation=pick(rng, ORIENTATIONS),
            holes=holes,
            tags=("target",),
            attributes={"target_key": key, "mapped_value": mapping[key]},
        )
        objects.append(target_object)
        occupied.append(target_object.bbox)
        object_index += 1

    scene = SceneSpec(
        height=grid_h,
        width=grid_w,
        background_color=0,
        border_color=None,
        outline_color=None,
        marker_position=None,
        objects=tuple(objects),
        legend=tuple(legend),
        attributes={"family": latent.family.value, "mapping": mapping},
    )
    selected = tuple(obj.object_id for obj in scene.objects if "target" in obj.tags)
    return ExampleSpec(
        example_id=example_id,
        input_scene=scene,
        selected_object_ids=selected,
        notes=tuple(notes),
        metadata={
            "diversity_token": (
                f"{key_type}:{value_type}:{tuple(legend_keys)}:{len(selected)}:{grid_h}x{grid_w}:{len(objects)}"
            ),
            "key_type": key_type,
            "value_type": value_type,
            "mapping": mapping,
            "legend_keys": tuple(legend_keys),
            "apply_mode": apply_mode.name,
        },
    )


def sample_episode_candidate(
    latent: TaskLatent,
    rng: random.Random,
    num_train: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple[ExampleSpec, ...], ExampleSpec]:
    bindings = resolve_role_bindings(latent, rng)
    episode_metadata: Dict[str, Any] = episode_visual_vocab(latent, bindings, rng)
    if latent.family == Family.RELATIONAL:
        src_selector, dst_selector, _ = latent.program.args
        if src_selector.kind == SelectorKind.ARGMAX_SIZE:
            ordered_shapes: List[str] = []
            if dst_selector.kind == SelectorKind.BY_SHAPE:
                ordered_shapes.append(resolve_value(dst_selector.args[0], bindings))
            ordered_shapes.extend((ShapeClass.RECT.value, ShapeClass.REGULAR.value))
            ordered_shapes.extend(episode_metadata["shape_vocab"])
            episode_metadata["shape_vocab"] = tuple(dict.fromkeys(ordered_shapes))[:3]
    train_examples: List[ExampleSpec] = []

    if latent.family == Family.COUNT_SELECT:
        group_by, reducer_selector, _ = latent.program.args
        if group_by == "shape" and reducer_selector.kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE):
            group_vocab = (ShapeClass.RECT.value, ShapeClass.LINE.value, ShapeClass.REGULAR.value)
        else:
            group_vocab = count_vocab(group_by, latent, bindings, rng)
        episode_metadata["group_vocab"] = group_vocab
        if reducer_selector.kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE):
            shape_base = [ShapeClass.RECT.value, ShapeClass.LINE.value, ShapeClass.REGULAR.value]
        else:
            shape_base = list(dict.fromkeys(latent.input_schema.shape_bias + (ShapeClass.REGULAR.value,)))
        rng.shuffle(shape_base)
        episode_metadata["shape_vocab"] = tuple(shape_base[:3])
        color_base = list(range(1, latent.input_schema.max_distinct_colors + 1))
        rng.shuffle(color_base)
        episode_metadata["color_vocab"] = tuple(color_base[: min(4, len(color_base))])
        episode_metadata["winner_schedule"] = choose_count_winner_schedule(rng, group_vocab, num_train)
    elif latent.family == Family.CONTEXTUAL:
        cue = latent.program.args[0]
        episode_metadata.update(contextual_episode_metadata(cue, bindings, rng))
        episode_metadata["cue_kind"] = cue.value
        schedule = choose_branch_schedule(rng, num_train)
        episode_metadata["branch_schedule"] = schedule
    elif latent.family == Family.SYMBOL_MAP:
        key_type, value_type, _ = latent.program.args
        episode_metadata["symbol_mapping"] = choose_symbol_map_mapping(latent, key_type, value_type, rng)
    else:
        schedule = ()

    for index in range(num_train):
        example_id = f"train_{index}"
        if latent.family == Family.UNARY_OBJECT:
            example = sample_unary_example(latent, bindings, rng, example_id, episode_metadata)
        elif latent.family == Family.RELATIONAL:
            example = sample_relational_example(latent, bindings, rng, example_id, episode_metadata)
        elif latent.family == Family.COUNT_SELECT:
            example = sample_count_select_example(
                latent,
                bindings,
                rng,
                example_id,
                episode_metadata,
                episode_metadata["winner_schedule"][index],
            )
        elif latent.family == Family.CONTEXTUAL:
            example = sample_contextual_example(
                latent,
                bindings,
                episode_metadata,
                rng,
                example_id,
                schedule[index],
            )
        else:
            example = sample_symbol_map_example(latent, bindings, rng, example_id, episode_metadata)
        train_examples.append(example)

    train_color_support = tuple(sorted(set().union(*(collect_colors(example.input_scene) for example in train_examples))))
    train_shape_support = tuple(sorted(set().union(*(collect_shapes(example.input_scene) for example in train_examples))))
    test_example_id = "test_0"
    if latent.family == Family.UNARY_OBJECT:
        test_example = sample_unary_example(
            latent,
            bindings,
            rng,
            test_example_id,
            episode_metadata,
            allowed_colors=train_color_support,
            allowed_shapes=train_shape_support,
        )
    elif latent.family == Family.RELATIONAL:
        test_example = sample_relational_example(
            latent,
            bindings,
            rng,
            test_example_id,
            episode_metadata,
            allowed_colors=train_color_support,
            allowed_shapes=train_shape_support,
        )
    elif latent.family == Family.COUNT_SELECT:
        train_color_support = tuple(sorted(set().union(*(collect_colors(example.input_scene) for example in train_examples))))
        train_shape_support = tuple(sorted(set().union(*(collect_shapes(example.input_scene) for example in train_examples))))
        test_example = sample_count_select_example(
            latent,
            bindings,
            rng,
            test_example_id,
            episode_metadata,
            episode_metadata["winner_schedule"][num_train],
            allowed_colors=train_color_support,
            allowed_shapes=train_shape_support,
        )
    elif latent.family == Family.CONTEXTUAL:
        test_example = sample_contextual_example(
            latent,
            bindings,
            episode_metadata,
            rng,
            test_example_id,
            schedule[num_train],
            allowed_colors=train_color_support,
            allowed_shapes=train_shape_support,
        )
    else:
        train_legend_keys = tuple(dict.fromkeys(entry.key for example in train_examples for entry in example.input_scene.legend))
        test_example = sample_symbol_map_example(
            latent,
            bindings,
            rng,
            test_example_id,
            episode_metadata,
            allowed_colors=train_color_support,
            allowed_shapes=train_shape_support,
            allowed_legend_keys=train_legend_keys,
        )

    return bindings, episode_metadata, tuple(train_examples), test_example


def validate_episode(latent: TaskLatent, train_examples: Tuple[ExampleSpec, ...], test_example: ExampleSpec) -> Tuple[str, ...]:
    if len(train_examples) < 3:
        return ("need_at_least_three_train_examples",)

    reasons: List[str] = []
    train_colors = set().union(*(collect_colors(example.input_scene) for example in train_examples))
    test_colors = collect_colors(test_example.input_scene)
    if not test_colors.issubset(train_colors):
        reasons.append("test_has_unseen_colors")

    train_shapes = set().union(*(collect_shapes(example.input_scene) for example in train_examples))
    test_shapes = collect_shapes(test_example.input_scene)
    if not test_shapes.issubset(train_shapes):
        reasons.append("test_has_unseen_shapes")

    diversity_tokens = {example.metadata["diversity_token"] for example in train_examples}
    if len(diversity_tokens) < 2:
        reasons.append("insufficient_train_diversity")

    grid_sizes = {(example.input_scene.height, example.input_scene.width) for example in train_examples}
    if len(grid_sizes) < 2:
        reasons.append("train_grid_sizes_do_not_vary")

    if latent.family == Family.COUNT_SELECT:
        group_by, reducer_selector, _ = latent.program.args
        intrinsic_size_winner = group_by == "size" and reducer_selector.kind in (
            SelectorKind.ARGMAX_SIZE,
            SelectorKind.ARGMIN_SIZE,
        )
        winner_keys = {str(example.metadata["winner_key"]) for example in train_examples}
        if len(winner_keys) < 2 and not intrinsic_size_winner:
            reasons.append("count_winner_constant_across_train")
        for example in train_examples:
            summary = example.metadata["group_summary"]
            reducer = example.metadata["reducer"]
            if reducer in (SelectorKind.ARGMAX_COUNT.value, SelectorKind.ARGMIN_COUNT.value):
                counts = [stats["count"] for stats in summary.values()]
                if len(counts) == 0:
                    reasons.append("winner_not_unique")
                    break
                winner_count = max(counts) if reducer == SelectorKind.ARGMAX_COUNT.value else min(counts)
                if counts.count(winner_count) > 1:
                    reasons.append("winner_not_unique")
                    break

    if latent.family == Family.CONTEXTUAL:
        branches = {example.metadata["branch"] for example in train_examples}
        if branches != {"then", "else"}:
            reasons.append("both_branches_not_realized")

    if latent.family == Family.SYMBOL_MAP:
        mapping_signatures = {
            tuple(sorted(example.metadata["mapping"].items(), key=lambda item: repr(item[0])))
            for example in train_examples + (test_example,)
        }
        if len(mapping_signatures) != 1:
            reasons.append("binding_rule_inconsistent_across_pairs")
        for example in train_examples + (test_example,):
            legend_keys = {entry.key for entry in example.input_scene.legend}
            target_keys = {
                obj.attributes["target_key"]
                for obj in example.input_scene.objects
                if "target" in obj.tags
            }
            if not target_keys.issubset(legend_keys):
                reasons.append("legend_incomplete_for_targets")
                break

    if latent.family == Family.RELATIONAL:
        for example in train_examples + (test_example,):
            source_ids = set(example.metadata.get("source_ids", ()))
            target_ids = set(example.metadata.get("target_ids", ()))
            if source_ids & target_ids:
                reasons.append("source_and_target_not_distinct")
                break

    return tuple(reasons)


def shortcut_checks_for(latent: TaskLatent) -> Tuple[str, ...]:
    checks = [
        "test visuals must be covered by train visuals",
        "train examples must vary in grid size or object layout",
        "family-specific ambiguity checks must pass",
    ]
    if latent.family == Family.CONTEXTUAL:
        checks.append("train examples must realize both control-flow branches")
    if (
        latent.family == Family.COUNT_SELECT
        and not (
            latent.program.args[0] == "size"
            and latent.program.args[1].kind in (SelectorKind.ARGMAX_SIZE, SelectorKind.ARGMIN_SIZE)
        )
    ):
        checks.append("winner identity must vary across train examples")
    if latent.family == Family.SYMBOL_MAP:
        checks.append("legend must cover every target key in every scene")
    return tuple(checks)


def sample_episode(
    latent: TaskLatent,
    *,
    seed: Optional[int] = None,
    num_train: int = 3,
    max_attempts: int = 256,
) -> EpisodeSpec:
    rng = random.Random(seed)
    rejection_counts: Counter[str] = Counter()

    for attempt in range(1, max_attempts + 1):
        try:
            bindings, episode_metadata, train_examples, test_example = sample_episode_candidate(latent, rng, num_train)
        except ValueError as exc:
            rejection_counts[f"construction_error:{exc}"] += 1
            continue

        reasons = validate_episode(latent, train_examples, test_example)
        if reasons:
            rejection_counts.update(reasons)
            continue

        return EpisodeSpec(
            latent=latent,
            role_bindings=bindings,
            episode_metadata=episode_metadata,
            train_examples=train_examples,
            test_example=test_example,
            shortcut_checks=shortcut_checks_for(latent),
            rejection_counts=dict(rejection_counts),
            sampling_attempts=attempt,
        )

    raise RuntimeError(
        f"failed to sample a valid episode for {latent.family.value} after {max_attempts} attempts: {dict(rejection_counts)}"
    )


if __name__ == "__main__":
    try:
        from .stage1_latent_sampler import sample_latent_rule
    except ImportError:  # pragma: no cover - direct script execution
        from stage1_latent_sampler import sample_latent_rule  # type: ignore

    latent = sample_latent_rule(seed=0)
    episode = sample_episode(latent, seed=0)
    print(json.dumps(episode.to_jsonable(), indent=2))
