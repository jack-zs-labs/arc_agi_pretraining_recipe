from __future__ import annotations

"""Stage 3 execution for ARC-style synthetic episodes.

This module turns the object-scene blueprints from Stage 2 into concrete grids
and approximate output scenes. The semantics are intentionally lightweight: the
goal is to create consistent supervision targets for trajectory pretraining,
not to exactly reproduce the full ARC-AGI distribution.
"""

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .stage1_latent_sampler import CueKind, Family, RelationAction, RoleVar, TaskLatent
    from .stage2_episode_sampler import (
        EpisodeSpec,
        ExampleSpec,
        ObjectSpec,
        SceneSpec,
        compute_mass,
        resolve_value,
    )
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import CueKind, Family, RelationAction, RoleVar, TaskLatent  # type: ignore
    from stage2_episode_sampler import (  # type: ignore
        EpisodeSpec,
        ExampleSpec,
        ObjectSpec,
        SceneSpec,
        compute_mass,
        resolve_value,
    )


Grid = Tuple[Tuple[int, ...], ...]
ORIENTATIONS = ("up", "right", "down", "left")
MARKER_COLOR = 8


@dataclass(frozen=True)
class ExecutedExample:
    example: ExampleSpec
    input_grid: Grid
    output_grid: Grid
    output_scene: SceneSpec
    trace_targets: Dict[str, Any]
    step_workspaces: Dict[str, Grid]
    step_scenes: Dict[str, SceneSpec]

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


@dataclass(frozen=True)
class ExecutedEpisode:
    episode: EpisodeSpec
    train_examples: Tuple[ExecutedExample, ...]
    test_example: ExecutedExample

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


def replace_object(obj: ObjectSpec, **changes: Any) -> ObjectSpec:
    shape = changes.get("shape", obj.shape)
    height = changes.get("height", obj.height)
    width = changes.get("width", obj.width)
    holes = changes.get("holes", obj.holes)
    attributes_changed = "attributes" in changes
    attributes = obj.attributes if not attributes_changed else dict(changes["attributes"])

    if (
        shape == obj.shape
        and height == obj.height
        and width == obj.width
        and holes == obj.holes
        and not (shape == "bitmap" and attributes_changed)
    ):
        mass = obj.mass
    else:
        filled_cells = None
        if shape == "bitmap":
            filled_cells = len(attributes.get("bitmap_cells", ()))
        mass = compute_mass(
            shape,
            height,
            width,
            holes,
            filled_cells=filled_cells,
        )

    return ObjectSpec(
        object_id=changes.get("object_id", obj.object_id),
        shape=shape,
        color=changes.get("color", obj.color),
        top=changes.get("top", obj.top),
        left=changes.get("left", obj.left),
        height=height,
        width=width,
        mass=mass,
        orientation=changes.get("orientation", obj.orientation),
        holes=holes,
        is_container=changes.get("is_container", obj.is_container),
        tags=changes.get("tags", obj.tags),
        attributes=attributes,
    )


def translate_object(obj: ObjectSpec, d_row: int = 0, d_col: int = 0) -> ObjectSpec:
    return replace_object(obj, top=obj.top + d_row, left=obj.left + d_col)


def rotate_orientation(orientation: str, quarter_turns: int) -> str:
    if orientation not in ORIENTATIONS:
        return orientation
    index = ORIENTATIONS.index(orientation)
    return ORIENTATIONS[(index + quarter_turns) % len(ORIENTATIONS)]


def reflect_orientation_h(orientation: str) -> str:
    return {
        "up": "up",
        "right": "left",
        "down": "down",
        "left": "right",
    }.get(orientation, orientation)


def reflect_orientation_v(orientation: str) -> str:
    return {
        "up": "down",
        "right": "right",
        "down": "up",
        "left": "left",
    }.get(orientation, orientation)


def selected_objects(scene: SceneSpec, object_ids: Iterable[str]) -> Tuple[ObjectSpec, ...]:
    wanted = set(object_ids)
    return tuple(obj for obj in scene.objects if obj.object_id in wanted)


def scene_from_objects(
    objects: Sequence[ObjectSpec],
    *,
    background_color: int = 0,
    border_color: Optional[int] = None,
    outline_color: Optional[int] = None,
    marker_position: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    min_height: int = 1,
    min_width: int = 1,
) -> SceneSpec:
    if objects:
        min_top = min(obj.top for obj in objects)
        min_left = min(obj.left for obj in objects)
        shift_row = -min(0, min_top)
        shift_col = -min(0, min_left)
        shifted = tuple(translate_object(obj, shift_row, shift_col) for obj in objects)
        height = max(min_height, max(obj.bottom for obj in shifted) + 1)
        width = max(min_width, max(obj.right for obj in shifted) + 1)
    else:
        shifted = tuple()
        height = max(1, min_height)
        width = max(1, min_width)
    return SceneSpec(
        height=height,
        width=width,
        background_color=background_color,
        border_color=border_color,
        outline_color=outline_color,
        marker_position=marker_position,
        objects=shifted,
        legend=tuple(),
        attributes=attributes or {},
    )


def normalized_objects(objects: Sequence[ObjectSpec]) -> Tuple[ObjectSpec, ...]:
    if not objects:
        return tuple()
    min_top = min(obj.top for obj in objects)
    min_left = min(obj.left for obj in objects)
    return tuple(translate_object(obj, -min_top, -min_left) for obj in objects)


def scene_like(
    base_scene: SceneSpec,
    objects: Sequence[ObjectSpec],
    *,
    border_color: Optional[int] | None = None,
    outline_color: Optional[int] | None = None,
    marker_position: Optional[str] | None = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> SceneSpec:
    return SceneSpec(
        height=base_scene.height,
        width=base_scene.width,
        background_color=base_scene.background_color,
        border_color=base_scene.border_color if border_color is None else border_color,
        outline_color=base_scene.outline_color if outline_color is None else outline_color,
        marker_position=base_scene.marker_position if marker_position is None else marker_position,
        objects=tuple(objects),
        legend=tuple(base_scene.legend),
        attributes=attributes if attributes is not None else base_scene.attributes,
    )


def palette_color(index: int) -> int:
    palette = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    return palette[index % len(palette)]


def segmented_workspace(scene: SceneSpec) -> Grid:
    return render_scene(segmented_scene(scene))


def segmented_scene(scene: SceneSpec) -> SceneSpec:
    recolored = tuple(replace_object(obj, color=palette_color(index)) for index, obj in enumerate(scene.objects))
    return scene_like(scene, recolored, attributes={"workspace": "segment"})


def highlight_workspace(
    scene: SceneSpec,
    *,
    primary_ids: Iterable[str] = (),
    secondary_ids: Iterable[str] = (),
    primary_color: int = 9,
    secondary_color: int = 8,
    other_color: Optional[int] = None,
) -> Grid:
    return render_scene(
        highlight_scene(
            scene,
            primary_ids=primary_ids,
            secondary_ids=secondary_ids,
            primary_color=primary_color,
            secondary_color=secondary_color,
            other_color=other_color,
        )
    )


def highlight_scene(
    scene: SceneSpec,
    *,
    primary_ids: Iterable[str] = (),
    secondary_ids: Iterable[str] = (),
    primary_color: int = 9,
    secondary_color: int = 8,
    other_color: Optional[int] = None,
) -> SceneSpec:
    primary = set(primary_ids)
    secondary = set(secondary_ids)
    objects = []
    for obj in scene.objects:
        if obj.object_id in primary:
            objects.append(replace_object(obj, color=primary_color))
        elif obj.object_id in secondary:
            objects.append(replace_object(obj, color=secondary_color))
        elif other_color is not None:
            objects.append(replace_object(obj, color=other_color))
        else:
            objects.append(obj)
    return scene_like(scene, objects, attributes={"workspace": "highlight"})


def grouped_workspace(scene: SceneSpec, group_assignments: Dict[str, Any]) -> Grid:
    return render_scene(grouped_scene(scene, group_assignments))


def grouped_scene(scene: SceneSpec, group_assignments: Dict[str, Any]) -> SceneSpec:
    palette = {key: palette_color(index) for index, key in enumerate(sorted(set(group_assignments.values()), key=repr))}
    objects = tuple(
        replace_object(obj, color=palette[group_assignments[obj.object_id]]) if obj.object_id in group_assignments else obj
        for obj in scene.objects
    )
    return scene_like(scene, objects, attributes={"workspace": "group"})


def preview_workspace(
    objects: Sequence[ObjectSpec],
    *,
    attributes: Optional[Dict[str, Any]] = None,
) -> Grid:
    return render_scene(preview_scene(objects, attributes=attributes))


def preview_scene(
    objects: Sequence[ObjectSpec],
    *,
    attributes: Optional[Dict[str, Any]] = None,
) -> SceneSpec:
    return scene_from_objects(
        normalized_objects(objects),
        background_color=0,
        min_height=1,
        min_width=1,
        attributes=attributes or {"workspace": "preview"},
    )


def cue_workspace(scene: SceneSpec, cue_kind: str, selected_ids: Iterable[str]) -> Grid:
    return render_scene(cue_scene(scene, cue_kind, selected_ids))


def cue_scene(scene: SceneSpec, cue_kind: str, selected_ids: Iterable[str]) -> SceneSpec:
    if cue_kind == CueKind.OUTLINE_COLOR.value:
        return scene_like(scene, scene.objects, outline_color=9, attributes={"workspace": "cue"})
    if cue_kind == CueKind.BORDER_COLOR.value:
        return scene_like(scene, scene.objects, border_color=9, attributes={"workspace": "cue"})
    if cue_kind == CueKind.MARKER_POSITION.value:
        return scene_like(scene, tuple(), marker_position=scene.marker_position, attributes={"workspace": "cue"})
    return highlight_scene(scene, primary_ids=selected_ids, primary_color=9, other_color=2)


def legend_workspace(scene: SceneSpec, mapping: Dict[Any, Any]) -> Grid:
    return render_scene(legend_scene(scene, mapping))


def legend_scene(scene: SceneSpec, mapping: Dict[Any, Any]) -> SceneSpec:
    legend_ids = {entry.object_id for entry in scene.legend}
    symbolic_values = sorted(
        {
            mapped_value
            for mapped_value in mapping.values()
            if not (isinstance(mapped_value, int) and 0 <= mapped_value <= 9)
        },
        key=repr,
    )
    symbolic_palette = {value: palette_color(index) for index, value in enumerate(symbolic_values)}
    objects = []
    for obj in scene.objects:
        if obj.object_id not in legend_ids:
            objects.append(replace_object(obj, color=1))
            continue
        mapped_value = mapping.get(obj.attributes.get("legend_key"))
        if isinstance(mapped_value, int) and 0 <= mapped_value <= 9:
            color = mapped_value
        else:
            color = symbolic_palette.get(mapped_value, 9)
        objects.append(replace_object(obj, color=color))
    return scene_like(scene, objects, attributes={"workspace": "legend"})


def matched_workspace(scene: SceneSpec, target_ids: Iterable[str]) -> Grid:
    return render_scene(matched_scene(scene, target_ids))


def matched_scene(scene: SceneSpec, target_ids: Iterable[str]) -> SceneSpec:
    target_set = set(target_ids)
    target_keys = {
        obj.object_id: obj.attributes.get("target_key")
        for obj in scene.objects
        if obj.object_id in target_set
    }
    return grouped_scene(scene, target_keys)


def arrange_row(
    objects: Sequence[ObjectSpec],
    *,
    gap: int = 1,
    descending: bool = False,
    repeat: int = 1,
) -> Tuple[ObjectSpec, ...]:
    ordered = sorted(objects, key=lambda obj: (obj.mass, obj.object_id), reverse=descending)
    placed: List[ObjectSpec] = []
    cursor = 0
    for _ in range(max(1, repeat)):
        for obj in ordered:
            base = replace_object(obj, top=0, left=cursor)
            placed.append(base)
            cursor += base.width + gap
    return tuple(placed)


def stack_side(objects: Sequence[ObjectSpec], *, side: str, canvas_height: int, canvas_width: int) -> Tuple[ObjectSpec, ...]:
    placed: List[ObjectSpec] = []
    cursor = 0
    ordered = sorted(objects, key=lambda obj: (obj.top, obj.left, obj.object_id))
    for obj in ordered:
        if side == "left":
            placed.append(replace_object(obj, top=cursor, left=0))
            cursor += obj.height + 1
        else:
            placed.append(replace_object(obj, top=cursor, left=max(0, canvas_width - obj.width)))
            cursor += obj.height + 1
    return tuple(placed)


def reflect_objects(objects: Sequence[ObjectSpec], *, axis: str, canvas_height: int, canvas_width: int) -> Tuple[ObjectSpec, ...]:
    reflected: List[ObjectSpec] = []
    for obj in objects:
        if axis == "h":
            new_left = max(0, canvas_width - obj.left - obj.width)
            reflected.append(replace_object(obj, left=new_left, orientation=reflect_orientation_h(obj.orientation)))
        else:
            new_top = max(0, canvas_height - obj.top - obj.height)
            reflected.append(replace_object(obj, top=new_top, orientation=reflect_orientation_v(obj.orientation)))
    return tuple(reflected)


def align_object_to(source: ObjectSpec, target: ObjectSpec, axis: str) -> ObjectSpec:
    top = source.top
    left = source.left
    if axis in {"x", "both"}:
        top = target.top
    if axis in {"y", "both"}:
        left = target.left
    return replace_object(source, top=top, left=left)


def move_until_contact(source: ObjectSpec, target: ObjectSpec, direction: str) -> ObjectSpec:
    if direction == "up":
        return replace_object(source, top=target.bottom + 1, left=target.left)
    if direction == "down":
        return replace_object(source, top=max(0, target.top - source.height), left=target.left)
    if direction == "left":
        return replace_object(source, top=target.top, left=target.right + 1)
    if direction == "right":
        return replace_object(source, top=target.top, left=max(0, target.left - source.width))

    source_center_row = source.top + source.height // 2
    source_center_col = source.left + source.width // 2
    target_center_row = target.top + target.height // 2
    target_center_col = target.left + target.width // 2
    if abs(target_center_row - source_center_row) >= abs(target_center_col - source_center_col):
        return move_until_contact(source, target, "down" if target_center_row > source_center_row else "up")
    return move_until_contact(source, target, "right" if target_center_col > source_center_col else "left")


def stack_next_to(source: ObjectSpec, target: ObjectSpec, side: str) -> ObjectSpec:
    if side == "left":
        return replace_object(source, top=target.top, left=max(0, target.left - source.width - 1))
    if side == "right":
        return replace_object(source, top=target.top, left=target.right + 1)
    if side == "top":
        return replace_object(source, top=max(0, target.top - source.height - 1), left=target.left)
    return replace_object(source, top=target.bottom + 1, left=target.left)


def place_in_container(source: ObjectSpec, container: ObjectSpec) -> ObjectSpec:
    inner_height = max(1, container.height - 2)
    inner_width = max(1, container.width - 2)
    top = container.top + 1 + max(0, (inner_height - source.height) // 2)
    left = container.left + 1 + max(0, (inner_width - source.width) // 2)
    return replace_object(source, top=top, left=left)


def connector_object(source: ObjectSpec, target: ObjectSpec) -> ObjectSpec:
    src_row = source.top + source.height // 2
    src_col = source.left + source.width // 2
    dst_row = target.top + target.height // 2
    dst_col = target.left + target.width // 2
    if src_row == dst_row:
        left = min(src_col, dst_col)
        width = max(1, abs(dst_col - src_col) + 1)
        return ObjectSpec(
            object_id="connector",
            shape="line",
            color=source.color,
            top=src_row,
            left=left,
            height=1,
            width=width,
            mass=width,
            orientation="right",
            holes=0,
            is_container=False,
            tags=("connector",),
            attributes={},
        )
    if src_col == dst_col:
        top = min(src_row, dst_row)
        height = max(1, abs(dst_row - src_row) + 1)
        return ObjectSpec(
            object_id="connector",
            shape="line",
            color=source.color,
            top=top,
            left=src_col,
            height=height,
            width=1,
            mass=height,
            orientation="down",
            holes=0,
            is_container=False,
            tags=("connector",),
            attributes={},
        )
    top = min(src_row, dst_row)
    left = min(src_col, dst_col)
    height = abs(dst_row - src_row) + 1
    width = abs(dst_col - src_col) + 1
    orientation = "up"
    if dst_row >= src_row and dst_col >= src_col:
        orientation = "right"
    elif dst_row >= src_row and dst_col < src_col:
        orientation = "down"
    elif dst_row < src_row and dst_col < src_col:
        orientation = "left"
    return ObjectSpec(
        object_id="connector",
        shape="l_shape",
        color=source.color,
        top=top,
        left=left,
        height=height,
        width=width,
        mass=compute_mass("l_shape", height, width, 0),
        orientation=orientation,
        holes=0,
        is_container=False,
        tags=("connector",),
        attributes={},
    )


def transformed_unary_object(obj: ObjectSpec, action_name: str, params: Dict[str, Any]) -> ObjectSpec:
    if obj.shape == "bitmap":
        return transformed_bitmap_object(obj, action_name, params)
    if action_name == "rotate_90":
        return replace_object(
            obj,
            height=obj.width,
            width=obj.height,
            orientation=rotate_orientation(obj.orientation, 1),
        )
    if action_name == "rotate_180":
        return replace_object(obj, orientation=rotate_orientation(obj.orientation, 2))
    if action_name == "reflect_h":
        return replace_object(obj, orientation=reflect_orientation_h(obj.orientation))
    if action_name == "reflect_v":
        return replace_object(obj, orientation=reflect_orientation_v(obj.orientation))
    if action_name == "translate":
        direction = str(params.get("direction", "right"))
        distance = int(params.get("distance", 1))
        if direction == "up":
            return translate_object(obj, -distance, 0)
        if direction == "down":
            return translate_object(obj, distance, 0)
        if direction == "left":
            return translate_object(obj, 0, -distance)
        if direction == "diag_up_right":
            return translate_object(obj, -distance, distance)
        return translate_object(obj, 0, distance)
    if action_name == "scale_up":
        factor = max(1, int(params.get("factor", 2)))
        return replace_object(obj, height=obj.height * factor, width=obj.width * factor)
    if action_name == "scale_down":
        factor = max(1, int(params.get("factor", 2)))
        return replace_object(obj, height=max(1, obj.height // factor), width=max(1, obj.width // factor))
    if action_name == "recolor":
        return replace_object(obj, color=int(params.get("to_color", obj.color)))
    if action_name == "fill_holes":
        return replace_object(obj, holes=0)
    return obj


def normalized_bitmap_cells(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    points = list(cells)
    if not points:
        return ((0, 0),)
    min_row = min(row for row, _ in points)
    min_col = min(col for _, col in points)
    return tuple(sorted((row - min_row, col - min_col) for row, col in points))


def bitmap_cell_bounds(cells: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    if not cells:
        return (1, 1)
    return (
        max(row for row, _ in cells) + 1,
        max(col for _, col in cells) + 1,
    )


def transformed_bitmap_object(obj: ObjectSpec, action_name: str, params: Dict[str, Any]) -> ObjectSpec:
    raw_cells = obj.attributes.get("bitmap_cells", ())
    cells = tuple((int(row), int(col)) for row, col in raw_cells) or ((0, 0),)
    if action_name == "translate":
        direction = str(params.get("direction", "right"))
        distance = int(params.get("distance", 1))
        if direction == "up":
            return translate_object(obj, -distance, 0)
        if direction == "down":
            return translate_object(obj, distance, 0)
        if direction == "left":
            return translate_object(obj, 0, -distance)
        if direction == "diag_up_right":
            return translate_object(obj, -distance, distance)
        return translate_object(obj, 0, distance)

    transformed = list(cells)
    if action_name == "rotate_90":
        transformed = [(col, obj.height - 1 - row) for row, col in cells]
    elif action_name == "rotate_180":
        transformed = [(obj.height - 1 - row, obj.width - 1 - col) for row, col in cells]
    elif action_name == "reflect_h":
        transformed = [(row, obj.width - 1 - col) for row, col in cells]
    elif action_name == "reflect_v":
        transformed = [(obj.height - 1 - row, col) for row, col in cells]
    elif action_name == "scale_up":
        factor = max(1, int(params.get("factor", 2)))
        transformed = [
            (row * factor + d_row, col * factor + d_col)
            for row, col in cells
            for d_row in range(factor)
            for d_col in range(factor)
        ]
    elif action_name == "scale_down":
        factor = max(1, int(params.get("factor", 2)))
        transformed = sorted({(row // factor, col // factor) for row, col in cells})
    elif action_name == "recolor":
        return replace_object(obj, color=int(params.get("to_color", obj.color)))
    elif action_name == "fill_holes":
        cell_set = set(cells)
        top = min(row for row, _ in cell_set)
        left = min(col for _, col in cell_set)
        bottom = max(row for row, _ in cell_set)
        right = max(col for _, col in cell_set)
        transformed = [
            (row, col)
            for row in range(top, bottom + 1)
            for col in range(left, right + 1)
        ]
    else:
        return obj

    normalized = normalized_bitmap_cells(transformed)
    height, width = bitmap_cell_bounds(normalized)
    attributes = dict(obj.attributes)
    attributes["bitmap_cells"] = normalized
    return replace_object(obj, height=height, width=width, attributes=attributes)


@lru_cache(maxsize=8192)
def cached_object_cells(
    shape: str,
    height: int,
    width: int,
    orientation: str,
    holes: int,
    is_container: bool,
    is_frame: bool,
    bitmap_cells: Tuple[Tuple[int, int], ...],
) -> Tuple[Tuple[int, int], ...]:
    if shape == "bitmap":
        return bitmap_cells

    if is_container or is_frame:
        return tuple(
            (row, col)
            for row in range(height)
            for col in range(width)
            if row in {0, height - 1} or col in {0, width - 1}
        )

    if shape in {"rect", "regular", "any"}:
        cells = [(row, col) for row in range(height) for col in range(width)]
    elif shape == "dot":
        cells = [(0, 0)]
    elif shape == "line":
        if height == 1:
            cells = [(0, col) for col in range(width)]
        elif width == 1:
            cells = [(row, 0) for row in range(height)]
        else:
            cells = [(row, width // 2) for row in range(height)]
    elif shape == "l_shape":
        if orientation == "up":
            cells = [(0, col) for col in range(width)] + [(row, 0) for row in range(height)]
        elif orientation == "right":
            cells = [(0, col) for col in range(width)] + [(row, width - 1) for row in range(height)]
        elif orientation == "down":
            cells = [(height - 1, col) for col in range(width)] + [(row, width - 1) for row in range(height)]
        else:
            cells = [(height - 1, col) for col in range(width)] + [(row, 0) for row in range(height)]
    elif shape == "cross":
        mid_row = height // 2
        mid_col = width // 2
        cells = [(mid_row, col) for col in range(width)] + [(row, mid_col) for row in range(height)]
    else:
        cells = [(row, col) for row in range(height) for col in range(width)]

    deduped = tuple(sorted(set(cells)))
    if holes <= 0:
        return deduped

    hole_cells = {(height // 2, width // 2)}
    if holes > 1 and width >= 3:
        hole_cells.add((height // 2, max(0, width // 2 - 1)))
    if holes > 2 and height >= 3:
        hole_cells.add((max(0, height // 2 - 1), width // 2))
    return tuple(cell for cell in deduped if cell not in hole_cells)


def object_cells(obj: ObjectSpec) -> Tuple[Tuple[int, int], ...]:
    bitmap_cells: Tuple[Tuple[int, int], ...] = ()
    if obj.shape == "bitmap":
        bitmap_cells = tuple(sorted((int(row), int(col)) for row, col in obj.attributes.get("bitmap_cells", ())))
    return cached_object_cells(
        obj.shape,
        obj.height,
        obj.width,
        obj.orientation,
        obj.holes,
        obj.is_container,
        "frame" in obj.tags,
        bitmap_cells,
    )


def render_scene(scene: SceneSpec) -> Grid:
    grid = [[scene.background_color for _ in range(scene.width)] for _ in range(scene.height)]

    if scene.border_color is not None:
        for col in range(scene.width):
            grid[0][col] = scene.border_color
            grid[scene.height - 1][col] = scene.border_color
        for row in range(scene.height):
            grid[row][0] = scene.border_color
            grid[row][scene.width - 1] = scene.border_color

    objects = sorted(scene.objects, key=lambda obj: (0 if (obj.is_container or "frame" in obj.tags) else 1, obj.mass))
    for obj in objects:
        for rel_row, rel_col in object_cells(obj):
            row = obj.top + rel_row
            col = obj.left + rel_col
            if 0 <= row < scene.height and 0 <= col < scene.width:
                grid[row][col] = obj.color

    if scene.marker_position is not None:
        marker_lookup = {
            "top_left": (0, 0),
            "top_right": (0, scene.width - 1),
            "bottom_left": (scene.height - 1, 0),
            "bottom_right": (scene.height - 1, scene.width - 1),
        }
        row, col = marker_lookup[scene.marker_position]
        grid[row][col] = MARKER_COLOR

    return tuple(tuple(row) for row in grid)


def execute_unary_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    example: ExampleSpec,
) -> Tuple[SceneSpec, Dict[str, Any], Dict[str, Grid], Dict[str, SceneSpec]]:
    _selector, action = latent.program.args
    params = resolve_value(action.params, bindings)
    chosen = selected_objects(example.input_scene, example.selected_object_ids)
    transformed = tuple(transformed_unary_object(obj, action.name, params) for obj in chosen)
    min_height = 1 if action.name == "crop_to_bbox" else example.input_scene.height
    min_width = 1 if action.name == "crop_to_bbox" else example.input_scene.width
    output_scene = scene_from_objects(
        normalized_objects(transformed) if action.name == "crop_to_bbox" else transformed,
        background_color=0,
        min_height=min_height,
        min_width=min_width,
        attributes={"family": latent.family.value, "action": action.name},
    )
    step_scenes = {
        "segment": segmented_scene(example.input_scene),
        "select": highlight_scene(example.input_scene, primary_ids=example.selected_object_ids, primary_color=9, other_color=2),
        "transform": preview_scene(transformed, attributes={"workspace": "transform_preview"}),
        "render": output_scene,
    }
    step_workspaces = {name: render_scene(scene) for name, scene in step_scenes.items() if name != "render"}
    return output_scene, {
        "segment": {"object_ids": [obj.object_id for obj in example.input_scene.objects]},
        "select": {"selected_object_ids": list(example.selected_object_ids)},
        "transform": {"action": action.name, "params": params},
        "render": {"output_scene_objects": [obj.object_id for obj in output_scene.objects]},
    }, step_workspaces, step_scenes


def execute_relational_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    example: ExampleSpec,
) -> Tuple[SceneSpec, Dict[str, Any], Dict[str, Grid], Dict[str, SceneSpec]]:
    _src_selector, _dst_selector, action = latent.program.args
    params = resolve_value(action.params, bindings)
    source_ids = tuple(example.metadata.get("source_ids", ()))
    target_ids = tuple(example.metadata.get("target_ids", ()))
    sources = list(selected_objects(example.input_scene, source_ids))
    targets = list(selected_objects(example.input_scene, target_ids))
    if not sources or not targets:
        chosen = selected_objects(example.input_scene, example.selected_object_ids)
        midpoint = max(1, len(chosen) // 2)
        sources = list(chosen[:midpoint])
        targets = list(chosen[midpoint:] or chosen[-1:])

    moved_sources: List[ObjectSpec] = []
    extra_objects: List[ObjectSpec] = []
    for source in sources:
        anchor = targets[0]
        if action.name == RelationAction.MOVE_UNTIL_CONTACT.value:
            moved_sources.append(move_until_contact(source, anchor, str(params["direction"])))
        elif action.name == RelationAction.ALIGN_TO.value:
            moved_sources.append(align_object_to(source, anchor, str(params["axis"])))
        elif action.name == RelationAction.STACK_NEXT_TO.value:
            moved_sources.append(stack_next_to(source, anchor, str(params["side"])))
        elif action.name == RelationAction.DRAW_CONNECTING_LINE.value:
            moved_sources.append(source)
            extra_objects.append(connector_object(source, anchor))
        else:
            moved_sources.append(place_in_container(source, anchor))

    output_scene = scene_from_objects(
        [*moved_sources, *targets, *extra_objects],
        background_color=0,
        min_height=example.input_scene.height,
        min_width=example.input_scene.width,
        attributes={"family": latent.family.value, "action": action.name},
    )
    step_scenes = {
        "segment": segmented_scene(example.input_scene),
        "pick_source": highlight_scene(example.input_scene, primary_ids=source_ids, primary_color=9, other_color=2),
        "pick_target": highlight_scene(
            example.input_scene,
            primary_ids=target_ids,
            secondary_ids=source_ids,
            primary_color=9,
            secondary_color=8,
            other_color=2,
        ),
        "relate": preview_scene([*moved_sources, *targets, *extra_objects], attributes={"workspace": "relate_preview"}),
        "render": output_scene,
    }
    step_workspaces = {name: render_scene(scene) for name, scene in step_scenes.items() if name != "render"}
    return output_scene, {
        "segment": {"object_ids": [obj.object_id for obj in example.input_scene.objects]},
        "pick_source": {"selected_object_ids": list(source_ids or example.selected_object_ids)},
        "pick_target": {"selected_object_ids": list(target_ids or example.selected_object_ids)},
        "relate": {"action": action.name, "params": params},
        "render": {"output_scene_objects": [obj.object_id for obj in output_scene.objects]},
    }, step_workspaces, step_scenes


def execute_count_select_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    example: ExampleSpec,
) -> Tuple[SceneSpec, Dict[str, Any], Dict[str, Grid], Dict[str, SceneSpec]]:
    group_by, reducer_selector, post_action = latent.program.args
    params = resolve_value(post_action.params, bindings)
    winners = selected_objects(example.input_scene, example.selected_object_ids)
    group_assignments = {
        obj.object_id: obj.attributes.get("group_key", obj.object_id)
        for obj in example.input_scene.objects
    }
    if post_action.name == "copy_selected":
        placed = normalized_objects(winners)
    elif post_action.name == "recolor_selected":
        placed = tuple(replace_object(obj, color=int(params["to_color"])) for obj in normalized_objects(winners))
    elif post_action.name == "sort_into_row":
        placed = arrange_row(
            normalized_objects(winners),
            gap=1,
            descending=str(params.get("order", "ascending")) == "descending",
        )
    else:
        placed = arrange_row(normalized_objects(winners), gap=1, repeat=max(1, int(params.get("n", 2))))

    output_scene = scene_from_objects(
        placed,
        background_color=0,
        min_height=1,
        min_width=1,
        attributes={"family": latent.family.value, "action": post_action.name},
    )
    step_scenes = {
        "segment": segmented_scene(example.input_scene),
        "group": grouped_scene(example.input_scene, group_assignments),
        "reduce": highlight_scene(example.input_scene, primary_ids=example.selected_object_ids, primary_color=9, other_color=2),
        "act": output_scene,
    }
    step_workspaces = {name: render_scene(scene) for name, scene in step_scenes.items()}
    return output_scene, {
        "segment": {"object_ids": [obj.object_id for obj in example.input_scene.objects]},
        "group": {"group_by": group_by, "group_summary": example.metadata.get("group_summary", {})},
        "reduce": {"winner_key": example.metadata.get("winner_key"), "reducer": reducer_selector.kind.value},
        "act": {"action": post_action.name, "params": params},
    }, step_workspaces, step_scenes


def execute_contextual_example(
    latent: TaskLatent,
    bindings: Dict[str, Any],
    example: ExampleSpec,
) -> Tuple[SceneSpec, Dict[str, Any], Dict[str, Grid], Dict[str, SceneSpec]]:
    cue, _selector, then_action, else_action = latent.program.args
    branch = str(example.metadata.get("branch", "then"))
    action = then_action if branch == "then" else else_action
    params = resolve_value(action.params, bindings)
    chosen = normalized_objects(selected_objects(example.input_scene, example.selected_object_ids))
    if action.name == "stack_left":
        placed = stack_side(chosen, side="left", canvas_height=example.input_scene.height, canvas_width=example.input_scene.width)
    elif action.name == "stack_right":
        placed = stack_side(chosen, side="right", canvas_height=example.input_scene.height, canvas_width=example.input_scene.width)
    elif action.name == "reflect_h":
        placed = reflect_objects(chosen, axis="h", canvas_height=example.input_scene.height, canvas_width=example.input_scene.width)
    elif action.name == "reflect_v":
        placed = reflect_objects(chosen, axis="v", canvas_height=example.input_scene.height, canvas_width=example.input_scene.width)
    else:
        placed = tuple(replace_object(obj, color=int(params["to_color"])) for obj in chosen)

    output_scene = scene_from_objects(
        placed,
        background_color=0,
        min_height=example.input_scene.height,
        min_width=example.input_scene.width,
        attributes={"family": latent.family.value, "action": action.name},
    )
    branch_color = 9 if branch == "then" else 8
    step_scenes = {
        "segment": segmented_scene(example.input_scene),
        "read_cue": cue_scene(example.input_scene, cue.value, example.selected_object_ids),
        "branch": highlight_scene(
            example.input_scene,
            primary_ids=example.selected_object_ids,
            primary_color=branch_color,
            other_color=2,
        ),
        "apply": output_scene,
    }
    step_workspaces = {name: render_scene(scene) for name, scene in step_scenes.items()}
    return output_scene, {
        "segment": {"object_ids": [obj.object_id for obj in example.input_scene.objects]},
        "read_cue": {"cue_kind": cue.value, "cue_value": example.metadata.get("cue_value")},
        "branch": {"branch": branch},
        "apply": {"action": action.name, "params": params},
    }, step_workspaces, step_scenes


def orientation_from_value(value: Any) -> str:
    if isinstance(value, str) and value in ORIENTATIONS:
        return value
    if value == "left":
        return "left"
    if value == "right":
        return "right"
    if value == "top":
        return "up"
    if value == "bottom":
        return "down"
    if isinstance(value, int):
        return ORIENTATIONS[value % len(ORIENTATIONS)]
    return "up"


def execute_symbol_map_example(
    latent: TaskLatent,
    _bindings: Dict[str, Any],
    example: ExampleSpec,
) -> Tuple[SceneSpec, Dict[str, Any], Dict[str, Grid], Dict[str, SceneSpec]]:
    _key_type, _value_type, apply_mode = latent.program.args
    mapping = dict(example.metadata.get("mapping", {}))
    targets = normalized_objects(selected_objects(example.input_scene, example.selected_object_ids))
    if apply_mode.name == "recolor_targets":
        placed = tuple(
            replace_object(obj, color=int(mapping.get(obj.attributes.get("target_key"), obj.color)))
            for obj in targets
        )
    elif apply_mode.name == "place_targets":
        placed = arrange_row(targets, gap=1)
    else:
        placed = tuple(
            replace_object(obj, orientation=orientation_from_value(mapping.get(obj.attributes.get("target_key"))))
            for obj in targets
        )

    output_scene = scene_from_objects(
        placed,
        background_color=0,
        min_height=1,
        min_width=1,
        attributes={"family": latent.family.value, "action": apply_mode.name},
    )
    target_keys = {
        obj.object_id: obj.attributes.get("target_key")
        for obj in selected_objects(example.input_scene, example.selected_object_ids)
    }
    step_scenes = {
        "segment": segmented_scene(example.input_scene),
        "bind": legend_scene(example.input_scene, mapping),
        "match": matched_scene(example.input_scene, example.selected_object_ids),
        "apply": output_scene,
    }
    step_workspaces = {name: render_scene(scene) for name, scene in step_scenes.items()}
    return output_scene, {
        "segment": {"object_ids": [obj.object_id for obj in example.input_scene.objects]},
        "bind": {"mapping": mapping},
        "match": {"target_keys": target_keys},
        "apply": {"action": apply_mode.name},
    }, step_workspaces, step_scenes


def execute_example(
    episode: EpisodeSpec,
    example: ExampleSpec,
) -> ExecutedExample:
    input_grid = render_scene(example.input_scene)
    if episode.latent.family == Family.UNARY_OBJECT:
        output_scene, trace_targets, step_workspaces, step_scenes = execute_unary_example(
            episode.latent,
            episode.role_bindings,
            example,
        )
    elif episode.latent.family == Family.RELATIONAL:
        output_scene, trace_targets, step_workspaces, step_scenes = execute_relational_example(
            episode.latent,
            episode.role_bindings,
            example,
        )
    elif episode.latent.family == Family.COUNT_SELECT:
        output_scene, trace_targets, step_workspaces, step_scenes = execute_count_select_example(
            episode.latent,
            episode.role_bindings,
            example,
        )
    elif episode.latent.family == Family.CONTEXTUAL:
        output_scene, trace_targets, step_workspaces, step_scenes = execute_contextual_example(
            episode.latent,
            episode.role_bindings,
            example,
        )
    else:
        output_scene, trace_targets, step_workspaces, step_scenes = execute_symbol_map_example(
            episode.latent,
            episode.role_bindings,
            example,
        )

    output_grid = render_scene(output_scene)
    trace_targets = {
        **trace_targets,
        "render": {
            **trace_targets.get("render", {}),
            "grid_shape": [len(output_grid), len(output_grid[0]) if output_grid else 0],
        },
    }
    step_workspaces = {
        **step_workspaces,
        "render": output_grid,
    }
    step_scenes = {
        **step_scenes,
        "render": output_scene,
    }
    return ExecutedExample(
        example=example,
        input_grid=input_grid,
        output_grid=output_grid,
        output_scene=output_scene,
        trace_targets=trace_targets,
        step_workspaces=step_workspaces,
        step_scenes=step_scenes,
    )


def execute_episode(episode: EpisodeSpec) -> ExecutedEpisode:
    train_examples = tuple(execute_example(episode, example) for example in episode.train_examples)
    test_example = execute_example(episode, episode.test_example)
    return ExecutedEpisode(
        episode=episode,
        train_examples=train_examples,
        test_example=test_example,
    )


if __name__ == "__main__":
    try:
        from .stage1_latent_sampler import sample_latent_rule
        from .stage2_episode_sampler import sample_episode
    except ImportError:  # pragma: no cover - direct script execution
        from stage1_latent_sampler import sample_latent_rule  # type: ignore
        from stage2_episode_sampler import sample_episode  # type: ignore

    latent = sample_latent_rule(seed=0)
    episode = sample_episode(latent, seed=0)
    executed = execute_episode(episode)
    print(json.dumps(executed.to_jsonable(), indent=2))
