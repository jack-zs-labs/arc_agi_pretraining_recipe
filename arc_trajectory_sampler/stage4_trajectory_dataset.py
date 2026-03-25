from __future__ import annotations

"""Stage 4 trajectory compilation with object-level states and verifier labels."""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .stage1_latent_sampler import Family, RoleVar
    from .stage2_episode_sampler import EpisodeSpec, ObjectSpec, SceneSpec, compute_mass
    from .stage3_grid_executor import ExecutedEpisode, ExecutedExample, Grid, execute_episode, render_scene, replace_object
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import Family, RoleVar  # type: ignore
    from stage2_episode_sampler import EpisodeSpec, ObjectSpec, SceneSpec, compute_mass  # type: ignore
    from stage3_grid_executor import ExecutedEpisode, ExecutedExample, Grid, execute_episode, render_scene, replace_object  # type: ignore


STEP_WEIGHTS = {
    "segment": 0.12,
    "select": 0.18,
    "pick_source": 0.14,
    "pick_target": 0.14,
    "group": 0.16,
    "reduce": 0.18,
    "read_cue": 0.18,
    "branch": 0.14,
    "bind": 0.18,
    "match": 0.18,
    "transform": 0.28,
    "relate": 0.28,
    "act": 0.26,
    "apply": 0.26,
    "render": 0.32,
    "extract_entities": 0.12,
    "bind_quantities": 0.18,
    "choose_operation": 0.18,
    "compute_answer": 0.26,
    "emit_ir": 0.22,
}
EXECUTION_STEPS = {"transform", "relate", "act", "apply", "render"}
State = Dict[str, Any]


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    name: str
    description: str
    action: Dict[str, Any]
    reward: float
    reward_terms: Dict[str, float]
    cumulative_reward: float
    progress: float
    stop_target: bool
    workspace_state: State
    verifier: Dict[str, Any]
    done: bool
    workspace_grid: Optional[Grid] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    split: str
    family: str
    difficulty: int
    source_modality: str
    concept_tags: Tuple[str, ...]
    trace_template: Tuple[str, ...]
    role_bindings: Dict[str, Any]
    episode_metadata: Dict[str, Any]
    shortcut_checks: Tuple[str, ...]
    example: Any
    input_state: State
    output_state: State
    steps: Tuple[TrajectoryStep, ...]
    total_reward: float
    total_possible_reward: float
    trajectory_role: str = "positive"
    variant_kind: str = "canonical"
    canonical_trace_template: Optional[Tuple[str, ...]] = None
    parent_trajectory_id: Optional[str] = None
    input_grid: Optional[Grid] = None
    output_grid: Optional[Grid] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


@dataclass
class CompileCache:
    selected_object_ids: Tuple[str, ...]
    selected_object_ids_flat: Dict[str, Any]
    scene_flat_by_id: Dict[int, Dict[str, Any]]
    scene_summary_by_id: Dict[int, Dict[str, Any]]
    scene_summary_flat_by_id: Dict[int, Dict[str, Any]]
    grid_flat_by_grid: Dict[Grid, Dict[str, Any]]
    scene_encoded_by_id: Dict[int, Dict[str, Any]]
    bindings_flat_by_id: Dict[int, Dict[str, Any]]
    bindings_encoded_by_id: Dict[int, Dict[str, Any]]
    focus_object_ids_by_step: Dict[str, List[str]]
    focus_flat_by_step: Dict[str, Dict[str, Any]]
    subgoal_state_by_completed: Dict[Tuple[str, ...], Tuple[Tuple[str, ...], Dict[str, Any], Dict[str, Any]]]
    input_state: Optional[State] = None
    output_state: Optional[State] = None
    final_workspace_flat: Optional[Dict[str, Any]] = None
    final_scene_flat: Optional[Dict[str, Any]] = None


def _encode(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_encode(item) for item in obj]
    if isinstance(obj, list):
        return [_encode(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if is_dataclass(obj):
        return {field.name: _encode(getattr(obj, field.name)) for field in fields(obj)}
    return obj


def grid_shape(grid: Grid) -> Tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def pad_grid(grid: Grid, height: int, width: int, fill: int = 0) -> Grid:
    rows = [list(row) + [fill] * max(0, width - len(row)) for row in grid]
    rows.extend([[fill] * width for _ in range(max(0, height - len(rows)))])
    return tuple(tuple(row[:width]) for row in rows[:height])


def grid_similarity(first: Grid, second: Grid) -> float:
    if first is second or first == second:
        return 1.0
    if len(first) == len(second) and (not first or len(first[0]) == len(second[0])):
        height = max(len(first), 1)
        width = max(len(first[0]) if first else 0, 1)
        matches = 0
        total = height * width
        for row in range(height):
            lhs_row = first[row]
            rhs_row = second[row]
            for col in range(width):
                matches += 1 if lhs_row[col] == rhs_row[col] else 0
        return matches / total
    height = max(len(first), len(second), 1)
    width = max(len(first[0]) if first else 0, len(second[0]) if second else 0, 1)
    lhs = pad_grid(first, height, width)
    rhs = pad_grid(second, height, width)
    matches = 0
    total = height * width
    for row in range(height):
        for col in range(width):
            matches += 1 if lhs[row][col] == rhs[row][col] else 0
    return matches / total


def changed_cell_fraction(input_grid: Grid, output_grid: Grid) -> float:
    return 1.0 - grid_similarity(input_grid, output_grid)


def flatten_scalar_sequence(values: Sequence[Any], prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}[{index}]": value.value if isinstance(value, Enum) else value
        for index, value in enumerate(values)
    }


def flatten_state(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, Enum):
        value = value.value

    if isinstance(value, dict):
        flattened: Dict[str, Any] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_state(value[key], child_prefix))
        return flattened

    if isinstance(value, (list, tuple)):
        flattened: Dict[str, Any] = {}
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            flattened.update(flatten_state(item, child_prefix))
        return flattened

    if is_dataclass(value):
        flattened: Dict[str, Any] = {}
        for field in fields(value):
            child_prefix = f"{prefix}.{field.name}" if prefix else field.name
            flattened.update(flatten_state(getattr(value, field.name), child_prefix))
        return flattened

    leaf_key = prefix or "<root>"
    return {leaf_key: value}


def state_similarity_from_flat(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> float:
    if lhs is rhs:
        return 1.0
    if lhs.keys() == rhs.keys():
        if not lhs:
            return 1.0
        matches = sum(1 for key, value in lhs.items() if rhs.get(key) == value)
        return matches / len(lhs)
    keys = set(lhs) | set(rhs)
    if not keys:
        return 1.0
    matches = sum(1 for key in keys if lhs.get(key) == rhs.get(key))
    return matches / len(keys)


def state_similarity(first: Any, second: Any) -> float:
    return state_similarity_from_flat(flatten_state(first), flatten_state(second))


def state_change_fraction(first: Any, second: Any) -> float:
    return 1.0 - state_similarity(first, second)


def symbolic_step_reward_terms(
    current_state: State,
    target_state: State,
    previous_state: State,
    final_state: State,
) -> Dict[str, float]:
    local_before = state_similarity(previous_state, target_state)
    local_after = state_similarity(current_state, target_state)
    output_before = state_similarity(previous_state, final_state)
    output_after = state_similarity(current_state, final_state)
    return {
        "local_progress": max(0.0, local_after - local_before),
        "output_progress": max(0.0, output_after - output_before),
        "state_delta": state_change_fraction(previous_state, current_state),
    }


def symbolic_step_reward_terms_from_flat(
    current_state_flat: Dict[str, Any],
    previous_state_flat: Dict[str, Any],
    final_state_flat: Dict[str, Any],
) -> Dict[str, float]:
    previous_to_current = state_similarity_from_flat(previous_state_flat, current_state_flat)
    output_before = state_similarity_from_flat(previous_state_flat, final_state_flat)
    output_after = state_similarity_from_flat(current_state_flat, final_state_flat)
    delta = 1.0 - previous_to_current
    return {
        "local_progress": delta,
        "output_progress": max(0.0, output_after - output_before),
        "state_delta": delta,
    }


def reward_from_terms(weight: float, reward_terms: Dict[str, float]) -> float:
    score = (
        0.5 * reward_terms["local_progress"]
        + 0.3 * reward_terms["output_progress"]
        + 0.2 * reward_terms["state_delta"]
    )
    return weight * score


def step_workspace_grid(step_name: str, executed: ExecutedExample) -> Grid:
    return executed.step_workspaces.get(step_name, executed.output_grid if step_name in EXECUTION_STEPS else executed.input_grid)


def step_workspace_scene(step_name: str, executed: ExecutedExample) -> SceneSpec:
    default_scene = executed.output_scene if step_name in EXECUTION_STEPS else executed.example.input_scene
    return executed.step_scenes.get(step_name, default_scene)


def focus_object_ids(step_name: str, executed: ExecutedExample) -> List[str]:
    payload = executed.trace_targets.get(step_name, {})
    if isinstance(payload.get("selected_object_ids"), list):
        return [str(item) for item in payload["selected_object_ids"]]
    if isinstance(payload.get("object_ids"), list):
        return [str(item) for item in payload["object_ids"]]
    if isinstance(payload.get("target_keys"), dict):
        return [str(item) for item in payload["target_keys"].keys()]
    if step_name == "bind":
        return [entry.object_id for entry in executed.example.input_scene.legend]
    if step_name in {"select", "reduce", "branch", "transform", "act", "apply", "match"}:
        return [str(item) for item in executed.example.selected_object_ids]
    return []


def scene_summary(scene: SceneSpec) -> Dict[str, Any]:
    return {
        "height": scene.height,
        "width": scene.width,
        "background_color": scene.background_color,
        "border_color": scene.border_color,
        "outline_color": scene.outline_color,
        "marker_position": scene.marker_position,
        "object_ids": [obj.object_id for obj in scene.objects],
        "object_count": len(scene.objects),
        "legend_object_ids": [entry.object_id for entry in scene.legend],
    }


def build_workspace_state(
    step_name: str,
    executed: ExecutedExample,
    *,
    canonical_trace_names: Sequence[str],
    completed_subgoals: Sequence[str],
    current_grid: Optional[Grid] = None,
    current_scene: Optional[SceneSpec] = None,
    action_bindings: Optional[Dict[str, Any]] = None,
    selected_object_ids_value: Optional[Sequence[str]] = None,
    focus_object_ids_value: Optional[Sequence[str]] = None,
    scene_summary_value: Optional[Dict[str, Any]] = None,
    remaining_subgoals_value: Optional[Sequence[str]] = None,
    subgoal_status_value: Optional[Dict[str, Any]] = None,
    encoded_scene_value: Optional[Dict[str, Any]] = None,
    encoded_bindings_value: Optional[Dict[str, Any]] = None,
) -> State:
    completed = tuple(completed_subgoals)
    completed_set = set(completed)
    if remaining_subgoals_value is None:
        remaining_subgoals = tuple(name for name in canonical_trace_names if name not in completed_set)
    else:
        remaining_subgoals = tuple(remaining_subgoals_value)
    scene = current_scene if current_scene is not None else step_workspace_scene(step_name, executed)
    if focus_object_ids_value is None:
        focus_ids = tuple(focus_object_ids(step_name, executed))
    else:
        focus_ids = tuple(focus_object_ids_value)
    if scene_summary_value is None:
        summary = scene_summary(scene)
    else:
        summary = scene_summary_value
    if selected_object_ids_value is None:
        selected_object_ids = tuple(executed.example.selected_object_ids)
    else:
        selected_object_ids = tuple(selected_object_ids_value)
    if subgoal_status_value is None:
        subgoal_status = {
            name: ("completed" if name in completed_set else "pending")
            for name in canonical_trace_names
        }
    else:
        subgoal_status = subgoal_status_value
    return {
        "modality": "object_scene",
        "step_name": step_name,
        "scene": encoded_scene_value if encoded_scene_value is not None else _encode(scene),
        "scene_summary": summary,
        "selected_object_ids": selected_object_ids,
        "focus_object_ids": focus_ids,
        "bindings": (
            encoded_bindings_value
            if encoded_bindings_value is not None
            else _encode(action_bindings if action_bindings is not None else executed.trace_targets.get(step_name, {}))
        ),
        "completed_subgoals": completed,
        "remaining_subgoals": remaining_subgoals,
        "subgoal_status": subgoal_status,
    }


def verifier_state(
    step_name: str,
    executed: ExecutedExample,
    *,
    canonical_trace_names: Sequence[str],
    completed_subgoals: Sequence[str],
    current_grid: Optional[Grid] = None,
    current_scene: Optional[SceneSpec] = None,
    current_scene_flat: Optional[Dict[str, Any]] = None,
    final_scene_flat: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    grid = current_grid if current_grid is not None else step_workspace_grid(step_name, executed)
    scene = current_scene if current_scene is not None else step_workspace_scene(step_name, executed)
    completed = list(completed_subgoals)
    completed_set = set(completed)
    remaining_subgoals = [name for name in canonical_trace_names if name not in completed_set]
    exact_grid_match = grid == executed.output_grid
    exact_scene_match = scene == executed.output_scene
    if exact_grid_match and not remaining_subgoals:
        non_terminal_reason = None
    elif remaining_subgoals:
        non_terminal_reason = f"remaining_subgoal:{remaining_subgoals[0]}"
    else:
        non_terminal_reason = "workspace_does_not_match_target_output"
    return {
        "exact_match": exact_grid_match and not remaining_subgoals,
        "exact_grid_match": exact_grid_match,
        "exact_scene_match": exact_scene_match,
        "should_stop": exact_grid_match and not remaining_subgoals,
        "grid_match": round(grid_similarity(grid, executed.output_grid), 6),
        "scene_match": round(
            state_similarity_from_flat(
                current_scene_flat if current_scene_flat is not None else flatten_state(scene),
                final_scene_flat if final_scene_flat is not None else flatten_state(executed.output_scene),
            ),
            6,
        ),
        "resolved_subgoal_count": len(completed),
        "unresolved_subgoal_count": len(remaining_subgoals),
        "next_subgoal": remaining_subgoals[0] if remaining_subgoals else None,
        "non_terminal_reason": non_terminal_reason,
    }


def arc_input_state(executed: ExecutedExample) -> State:
    return {
        "modality": "object_scene",
        "scene": _encode(executed.example.input_scene),
        "scene_summary": scene_summary(executed.example.input_scene),
        "input_grid": _encode(executed.input_grid),
        "selected_object_ids": list(executed.example.selected_object_ids),
    }


def arc_output_state(executed: ExecutedExample) -> State:
    return {
        "modality": "object_scene",
        "scene": _encode(executed.output_scene),
        "scene_summary": scene_summary(executed.output_scene),
        "output_grid": _encode(executed.output_grid),
        "trace_targets": _encode(executed.trace_targets),
    }


def step_payload(
    step_name: str,
    executed: ExecutedExample,
    *,
    current_grid: Optional[Grid] = None,
    action_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(executed.trace_targets.get(step_name, {}))
    if action_override:
        payload.update(action_override)
    workspace_grid = current_grid if current_grid is not None else step_workspace_grid(step_name, executed)
    if step_name in EXECUTION_STEPS:
        payload["workspace_grid_shape"] = list(grid_shape(workspace_grid))
        payload["target_grid_shape"] = list(grid_shape(executed.output_grid))
        payload["changed_cell_fraction"] = round(changed_cell_fraction(executed.input_grid, workspace_grid), 6)
    return payload


def trace_step_descriptions(episode: EpisodeSpec) -> Dict[str, str]:
    return {
        step.name: step.description
        for step in episode.latent.trace_template
    }


def swapped_trace_names(trace_names: Sequence[str], first: str, second: str) -> Tuple[str, ...]:
    swapped = list(trace_names)
    first_index = swapped.index(first)
    second_index = swapped.index(second)
    swapped[first_index], swapped[second_index] = swapped[second_index], swapped[first_index]
    return tuple(swapped)


def alternate_trace_variants(episode: EpisodeSpec) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    canonical = tuple(step.name for step in episode.latent.trace_template)
    variants: List[Tuple[str, Tuple[str, ...]]] = []
    if episode.latent.family == Family.RELATIONAL and "pick_source" in canonical and "pick_target" in canonical:
        variants.append(("swap_pick_order", swapped_trace_names(canonical, "pick_source", "pick_target")))
    if episode.latent.family == Family.SYMBOL_MAP and "bind" in canonical and "match" in canonical:
        variants.append(("swap_bind_match", swapped_trace_names(canonical, "bind", "match")))
    return tuple(
        (variant_kind, trace_names)
        for variant_kind, trace_names in variants
        if trace_names != canonical
    )


def scene_with_objects(
    base_scene: SceneSpec,
    objects: Sequence[ObjectSpec],
    *,
    attributes: Optional[Dict[str, Any]] = None,
) -> SceneSpec:
    merged_attributes = dict(base_scene.attributes)
    if attributes:
        merged_attributes.update(attributes)
    return SceneSpec(
        height=base_scene.height,
        width=base_scene.width,
        background_color=base_scene.background_color,
        border_color=base_scene.border_color,
        outline_color=base_scene.outline_color,
        marker_position=base_scene.marker_position,
        objects=tuple(objects),
        legend=tuple(base_scene.legend),
        attributes=merged_attributes,
    )


def wrong_color(color: int) -> int:
    return 1 if color != 1 else 2


def wrong_dot_object(scene: SceneSpec) -> ObjectSpec:
    return ObjectSpec(
        object_id="wrong_final_dot",
        shape="dot",
        color=wrong_color(scene.background_color),
        top=0,
        left=0,
        height=1,
        width=1,
        mass=compute_mass("dot", 1, 1, 0),
        orientation="up",
        holes=0,
        is_container=False,
        tags=("negative",),
        attributes={},
    )


def build_wrong_final_scene(executed: ExecutedExample) -> Tuple[SceneSpec, Dict[str, Any]]:
    base_scene = executed.output_scene
    prioritized_ids = list(executed.example.selected_object_ids)
    prioritized_ids.extend(obj.object_id for obj in base_scene.objects if obj.object_id not in prioritized_ids)

    for object_id in prioritized_ids:
        obj = next((item for item in base_scene.objects if item.object_id == object_id), None)
        if obj is None:
            continue
        candidates: List[Tuple[ObjectSpec, Dict[str, Any]]] = []
        if obj.left + obj.width < base_scene.width:
            candidates.append(
                (
                    replace_object(obj, left=obj.left + 1),
                    {"negative_edit": "shift_right", "edited_object_id": object_id},
                )
            )
        if obj.top + obj.height < base_scene.height:
            candidates.append(
                (
                    replace_object(obj, top=obj.top + 1),
                    {"negative_edit": "shift_down", "edited_object_id": object_id},
                )
            )
        candidates.append(
            (
                replace_object(obj, color=wrong_color(obj.color)),
                {"negative_edit": "recolor", "edited_object_id": object_id},
            )
        )
        for candidate, metadata in candidates:
            objects = [candidate if item.object_id == object_id else item for item in base_scene.objects]
            wrong_scene = scene_with_objects(base_scene, objects, attributes={"variant": "wrong_final_state"})
            if render_scene(wrong_scene) != executed.output_grid:
                return wrong_scene, metadata

    if base_scene.objects:
        first = base_scene.objects[0]
        fallback = replace_object(first, color=wrong_color(first.color))
        wrong_scene = scene_with_objects(
            base_scene,
            [fallback if item.object_id == first.object_id else item for item in base_scene.objects],
            attributes={"variant": "wrong_final_state"},
        )
        return wrong_scene, {"negative_edit": "fallback_recolor", "edited_object_id": first.object_id}

    wrong_scene = scene_with_objects(
        base_scene,
        [wrong_dot_object(base_scene)],
        attributes={"variant": "wrong_final_state"},
    )
    return wrong_scene, {"negative_edit": "inject_dot", "edited_object_id": "wrong_final_dot"}


def cached_scene_flat(cache: CompileCache, scene: SceneSpec) -> Dict[str, Any]:
    scene_key = id(scene)
    flat = cache.scene_flat_by_id.get(scene_key)
    if flat is None:
        flat = flatten_state(scene, prefix="scene")
        cache.scene_flat_by_id[scene_key] = flat
    return flat


def cached_scene_summary(cache: CompileCache, scene: SceneSpec) -> Dict[str, Any]:
    scene_key = id(scene)
    summary = cache.scene_summary_by_id.get(scene_key)
    if summary is None:
        summary = scene_summary(scene)
        cache.scene_summary_by_id[scene_key] = summary
    return summary


def cached_scene_summary_flat(cache: CompileCache, scene: SceneSpec) -> Dict[str, Any]:
    scene_key = id(scene)
    flat = cache.scene_summary_flat_by_id.get(scene_key)
    if flat is None:
        flat = flatten_state(cached_scene_summary(cache, scene), prefix="scene_summary")
        cache.scene_summary_flat_by_id[scene_key] = flat
    return flat


def cached_grid_flat(cache: CompileCache, grid: Grid) -> Dict[str, Any]:
    flat = cache.grid_flat_by_grid.get(grid)
    if flat is None:
        flat = {}
        for row_index, row in enumerate(grid):
            row_prefix = f"workspace_grid[{row_index}]"
            for col_index, value in enumerate(row):
                flat[f"{row_prefix}[{col_index}]"] = value
        cache.grid_flat_by_grid[grid] = flat
    return flat


def cached_scene_encoded(cache: CompileCache, scene: SceneSpec) -> Dict[str, Any]:
    scene_key = id(scene)
    encoded = cache.scene_encoded_by_id.get(scene_key)
    if encoded is None:
        encoded = _encode(scene)
        cache.scene_encoded_by_id[scene_key] = encoded
    return encoded


def cached_bindings_flat(cache: CompileCache, bindings: Dict[str, Any]) -> Dict[str, Any]:
    binding_key = id(bindings)
    flat = cache.bindings_flat_by_id.get(binding_key)
    if flat is None:
        flat = flatten_state(bindings, prefix="bindings") if bindings else {}
        cache.bindings_flat_by_id[binding_key] = flat
    return flat


def cached_bindings_encoded(cache: CompileCache, bindings: Dict[str, Any]) -> Dict[str, Any]:
    binding_key = id(bindings)
    encoded = cache.bindings_encoded_by_id.get(binding_key)
    if encoded is None:
        encoded = _encode(bindings) if bindings else {}
        cache.bindings_encoded_by_id[binding_key] = encoded
    return encoded


def cached_focus_object_ids(cache: CompileCache, step_name: str, executed: ExecutedExample) -> List[str]:
    ids = cache.focus_object_ids_by_step.get(step_name)
    if ids is None:
        ids = focus_object_ids(step_name, executed)
        cache.focus_object_ids_by_step[step_name] = ids
    return ids


def cached_focus_flat(cache: CompileCache, step_name: str, executed: ExecutedExample) -> Dict[str, Any]:
    flat = cache.focus_flat_by_step.get(step_name)
    if flat is None:
        flat = flatten_scalar_sequence(cached_focus_object_ids(cache, step_name, executed), "focus_object_ids")
        cache.focus_flat_by_step[step_name] = flat
    return flat


def cached_subgoal_state(
    cache: CompileCache,
    canonical_trace_names: Sequence[str],
    completed_subgoals: Sequence[str],
) -> Tuple[Tuple[str, ...], Dict[str, Any], Dict[str, Any]]:
    completed_key = tuple(completed_subgoals)
    cached = cache.subgoal_state_by_completed.get(completed_key)
    if cached is not None:
        return cached

    completed_set = set(completed_key)
    remaining = tuple(name for name in canonical_trace_names if name not in completed_set)
    flat = {}
    flat.update(flatten_scalar_sequence(completed_key, "completed_subgoals"))
    flat.update(flatten_scalar_sequence(remaining, "remaining_subgoals"))
    subgoal_status = {}
    for name in canonical_trace_names:
        status = "completed" if name in completed_set else "pending"
        flat[f"subgoal_status.{name}"] = status
        subgoal_status[name] = status
    cache.subgoal_state_by_completed[completed_key] = (remaining, flat, subgoal_status)
    return remaining, flat, subgoal_status


def workspace_state_flat(
    step_name: str,
    executed: ExecutedExample,
    *,
    canonical_trace_names: Sequence[str],
    completed_subgoals: Sequence[str],
    current_grid: Grid,
    current_scene: SceneSpec,
    action_bindings: Dict[str, Any],
    bindings_flat_value: Optional[Dict[str, Any]] = None,
    cache: CompileCache,
) -> Tuple[Dict[str, Any], Tuple[str, ...], Dict[str, Any], List[str]]:
    remaining_subgoals, subgoal_flat, _subgoal_status = cached_subgoal_state(cache, canonical_trace_names, completed_subgoals)
    scene_summary_value = cached_scene_summary(cache, current_scene)
    focus_ids = cached_focus_object_ids(cache, step_name, executed)
    flat: Dict[str, Any] = {
        "modality": "object_scene",
        "step_name": step_name,
    }
    flat.update(cached_scene_flat(cache, current_scene))
    flat.update(cached_scene_summary_flat(cache, current_scene))
    flat.update(cached_grid_flat(cache, current_grid))
    flat.update(cache.selected_object_ids_flat)
    flat.update(cached_focus_flat(cache, step_name, executed))
    if action_bindings:
        flat.update(bindings_flat_value if bindings_flat_value is not None else cached_bindings_flat(cache, action_bindings))
    flat.update(subgoal_flat)
    return flat, remaining_subgoals, scene_summary_value, focus_ids


def compile_trajectory(
    episode: EpisodeSpec,
    executed: ExecutedExample,
    *,
    split: str,
    trajectory_index: int,
    step_names: Optional[Sequence[str]] = None,
    trajectory_role: str = "positive",
    variant_kind: str = "canonical",
    parent_trajectory_id: Optional[str] = None,
    scene_overrides: Optional[Dict[str, SceneSpec]] = None,
    action_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    cache: Optional[CompileCache] = None,
) -> TrajectoryRecord:
    canonical_trace_names = tuple(step.name for step in episode.latent.trace_template)
    trace_names = tuple(step_names) if step_names is not None else canonical_trace_names
    descriptions = trace_step_descriptions(episode)
    total_possible_reward = sum(STEP_WEIGHTS.get(name, 0.15) for name in canonical_trace_names)
    compile_cache = cache or CompileCache(
        selected_object_ids=tuple(executed.example.selected_object_ids),
        selected_object_ids_flat=flatten_scalar_sequence(executed.example.selected_object_ids, "selected_object_ids"),
        scene_flat_by_id={},
        scene_summary_by_id={},
        scene_summary_flat_by_id={},
        grid_flat_by_grid={},
        scene_encoded_by_id={},
        bindings_flat_by_id={},
        bindings_encoded_by_id={},
        focus_object_ids_by_step={},
        focus_flat_by_step={},
        subgoal_state_by_completed={},
    )
    cumulative = 0.0
    steps: List[TrajectoryStep] = []
    if compile_cache.input_state is None:
        compile_cache.input_state = {
            "modality": "object_scene",
            "scene": cached_scene_encoded(compile_cache, executed.example.input_scene),
            "scene_summary": cached_scene_summary(compile_cache, executed.example.input_scene),
            "input_grid": executed.input_grid,
            "selected_object_ids": compile_cache.selected_object_ids,
        }
    if compile_cache.output_state is None:
        compile_cache.output_state = {
            "modality": "object_scene",
            "scene": cached_scene_encoded(compile_cache, executed.output_scene),
            "scene_summary": cached_scene_summary(compile_cache, executed.output_scene),
            "output_grid": executed.output_grid,
            "trace_targets": _encode(executed.trace_targets),
        }
    input_state = compile_cache.input_state
    output_state = compile_cache.output_state
    final_action_bindings = dict(executed.trace_targets.get(canonical_trace_names[-1], {}))
    if compile_cache.final_workspace_flat is None:
        compile_cache.final_workspace_flat, _, _, _ = workspace_state_flat(
            canonical_trace_names[-1],
            executed,
            canonical_trace_names=canonical_trace_names,
            completed_subgoals=canonical_trace_names,
            current_grid=executed.output_grid,
            current_scene=executed.output_scene,
            action_bindings=final_action_bindings,
            cache=compile_cache,
        )
    if compile_cache.final_scene_flat is None:
        compile_cache.final_scene_flat = cached_scene_flat(compile_cache, executed.output_scene)
    final_workspace_flat = compile_cache.final_workspace_flat
    final_scene_flat = compile_cache.final_scene_flat
    previous_state: State = input_state
    previous_state_flat = flatten_state(previous_state)
    completed_subgoals: List[str] = []

    for index, step_name in enumerate(trace_names):
        current_scene = (scene_overrides or {}).get(step_name, step_workspace_scene(step_name, executed))
        current_grid = render_scene(current_scene) if step_name in (scene_overrides or {}) else step_workspace_grid(step_name, executed)
        action_payload_override = (action_overrides or {}).get(step_name, {})
        completed_subgoals.append(step_name)
        base_action_bindings = executed.trace_targets.get(step_name, {})
        if action_payload_override:
            action_bindings = {**base_action_bindings, **action_payload_override}
            bindings_flat = flatten_state(action_bindings, prefix="bindings") if action_bindings else {}
            encoded_bindings = _encode(action_bindings) if action_bindings else {}
        else:
            action_bindings = base_action_bindings
            bindings_flat = cached_bindings_flat(compile_cache, action_bindings)
            encoded_bindings = cached_bindings_encoded(compile_cache, action_bindings)
        encoded_scene = cached_scene_encoded(compile_cache, current_scene)
        _, _, subgoal_status = cached_subgoal_state(compile_cache, canonical_trace_names, completed_subgoals)
        current_state_flat, remaining_subgoals, scene_summary_value, focus_ids = workspace_state_flat(
            step_name,
            executed,
            canonical_trace_names=canonical_trace_names,
            completed_subgoals=completed_subgoals,
            current_grid=current_grid,
            current_scene=current_scene,
            action_bindings=action_bindings,
            bindings_flat_value=bindings_flat,
            cache=compile_cache,
        )
        current_state = build_workspace_state(
            step_name,
            executed,
            canonical_trace_names=canonical_trace_names,
            completed_subgoals=completed_subgoals,
            current_grid=current_grid,
            current_scene=current_scene,
            action_bindings=action_bindings,
            selected_object_ids_value=compile_cache.selected_object_ids,
            focus_object_ids_value=focus_ids,
            scene_summary_value=scene_summary_value,
            remaining_subgoals_value=remaining_subgoals,
            subgoal_status_value=subgoal_status,
            encoded_scene_value=encoded_scene,
            encoded_bindings_value=encoded_bindings,
        )
        reward_terms = symbolic_step_reward_terms_from_flat(
            current_state_flat,
            previous_state_flat,
            final_workspace_flat,
        )
        weight = STEP_WEIGHTS.get(step_name, 0.15)
        reward = reward_from_terms(weight, reward_terms)
        cumulative += reward
        current_scene_flat = cached_scene_flat(compile_cache, current_scene)
        verifier = verifier_state(
            step_name,
            executed,
            canonical_trace_names=canonical_trace_names,
            completed_subgoals=completed_subgoals,
            current_grid=current_grid,
            current_scene=current_scene,
            current_scene_flat=current_scene_flat,
            final_scene_flat=final_scene_flat,
        )
        steps.append(
            TrajectoryStep(
                index=index,
                name=step_name,
                description=descriptions[step_name],
                action=step_payload(
                    step_name,
                    executed,
                    current_grid=current_grid,
                    action_override=action_payload_override,
                ),
                reward=reward,
                reward_terms=reward_terms,
                cumulative_reward=cumulative,
                progress=len(completed_subgoals) / max(1, len(canonical_trace_names)),
                stop_target=bool(verifier["should_stop"]),
                workspace_state=current_state,
                verifier=verifier,
                done=index == len(trace_names) - 1,
                workspace_grid=current_grid,
            )
        )
        previous_state = current_state
        previous_state_flat = current_state_flat

    return TrajectoryRecord(
        trajectory_id=f"{episode.latent.family.value}:{split}:{executed.example.example_id}:{trajectory_index}:{variant_kind}",
        split=split,
        family=episode.latent.family.value,
        difficulty=episode.latent.difficulty,
        source_modality="object_scene",
        concept_tags=episode.latent.concept_tags,
        trace_template=trace_names,
        role_bindings=episode.role_bindings,
        episode_metadata=episode.episode_metadata,
        shortcut_checks=episode.shortcut_checks,
        example=executed.example,
        input_state=input_state,
        output_state=output_state,
        trajectory_role=trajectory_role,
        variant_kind=variant_kind,
        canonical_trace_template=canonical_trace_names,
        parent_trajectory_id=parent_trajectory_id,
        input_grid=executed.input_grid,
        output_grid=executed.output_grid,
        steps=tuple(steps),
        total_reward=cumulative,
        total_possible_reward=total_possible_reward,
    )


def compile_episode_trajectories(
    executed_episode: ExecutedEpisode,
    *,
    include_test: bool = True,
    include_alternates: bool = True,
    include_negatives: bool = True,
) -> Tuple[TrajectoryRecord, ...]:
    trajectories: List[TrajectoryRecord] = []
    example_specs: List[Tuple[str, ExecutedExample]] = [("train", executed) for executed in executed_episode.train_examples]
    if include_test:
        example_specs.append(("test", executed_episode.test_example))

    trajectory_index = 0
    for split, executed in example_specs:
        compile_cache = CompileCache(
            selected_object_ids=tuple(executed.example.selected_object_ids),
            selected_object_ids_flat=flatten_scalar_sequence(executed.example.selected_object_ids, "selected_object_ids"),
            scene_flat_by_id={},
            scene_summary_by_id={},
            scene_summary_flat_by_id={},
            grid_flat_by_grid={},
            scene_encoded_by_id={},
            bindings_flat_by_id={},
            bindings_encoded_by_id={},
            focus_object_ids_by_step={},
            focus_flat_by_step={},
            subgoal_state_by_completed={},
        )
        canonical = compile_trajectory(
            executed_episode.episode,
            executed,
            split=split,
            trajectory_index=trajectory_index,
            variant_kind="canonical",
            cache=compile_cache,
        )
        trajectories.append(canonical)
        parent_trajectory_id = canonical.trajectory_id
        trajectory_index += 1

        if include_alternates:
            for variant_kind, trace_names in alternate_trace_variants(executed_episode.episode):
                trajectories.append(
                    compile_trajectory(
                        executed_episode.episode,
                        executed,
                        split=split,
                        trajectory_index=trajectory_index,
                        step_names=trace_names,
                        trajectory_role="positive",
                        variant_kind=variant_kind,
                        parent_trajectory_id=parent_trajectory_id,
                        cache=compile_cache,
                    )
                )
                trajectory_index += 1

        if include_negatives:
            canonical_trace_names = tuple(step.name for step in executed_episode.episode.latent.trace_template)
            trajectories.append(
                compile_trajectory(
                    executed_episode.episode,
                    executed,
                    split=split,
                    trajectory_index=trajectory_index,
                    step_names=canonical_trace_names[:-1],
                    trajectory_role="negative",
                    variant_kind="one_step_short",
                    parent_trajectory_id=parent_trajectory_id,
                    cache=compile_cache,
                )
            )
            trajectory_index += 1

            wrong_scene, wrong_metadata = build_wrong_final_scene(executed)
            wrong_final_step = canonical_trace_names[-1]
            trajectories.append(
                compile_trajectory(
                    executed_episode.episode,
                    executed,
                    split=split,
                    trajectory_index=trajectory_index,
                    step_names=canonical_trace_names,
                    trajectory_role="negative",
                    variant_kind="wrong_final_state",
                    parent_trajectory_id=parent_trajectory_id,
                    scene_overrides={wrong_final_step: wrong_scene},
                    action_overrides={wrong_final_step: wrong_metadata},
                    cache=compile_cache,
                )
            )
            trajectory_index += 1
    return tuple(trajectories)


def build_trajectories(
    episode: EpisodeSpec,
    *,
    include_test: bool = True,
    include_alternates: bool = True,
    include_negatives: bool = True,
) -> Tuple[TrajectoryRecord, ...]:
    executed = execute_episode(episode)
    return compile_episode_trajectories(
        executed,
        include_test=include_test,
        include_alternates=include_alternates,
        include_negatives=include_negatives,
    )


def write_jsonl(path: str | Path, records: Sequence[TrajectoryRecord]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_jsonable()))
            handle.write("\n")
    return output_path


if __name__ == "__main__":
    try:
        from .stage1_latent_sampler import sample_latent_rule
        from .stage2_episode_sampler import sample_episode
    except ImportError:  # pragma: no cover - direct script execution
        from stage1_latent_sampler import sample_latent_rule  # type: ignore
        from stage2_episode_sampler import sample_episode  # type: ignore

    latent = sample_latent_rule(seed=0)
    episode = sample_episode(latent, seed=0)
    trajectories = build_trajectories(episode)
    print(json.dumps(trajectories[0].to_jsonable(), indent=2))
