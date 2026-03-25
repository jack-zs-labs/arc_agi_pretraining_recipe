from __future__ import annotations

"""Modality-neutral reasoning-state adapter for structured trajectory prompts."""

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from .stage4_trajectory_dataset import TrajectoryRecord
except ImportError:  # pragma: no cover - direct script execution
    from stage4_trajectory_dataset import TrajectoryRecord  # type: ignore


State = Dict[str, Any]
VerifierTargets = Dict[str, Any]

FORMAT_HEADER = "REASONING_STATE_V1"
TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9_./%-]+")
TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class EncodedWorkspace:
    tokens: Tuple[str, ...]
    scalar_features: Tuple[float, ...]
    metadata: Dict[str, Any]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "scalar_features": list(self.scalar_features),
            "metadata": self.metadata,
        }


def trace_names_for(trajectory: TrajectoryRecord) -> tuple[str, ...]:
    return trajectory.canonical_trace_template or trajectory.trace_template


def normalize_token_atom(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    if isinstance(value, (int, str)):
        text = str(value)
        return re.sub(r"\s+", "_", text.strip()) or "empty"
    return re.sub(r"\s+", "_", json.dumps(value, separators=(",", ":"), sort_keys=True))


def flatten_value_map(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_value_map(value[key], child_prefix))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened: dict[str, str] = {}
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            flattened.update(flatten_value_map(item, child_prefix))
        return flattened
    leaf_key = prefix or "<root>"
    return {leaf_key: json.dumps(value, separators=(",", ":"), sort_keys=True)}


def flatten_tokens(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        tokens: list[str] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            tokens.extend(flatten_tokens(value[key], child_prefix))
        return tokens
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            tokens.extend(flatten_tokens(item, child_prefix))
        return tokens
    leaf_key = prefix or "<root>"
    return [f"{leaf_key}={normalize_token_atom(value)}"]


def flattened_match_fraction(first: Any, second: Any) -> float:
    lhs = flatten_value_map(first)
    rhs = flatten_value_map(second)
    keys = set(lhs) | set(rhs)
    if not keys:
        return 1.0
    matches = sum(1 for key in keys if lhs.get(key) == rhs.get(key))
    return matches / len(keys)


def grid_similarity(
    first: tuple[tuple[int, ...], ...] | None,
    second: tuple[tuple[int, ...], ...] | None,
) -> float:
    lhs = first or ()
    rhs = second or ()
    height = max(len(lhs), len(rhs), 1)
    width = max(len(lhs[0]) if lhs else 0, len(rhs[0]) if rhs else 0, 1)
    matches = 0
    total = height * width
    for row in range(height):
        lhs_row = lhs[row] if row < len(lhs) else ()
        rhs_row = rhs[row] if row < len(rhs) else ()
        for col in range(width):
            lhs_value = lhs_row[col] if col < len(lhs_row) else 0
            rhs_value = rhs_row[col] if col < len(rhs_row) else 0
            matches += int(lhs_value == rhs_value)
    return matches / total


def detect_source_modality(trajectory: TrajectoryRecord, workspace_state: Optional[State] = None) -> str:
    if workspace_state is not None:
        for key in ("source_modality", "modality"):
            value = workspace_state.get(key)
            if value:
                return str(value)
    if trajectory.source_modality:
        return str(trajectory.source_modality)
    for state in (trajectory.input_state, trajectory.output_state):
        for key in ("source_modality", "modality"):
            value = state.get(key)
            if value:
                return str(value)
    return "unknown"


def dataset_name_for(trajectory: TrajectoryRecord) -> str:
    metadata = trajectory.episode_metadata or {}
    modality = detect_source_modality(trajectory)
    if "gsm8k_index" in metadata or "gsm8k_split" in metadata:
        return "gsm8k"
    if "mmlu_index" in metadata or modality == "text_mcq":
        return "mmlu"
    if metadata.get("source_dataset"):
        return str(metadata["source_dataset"])
    if modality == "object_scene":
        return "arc_synthetic"
    if trajectory.trajectory_id.startswith("word_problem_translation:"):
        return "word_problem_translation"
    return trajectory.trajectory_id.split(":", 1)[0]


def answer_format_for(trajectory: TrajectoryRecord, workspace_state: Optional[State] = None) -> str:
    if workspace_state is not None:
        value = workspace_state.get("answer_format")
        if value:
            return str(value)
    for state in (trajectory.input_state, trajectory.output_state):
        value = state.get("answer_format")
        if value:
            return str(value)
    return "open"


def decision_trace_step(trajectory: TrajectoryRecord, step_index: int) -> str:
    if 0 <= step_index < len(trajectory.steps):
        return trajectory.steps[step_index].name
    return "<terminal>"


def with_subgoal_context(trajectory: TrajectoryRecord, step_index: int, base_state: State) -> State:
    trace_names = trace_names_for(trajectory)
    completed = list(trace_names[: max(0, step_index)])
    remaining = list(trace_names[max(0, step_index) :])
    state = dict(base_state)
    state.setdefault("source_modality", detect_source_modality(trajectory, base_state))
    if "source_modality" not in state and "modality" in state:
        state["source_modality"] = state["modality"]
    state.setdefault("completed_subgoals", completed)
    state.setdefault("remaining_subgoals", remaining)
    state["subgoal_status"] = {
        name: ("done" if name in completed else "pending")
        for name in trace_names
    }
    state.setdefault("focus_object_ids", [])
    state.setdefault("selected_object_ids", [])
    return state


def initial_workspace_state(trajectory: TrajectoryRecord) -> State:
    trace_names = trace_names_for(trajectory)
    input_state = trajectory.input_state
    modality = detect_source_modality(trajectory, input_state)
    state = dict(input_state)
    state.setdefault("source_modality", modality)
    state.setdefault("step_name", "<start>")
    state.setdefault("completed_subgoals", [])
    state.setdefault("remaining_subgoals", list(trace_names))
    state["subgoal_status"] = {name: "pending" for name in trace_names}
    state.setdefault("focus_object_ids", list(input_state.get("selected_object_ids", [])))
    state.setdefault("selected_object_ids", list(input_state.get("selected_object_ids", [])))
    if modality == "object_scene":
        state.setdefault("scene", input_state.get("scene", {}))
        state.setdefault("scene_summary", input_state.get("scene_summary", {}))
        state.setdefault("bindings", {})
    return state


def initial_verifier_state(trajectory: TrajectoryRecord) -> VerifierTargets:
    trace_names = trace_names_for(trajectory)
    input_state = trajectory.input_state
    output_state = trajectory.output_state
    input_scene = input_state.get("scene", {})
    output_scene = output_state.get("scene", {})
    input_grid = trajectory.input_grid
    output_grid = trajectory.output_grid
    exact_state_match = input_state == output_state
    exact_grid_match = bool(input_grid == output_grid and input_grid is not None and output_grid is not None)
    exact_scene_match = input_scene == output_scene if input_scene or output_scene else exact_state_match
    next_subgoal = trace_names[0] if trace_names else None
    if exact_state_match and not trace_names:
        non_terminal_reason = None
    elif next_subgoal is not None:
        non_terminal_reason = f"remaining_subgoal:{next_subgoal}"
    else:
        non_terminal_reason = "workspace_does_not_match_target_output"
    return {
        "exact_match": exact_state_match and not trace_names,
        "exact_grid_match": exact_grid_match,
        "exact_scene_match": exact_scene_match,
        "exact_state_match": exact_state_match,
        "should_stop": exact_state_match and not trace_names,
        "grid_match": round(grid_similarity(input_grid, output_grid), 6) if (input_grid or output_grid) else 0.0,
        "scene_match": round(flattened_match_fraction(input_scene, output_scene), 6)
        if (input_scene or output_scene)
        else round(float(exact_state_match), 6),
        "state_match": round(flattened_match_fraction(input_state, output_state), 6),
        "resolved_subgoal_count": 0,
        "unresolved_subgoal_count": len(trace_names),
        "next_subgoal": next_subgoal,
        "non_terminal_reason": non_terminal_reason,
    }


def decision_workspace_state(trajectory: TrajectoryRecord, step_index: int) -> State:
    if step_index <= 0:
        return initial_workspace_state(trajectory)
    return with_subgoal_context(trajectory, step_index, trajectory.steps[step_index - 1].workspace_state)


def verifier_targets(trajectory: TrajectoryRecord, step_index: int) -> VerifierTargets:
    if step_index <= 0:
        return initial_verifier_state(trajectory)
    return trajectory.steps[step_index - 1].verifier


def normalized_bucket(value: int | float | None, maximum: int) -> str:
    if value is None:
        return "none"
    if maximum <= 1:
        return "0"
    clipped = max(0.0, min(float(value), float(maximum)))
    bucket = int(max(0, min(3, int((clipped / max(float(maximum), 1.0)) * 4.0))))
    return str(bucket)


def lexical_focus_tokens(text: str, *, limit: int = 12) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in TEXT_TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("._")
        if not token or token in TEXT_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def extend_limited(tokens: list[str], values: Sequence[str], *, prefix: str, limit: int) -> None:
    for index, value in enumerate(values[:limit]):
        tokens.append(f"{prefix}[{index}]={normalize_token_atom(value)}")


def extend_flattened(tokens: list[str], value: Any, *, prefix: str, limit: int = 24) -> None:
    for token in flatten_tokens(value, prefix)[:limit]:
        tokens.append(token)


def object_count_for(workspace_state: State) -> int:
    scene_summary = workspace_state.get("scene_summary", {})
    if isinstance(scene_summary, dict) and scene_summary.get("object_count") is not None:
        return int(scene_summary.get("object_count", 0) or 0)
    scene = workspace_state.get("scene", {})
    if isinstance(scene, dict):
        return len(scene.get("objects", []))
    return 0


def append_subgoal_tokens(tokens: list[str], trajectory: TrajectoryRecord, workspace_state: State) -> None:
    for name in trace_names_for(trajectory):
        status = workspace_state.get("subgoal_status", {}).get(name, "pending")
        tokens.append(f"subgoal.{name}={normalize_token_atom(status)}")


def append_object_scene_tokens(
    tokens: list[str],
    trajectory: TrajectoryRecord,
    workspace_state: State,
) -> None:
    scene = workspace_state.get("scene", {})
    scene = scene if isinstance(scene, dict) else {}
    scene_summary = workspace_state.get("scene_summary", {})
    scene_summary = scene_summary if isinstance(scene_summary, dict) else {}
    scene_height = int(scene_summary.get("height", scene.get("height", 0)) or 0)
    scene_width = int(scene_summary.get("width", scene.get("width", 0)) or 0)
    tokens.extend(
        [
            f"scene.height={scene_height}",
            f"scene.width={scene_width}",
            f"scene.background={normalize_token_atom(scene.get('background_color', 0))}",
            f"scene.object_count={object_count_for(workspace_state)}",
        ]
    )

    focus_ids = set(str(item) for item in workspace_state.get("focus_object_ids", ()))
    selected_ids = set(str(item) for item in workspace_state.get("selected_object_ids", ()))
    objects = scene.get("objects", [])
    for object_index, obj in enumerate(objects):
        prefix = f"obj[{object_index}]"
        object_id = str(obj.get("object_id", ""))
        tokens.extend(
            [
                f"{prefix}.id={normalize_token_atom(object_id)}",
                f"{prefix}.shape={normalize_token_atom(obj.get('shape', '?'))}",
                f"{prefix}.color={normalize_token_atom(obj.get('color', '?'))}",
                f"{prefix}.orientation={normalize_token_atom(obj.get('orientation', '?'))}",
                f"{prefix}.mass_bucket={normalized_bucket(obj.get('mass'), 9)}",
                f"{prefix}.height_bucket={normalized_bucket(obj.get('height'), max(scene_height, 1))}",
                f"{prefix}.width_bucket={normalized_bucket(obj.get('width'), max(scene_width, 1))}",
                f"{prefix}.top_bucket={normalized_bucket(obj.get('top'), max(scene_height, 1))}",
                f"{prefix}.left_bucket={normalized_bucket(obj.get('left'), max(scene_width, 1))}",
                f"{prefix}.holes={min(int(obj.get('holes', 0) or 0), 3)}",
            ]
        )
        if obj.get("is_container"):
            tokens.append(f"{prefix}.container=1")
        if object_id in focus_ids:
            tokens.append(f"{prefix}.focus=1")
        if object_id in selected_ids:
            tokens.append(f"{prefix}.selected=1")
        for tag_index, tag in enumerate(sorted(obj.get("tags", ()))):
            tokens.append(f"{prefix}.tag[{tag_index}]={normalize_token_atom(tag)}")
        attributes = obj.get("attributes", {})
        for attr_key in ("shape_class", "size_bucket", "cue_kind", "cue_value", "target_key", "group_value", "role"):
            if attr_key in attributes:
                tokens.append(f"{prefix}.attr.{attr_key}={normalize_token_atom(attributes[attr_key])}")

    for legend_index, entry in enumerate(scene.get("legend", [])):
        tokens.append(f"legend[{legend_index}].shape={normalize_token_atom(entry.get('shape', '?'))}")
        tokens.append(f"legend[{legend_index}].color={normalize_token_atom(entry.get('color', '?'))}")

    bindings = workspace_state.get("bindings", {})
    if isinstance(bindings, dict):
        for binding_token in flatten_tokens(bindings, "binding"):
            if ".selected_object_ids" in binding_token or ".object_ids" in binding_token:
                continue
            tokens.append(binding_token)


def append_text_tokens(tokens: list[str], workspace_state: State) -> None:
    source_text = str(workspace_state.get("source_text", ""))
    extend_limited(tokens, lexical_focus_tokens(source_text), prefix="src.focus", limit=12)

    if "subject" in workspace_state:
        tokens.append(f"subject={normalize_token_atom(workspace_state['subject'])}")

    goal = workspace_state.get("goal")
    if goal:
        extend_flattened(tokens, goal, prefix="goal", limit=8)

    entities = workspace_state.get("entities", [])
    for index, entity in enumerate(entities):
        prefix = f"ent[{index}]"
        tokens.append(f"{prefix}.id={normalize_token_atom(entity.get('entity_id'))}")
        tokens.append(f"{prefix}.kind={normalize_token_atom(entity.get('kind'))}")
        label = entity.get("label")
        if label:
            tokens.append(f"{prefix}.label={normalize_token_atom(label)}")

    quantities = workspace_state.get("quantities", [])
    for index, quantity in enumerate(quantities):
        prefix = f"qty[{index}]"
        tokens.append(f"{prefix}.id={normalize_token_atom(quantity.get('quantity_id'))}")
        if "value" in quantity:
            tokens.append(f"{prefix}.value={normalize_token_atom(quantity.get('value'))}")
        if quantity.get("unit"):
            tokens.append(f"{prefix}.unit={normalize_token_atom(quantity.get('unit'))}")
        if quantity.get("role"):
            tokens.append(f"{prefix}.role={normalize_token_atom(quantity.get('role'))}")
        if quantity.get("owner_id"):
            tokens.append(f"{prefix}.owner={normalize_token_atom(quantity.get('owner_id'))}")

    relations = workspace_state.get("relations", [])
    for index, relation in enumerate(relations[:8]):
        extend_flattened(tokens, relation, prefix=f"rel[{index}]", limit=8)

    program = workspace_state.get("program")
    if program:
        extend_flattened(tokens, program, prefix="prog", limit=12)

    for key in (
        "quantity_ids",
        "derived_quantity_ids",
        "relation_types",
        "reduce_input_ids",
        "final_input_ids",
        "supported_choice_ids",
        "eliminated_choice_ids",
        "remaining_choice_ids",
        "question_focus_tokens",
        "question_numeric_fragments",
    ):
        value = workspace_state.get(key)
        if value:
            extend_flattened(tokens, value, prefix=key, limit=12)

    if workspace_state.get("choice_signatures"):
        extend_flattened(tokens, workspace_state["choice_signatures"], prefix="choice_signature", limit=16)
    if workspace_state.get("prompt_structure"):
        extend_flattened(tokens, workspace_state["prompt_structure"], prefix="prompt_structure", limit=12)

    choices = workspace_state.get("choices", [])
    for index, choice in enumerate(choices[:4]):
        choice_id = choice.get("choice_id", chr(ord("A") + index))
        prefix = f"choice[{normalize_token_atom(choice_id)}]"
        tokens.append(f"{prefix}.id={normalize_token_atom(choice_id)}")
        attributes = choice.get("attributes", {})
        if "token_count" in attributes:
            tokens.append(f"{prefix}.token_count={normalize_token_atom(attributes['token_count'])}")
        if attributes.get("numeric_fragments"):
            extend_flattened(tokens, attributes["numeric_fragments"], prefix=f"{prefix}.num", limit=6)
        keyword_tokens = attributes.get("keyword_tokens")
        if keyword_tokens:
            extend_limited(tokens, [str(item) for item in keyword_tokens], prefix=f"{prefix}.kw", limit=6)
        elif "text" in choice:
            extend_limited(
                tokens,
                lexical_focus_tokens(str(choice.get("text", "")), limit=6),
                prefix=f"{prefix}.kw",
                limit=6,
            )

    if workspace_state.get("selected_choice_id") is not None:
        tokens.append(f"selected_choice_id={normalize_token_atom(workspace_state['selected_choice_id'])}")
    if workspace_state.get("selected_choice_text"):
        extend_limited(
            tokens,
            lexical_focus_tokens(str(workspace_state["selected_choice_text"]), limit=6),
            prefix="selected_choice.kw",
            limit=6,
        )
    if workspace_state.get("answer") is not None:
        tokens.append(f"answer={normalize_token_atom(workspace_state['answer'])}")
    if workspace_state.get("reducer") is not None:
        tokens.append(f"reducer={normalize_token_atom(workspace_state['reducer'])}")
    if workspace_state.get("program_op") is not None:
        tokens.append(f"program_op={normalize_token_atom(workspace_state['program_op'])}")
    if workspace_state.get("family_name") is not None:
        tokens.append(f"family_name={normalize_token_atom(workspace_state['family_name'])}")
    if workspace_state.get("ir_keys"):
        extend_flattened(tokens, workspace_state["ir_keys"], prefix="ir_key", limit=12)


def workspace_tokens(
    trajectory: TrajectoryRecord,
    *,
    step_index: int,
    workspace_state: State,
) -> list[str]:
    modality = detect_source_modality(trajectory, workspace_state)
    tokens = [f"src.kind={normalize_token_atom(modality)}"]
    append_subgoal_tokens(tokens, trajectory, workspace_state)
    if modality == "object_scene":
        append_object_scene_tokens(tokens, trajectory, workspace_state)
    else:
        append_text_tokens(tokens, workspace_state)
    return tokens


def workspace_scalar_features(
    trajectory: TrajectoryRecord,
    *,
    step_index: int,
    workspace_state: State,
) -> tuple[float, ...]:
    trace_names = trace_names_for(trajectory)
    scene_summary = workspace_state.get("scene_summary", {})
    scene_summary = scene_summary if isinstance(scene_summary, dict) else {}
    scene = workspace_state.get("scene", {})
    scene = scene if isinstance(scene, dict) else {}
    scene_height = float(scene_summary.get("height", scene.get("height", 0)) or 0)
    scene_width = float(scene_summary.get("width", scene.get("width", 0)) or 0)
    object_count = float(object_count_for(workspace_state))
    entity_count = float(len(workspace_state.get("entities", ())))
    quantity_count = float(len(workspace_state.get("quantities", ())))
    relation_count = float(len(workspace_state.get("relations", ())))
    choice_count = float(len(workspace_state.get("choices", ())))
    source_text = str(workspace_state.get("source_text", ""))
    source_token_count = float(len(TEXT_TOKEN_RE.findall(source_text)))
    selected_count = float(len(workspace_state.get("selected_object_ids", ())))
    focus_count = float(len(workspace_state.get("focus_object_ids", ())))
    resolved = float(len(workspace_state.get("completed_subgoals", ())))
    unresolved = float(len(workspace_state.get("remaining_subgoals", ())))
    return (
        float(step_index) / max(len(trajectory.steps), 1),
        float(len(trace_names)) / 8.0,
        float(trajectory.difficulty) / 5.0,
        min(source_token_count, 64.0) / 64.0,
        min(scene_height, 30.0) / 30.0,
        min(scene_width, 30.0) / 30.0,
        min(object_count, 16.0) / 16.0,
        min(entity_count, 16.0) / 16.0,
        min(quantity_count, 16.0) / 16.0,
        min(relation_count, 16.0) / 16.0,
        min(choice_count, 4.0) / 4.0,
        min(selected_count, 8.0) / 8.0,
        min(focus_count, 8.0) / 8.0,
        resolved / max(len(trace_names), 1),
        unresolved / max(len(trace_names), 1),
    )


def verifier_token_list(verifier_state: VerifierTargets) -> list[str]:
    tokens: list[str] = []
    for key in ("next_subgoal", "non_terminal_reason"):
        value = verifier_state.get(key)
        if value is not None:
            tokens.append(f"verifier.{key}={normalize_token_atom(value)}")
    for key in ("exact_grid_match", "exact_scene_match", "exact_state_match", "should_stop"):
        if verifier_state.get(key):
            tokens.append(f"verifier.{key}=1")
    return tokens


def verifier_scalar_features(trajectory: TrajectoryRecord, *, verifier_state: VerifierTargets) -> tuple[float, ...]:
    trace_names = trace_names_for(trajectory)
    return (
        float(verifier_state.get("grid_match", 0.0)),
        float(verifier_state.get("scene_match", 0.0)),
        float(verifier_state.get("state_match", 0.0)),
        float(verifier_state.get("resolved_subgoal_count", 0)) / max(len(trace_names), 1),
        float(verifier_state.get("unresolved_subgoal_count", 0)) / max(len(trace_names), 1),
        float(bool(verifier_state.get("exact_grid_match"))),
        float(bool(verifier_state.get("exact_state_match"))),
        float(bool(verifier_state.get("should_stop"))),
    )


def encode_workspace(
    trajectory: TrajectoryRecord,
    step_index: int,
    *,
    include_verifier: bool = False,
    verifier_state: Optional[VerifierTargets] = None,
) -> EncodedWorkspace:
    workspace_state = decision_workspace_state(trajectory, step_index)
    trace_names = trace_names_for(trajectory)
    tokens = workspace_tokens(trajectory, step_index=step_index, workspace_state=workspace_state)
    scalar_features = list(
        workspace_scalar_features(
            trajectory,
            step_index=step_index,
            workspace_state=workspace_state,
        )
    )
    if include_verifier:
        resolved_verifier = verifier_state if verifier_state is not None else verifier_targets(trajectory, step_index)
        tokens.extend(verifier_token_list(resolved_verifier))
        scalar_features.extend(verifier_scalar_features(trajectory, verifier_state=resolved_verifier))
    return EncodedWorkspace(
        tokens=tuple(tokens),
        scalar_features=tuple(float(value) for value in scalar_features),
        metadata={
            "dataset": dataset_name_for(trajectory),
            "trajectory_id": trajectory.trajectory_id,
            "family": trajectory.family,
            "difficulty": trajectory.difficulty,
            "step_index": step_index,
            "trace_length": len(trace_names),
            "trace_step": decision_trace_step(trajectory, step_index),
            "source_modality": detect_source_modality(trajectory, workspace_state),
            "answer_format": answer_format_for(trajectory, workspace_state),
            "variant_kind": trajectory.variant_kind,
            "trajectory_role": trajectory.trajectory_role,
            "record_type": "decision_action",
        },
    )


def previous_action_name(previous_action: str | None) -> str:
    if previous_action in (None, "", "<start>"):
        return "<start>"
    try:
        payload = json.loads(previous_action)
    except json.JSONDecodeError:
        return str(previous_action)
    if isinstance(payload, dict) and payload.get("name"):
        return str(payload["name"])
    return str(previous_action)


def candidate_bucket_for(encoded_workspace: EncodedWorkspace, previous_action: str | None) -> str:
    return (
        f"family:{encoded_workspace.metadata['family']}"
        f"|step:{encoded_workspace.metadata['step_index']}"
        f"|prev:{previous_action_name(previous_action)}"
    )


def serialize_reasoning_state_text(
    encoded_workspace: EncodedWorkspace,
    *,
    previous_action: str | None = None,
    target_action: str | None = None,
    target_stop: bool | None = None,
    target_answer: str | int | float | None = None,
    verifier_state: Optional[VerifierTargets] = None,
    record_type: str | None = None,
) -> str:
    target_count = sum(
        value is not None for value in (target_action, target_stop, target_answer)
    )
    if record_type is None:
        if target_stop is not None and target_count == 1:
            record_type = "decision_stop"
        elif target_answer is not None and target_count == 1:
            record_type = "terminal_answer"
        else:
            record_type = "decision_action"

    resolved_previous_action = "<start>" if previous_action is None else previous_action
    lines = [
        FORMAT_HEADER,
        f"record_type={record_type}",
        f"dataset={encoded_workspace.metadata['dataset']}",
        f"source_modality={encoded_workspace.metadata['source_modality']}",
        f"family={encoded_workspace.metadata['family']}",
        f"difficulty={encoded_workspace.metadata['difficulty']}",
        f"trajectory_id={encoded_workspace.metadata['trajectory_id']}",
        f"step_index={encoded_workspace.metadata['step_index']}",
        f"trace_length={encoded_workspace.metadata['trace_length']}",
        f"trace_step={encoded_workspace.metadata['trace_step']}",
        f"previous_action={resolved_previous_action}",
        f"candidate_bucket={candidate_bucket_for(encoded_workspace, resolved_previous_action)}",
        f"answer_format={encoded_workspace.metadata['answer_format']}",
        f"variant_kind={encoded_workspace.metadata['variant_kind']}",
        f"trajectory_role={encoded_workspace.metadata['trajectory_role']}",
        "state_tokens=" + " ".join(encoded_workspace.tokens),
        "state_scalars=" + ",".join(
            (f"{value:.6f}".rstrip("0").rstrip(".") or "0")
            for value in encoded_workspace.scalar_features
        ),
    ]
    if verifier_state is not None:
        verifier_items = sorted(verifier_state.items())
        lines.append("verifier_context=" + json.dumps(dict(verifier_items), separators=(",", ":"), sort_keys=True))
    if target_action is not None:
        lines.append(f"target_action={target_action}")
    if target_stop is not None:
        lines.append(f"target_stop={'true' if target_stop else 'false'}")
    if target_answer is not None:
        lines.append(f"target_answer={normalize_token_atom(target_answer)}")
    return "\n".join(lines) + "\n"


def serialize_workspace_text(
    encoded_workspace: EncodedWorkspace,
    *,
    previous_action: str | None = None,
    target_action: str | None = None,
    target_stop: bool | None = None,
    target_answer: str | int | float | None = None,
    verifier_state: Optional[VerifierTargets] = None,
    record_type: str | None = None,
) -> str:
    return serialize_reasoning_state_text(
        encoded_workspace,
        previous_action=previous_action,
        target_action=target_action,
        target_stop=target_stop,
        target_answer=target_answer,
        verifier_state=verifier_state,
        record_type=record_type,
    )
