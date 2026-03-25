from __future__ import annotations

"""Stage 1 starter for an ARC-style synthetic task generator.

This module only does one thing: sample a latent task family plus latent rule.
It does not generate concrete input/output grids. The intended workflow is:

    stage 1: sample_latent_rule()     -> choose the hidden rule/program
    stage 2: sample_episode(latent)   -> create train/test inputs that expose it
    stage 3: execute(latent, input)   -> exact output grid
    stage 4: trace(latent, input)     -> subgoals / actions / verifier targets

The design is intentionally narrow and typed. A broad unrestricted CFG tends to
produce too many invalid or semantically redundant programs. This starter uses a
small library of high-value ARC-like families and slot-samples within them.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import random
from typing import Any, Dict, Optional, Sequence, Tuple


class Family(str, Enum):
    UNARY_OBJECT = "unary_object"
    RELATIONAL = "relational"
    COUNT_SELECT = "count_select"
    CONTEXTUAL = "contextual"
    SYMBOL_MAP = "symbol_map"


class Transform(str, Enum):
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    REFLECT_H = "reflect_h"
    REFLECT_V = "reflect_v"
    TRANSLATE = "translate"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RECOLOR = "recolor"
    CROP_TO_BBOX = "crop_to_bbox"
    FILL_HOLES = "fill_holes"


class RelationAction(str, Enum):
    MOVE_UNTIL_CONTACT = "move_until_contact"
    ALIGN_TO = "align_to"
    STACK_NEXT_TO = "stack_next_to"
    DRAW_CONNECTING_LINE = "draw_connecting_line"
    PLACE_IN_CONTAINER = "place_in_container"


class SelectorKind(str, Enum):
    ALL_OBJECTS = "all_objects"
    BY_COLOR = "by_color"
    BY_SHAPE = "by_shape"
    BY_SIZE = "by_size"
    ARGMAX_SIZE = "argmax_size"
    ARGMIN_SIZE = "argmin_size"
    BY_HOLES = "by_holes"
    ARGMAX_COUNT = "argmax_count"
    ARGMIN_COUNT = "argmin_count"
    INSIDE_FRAME = "inside_frame"
    OUTSIDE_FRAME = "outside_frame"
    BY_BORDER_COLOR = "by_border_color"
    BY_OUTLINE_COLOR = "by_outline_color"


class CueKind(str, Enum):
    OUTLINE_COLOR = "outline_color"
    BORDER_COLOR = "border_color"
    MARKER_POSITION = "marker_position"
    PARITY_OF_COUNT = "parity_of_count"
    HOLE_COUNT = "hole_count"


class ShapeClass(str, Enum):
    ANY = "any"
    RECT = "rect"
    LINE = "line"
    DOT = "dot"
    L_SHAPE = "l_shape"
    CROSS = "cross"
    REGULAR = "regular"


@dataclass(frozen=True)
class RoleVar:
    """A symbolic role, not a literal integer/color/shape id."""

    name: str
    domain: str


@dataclass(frozen=True)
class Selector:
    kind: SelectorKind
    args: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class Action:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Program:
    op: str
    args: Tuple[Any, ...]


@dataclass(frozen=True)
class TraceStep:
    name: str
    description: str


@dataclass(frozen=True)
class InputSchema:
    """Nuisance variables and structural preconditions for stage 2."""

    grid_h_range: Tuple[int, int]
    grid_w_range: Tuple[int, int]
    num_objects_range: Tuple[int, int]
    shape_bias: Tuple[str, ...]
    max_distinct_colors: int
    extra_requirements: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskLatent:
    family: Family
    concept_tags: Tuple[str, ...]
    input_schema: InputSchema
    program: Program
    trace_template: Tuple[TraceStep, ...]
    constraints: Tuple[str, ...]
    difficulty: int

    def to_jsonable(self) -> Dict[str, Any]:
        def encode(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, RoleVar):
                return asdict(obj)
            if isinstance(obj, tuple):
                return [encode(x) for x in obj]
            if isinstance(obj, list):
                return [encode(x) for x in obj]
            if isinstance(obj, dict):
                return {k: encode(v) for k, v in obj.items()}
            if hasattr(obj, "__dataclass_fields__"):
                return {k: encode(v) for k, v in asdict(obj).items()}
            return obj

        return encode(self)


COLOR_ROLES = [
    RoleVar("C_src", "color"),
    RoleVar("C_dst", "color"),
    RoleVar("C_frame", "color"),
    RoleVar("C_outline", "color"),
    RoleVar("C_marker", "color"),
]

SHAPE_ROLES = [
    RoleVar("S_target", "shape"),
    RoleVar("S_container", "shape"),
    RoleVar("S_group", "shape"),
]

COUNT_ROLES = [
    RoleVar("N_repeat", "small_int"),
    RoleVar("N_holes", "small_int"),
]


def pick(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


def maybe(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def weighted_choice(rng: random.Random, items: Sequence[Tuple[float, Any]]) -> Any:
    total = sum(weight for weight, _ in items)
    x = rng.random() * total
    acc = 0.0
    for weight, item in items:
        acc += weight
        if x <= acc:
            return item
    return items[-1][1]


def make_unary_object_family(rng: random.Random) -> TaskLatent:
    transform = weighted_choice(
        rng,
        [
            (2.0, Transform.ROTATE_90),
            (1.5, Transform.ROTATE_180),
            (1.5, Transform.REFLECT_H),
            (1.5, Transform.REFLECT_V),
            (1.2, Transform.TRANSLATE),
            (1.0, Transform.SCALE_UP),
            (0.8, Transform.SCALE_DOWN),
            (1.0, Transform.RECOLOR),
            (1.0, Transform.CROP_TO_BBOX),
            (0.8, Transform.FILL_HOLES),
        ],
    )

    selector = weighted_choice(
        rng,
        [
            (2.0, Selector(SelectorKind.BY_COLOR, (pick(rng, COLOR_ROLES),))),
            (1.6, Selector(SelectorKind.BY_SHAPE, (pick(rng, SHAPE_ROLES),))),
            (1.2, Selector(SelectorKind.ARGMAX_SIZE, ())),
            (1.0, Selector(SelectorKind.ARGMIN_SIZE, ())),
            (0.8, Selector(SelectorKind.BY_HOLES, (pick(rng, COUNT_ROLES),))),
            (0.6, Selector(SelectorKind.INSIDE_FRAME, ())),
        ],
    )

    params: Dict[str, Any] = {}
    tags = ["objectness", "geometry"]
    trace = [
        TraceStep("segment", "Parse the grid into objects or zones."),
        TraceStep("select", "Select the target object set."),
        TraceStep("transform", f"Apply {transform.value} exactly."),
        TraceStep("render", "Write transformed object(s) to the output canvas."),
    ]
    constraints = ["selected_set_nonempty", "transform_preserves_grid_bounds_or_stage2_rejects"]

    if transform == Transform.TRANSLATE:
        params["direction"] = pick(rng, ["up", "down", "left", "right", "diag_up_right"])
        params["distance"] = RoleVar("N_repeat", "small_int") if maybe(rng, 0.5) else pick(rng, [1, 2, 3])
    elif transform in (Transform.SCALE_UP, Transform.SCALE_DOWN):
        params["factor"] = pick(rng, [2, 3])
    elif transform == Transform.RECOLOR:
        params["to_color"] = pick(rng, COLOR_ROLES)
        tags.append("symbolic_color")
    elif transform == Transform.FILL_HOLES:
        tags.append("topology")

    schema = InputSchema(
        grid_h_range=(5, 15),
        grid_w_range=(5, 15),
        num_objects_range=(1, 5),
        shape_bias=(ShapeClass.RECT.value, ShapeClass.LINE.value, ShapeClass.REGULAR.value),
        max_distinct_colors=5,
    )

    program = Program(
        op="MapEach",
        args=(selector, Action(transform.value, params)),
    )

    difficulty = 1 + (1 if transform in (Transform.SCALE_UP, Transform.FILL_HOLES) else 0)
    return TaskLatent(
        family=Family.UNARY_OBJECT,
        concept_tags=tuple(tags),
        input_schema=schema,
        program=program,
        trace_template=tuple(trace),
        constraints=tuple(constraints),
        difficulty=difficulty,
    )


def make_relational_family(rng: random.Random) -> TaskLatent:
    action = weighted_choice(
        rng,
        [
            (2.0, RelationAction.MOVE_UNTIL_CONTACT),
            (1.2, RelationAction.ALIGN_TO),
            (1.1, RelationAction.STACK_NEXT_TO),
            (1.0, RelationAction.DRAW_CONNECTING_LINE),
            (0.9, RelationAction.PLACE_IN_CONTAINER),
        ],
    )
    src_selector = weighted_choice(
        rng,
        [
            (1.8, Selector(SelectorKind.BY_COLOR, (RoleVar("C_src", "color"),))),
            (1.5, Selector(SelectorKind.BY_SHAPE, (RoleVar("S_target", "shape"),))),
            (1.0, Selector(SelectorKind.ARGMAX_SIZE, ())),
        ],
    )
    dst_selector = weighted_choice(
        rng,
        [
            (1.8, Selector(SelectorKind.BY_COLOR, (RoleVar("C_dst", "color"),))),
            (1.2, Selector(SelectorKind.BY_SHAPE, (RoleVar("S_container", "shape"),))),
            (0.8, Selector(SelectorKind.OUTSIDE_FRAME, ())),
        ],
    )
    params: Dict[str, Any] = {}
    tags = ["objectness", "contact", "geometry"]
    trace = [
        TraceStep("segment", "Parse the grid into objects and their relations."),
        TraceStep("pick_source", "Choose the movable/source objects."),
        TraceStep("pick_target", "Choose the reference or target objects."),
        TraceStep("relate", f"Apply {action.value} under exact geometry rules."),
        TraceStep("render", "Render the relational result."),
    ]
    constraints = [
        "source_and_target_exist",
        "source_and_target_distinct",
        "stage2_must_sample_nontrivial_spatial_relation",
    ]

    if action == RelationAction.MOVE_UNTIL_CONTACT:
        params["direction"] = pick(rng, ["up", "down", "left", "right", "towards_target_centroid"])
    elif action == RelationAction.ALIGN_TO:
        params["axis"] = pick(rng, ["x", "y", "both"])
    elif action == RelationAction.STACK_NEXT_TO:
        params["side"] = pick(rng, ["left", "right", "top", "bottom"])
    elif action == RelationAction.DRAW_CONNECTING_LINE:
        params["line_type"] = pick(rng, ["orthogonal", "diagonal_if_possible"])
        tags.append("line_drawing")
    elif action == RelationAction.PLACE_IN_CONTAINER:
        params["fit_mode"] = pick(rng, ["center", "same_shape_slot"])
        tags.append("containment")

    schema = InputSchema(
        grid_h_range=(6, 18),
        grid_w_range=(6, 18),
        num_objects_range=(2, 6),
        shape_bias=(ShapeClass.RECT.value, ShapeClass.LINE.value, ShapeClass.REGULAR.value),
        max_distinct_colors=6,
        extra_requirements=("at_least_two_salient_objects",),
    )
    program = Program(
        op="Relate",
        args=(src_selector, dst_selector, Action(action.value, params)),
    )
    difficulty = 2 if action in (RelationAction.MOVE_UNTIL_CONTACT, RelationAction.ALIGN_TO) else 3
    return TaskLatent(
        family=Family.RELATIONAL,
        concept_tags=tuple(tags),
        input_schema=schema,
        program=program,
        trace_template=tuple(trace),
        constraints=tuple(constraints),
        difficulty=difficulty,
    )


def make_count_select_family(rng: random.Random) -> TaskLatent:
    group_by = weighted_choice(
        rng,
        [
            (1.8, "shape"),
            (1.5, "color"),
            (1.0, "size"),
            (0.8, "hole_count"),
        ],
    )
    reducer = weighted_choice(
        rng,
        [
            (1.7, SelectorKind.ARGMAX_COUNT),
            (1.4, SelectorKind.ARGMIN_COUNT),
            (1.0, SelectorKind.ARGMAX_SIZE),
            (0.9, SelectorKind.ARGMIN_SIZE),
        ],
    )
    post = weighted_choice(
        rng,
        [
            (1.8, Action("copy_selected", {})),
            (1.4, Action("recolor_selected", {"to_color": RoleVar("C_dst", "color")})),
            (1.2, Action("sort_into_row", {"order": pick(rng, ["ascending", "descending"])})),
            (1.0, Action("repeat_selected", {"n": RoleVar("N_repeat", "small_int")})),
        ],
    )

    tags = ["numbers", "selection", "comparison"]
    trace = [
        TraceStep("segment", "Parse the grid into comparable units."),
        TraceStep("group", f"Group objects by {group_by}."),
        TraceStep("reduce", f"Choose the {reducer.value} group or object."),
        TraceStep("act", f"Apply {post.name} to the winner."),
    ]
    constraints = [
        "at_least_two_comparable_groups",
        "winner_unique_or_stage2_rejects",
        "small_numbers_only",
    ]
    schema = InputSchema(
        grid_h_range=(5, 16),
        grid_w_range=(5, 16),
        num_objects_range=(3, 8),
        shape_bias=(ShapeClass.RECT.value, ShapeClass.DOT.value, ShapeClass.LINE.value),
        max_distinct_colors=7,
    )
    program = Program(
        op="ReduceThenAct",
        args=(group_by, Selector(reducer), post),
    )
    difficulty = 2 + (1 if post.name in ("sort_into_row", "repeat_selected") else 0)
    return TaskLatent(
        family=Family.COUNT_SELECT,
        concept_tags=tuple(tags),
        input_schema=schema,
        program=program,
        trace_template=tuple(trace),
        constraints=tuple(constraints),
        difficulty=difficulty,
    )


def make_contextual_family(rng: random.Random) -> TaskLatent:
    cue = weighted_choice(
        rng,
        [
            (1.8, CueKind.OUTLINE_COLOR),
            (1.4, CueKind.BORDER_COLOR),
            (1.2, CueKind.MARKER_POSITION),
            (0.9, CueKind.PARITY_OF_COUNT),
            (0.8, CueKind.HOLE_COUNT),
        ],
    )
    then_action = pick(
        rng,
        [
            Action("stack_left", {}),
            Action("stack_right", {}),
            Action("reflect_h", {}),
            Action("reflect_v", {}),
            Action("recolor", {"to_color": RoleVar("C_dst", "color")}),
        ],
    )
    else_action = pick(
        rng,
        [
            action
            for action in [
                Action("stack_left", {}),
                Action("stack_right", {}),
                Action("reflect_h", {}),
                Action("reflect_v", {}),
                Action("recolor", {"to_color": RoleVar("C_src", "color")}),
            ]
            if action != then_action
        ],
    )

    selector = weighted_choice(
        rng,
        [
            (1.8, Selector(SelectorKind.ALL_OBJECTS, ())),
            (1.4, Selector(SelectorKind.BY_SHAPE, (RoleVar("S_group", "shape"),))),
            (1.0, Selector(SelectorKind.BY_COLOR, (RoleVar("C_src", "color"),))),
        ],
    )
    tags = ["context", "control_flow", "compositionality"]
    trace = [
        TraceStep("segment", "Parse objects and cue-bearing context."),
        TraceStep("read_cue", f"Read contextual cue {cue.value}."),
        TraceStep("branch", "Choose the guarded branch."),
        TraceStep("apply", "Apply the branch-specific action exactly."),
    ]
    constraints = [
        "cue_visible_in_train_pairs",
        "both_branches_realizable_across_episode",
        "no_spurious_shortcuts_like_constant_side",
    ]
    schema = InputSchema(
        grid_h_range=(6, 18),
        grid_w_range=(6, 18),
        num_objects_range=(2, 7),
        shape_bias=(ShapeClass.RECT.value, ShapeClass.LINE.value, ShapeClass.REGULAR.value),
        max_distinct_colors=7,
        extra_requirements=("explicit_context_object_or_border",),
    )
    program = Program(
        op="IfThenElse",
        args=(cue, selector, then_action, else_action),
    )
    difficulty = 3
    return TaskLatent(
        family=Family.CONTEXTUAL,
        concept_tags=tuple(tags),
        input_schema=schema,
        program=program,
        trace_template=tuple(trace),
        constraints=tuple(constraints),
        difficulty=difficulty,
    )


def make_symbol_map_family(rng: random.Random) -> TaskLatent:
    key = weighted_choice(
        rng,
        [
            (1.8, "hole_count"),
            (1.4, "shape_class"),
            (0.9, "size_bucket"),
        ],
    )
    value = weighted_choice(
        rng,
        [
            (2.0, "color"),
            (1.0, "orientation"),
            (0.8, "placement_side"),
        ],
    )
    if value == "color":
        apply_mode = Action("recolor_targets", {})
    elif value == "orientation":
        apply_mode = Action("rotate_targets", {})
    else:
        apply_mode = Action("place_targets", {"layout": pick(rng, ["row", "column"])})

    tags = ["in_context_symbol_definition", "binding", "compositionality"]
    trace = [
        TraceStep("segment", "Parse legend objects and target objects."),
        TraceStep("bind", f"Construct a dictionary from {key} to {value}."),
        TraceStep("match", "Match each target to its legend key."),
        TraceStep("apply", f"Apply {apply_mode.name} using the bound value."),
    ]
    constraints = [
        "legend_complete_for_targets",
        "legend_and_targets_use_disjoint_visual_roles_when_possible",
        "binding_rule_consistent_across_pairs",
    ]
    schema = InputSchema(
        grid_h_range=(7, 20),
        grid_w_range=(7, 20),
        num_objects_range=(4, 9),
        shape_bias=(ShapeClass.RECT.value, ShapeClass.REGULAR.value),
        max_distinct_colors=8,
        extra_requirements=("legend_region_present", "targets_present"),
    )
    program = Program(
        op="BindMapThenApply",
        args=(key, value, apply_mode),
    )
    difficulty = 4
    return TaskLatent(
        family=Family.SYMBOL_MAP,
        concept_tags=tuple(tags),
        input_schema=schema,
        program=program,
        trace_template=tuple(trace),
        constraints=tuple(constraints),
        difficulty=difficulty,
    )


FAMILY_BUILDERS = [
    (3.0, make_unary_object_family),
    (2.2, make_relational_family),
    (1.8, make_count_select_family),
    (1.1, make_contextual_family),
    (0.8, make_symbol_map_family),
]


def validate_latent(latent: TaskLatent) -> None:
    """Cheap structural validation only.

    Stage 2 should do semantic validation by trying to sample episodes and reject
    unsatisfiable or shortcut-prone latents.
    """

    if latent.input_schema.grid_h_range[0] < 1 or latent.input_schema.grid_w_range[0] < 1:
        raise ValueError("grid ranges must be positive")
    if latent.input_schema.num_objects_range[0] < 1:
        raise ValueError("must allow at least one object")
    if latent.difficulty < 1:
        raise ValueError("difficulty must be positive")
    if not latent.trace_template:
        raise ValueError("trace template required")
    if not latent.constraints:
        raise ValueError("constraints required")


def sample_latent_rule(seed: Optional[int] = None) -> TaskLatent:
    rng = random.Random(seed)
    builder = weighted_choice(rng, FAMILY_BUILDERS)
    latent = builder(rng)
    validate_latent(latent)
    return latent


if __name__ == "__main__":
    for seed in range(5):
        latent = sample_latent_rule(seed)
        print(json.dumps(latent.to_jsonable(), indent=2))
        print("-" * 80)
