from __future__ import annotations

"""Synthetic word-problem translation dataset compiled into abstract reasoning IR."""

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Sequence, Tuple

try:
    from .reasoning_ir import AbstractReasoningTask, EntitySpec, GoalSpec, QuantitySpec
    from .stage1_latent_sampler import Program, TraceStep
    from .stage4_trajectory_dataset import (
        STEP_WEIGHTS,
        State,
        TrajectoryRecord,
        TrajectoryStep,
        reward_from_terms,
        symbolic_step_reward_terms,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution
    from reasoning_ir import AbstractReasoningTask, EntitySpec, GoalSpec, QuantitySpec  # type: ignore
    from stage1_latent_sampler import Program, TraceStep  # type: ignore
    from stage4_trajectory_dataset import (  # type: ignore
        STEP_WEIGHTS,
        State,
        TrajectoryRecord,
        TrajectoryStep,
        reward_from_terms,
        symbolic_step_reward_terms,
        write_jsonl,
    )


PEOPLE = ("Alice", "Ben", "Carla", "Diego", "Emma", "Farah", "Gabe", "Hana")
PERSON_PRONOUNS = {
    "Alice": ("she", "her", "her"),
    "Ben": ("he", "him", "his"),
    "Carla": ("she", "her", "her"),
    "Diego": ("he", "him", "his"),
    "Emma": ("she", "her", "her"),
    "Farah": ("she", "her", "her"),
    "Gabe": ("he", "him", "his"),
    "Hana": ("she", "her", "her"),
}
ITEMS = (
    ("apple", "apples"),
    ("book", "books"),
    ("marble", "marbles"),
    ("sticker", "stickers"),
    ("shell", "shells"),
)
CONTAINERS = (("bag", "bags"), ("box", "boxes"), ("crate", "crates"))
PURCHASE_ITEMS = ("wallet", "backpack", "jacket", "bike")

WORD_PROBLEM_STEP_WEIGHTS = {
    "extract_entities": 0.12,
    "bind_quantities": 0.18,
    "choose_operation": 0.18,
    "compute_answer": 0.26,
    "emit_ir": 0.22,
}


@dataclass(frozen=True)
class WordProblemExample:
    example_id: str
    source_text: str
    abstract_task: AbstractReasoningTask
    family_name: str
    template_name: str
    notes: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


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
        return {key: _encode(value) for key, value in asdict(obj).items()}
    return obj


def slug(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def pick(rng: random.Random, values: Sequence[Any]) -> Any:
    return values[rng.randrange(len(values))]


def pronouns(person: str) -> Tuple[str, str, str]:
    return PERSON_PRONOUNS[person]


def weighted_choice(rng: random.Random, items: Sequence[Tuple[float, Any]]) -> Any:
    total = sum(weight for weight, _ in items)
    target = rng.random() * total
    running = 0.0
    for weight, item in items:
        running += weight
        if target <= running:
            return item
    return items[-1][1]


def translation_trace() -> Tuple[TraceStep, ...]:
    return (
        TraceStep("extract_entities", "Identify entities, units, and the question target."),
        TraceStep("bind_quantities", "Bind surface numbers to typed quantity roles."),
        TraceStep("choose_operation", "Infer the abstract operator from the text pattern."),
        TraceStep("compute_answer", "Compute the numeric result in the abstract state."),
        TraceStep("emit_ir", "Emit the full typed reasoning IR."),
    )


def compose_trace(reducer: str) -> Tuple[TraceStep, ...]:
    reduce_description = "Sum the derived terms into one result." if reducer == "sum" else "Subtract to answer the final query."
    return (
        TraceStep("segment", "Parse the text into primitive numeric facts."),
        TraceStep("bind", "Bind named quantities and derivation rules."),
        TraceStep("apply", "Derive the hidden intermediate quantities locally."),
        TraceStep("reduce", reduce_description),
        TraceStep("render", "Emit the final answer value."),
    )


def rate_scale_trace() -> Tuple[TraceStep, ...]:
    return (
        TraceStep("segment", "Parse base quantity, rate, and repetition extent."),
        TraceStep("bind", "Bind the base quantity and rate relation."),
        TraceStep("apply", "Apply the rate or repetition transform."),
        TraceStep("render", "Emit the scaled result."),
    )


def partition_inverse_trace() -> Tuple[TraceStep, ...]:
    return (
        TraceStep("segment", "Parse whole quantity and partition rule."),
        TraceStep("bind", "Bind the group size or number of shares."),
        TraceStep("reduce", "Divide to recover the missing part or group count."),
        TraceStep("render", "Emit the final answer value."),
    )


def make_add_change_problem(rng: random.Random, example_id: str) -> WordProblemExample:
    person = pick(rng, PEOPLE)
    item_singular, item_plural = pick(rng, ITEMS)
    start = rng.randint(2, 12)
    delta = rng.randint(1, 9)
    person_id = slug(person)
    item_id = slug(item_plural)
    source_text = (
        f"{person} has {start} {item_plural} and gets {delta} more. "
        f"How many {item_plural} does {person} have now?"
    )
    task = AbstractReasoningTask(
        task_id=f"{example_id}_task",
        source_modality="text",
        source_text=source_text,
        entities=(
            EntitySpec(entity_id=person_id, label=person, kind="person"),
            EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
        ),
        quantities=(
            QuantitySpec(quantity_id="q_start", value=start, unit=item_plural, owner_id=person_id, role="start"),
            QuantitySpec(quantity_id="q_delta", value=delta, unit=item_plural, owner_id=person_id, role="increase"),
        ),
        relations=(
            {"type": "ownership", "owner_id": person_id, "item_id": item_id},
            {"type": "change", "effect": "increase", "quantity_id": "q_delta"},
        ),
        goal=GoalSpec(
            target_id="answer",
            query=f"current_{item_plural}_for_{person_id}",
            unit=item_plural,
        ),
        program=Program(op="add", args=("q_start", "q_delta")),
        trace_template=translation_trace(),
        answer=start + delta,
        concept_tags=("word_problem", "translation", "arithmetic", "addition"),
        difficulty=1,
        metadata={"template": "add_change", "answer_unit": item_plural},
    )
    return WordProblemExample(
        example_id=example_id,
        source_text=source_text,
        abstract_task=task,
        family_name="add_change",
        template_name="add_change",
        notes=("change event adds to an existing quantity",),
        metadata={"operation": "add", "answer_unit": item_plural},
    )


def make_subtract_change_problem(rng: random.Random, example_id: str) -> WordProblemExample:
    person = pick(rng, PEOPLE)
    item_singular, item_plural = pick(rng, ITEMS)
    start = rng.randint(4, 14)
    delta = rng.randint(1, start - 1)
    person_id = slug(person)
    item_id = slug(item_plural)
    source_text = (
        f"{person} had {start} {item_plural} and gave away {delta}. "
        f"How many {item_plural} are left?"
    )
    task = AbstractReasoningTask(
        task_id=f"{example_id}_task",
        source_modality="text",
        source_text=source_text,
        entities=(
            EntitySpec(entity_id=person_id, label=person, kind="person"),
            EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
        ),
        quantities=(
            QuantitySpec(quantity_id="q_start", value=start, unit=item_plural, owner_id=person_id, role="start"),
            QuantitySpec(quantity_id="q_delta", value=delta, unit=item_plural, owner_id=person_id, role="decrease"),
        ),
        relations=(
            {"type": "ownership", "owner_id": person_id, "item_id": item_id},
            {"type": "change", "effect": "decrease", "quantity_id": "q_delta"},
        ),
        goal=GoalSpec(
            target_id="answer",
            query=f"remaining_{item_plural}_for_{person_id}",
            unit=item_plural,
        ),
        program=Program(op="subtract", args=("q_start", "q_delta")),
        trace_template=translation_trace(),
        answer=start - delta,
        concept_tags=("word_problem", "translation", "arithmetic", "subtraction"),
        difficulty=1,
        metadata={"template": "subtract_change", "answer_unit": item_plural},
    )
    return WordProblemExample(
        example_id=example_id,
        source_text=source_text,
        abstract_task=task,
        family_name="subtract_change",
        template_name="subtract_change",
        notes=("change event subtracts from an existing quantity",),
        metadata={"operation": "subtract", "answer_unit": item_plural},
    )


def make_compare_problem(rng: random.Random, example_id: str) -> WordProblemExample:
    first_person = pick(rng, PEOPLE)
    second_person = pick(rng, [name for name in PEOPLE if name != first_person])
    item_singular, item_plural = pick(rng, ITEMS)
    bigger = rng.randint(6, 15)
    smaller = rng.randint(1, bigger - 1)
    first_id = slug(first_person)
    second_id = slug(second_person)
    item_id = slug(item_plural)
    source_text = (
        f"{first_person} has {bigger} {item_plural}. "
        f"{second_person} has {smaller} {item_plural}. "
        f"How many more {item_plural} does {first_person} have than {second_person}?"
    )
    task = AbstractReasoningTask(
        task_id=f"{example_id}_task",
        source_modality="text",
        source_text=source_text,
        entities=(
            EntitySpec(entity_id=first_id, label=first_person, kind="person"),
            EntitySpec(entity_id=second_id, label=second_person, kind="person"),
            EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
        ),
        quantities=(
            QuantitySpec(quantity_id="q_first", value=bigger, unit=item_plural, owner_id=first_id, role="lhs"),
            QuantitySpec(quantity_id="q_second", value=smaller, unit=item_plural, owner_id=second_id, role="rhs"),
        ),
        relations=(
            {"type": "ownership", "owner_id": first_id, "item_id": item_id},
            {"type": "ownership", "owner_id": second_id, "item_id": item_id},
            {"type": "comparison", "relation": "difference", "lhs": "q_first", "rhs": "q_second"},
        ),
        goal=GoalSpec(
            target_id="answer",
            query=f"difference_{item_plural}_{first_id}_{second_id}",
            unit=item_plural,
        ),
        program=Program(op="subtract", args=("q_first", "q_second")),
        trace_template=translation_trace(),
        answer=bigger - smaller,
        concept_tags=("word_problem", "translation", "arithmetic", "comparison"),
        difficulty=2,
        metadata={"template": "compare_difference", "answer_unit": item_plural},
    )
    return WordProblemExample(
        example_id=example_id,
        source_text=source_text,
        abstract_task=task,
        family_name="compare_difference",
        template_name="compare_difference",
        notes=("comparison asks for a positive difference",),
        metadata={"operation": "subtract", "answer_unit": item_plural},
    )


def make_multiply_group_problem(rng: random.Random, example_id: str) -> WordProblemExample:
    container_singular, container_plural = pick(rng, CONTAINERS)
    item_singular, item_plural = pick(rng, ITEMS)
    num_groups = rng.randint(2, 6)
    per_group = rng.randint(2, 5)
    container_id = slug(container_plural)
    item_id = slug(item_plural)
    source_text = (
        f"There are {num_groups} {container_plural} with {per_group} {item_plural} in each {container_singular}. "
        f"How many {item_plural} are there altogether?"
    )
    task = AbstractReasoningTask(
        task_id=f"{example_id}_task",
        source_modality="text",
        source_text=source_text,
        entities=(
            EntitySpec(entity_id=container_id, label=container_plural, kind="container"),
            EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
        ),
        quantities=(
            QuantitySpec(quantity_id="q_groups", value=num_groups, unit=container_plural, role="groups"),
            QuantitySpec(quantity_id="q_per_group", value=per_group, unit=item_plural, role="per_group"),
        ),
        relations=(
            {"type": "grouping", "container_id": container_id, "item_id": item_id},
            {"type": "rate", "numerator": "q_per_group", "denominator": "q_groups"},
        ),
        goal=GoalSpec(
            target_id="answer",
            query=f"total_{item_plural}",
            unit=item_plural,
        ),
        program=Program(op="multiply", args=("q_groups", "q_per_group")),
        trace_template=translation_trace(),
        answer=num_groups * per_group,
        concept_tags=("word_problem", "translation", "arithmetic", "multiplication"),
        difficulty=2,
        metadata={"template": "multiply_groups", "answer_unit": item_plural},
    )
    return WordProblemExample(
        example_id=example_id,
        source_text=source_text,
        abstract_task=task,
        family_name="multiply_groups",
        template_name="multiply_groups",
        notes=("groups times per-group count",),
        metadata={"operation": "multiply", "answer_unit": item_plural},
    )


def make_compose_total_problem(rng: random.Random, example_id: str) -> WordProblemExample:
    variant = pick(rng, ("offset_chain_total", "scaled_groups_plus_extra", "weekday_rate_plus_weekend_total"))
    item_singular, item_plural = pick(rng, ITEMS)

    if variant == "offset_chain_total":
        first_person = pick(rng, PEOPLE)
        second_person = pick(rng, [name for name in PEOPLE if name != first_person])
        third_person = pick(rng, [name for name in PEOPLE if name not in {first_person, second_person}])
        first_id = slug(first_person)
        second_id = slug(second_person)
        third_id = slug(third_person)
        item_id = slug(item_plural)
        base = rng.randint(6, 16)
        delta_up = rng.randint(3, 9)
        second_value = base + delta_up
        delta_down = rng.randint(1, max(1, second_value - 3))
        third_value = second_value - delta_down
        source_text = (
            f"{first_person} has {base} {item_plural}. "
            f"{second_person} has {delta_up} more {item_plural} than {first_person}. "
            f"{third_person} has {delta_down} fewer {item_plural} than {second_person}. "
            f"How many {item_plural} do they have altogether?"
        )
        task = AbstractReasoningTask(
            task_id=f"{example_id}_task",
            source_modality="text",
            source_text=source_text,
            entities=(
                EntitySpec(entity_id=first_id, label=first_person, kind="person"),
                EntitySpec(entity_id=second_id, label=second_person, kind="person"),
                EntitySpec(entity_id=third_id, label=third_person, kind="person"),
                EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
            ),
            quantities=(
                QuantitySpec(quantity_id="q_first", value=base, unit=item_plural, owner_id=first_id, role="base"),
                QuantitySpec(quantity_id="q_delta_up", value=delta_up, unit=item_plural, role="offset_up"),
                QuantitySpec(quantity_id="q_second", value=second_value, unit=item_plural, owner_id=second_id, role="derived_term"),
                QuantitySpec(quantity_id="q_delta_down", value=delta_down, unit=item_plural, role="offset_down"),
                QuantitySpec(quantity_id="q_third", value=third_value, unit=item_plural, owner_id=third_id, role="derived_term"),
            ),
            relations=(
                {"type": "ownership", "owner_id": first_id, "item_id": item_id},
                {"type": "ownership", "owner_id": second_id, "item_id": item_id},
                {"type": "ownership", "owner_id": third_id, "item_id": item_id},
                {"type": "derivation", "target": "q_second", "op": "add", "inputs": ("q_first", "q_delta_up")},
                {"type": "derivation", "target": "q_third", "op": "subtract", "inputs": ("q_second", "q_delta_down")},
            ),
            goal=GoalSpec(
                target_id="answer",
                query=f"total_{item_plural}_{first_id}_{second_id}_{third_id}",
                unit=item_plural,
            ),
            program=Program(
                op="ComposeThenReduce",
                args=(
                    ("q_first", "q_second", "q_third"),
                    {"term_ops": ("offset",), "reducer": "sum"},
                ),
            ),
            trace_template=compose_trace("sum"),
            answer=base + second_value + third_value,
            concept_tags=("word_problem", "composition", "reduction", "addition"),
            difficulty=3,
            metadata={
                "template": variant,
                "answer_unit": item_plural,
                "primitive_quantity_ids": ("q_first", "q_delta_up", "q_delta_down"),
                "derived_quantity_ids": ("q_second", "q_third"),
                "reduce_input_ids": ("q_first", "q_second", "q_third"),
                "derivation_rules": (
                    {"target": "q_second", "op": "add", "inputs": ("q_first", "q_delta_up")},
                    {"target": "q_third", "op": "subtract", "inputs": ("q_second", "q_delta_down")},
                ),
                "reducer": "sum",
                "family_name": "compose_total",
            },
        )
        notes = ("derive two hidden term values before summing all terms",)
    elif variant == "scaled_groups_plus_extra":
        container_singular, container_plural = pick(rng, CONTAINERS)
        container_id = slug(container_plural)
        item_id = slug(item_plural)
        table_id = "table"
        num_groups = rng.randint(3, 7)
        per_group = rng.randint(2, 6)
        extra = rng.randint(2, 11)
        boxed_total = num_groups * per_group
        source_text = (
            f"There are {num_groups} {container_plural} with {per_group} {item_plural} in each {container_singular}. "
            f"There are also {extra} extra {item_plural} on a table. "
            f"How many {item_plural} are there altogether?"
        )
        task = AbstractReasoningTask(
            task_id=f"{example_id}_task",
            source_modality="text",
            source_text=source_text,
            entities=(
                EntitySpec(entity_id=container_id, label=container_plural, kind="container"),
                EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
                EntitySpec(entity_id=table_id, label="table", kind="location"),
            ),
            quantities=(
                QuantitySpec(quantity_id="q_groups", value=num_groups, unit=container_plural, role="groups"),
                QuantitySpec(quantity_id="q_per_group", value=per_group, unit=item_plural, role="per_group"),
                QuantitySpec(quantity_id="q_boxed_total", value=boxed_total, unit=item_plural, role="derived_term"),
                QuantitySpec(quantity_id="q_extra", value=extra, unit=item_plural, role="extra"),
            ),
            relations=(
                {"type": "grouping", "container_id": container_id, "item_id": item_id},
                {"type": "location", "location_id": table_id, "item_id": item_id},
                {"type": "derivation", "target": "q_boxed_total", "op": "multiply", "inputs": ("q_groups", "q_per_group")},
            ),
            goal=GoalSpec(
                target_id="answer",
                query=f"total_{item_plural}",
                unit=item_plural,
            ),
            program=Program(
                op="ComposeThenReduce",
                args=(
                    ("q_boxed_total", "q_extra"),
                    {"term_ops": ("scale",), "reducer": "sum"},
                ),
            ),
            trace_template=compose_trace("sum"),
            answer=boxed_total + extra,
            concept_tags=("word_problem", "composition", "reduction", "addition"),
            difficulty=3,
            metadata={
                "template": variant,
                "answer_unit": item_plural,
                "primitive_quantity_ids": ("q_groups", "q_per_group", "q_extra"),
                "derived_quantity_ids": ("q_boxed_total",),
                "reduce_input_ids": ("q_boxed_total", "q_extra"),
                "derivation_rules": (
                    {"target": "q_boxed_total", "op": "multiply", "inputs": ("q_groups", "q_per_group")},
                ),
                "reducer": "sum",
                "family_name": "compose_total",
            },
        )
        notes = ("apply a local scale rule before summing the final terms",)
    else:
        shop_id = "coffee_shop"
        item_singular = "coffee cup"
        item_plural = "coffee cups"
        item_id = slug(item_plural)
        per_hour = rng.randint(6, 18)
        hours_per_day = rng.randint(4, 8)
        weekdays = 5
        weekend_total = rng.randint(80, 180)
        weekday_daily = per_hour * hours_per_day
        weekday_total = weekday_daily * weekdays
        source_text = (
            f"A coffee shop brews {per_hour} {item_plural} per hour on a weekday and {weekend_total} {item_plural} "
            f"in total over the weekend. If the coffee shop is open {hours_per_day} hours a day every single day, "
            f"how many {item_plural} are brewed in 1 week?"
        )
        task = AbstractReasoningTask(
            task_id=f"{example_id}_task",
            source_modality="text",
            source_text=source_text,
            entities=(
                EntitySpec(entity_id=shop_id, label="coffee shop", kind="business"),
                EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
            ),
            quantities=(
                QuantitySpec(quantity_id="q_rate", value=per_hour, unit=item_plural, owner_id=shop_id, role="rate"),
                QuantitySpec(quantity_id="q_hours_per_day", value=hours_per_day, unit="hours", role="time"),
                QuantitySpec(quantity_id="q_weekdays", value=weekdays, unit="days", role="count"),
                QuantitySpec(
                    quantity_id="q_weekend_total",
                    value=weekend_total,
                    unit=item_plural,
                    owner_id=shop_id,
                    role="weekend_total",
                ),
                QuantitySpec(
                    quantity_id="q_weekday_daily",
                    value=weekday_daily,
                    unit=item_plural,
                    owner_id=shop_id,
                    role="derived_term",
                ),
                QuantitySpec(
                    quantity_id="q_weekday_total",
                    value=weekday_total,
                    unit=item_plural,
                    owner_id=shop_id,
                    role="derived_term",
                ),
            ),
            relations=(
                {"type": "production_rate", "owner_id": shop_id, "item_id": item_id, "period": "weekday"},
                {"type": "aggregate_total", "owner_id": shop_id, "item_id": item_id, "period": "weekend"},
                {"type": "derivation", "target": "q_weekday_daily", "op": "multiply", "inputs": ("q_rate", "q_hours_per_day")},
                {"type": "derivation", "target": "q_weekday_total", "op": "multiply", "inputs": ("q_weekday_daily", "q_weekdays")},
            ),
            goal=GoalSpec(
                target_id="answer",
                query="weekly_total_coffee_cups",
                unit=item_plural,
            ),
            program=Program(
                op="ComposeThenReduce",
                args=(
                    ("q_weekday_total", "q_weekend_total"),
                    {"term_ops": ("scale", "scale"), "reducer": "sum"},
                ),
            ),
            trace_template=compose_trace("sum"),
            answer=weekday_total + weekend_total,
            concept_tags=("word_problem", "composition", "reduction", "addition"),
            difficulty=3,
            metadata={
                "template": variant,
                "answer_unit": item_plural,
                "primitive_quantity_ids": ("q_rate", "q_hours_per_day", "q_weekdays", "q_weekend_total"),
                "derived_quantity_ids": ("q_weekday_daily", "q_weekday_total"),
                "reduce_input_ids": ("q_weekday_total", "q_weekend_total"),
                "derivation_rules": (
                    {"target": "q_weekday_daily", "op": "multiply", "inputs": ("q_rate", "q_hours_per_day")},
                    {"target": "q_weekday_total", "op": "multiply", "inputs": ("q_weekday_daily", "q_weekdays")},
                ),
                "reducer": "sum",
                "family_name": "compose_total",
            },
        )
        notes = ("derive a daily total, then a weekday total, before summing with the weekend total",)

    return WordProblemExample(
        example_id=example_id,
        source_text=source_text,
        abstract_task=task,
        family_name="compose_total",
        template_name=variant,
        notes=notes,
        metadata={"operation": "ComposeThenReduce", "answer_unit": item_plural, "family_name": "compose_total"},
    )


def make_compose_difference_problem(rng: random.Random, example_id: str) -> WordProblemExample:
    variant = pick(rng, ("remaining_budget", "derived_compare", "fractional_budget_gap"))
    item_singular, item_plural = pick(rng, ITEMS)

    if variant == "remaining_budget":
        person = pick(rng, PEOPLE)
        person_id = slug(person)
        aunt_gift = rng.randint(5, 15)
        uncle_factor = pick(rng, (2, 3))
        uncle_gift = aunt_gift * uncle_factor
        saved = rng.randint(10, 35)
        answer = rng.randint(4, 20)
        available = saved + aunt_gift + uncle_gift
        target = available + answer
        source_text = (
            f"{person} needs ${target} to buy a new {item_singular}. "
            f"{person} already has ${saved} saved. "
            f"{person}'s aunt gives ${aunt_gift}, and an uncle gives {uncle_factor} times that amount. "
            f"How much more money does {person} need?"
        )
        task = AbstractReasoningTask(
            task_id=f"{example_id}_task",
            source_modality="text",
            source_text=source_text,
            entities=(
                EntitySpec(entity_id=person_id, label=person, kind="person"),
                EntitySpec(entity_id=slug(item_singular), label=item_singular, kind="item"),
            ),
            quantities=(
                QuantitySpec(quantity_id="q_target", value=target, unit="dollars", owner_id=person_id, role="target"),
                QuantitySpec(quantity_id="q_saved", value=saved, unit="dollars", owner_id=person_id, role="saved"),
                QuantitySpec(quantity_id="q_aunt", value=aunt_gift, unit="dollars", role="gift"),
                QuantitySpec(quantity_id="q_uncle_factor", value=uncle_factor, unit="multiplier", role="scale"),
                QuantitySpec(quantity_id="q_uncle", value=uncle_gift, unit="dollars", role="derived_term"),
                QuantitySpec(quantity_id="q_available", value=available, unit="dollars", owner_id=person_id, role="derived_term"),
            ),
            relations=(
                {"type": "goal_budget", "owner_id": person_id, "item": item_singular},
                {"type": "derivation", "target": "q_uncle", "op": "multiply", "inputs": ("q_aunt", "q_uncle_factor")},
                {"type": "derivation", "target": "q_available", "op": "add", "inputs": ("q_saved", "q_aunt", "q_uncle")},
            ),
            goal=GoalSpec(
                target_id="answer",
                query=f"remaining_budget_for_{person_id}",
                unit="dollars",
            ),
            program=Program(
                op="ComposeThenReduce",
                args=(
                    ("q_target",),
                    ("q_available",),
                    {"term_ops": ("scale", "sum"), "reducer": "difference"},
                ),
            ),
            trace_template=compose_trace("difference"),
            answer=answer,
            concept_tags=("word_problem", "composition", "comparison", "subtraction"),
            difficulty=3,
            metadata={
                "template": variant,
                "answer_unit": "dollars",
                "primitive_quantity_ids": ("q_target", "q_saved", "q_aunt", "q_uncle_factor"),
                "derived_quantity_ids": ("q_uncle", "q_available"),
                "reduce_input_ids": (("q_target",), ("q_available",)),
                "derivation_rules": (
                    {"target": "q_uncle", "op": "multiply", "inputs": ("q_aunt", "q_uncle_factor")},
                    {"target": "q_available", "op": "add", "inputs": ("q_saved", "q_aunt", "q_uncle")},
                ),
                "reducer": "difference",
                "family_name": "compose_difference",
            },
        )
        notes = ("compose available money from multiple sources, then subtract from the goal",)
    elif variant == "fractional_budget_gap":
        person = pick(rng, PEOPLE)
        subject_pronoun, object_pronoun, possessive_pronoun = pronouns(person)
        person_id = slug(person)
        purchase = pick(rng, PURCHASE_ITEMS)
        parent_gift = rng.randint(5, 18)
        grandparent_factor = 2
        answer = rng.randint(4, 20)
        target = 2 * (answer + parent_gift * (1 + grandparent_factor))
        saved = target // 2
        grandparent_gift = parent_gift * grandparent_factor
        available = saved + parent_gift + grandparent_gift
        source_text = (
            f"{person} is saving money for a new {purchase} which costs ${target}. "
            f"{person} has only half of the money {subject_pronoun} needs. "
            f"{possessive_pronoun.capitalize()} parents decided to give {object_pronoun} ${parent_gift} for that purpose, "
            f"and {possessive_pronoun} grandparents twice as much as {possessive_pronoun} parents. "
            f"How much more money does {person} need to buy the {purchase}?"
        )
        task = AbstractReasoningTask(
            task_id=f"{example_id}_task",
            source_modality="text",
            source_text=source_text,
            entities=(
                EntitySpec(entity_id=person_id, label=person, kind="person"),
                EntitySpec(entity_id=slug(purchase), label=purchase, kind="item"),
            ),
            quantities=(
                QuantitySpec(quantity_id="q_target", value=target, unit="dollars", owner_id=person_id, role="target"),
                QuantitySpec(quantity_id="q_saved_divisor", value=2, unit="divisor", role="partition"),
                QuantitySpec(quantity_id="q_parents", value=parent_gift, unit="dollars", role="gift"),
                QuantitySpec(quantity_id="q_grand_factor", value=grandparent_factor, unit="multiplier", role="scale"),
                QuantitySpec(quantity_id="q_saved", value=saved, unit="dollars", owner_id=person_id, role="derived_term"),
                QuantitySpec(
                    quantity_id="q_grandparents",
                    value=grandparent_gift,
                    unit="dollars",
                    role="derived_term",
                ),
                QuantitySpec(
                    quantity_id="q_available",
                    value=available,
                    unit="dollars",
                    owner_id=person_id,
                    role="derived_term",
                ),
            ),
            relations=(
                {"type": "goal_budget", "owner_id": person_id, "item": purchase},
                {"type": "derivation", "target": "q_saved", "op": "divide", "inputs": ("q_target", "q_saved_divisor")},
                {"type": "derivation", "target": "q_grandparents", "op": "multiply", "inputs": ("q_parents", "q_grand_factor")},
                {
                    "type": "derivation",
                    "target": "q_available",
                    "op": "add",
                    "inputs": ("q_saved", "q_parents", "q_grandparents"),
                },
            ),
            goal=GoalSpec(
                target_id="answer",
                query=f"remaining_budget_for_{person_id}",
                unit="dollars",
            ),
            program=Program(
                op="ComposeThenReduce",
                args=(
                    ("q_target",),
                    ("q_available",),
                    {"term_ops": ("partition", "scale", "sum"), "reducer": "difference"},
                ),
            ),
            trace_template=compose_trace("difference"),
            answer=answer,
            concept_tags=("word_problem", "composition", "comparison", "subtraction"),
            difficulty=3,
            metadata={
                "template": variant,
                "answer_unit": "dollars",
                "primitive_quantity_ids": ("q_target", "q_saved_divisor", "q_parents", "q_grand_factor"),
                "derived_quantity_ids": ("q_saved", "q_grandparents", "q_available"),
                "reduce_input_ids": (("q_target",), ("q_available",)),
                "derivation_rules": (
                    {"target": "q_saved", "op": "divide", "inputs": ("q_target", "q_saved_divisor")},
                    {"target": "q_grandparents", "op": "multiply", "inputs": ("q_parents", "q_grand_factor")},
                    {"target": "q_available", "op": "add", "inputs": ("q_saved", "q_parents", "q_grandparents")},
                ),
                "reducer": "difference",
                "family_name": "compose_difference",
            },
        )
        notes = ("derive a saved fraction and a scaled family gift before subtracting from the goal",)
    else:
        first_person = pick(rng, PEOPLE)
        second_person = pick(rng, [name for name in PEOPLE if name != first_person])
        first_id = slug(first_person)
        second_id = slug(second_person)
        item_id = slug(item_plural)
        base = rng.randint(5, 12)
        factor = pick(rng, (2, 3, 4))
        scaled = base * factor
        delta = rng.randint(1, max(1, scaled - base - 1))
        second_value = scaled - delta
        answer = second_value - base
        source_text = (
            f"{first_person} has {base} {item_plural}. "
            f"{second_person} has {factor} times as many {item_plural} as {first_person}, but then gives away {delta}. "
            f"How many more {item_plural} does {second_person} have than {first_person}?"
        )
        task = AbstractReasoningTask(
            task_id=f"{example_id}_task",
            source_modality="text",
            source_text=source_text,
            entities=(
                EntitySpec(entity_id=first_id, label=first_person, kind="person"),
                EntitySpec(entity_id=second_id, label=second_person, kind="person"),
                EntitySpec(entity_id=item_id, label=item_plural, kind="item"),
            ),
            quantities=(
                QuantitySpec(quantity_id="q_first", value=base, unit=item_plural, owner_id=first_id, role="base"),
                QuantitySpec(quantity_id="q_factor", value=factor, unit="multiplier", role="scale"),
                QuantitySpec(quantity_id="q_second_scaled", value=scaled, unit=item_plural, owner_id=second_id, role="derived_term"),
                QuantitySpec(quantity_id="q_delta", value=delta, unit=item_plural, role="offset_down"),
                QuantitySpec(quantity_id="q_second_final", value=second_value, unit=item_plural, owner_id=second_id, role="derived_term"),
            ),
            relations=(
                {"type": "ownership", "owner_id": first_id, "item_id": item_id},
                {"type": "ownership", "owner_id": second_id, "item_id": item_id},
                {"type": "derivation", "target": "q_second_scaled", "op": "multiply", "inputs": ("q_first", "q_factor")},
                {"type": "derivation", "target": "q_second_final", "op": "subtract", "inputs": ("q_second_scaled", "q_delta")},
            ),
            goal=GoalSpec(
                target_id="answer",
                query=f"difference_{item_plural}_{second_id}_{first_id}",
                unit=item_plural,
            ),
            program=Program(
                op="ComposeThenReduce",
                args=(
                    ("q_second_final",),
                    ("q_first",),
                    {"term_ops": ("scale", "offset"), "reducer": "difference"},
                ),
            ),
            trace_template=compose_trace("difference"),
            answer=answer,
            concept_tags=("word_problem", "composition", "comparison", "subtraction"),
            difficulty=3,
            metadata={
                "template": variant,
                "answer_unit": item_plural,
                "primitive_quantity_ids": ("q_first", "q_factor", "q_delta"),
                "derived_quantity_ids": ("q_second_scaled", "q_second_final"),
                "reduce_input_ids": (("q_second_final",), ("q_first",)),
                "derivation_rules": (
                    {"target": "q_second_scaled", "op": "multiply", "inputs": ("q_first", "q_factor")},
                    {"target": "q_second_final", "op": "subtract", "inputs": ("q_second_scaled", "q_delta")},
                ),
                "reducer": "difference",
                "family_name": "compose_difference",
            },
        )
        notes = ("derive a comparison term before taking the final difference",)

    return WordProblemExample(
        example_id=example_id,
        source_text=source_text,
        abstract_task=task,
        family_name="compose_difference",
        template_name=variant,
        notes=notes,
        metadata={"operation": "ComposeThenReduce", "answer_unit": task.goal.unit, "family_name": "compose_difference"},
    )


TEMPLATE_BUILDERS = (
    (1.8, make_compose_total_problem),
    (1.8, make_compose_difference_problem),
    (2.0, make_add_change_problem),
    (2.0, make_subtract_change_problem),
    (1.3, make_compare_problem),
    (1.1, make_multiply_group_problem),
)


def sample_word_problem_example(seed: int, example_index: int | None = None) -> WordProblemExample:
    rng = random.Random(seed)
    example_id = f"word_problem_{example_index if example_index is not None else seed}"
    builder = weighted_choice(rng, TEMPLATE_BUILDERS)
    return builder(rng, example_id)


def encoded_quantities(task: AbstractReasoningTask, quantity_ids: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = set(quantity_ids)
    return [_encode(quantity) for quantity in task.quantities if quantity.quantity_id in wanted]


def translation_step_states(example: WordProblemExample) -> Dict[str, State]:
    task = example.abstract_task
    final_state = task.to_state()
    prefix = {
        "source_modality": task.source_modality,
        "source_text": example.source_text,
        "answer_format": task.answer_format,
        "choices": final_state["choices"],
        "correct_choice": final_state["correct_choice"],
    }
    return {
        "extract_entities": {
            **prefix,
            "entities": final_state["entities"],
            "goal": final_state["goal"],
        },
        "bind_quantities": {
            **prefix,
            "entities": final_state["entities"],
            "quantities": final_state["quantities"],
            "goal": final_state["goal"],
        },
        "choose_operation": {
            **prefix,
            "entities": final_state["entities"],
            "quantities": final_state["quantities"],
            "relations": final_state["relations"],
            "goal": final_state["goal"],
            "program": final_state["program"],
        },
        "compute_answer": {
            **prefix,
            "entities": final_state["entities"],
            "quantities": final_state["quantities"],
            "relations": final_state["relations"],
            "goal": final_state["goal"],
            "program": final_state["program"],
            "answer": final_state["answer"],
        },
        "emit_ir": final_state,
    }


def compose_step_states(example: WordProblemExample) -> Dict[str, State]:
    task = example.abstract_task
    final_state = task.to_state()
    metadata = task.metadata
    primitive_ids = metadata.get("primitive_quantity_ids", ())
    derived_ids = metadata.get("derived_quantity_ids", ())
    prefix = {
        "source_modality": task.source_modality,
        "source_text": example.source_text,
        "answer_format": task.answer_format,
        "choices": final_state["choices"],
        "correct_choice": final_state["correct_choice"],
    }
    segment_state = {
        **prefix,
        "entities": final_state["entities"],
        "goal": final_state["goal"],
    }
    bind_state = {
        **segment_state,
        "quantities": encoded_quantities(task, primitive_ids),
        "relations": [relation for relation in final_state["relations"] if relation.get("type") != "derivation"],
        "program": final_state["program"],
    }
    apply_state = {
        **bind_state,
        "derived_quantities": encoded_quantities(task, derived_ids),
        "derivation_rules": _encode(metadata.get("derivation_rules", ())),
    }
    reduce_state = {
        **apply_state,
        "reducer": metadata.get("reducer"),
        "reduce_input_ids": _encode(metadata.get("reduce_input_ids", ())),
        "answer": final_state["answer"],
    }
    return {
        "segment": segment_state,
        "bind": bind_state,
        "apply": apply_state,
        "reduce": reduce_state,
        "render": final_state,
    }


def rate_scale_step_states(example: WordProblemExample) -> Dict[str, State]:
    task = example.abstract_task
    final_state = task.to_state()
    metadata = task.metadata
    primitive_ids = metadata.get("primitive_quantity_ids", ())
    derived_ids = metadata.get("derived_quantity_ids", ())
    prefix = {
        "source_modality": task.source_modality,
        "source_text": example.source_text,
        "answer_format": task.answer_format,
        "choices": final_state["choices"],
        "correct_choice": final_state["correct_choice"],
    }
    segment_state = {
        **prefix,
        "entities": final_state["entities"],
        "goal": final_state["goal"],
    }
    bind_state = {
        **segment_state,
        "quantities": encoded_quantities(task, primitive_ids),
        "relations": [relation for relation in final_state["relations"] if relation.get("type") != "derivation"],
        "program": final_state["program"],
    }
    apply_state = {
        **bind_state,
        "derived_quantities": encoded_quantities(task, derived_ids),
        "derivation_rules": _encode(metadata.get("derivation_rules", ())),
        "final_input_ids": _encode(metadata.get("final_input_ids", ())),
        "reducer": metadata.get("reducer"),
    }
    return {
        "segment": segment_state,
        "bind": bind_state,
        "apply": apply_state,
        "render": final_state,
    }


def partition_inverse_step_states(example: WordProblemExample) -> Dict[str, State]:
    task = example.abstract_task
    final_state = task.to_state()
    metadata = task.metadata
    primitive_ids = metadata.get("primitive_quantity_ids", ())
    derived_ids = metadata.get("derived_quantity_ids", ())
    prefix = {
        "source_modality": task.source_modality,
        "source_text": example.source_text,
        "answer_format": task.answer_format,
        "choices": final_state["choices"],
        "correct_choice": final_state["correct_choice"],
    }
    segment_state = {
        **prefix,
        "entities": final_state["entities"],
        "goal": final_state["goal"],
    }
    bind_state = {
        **segment_state,
        "quantities": encoded_quantities(task, primitive_ids),
        "relations": [relation for relation in final_state["relations"] if relation.get("type") != "derivation"],
        "program": final_state["program"],
    }
    reduce_state = {
        **bind_state,
        "derived_quantities": encoded_quantities(task, derived_ids),
        "derivation_rules": _encode(metadata.get("derivation_rules", ())),
        "final_input_ids": _encode(metadata.get("final_input_ids", ())),
        "reducer": metadata.get("reducer"),
        "answer": final_state["answer"],
    }
    return {
        "segment": segment_state,
        "bind": bind_state,
        "reduce": reduce_state,
        "render": final_state,
    }


def example_step_states(example: WordProblemExample) -> Dict[str, State]:
    step_names = tuple(step.name for step in example.abstract_task.trace_template)
    if step_names == ("extract_entities", "bind_quantities", "choose_operation", "compute_answer", "emit_ir"):
        return translation_step_states(example)
    if step_names == ("segment", "bind", "apply", "render"):
        return rate_scale_step_states(example)
    if step_names == ("segment", "bind", "reduce", "render"):
        return partition_inverse_step_states(example)
    return compose_step_states(example)


def translation_step_action(step_name: str, example: WordProblemExample) -> Dict[str, Any]:
    task = example.abstract_task
    if step_name == "extract_entities":
        return {
            "entity_ids": [entity.entity_id for entity in task.entities],
            "goal_query": task.goal.query,
        }
    if step_name == "bind_quantities":
        return {
            "quantity_ids": [quantity.quantity_id for quantity in task.quantities],
            "quantity_roles": {quantity.quantity_id: quantity.role for quantity in task.quantities},
        }
    if step_name == "choose_operation":
        return {
            "op": task.program.op,
            "args": list(task.program.args),
        }
    if step_name == "compute_answer":
        return {
            "answer": task.answer,
            "unit": task.goal.unit,
        }
    return {
        "ir_keys": sorted(task.to_state()),
        "template_name": example.template_name,
    }


def arc_reasoning_step_action(step_name: str, example: WordProblemExample) -> Dict[str, Any]:
    task = example.abstract_task
    metadata = task.metadata
    if step_name == "segment":
        return {
            "entity_ids": [entity.entity_id for entity in task.entities],
            "goal_query": task.goal.query,
        }
    if step_name == "bind":
        return {
            "quantity_ids": list(metadata.get("primitive_quantity_ids", ())),
            "relation_types": sorted({relation["type"] for relation in task.relations}),
            "program_op": task.program.op,
        }
    if step_name == "apply":
        payload = {
            "derived_quantity_ids": list(metadata.get("derived_quantity_ids", ())),
            "derivation_rules": _encode(metadata.get("derivation_rules", ())),
        }
        if "final_input_ids" in metadata:
            payload["final_input_ids"] = _encode(metadata.get("final_input_ids", ()))
        if "reducer" in metadata:
            payload["reducer"] = metadata.get("reducer")
        return payload
    if step_name == "reduce":
        payload = {
            "reducer": metadata.get("reducer"),
            "answer": task.answer,
            "unit": task.goal.unit,
        }
        if "reduce_input_ids" in metadata:
            payload["reduce_input_ids"] = _encode(metadata.get("reduce_input_ids", ()))
        if "final_input_ids" in metadata:
            payload["final_input_ids"] = _encode(metadata.get("final_input_ids", ()))
        return payload
    return {
        "ir_keys": sorted(task.to_state()),
        "family_name": example.family_name,
    }


def step_action(step_name: str, example: WordProblemExample) -> Dict[str, Any]:
    if step_name in {"extract_entities", "bind_quantities", "choose_operation", "compute_answer", "emit_ir"}:
        return translation_step_action(step_name, example)
    return arc_reasoning_step_action(step_name, example)


def compile_word_problem_trajectory(
    example: WordProblemExample,
    *,
    split: str,
    trajectory_index: int,
) -> TrajectoryRecord:
    final_state = example.abstract_task.to_state()
    step_states = example_step_states(example)
    total_possible_reward = sum(
        WORD_PROBLEM_STEP_WEIGHTS.get(step.name, STEP_WEIGHTS.get(step.name, 0.15))
        for step in example.abstract_task.trace_template
    )
    previous_state: State = {
        "source_modality": example.abstract_task.source_modality,
        "source_text": example.source_text,
        "answer_format": example.abstract_task.answer_format,
        "choices": final_state["choices"],
        "correct_choice": final_state["correct_choice"],
    }
    initial_input_state = previous_state
    cumulative = 0.0
    steps: List[TrajectoryStep] = []

    for index, trace_step in enumerate(example.abstract_task.trace_template):
        current_state = step_states[trace_step.name]
        reward_terms = symbolic_step_reward_terms(
            current_state=current_state,
            target_state=current_state,
            previous_state=previous_state,
            final_state=final_state,
        )
        weight = WORD_PROBLEM_STEP_WEIGHTS.get(trace_step.name, STEP_WEIGHTS.get(trace_step.name, 0.15))
        reward = reward_from_terms(weight, reward_terms)
        cumulative += reward
        steps.append(
            TrajectoryStep(
                index=index,
                name=trace_step.name,
                description=trace_step.description,
                action=step_action(trace_step.name, example),
                reward=reward,
                reward_terms=reward_terms,
                cumulative_reward=cumulative,
                progress=(index + 1) / max(1, len(example.abstract_task.trace_template)),
                stop_target=index == len(example.abstract_task.trace_template) - 1,
                workspace_state=current_state,
                verifier={
                    "exact_match": index == len(example.abstract_task.trace_template) - 1,
                    "should_stop": index == len(example.abstract_task.trace_template) - 1,
                    "resolved_subgoal_count": index + 1,
                    "unresolved_subgoal_count": len(example.abstract_task.trace_template) - index - 1,
                    "next_subgoal": (
                        example.abstract_task.trace_template[index + 1].name
                        if index + 1 < len(example.abstract_task.trace_template)
                        else None
                    ),
                    "non_terminal_reason": (
                        None
                        if index == len(example.abstract_task.trace_template) - 1
                        else f"remaining_subgoal:{example.abstract_task.trace_template[index + 1].name}"
                    ),
                },
                done=index == len(example.abstract_task.trace_template) - 1,
            )
        )
        previous_state = current_state

    return TrajectoryRecord(
        trajectory_id=f"word_problem_translation:{split}:{example.example_id}:{trajectory_index}",
        split=split,
        family=example.family_name,
        difficulty=example.abstract_task.difficulty,
        source_modality=example.abstract_task.source_modality,
        concept_tags=example.abstract_task.concept_tags,
        trace_template=tuple(step.name for step in example.abstract_task.trace_template),
        role_bindings={},
        episode_metadata=example.metadata,
        shortcut_checks=(
            "operation must be recoverable from lexical template",
            "derived quantities must be recoverable from primitive quantities",
            "final reducer must match the question type",
        ),
        example=example,
        input_state=initial_input_state,
        output_state=final_state,
        steps=tuple(steps),
        total_reward=cumulative,
        total_possible_reward=total_possible_reward,
    )


def build_word_problem_dataset(
    *,
    num_examples: int,
    seed_start: int = 0,
) -> Tuple[WordProblemExample, ...]:
    examples: List[WordProblemExample] = []
    for offset in range(num_examples):
        examples.append(sample_word_problem_example(seed=seed_start + offset, example_index=offset))
    return tuple(examples)


def compile_word_problem_examples(
    examples: Sequence[WordProblemExample],
    *,
    num_test: int = 0,
) -> Tuple[TrajectoryRecord, ...]:
    records: List[TrajectoryRecord] = []
    cutoff = max(0, len(examples) - num_test)
    for index, example in enumerate(examples):
        split = "train" if index < cutoff else "test"
        records.append(compile_word_problem_trajectory(example, split=split, trajectory_index=index))
    return tuple(records)


def build_word_problem_trajectories(
    *,
    num_examples: int,
    seed_start: int = 0,
    num_test: int = 0,
) -> Tuple[TrajectoryRecord, ...]:
    examples = build_word_problem_dataset(num_examples=num_examples, seed_start=seed_start)
    return compile_word_problem_examples(examples, num_test=num_test)


def write_translation_jsonl(path: str | Path, examples: Sequence[WordProblemExample]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_jsonable()))
            handle.write("\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic word problems paired with abstract reasoning IR.")
    parser.add_argument("--num-examples", type=int, default=32, help="Number of examples to sample.")
    parser.add_argument("--seed-start", type=int, default=0, help="Initial random seed.")
    parser.add_argument("--num-test", type=int, default=4, help="Number of held-out test examples at the end.")
    parser.add_argument(
        "--translation-output",
        type=str,
        default="arc_trajectory_sampler/results/word_problem_translation.jsonl",
        help="Destination JSONL path for text-to-IR pairs.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default="arc_trajectory_sampler/results/word_problem_translation_trajectories.jsonl",
        help="Destination JSONL path for canonical trajectory records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_word_problem_dataset(
        num_examples=args.num_examples,
        seed_start=args.seed_start,
    )
    trajectories = compile_word_problem_examples(examples, num_test=args.num_test)
    write_translation_jsonl(args.translation_output, examples)
    write_jsonl(args.trajectory_output, trajectories)
    print(json.dumps(trajectories[0].to_jsonable(), indent=2))


if __name__ == "__main__":
    main()
