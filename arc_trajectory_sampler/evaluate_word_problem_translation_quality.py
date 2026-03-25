from __future__ import annotations

"""Evaluate synthetic word-problem translation dataset quality."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    from .reasoning_ir import AbstractReasoningTask
    from .word_problem_translation_dataset import (
        WordProblemExample,
        build_word_problem_dataset,
        compile_word_problem_examples,
    )
except ImportError:  # pragma: no cover - direct script execution
    from reasoning_ir import AbstractReasoningTask  # type: ignore
    from word_problem_translation_dataset import (  # type: ignore
        WordProblemExample,
        build_word_problem_dataset,
        compile_word_problem_examples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic word-problem translation dataset quality.")
    parser.add_argument("--num-examples", type=int, default=500, help="Number of examples to sample.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting random seed.")
    parser.add_argument("--num-test", type=int, default=50, help="Number of held-out examples from the tail.")
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/word_problem_quality_summary.json",
        help="Destination summary JSON path.",
    )
    return parser.parse_args()


def encode(obj: Any) -> Any:
    if hasattr(obj, "to_jsonable"):
        return obj.to_jsonable()
    return obj


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def canonical_task_state(task: AbstractReasoningTask) -> Dict[str, Any]:
    state = task.to_state()
    return {key: value for key, value in state.items() if key != "task_id"}


def to_fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(str(value))


def exact_quantity_fraction(quantity: Any) -> Fraction:
    if hasattr(quantity, "attributes") and hasattr(quantity, "value"):
        exact_value = quantity.attributes.get("exact_value")
        if isinstance(exact_value, dict):
            numerator = exact_value.get("numerator")
            denominator = exact_value.get("denominator")
            if isinstance(numerator, int) and isinstance(denominator, int) and denominator != 0:
                return Fraction(numerator, denominator)
        return to_fraction(quantity.value)
    return to_fraction(quantity)


def number_from_fraction(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def values_match(lhs: int | float | Fraction, rhs: int | float | Fraction) -> bool:
    return to_fraction(lhs) == to_fraction(rhs)


def apply_numeric_op(op: str, values: Sequence[int | float | Fraction]) -> Fraction:
    fractions = [to_fraction(value) for value in values]
    if op == "add":
        return sum(fractions, Fraction(0, 1))
    if op == "subtract":
        if len(fractions) < 2:
            raise ValueError("subtract requires at least two inputs")
        return fractions[0] - sum(fractions[1:], Fraction(0, 1))
    if op == "multiply":
        product = Fraction(1, 1)
        for value in fractions:
            product *= value
        return product
    if op == "divide":
        if len(fractions) != 2:
            raise ValueError("divide requires exactly two inputs")
        numerator, denominator = fractions
        if denominator == 0:
            raise ValueError("divide requires a non-zero denominator")
        return numerator / denominator
    raise ValueError(f"unsupported numeric op: {op}")


def build_value_resolver(task: AbstractReasoningTask, quantities: Dict[str, Any]):
    rules = {rule["target"]: rule for rule in task.metadata.get("derivation_rules", ())}
    cache: Dict[str, Fraction] = {}

    def value_of(quantity_id: str) -> Fraction:
        if quantity_id in cache:
            return cache[quantity_id]
        if quantity_id in rules:
            rule = rules[quantity_id]
            computed = apply_numeric_op(rule["op"], [value_of(arg) for arg in rule["inputs"]])
            stored = quantities.get(quantity_id)
            if stored is not None and exact_quantity_fraction(stored) != computed:
                raise ValueError(
                    f"derived quantity mismatch for {quantity_id}: stored={stored} computed={number_from_fraction(computed)}"
                )
            cache[quantity_id] = computed
            return computed
        if quantity_id not in quantities:
            raise KeyError(f"unknown quantity id: {quantity_id}")
        cache[quantity_id] = exact_quantity_fraction(quantities[quantity_id])
        return cache[quantity_id]

    return value_of


def solve_compose_then_reduce(task: AbstractReasoningTask, quantities: Dict[str, Any]) -> Fraction:
    value_of = build_value_resolver(task, quantities)
    reducer = task.metadata.get("reducer")
    reduce_input_ids = task.metadata.get("reduce_input_ids", ())
    if reducer == "sum":
        return sum((value_of(quantity_id) for quantity_id in reduce_input_ids), Fraction(0, 1))
    if reducer == "difference":
        if len(reduce_input_ids) != 2:
            raise ValueError("difference reducer requires lhs/rhs groups")
        lhs_ids, rhs_ids = reduce_input_ids
        return sum((value_of(quantity_id) for quantity_id in lhs_ids), Fraction(0, 1)) - sum(
            (value_of(quantity_id) for quantity_id in rhs_ids), Fraction(0, 1)
        )
    raise ValueError(f"unsupported reducer: {reducer}")


def solve_apply_rate(task: AbstractReasoningTask, quantities: Dict[str, Any]) -> Fraction:
    value_of = build_value_resolver(task, quantities)
    final_input_ids = task.metadata.get("final_input_ids", ())
    if not final_input_ids:
        raise ValueError("ApplyRate requires final_input_ids metadata")
    return apply_numeric_op("multiply", [value_of(quantity_id) for quantity_id in final_input_ids])


def solve_partition_inverse(task: AbstractReasoningTask, quantities: Dict[str, Any]) -> Fraction:
    value_of = build_value_resolver(task, quantities)
    final_input_ids = task.metadata.get("final_input_ids", ())
    if len(final_input_ids) != 2:
        raise ValueError("PartitionInverse requires exactly two final_input_ids")
    return apply_numeric_op("divide", [value_of(quantity_id) for quantity_id in final_input_ids])


def solve_task(task: AbstractReasoningTask) -> int | float:
    quantities = {quantity.quantity_id: quantity for quantity in task.quantities}
    if task.program.op == "add":
        return number_from_fraction(apply_numeric_op("add", [exact_quantity_fraction(quantities[arg]) for arg in task.program.args]))
    if task.program.op == "subtract":
        return number_from_fraction(
            apply_numeric_op("subtract", [exact_quantity_fraction(quantities[arg]) for arg in task.program.args])
        )
    if task.program.op == "multiply":
        return number_from_fraction(
            apply_numeric_op("multiply", [exact_quantity_fraction(quantities[arg]) for arg in task.program.args])
        )
    if task.program.op == "ComposeThenReduce":
        return number_from_fraction(solve_compose_then_reduce(task, quantities))
    if task.program.op == "ApplyRate":
        return number_from_fraction(solve_apply_rate(task, quantities))
    if task.program.op == "PartitionInverse":
        return number_from_fraction(solve_partition_inverse(task, quantities))
    raise ValueError(f"unsupported program op: {task.program.op}")


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def min_or_none(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def max_or_none(values: Sequence[float]) -> float | None:
    return max(values) if values else None


def collect_vocab(examples: Iterable[WordProblemExample], *, kind: str) -> List[str]:
    values: List[str] = []
    for example in examples:
        for entity in example.abstract_task.entities:
            if entity.kind == kind:
                values.append(entity.label)
    return values


def main() -> None:
    args = parse_args()
    failures = Counter()

    try:
        examples = build_word_problem_dataset(num_examples=args.num_examples, seed_start=args.seed_start)
        trajectories = compile_word_problem_examples(examples, num_test=args.num_test)
    except Exception as exc:  # pragma: no cover - summary should capture failures
        failures[f"{type(exc).__name__}: {exc}"] += 1
        examples = tuple()
        trajectories = tuple()

    template_counts = Counter()
    family_counts = Counter()
    operation_counts = Counter()
    difficulty_counts = Counter()
    split_counts = Counter()
    entity_hist = Counter()
    quantity_hist = Counter()
    relation_hist = Counter()
    answer_values: List[int | float] = []
    source_lengths = []
    total_rewards = []
    total_possible = []
    progress_terminal = []
    local_progress = []
    output_progress = []
    state_delta = []
    step_reward_values = Counter()
    per_step_reward = defaultdict(list)
    exact_text_hashes = Counter()
    exact_ir_hashes = Counter()
    template_surface_hashes = defaultdict(set)
    template_program_ops = defaultdict(set)
    template_units = defaultdict(set)
    split_templates = defaultdict(set)
    step_examples = []
    derivation_failures = []

    for example, trajectory in zip(examples, trajectories, strict=False):
        task = example.abstract_task
        template_counts[example.template_name] += 1
        family_counts[example.family_name] += 1
        operation_counts[task.program.op] += 1
        difficulty_counts[task.difficulty] += 1
        split_counts[trajectory.split] += 1
        split_templates[trajectory.split].add(example.template_name)
        entity_hist[len(task.entities)] += 1
        quantity_hist[len(task.quantities)] += 1
        relation_hist[len(task.relations)] += 1
        answer_values.append(task.answer)
        source_lengths.append(len(task.source_text.split()))
        template_program_ops[example.template_name].add(task.program.op)
        template_units[example.template_name].add(task.goal.unit)
        total_rewards.append(trajectory.total_reward)
        total_possible.append(trajectory.total_possible_reward)
        progress_terminal.append(trajectory.steps[-1].progress if trajectory.steps else 0.0)

        source_payload = {
            "source_text": example.source_text,
            "template_name": example.template_name,
        }
        exact_text_hashes[digest(source_payload)] += 1
        exact_ir_hashes[digest(canonical_task_state(task))] += 1
        template_surface_hashes[example.template_name].add(digest({"source_text": example.source_text}))

        try:
            solved = solve_task(task)
            if not values_match(solved, task.answer):
                derivation_failures.append(
                    {
                        "example_id": example.example_id,
                        "program": encode(task.program),
                        "expected": task.answer,
                        "derived": solved,
                    }
                )
        except Exception as exc:  # pragma: no cover - summary should capture failures
            derivation_failures.append(
                {
                    "example_id": example.example_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        for step in trajectory.steps:
            step_reward_values[round(step.reward, 6)] += 1
            per_step_reward[step.name].append(step.reward)
            local_progress.append(float(step.reward_terms.get("local_progress", 0.0)))
            output_progress.append(float(step.reward_terms.get("output_progress", 0.0)))
            state_delta.append(float(step.reward_terms.get("state_delta", 0.0)))
            if len(step_examples) < 8:
                step_examples.append(
                    {
                        "template": example.template_name,
                        "step": step.name,
                        "reward": step.reward,
                        "reward_terms": step.reward_terms,
                    }
                )

    summary = {
        "num_examples": len(examples),
        "seed_start": args.seed_start,
        "num_test": args.num_test,
        "failure_counts": dict(failures),
        "template_counts": dict(template_counts),
        "family_counts": dict(family_counts),
        "operation_counts": dict(operation_counts),
        "difficulty_counts": dict(difficulty_counts),
        "split_counts": dict(split_counts),
        "entity_count_histogram": dict(entity_hist),
        "quantity_count_histogram": dict(quantity_hist),
        "relation_count_histogram": dict(relation_hist),
        "answer": {
            "min": min_or_none(answer_values),
            "max": max_or_none(answer_values),
            "mean": mean_or_none(answer_values),
        },
        "source_length_tokens": {
            "min": min_or_none(source_lengths),
            "max": max_or_none(source_lengths),
            "mean": mean_or_none(source_lengths),
        },
        "reward_min": min_or_none(total_rewards),
        "reward_max": max_or_none(total_rewards),
        "reward_mean": mean_or_none(total_rewards),
        "possible_unique": sorted(set(total_possible)),
        "all_total_reward_equal_possible": all(abs(a - b) < 1e-9 for a, b in zip(total_rewards, total_possible)),
        "terminal_progress_min": min_or_none(progress_terminal),
        "terminal_progress_max": max_or_none(progress_terminal),
        "terminal_progress_mean": mean_or_none(progress_terminal),
        "unique_step_reward_values_count": len(step_reward_values),
        "unique_step_reward_values_sample": sorted(step_reward_values)[:30],
        "local_progress": {
            "min": min_or_none(local_progress),
            "max": max_or_none(local_progress),
            "mean": mean_or_none(local_progress),
        },
        "output_progress": {
            "min": min_or_none(output_progress),
            "max": max_or_none(output_progress),
            "mean": mean_or_none(output_progress),
        },
        "state_delta": {
            "min": min_or_none(state_delta),
            "max": max_or_none(state_delta),
            "mean": mean_or_none(state_delta),
        },
        "mean_reward_by_step": {name: mean_or_none(values) for name, values in per_step_reward.items()},
        "duplicate_source_texts": sum(count - 1 for count in exact_text_hashes.values() if count > 1),
        "unique_source_texts": len(exact_text_hashes),
        "duplicate_ir_states": sum(count - 1 for count in exact_ir_hashes.values() if count > 1),
        "unique_ir_states": len(exact_ir_hashes),
        "template_surface_diversity": {
            name: len(hashes) for name, hashes in template_surface_hashes.items()
        },
        "template_program_ops": {
            name: sorted(values) for name, values in template_program_ops.items()
        },
        "template_units": {
            name: sorted(values) for name, values in template_units.items()
        },
        "train_test_template_overlap": sorted(split_templates["train"] & split_templates["test"]),
        "train_only_templates": sorted(split_templates["train"] - split_templates["test"]),
        "test_only_templates": sorted(split_templates["test"] - split_templates["train"]),
        "person_vocab_size": len(set(collect_vocab(examples, kind="person"))),
        "item_vocab_size": len(set(collect_vocab(examples, kind="item"))),
        "container_vocab_size": len(set(collect_vocab(examples, kind="container"))),
        "answer_derivation_failures": derivation_failures[:20],
        "answer_derivation_failure_count": len(derivation_failures),
        "step_examples": step_examples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
