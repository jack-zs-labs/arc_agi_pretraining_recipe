from __future__ import annotations

"""Evaluate OlymMATH parser coverage and canonical trajectory compilation."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from .olympiad_math_parser import (
        DEFAULT_OLYMPIAD_MATH_CONFIGS,
        OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        build_olympiad_math_examples,
        compile_olympiad_math_examples,
        load_olympiad_math_rows,
    )
except ImportError:  # pragma: no cover - direct script execution
    from olympiad_math_parser import (  # type: ignore
        DEFAULT_OLYMPIAD_MATH_CONFIGS,
        OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        build_olympiad_math_examples,
        compile_olympiad_math_examples,
        load_olympiad_math_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OlymMATH parser coverage and trajectory consistency.")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=list(DEFAULT_OLYMPIAD_MATH_CONFIGS),
        help="OlymMATH configs to evaluate. Defaults to English easy/hard benchmark sets.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/olympiad_math_parser_eval_summary.json",
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of rows to evaluate for a quick smoke run.",
    )
    return parser.parse_args()


def sorted_counter(counter: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def evaluate_olympiad_math_parser(
    *,
    configs: Sequence[str],
    max_rows: int | None = None,
) -> Dict[str, Any]:
    rows = load_olympiad_math_rows(
        configs=configs,
        allow_eval_configs=True,
        max_rows=max_rows,
    )
    examples, failure_counts = build_olympiad_math_examples(
        configs=configs,
        allow_eval_configs=True,
        max_rows=max_rows,
    )
    trajectories = compile_olympiad_math_examples(examples)

    config_counts = Counter(example.row.config for example in examples)
    family_counts = Counter(example.family_name for example in examples)
    subject_counts = Counter(example.row.subject for example in examples)
    language_counts = Counter(example.row.language for example in examples)
    task_variant_counts = Counter(example.row.task_variant for example in examples)
    answer_kind_counts = Counter(
        str(example.abstract_task.metadata.get("answer_metadata", {}).get("kind", "unknown"))
        for example in examples
    )

    consistency_failures = []
    sample_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example, trajectory in zip(examples, trajectories):
        canonical = example.abstract_task.metadata.get("answer_metadata", {}).get("canonical")
        final_canonical = trajectory.steps[-2].workspace_state.get("canonical_answer") if len(trajectory.steps) >= 2 else None
        if canonical != final_canonical:
            consistency_failures.append(
                {
                    "unique_id": example.row.unique_id,
                    "config": example.row.config,
                    "expected_canonical": canonical,
                    "trajectory_canonical": final_canonical,
                }
            )
        if len(sample_examples[example.family_name]) < 5:
            sample_examples[example.family_name].append(
                {
                    "config": example.row.config,
                    "unique_id": example.row.unique_id,
                    "subject": example.row.subject,
                    "problem": example.row.problem,
                    "answer": example.row.answer,
                    "canonical_answer": canonical,
                }
            )

    return {
        "row_count": len(rows),
        "configs": list(configs),
        "parsed_total": len(examples),
        "parsed_rate": (len(examples) / len(rows)) if rows else 0.0,
        "config_counts": sorted_counter(config_counts),
        "family_counts": sorted_counter(family_counts),
        "subject_counts": sorted_counter(subject_counts),
        "language_counts": sorted_counter(language_counts),
        "task_variant_counts": sorted_counter(task_variant_counts),
        "answer_kind_counts": sorted_counter(answer_kind_counts),
        "failure_counts": dict(sorted(failure_counts.items())),
        "trajectory_records": len(trajectories),
        "canonical_consistency_failure_count": len(consistency_failures),
        "canonical_consistency_failures": consistency_failures[:20],
        "sample_examples": dict(sample_examples),
        "notes": [
            "This parser treats OlymMATH as an eval-only benchmark surface and maps each row into a subject-specific Olympiad reasoning trace scaffold.",
            "Open-answer rows use a SymPy-backed LaTeX adapter for scalar expressions and interval answers.",
            "Lean rows are treated as formal-proof concordance tasks with bilingual informal text plus canonical Lean statement metadata.",
            "Current lexical focus extraction is strongest for English; Chinese rows still preserve subject, difficulty, and symbolic-answer metadata.",
        ],
    }


def main() -> None:
    args = parse_args()
    summary = evaluate_olympiad_math_parser(configs=args.configs, max_rows=args.max_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
