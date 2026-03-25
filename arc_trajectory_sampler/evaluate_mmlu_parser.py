from __future__ import annotations

"""Evaluate MMLU parser coverage and canonical trajectory compilation."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from .mmlu_parser import (
        ALL_MMLU_SPLITS,
        DEFAULT_MMLU_AUDIT_SPLITS,
        build_mmlu_examples,
        compile_mmlu_examples,
        load_mmlu_rows,
    )
except ImportError:  # pragma: no cover - direct script execution
    from mmlu_parser import (  # type: ignore
        ALL_MMLU_SPLITS,
        DEFAULT_MMLU_AUDIT_SPLITS,
        build_mmlu_examples,
        compile_mmlu_examples,
        load_mmlu_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MMLU parser coverage and trajectory consistency.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="arc_trajectory_sampler/data/mmlu",
        help="Directory containing official MMLU CSV folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_MMLU_SPLITS,
        default=DEFAULT_MMLU_AUDIT_SPLITS,
        help="MMLU splits to evaluate. Defaults to the official benchmark splits.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/mmlu_parser_eval_summary.json",
        help="Path to write the JSON summary.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of MMLU rows to evaluate for a quick smoke run.",
    )
    return parser.parse_args()


def sorted_dict(counter: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def evaluate_mmlu_parser(
    *,
    data_dir: str | Path,
    splits: Sequence[str],
    max_rows: int | None = None,
) -> Dict[str, Any]:
    rows = load_mmlu_rows(data_dir, splits=splits, max_rows=max_rows)
    examples, failure_counts = build_mmlu_examples(
        data_dir=data_dir,
        splits=splits,
        allow_eval_splits=True,
        max_rows=max_rows,
    )
    trajectories = compile_mmlu_examples(examples)

    family_counts = Counter(example.family_name for example in examples)
    split_counts = Counter(example.row.split for example in examples)
    subject_counts = Counter(example.row.subject for example in examples)

    consistency_failures = []
    for record in trajectories:
        output_state = record.output_state
        correct_choice = output_state.get("correct_choice")
        answer = output_state.get("answer")
        final_step_choice = record.steps[-2].workspace_state.get("selected_choice_id") if len(record.steps) >= 2 else None
        if correct_choice != answer or final_step_choice != correct_choice:
            consistency_failures.append(
                {
                    "trajectory_id": record.trajectory_id,
                    "family": record.family,
                    "correct_choice": correct_choice,
                    "answer": answer,
                    "selected_choice_id": final_step_choice,
                }
            )

    sample_examples = {}
    for family in sorted(family_counts):
        sample_examples[family] = [
            {
                "split": example.row.split,
                "subject": example.row.subject,
                "index": example.row.index,
                "question": example.row.question,
                "answer": example.abstract_task.answer,
            }
            for example in examples
            if example.family_name == family
        ][:5]

    return {
        "row_count": len(rows),
        "splits": list(splits),
        "parsed_total": len(examples),
        "parsed_rate": (len(examples) / len(rows)) if rows else 0.0,
        "family_counts": sorted_dict(family_counts),
        "split_counts": sorted_dict(split_counts),
        "subject_counts": sorted_dict(subject_counts),
        "failure_counts": dict(sorted(failure_counts.items())),
        "trajectory_records": len(trajectories),
        "multiple_choice_consistency_failure_count": len(consistency_failures),
        "multiple_choice_consistency_failures": consistency_failures[:20],
        "sample_examples": sample_examples,
        "notes": [
            "This parser currently uses ARC-style multiple-choice families for quantitative, factual, case-application, passage-reference, negation-exception, completion, rule-application, statement-evaluation, descriptor-match, comparative-inference, and concept-identification rows.",
            "Coverage is measured on the requested MMLU CSV splits using subject and lexical heuristics.",
            "Trainable exports should default to auxiliary_train; official dev/val/test rows are benchmark material and require explicit opt-in in the parser/export path.",
            "The former broad residual MMLU bucket is now split into structural subfamilies instead of one general elimination label.",
            "Trajectory consistency checks only verify canonical IR/choice alignment, not independent question solving.",
        ],
    }


def main() -> None:
    args = parse_args()
    summary = evaluate_mmlu_parser(data_dir=args.data_dir, splits=args.splits, max_rows=args.max_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
