from __future__ import annotations

"""Evaluate GSM8K family coverage and supervision-time parsing into canonical IR."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .analyze_gsm8k_template_fit import (
        ALL_GSM8K_SPLITS,
        ensure_gsm8k_files,
        final_binary_expression,
        load_rows,
        loose_match,
        strict_match,
    )
    from .evaluate_word_problem_translation_quality import solve_task, values_match
    from .gsm8k_reasoning_parser import (
        build_gsm8k_examples,
        compile_gsm8k_examples,
        parse_gsm8k_row,
    )
    from .recommend_gsm8k_arc_families import (
        COMPARE_RE,
        PARTITION_RE,
        RATE_RE,
        REMAIN_RE,
        TOTAL_RE,
        UNIT_RE,
        classify_family,
        final_op,
    )
except ImportError:  # pragma: no cover - direct script execution
    from analyze_gsm8k_template_fit import (  # type: ignore
        ALL_GSM8K_SPLITS,
        ensure_gsm8k_files,
        final_binary_expression,
        load_rows,
        loose_match,
        strict_match,
    )
    from evaluate_word_problem_translation_quality import solve_task, values_match  # type: ignore
    from gsm8k_reasoning_parser import (  # type: ignore
        build_gsm8k_examples,
        compile_gsm8k_examples,
        parse_gsm8k_row,
    )
    from recommend_gsm8k_arc_families import (  # type: ignore
        COMPARE_RE,
        PARTITION_RE,
        RATE_RE,
        REMAIN_RE,
        TOTAL_RE,
        UNIT_RE,
        classify_family,
        final_op,
    )


IMPLEMENTED_GSM_FAMILIES = {
    "compose_total",
    "compose_difference",
    "rate_scale",
    "partition_inverse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current GSM8K parser/classifier coverage.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="arc_trajectory_sampler/data/gsm8k",
        help="Directory where GSM8K JSONL files are stored.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/gsm8k_parser_eval_summary.json",
        help="Destination summary JSON path.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_GSM8K_SPLITS,
        default=ALL_GSM8K_SPLITS,
        help="GSM8K splits to audit. Defaults to the full official corpus.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of GSM8K rows to evaluate for a quick smoke run.",
    )
    return parser.parse_args()


def family_failure_reason(question: str, answer: str) -> str:
    op = final_op(answer)
    if op is None:
        return "no_final_tagged_expression"
    if op == "mix":
        return "mixed_final_ops"

    if op == "+":
        if TOTAL_RE.search(question):
            return "compose_total_not_bound"
        return "missing_total_cue"
    if op == "-":
        if REMAIN_RE.search(question) or COMPARE_RE.search(question):
            return "compose_difference_not_bound"
        return "missing_difference_cue"
    if op == "*":
        if RATE_RE.search(question):
            return "rate_scale_not_implemented"
        if UNIT_RE.search(question):
            return "unit_convert_not_implemented"
        return "missing_rate_or_unit_cue"
    if op == "/":
        if PARTITION_RE.search(question):
            return "partition_inverse_not_implemented"
        if UNIT_RE.search(question):
            return "unit_convert_not_implemented"
        return "missing_partition_or_unit_cue"
    return f"unsupported_final_op:{op}"


def final_answer_value(answer: str) -> str | None:
    if "####" not in answer:
        return None
    return answer.split("####", 1)[1].strip()


def add_example(bucket: Dict[str, List[Dict[str, Any]]], key: str, row: Dict[str, Any], *, extra: Dict[str, Any] | None = None) -> None:
    if len(bucket[key]) >= 5:
        return
    payload = {
        "split": row["split"],
        "index": row["index"],
        "question": row["question"],
    }
    if extra:
        payload.update(extra)
    bucket[key].append(payload)


def evaluate_gsm8k_parser(
    *,
    data_dir: str | Path,
    splits: List[str] | tuple[str, ...] = ALL_GSM8K_SPLITS,
    max_rows: int | None = None,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    paths = ensure_gsm8k_files(data_dir)
    rows = load_rows(paths, splits=splits, max_rows=max_rows)

    family_counts = Counter()
    implemented_counts = Counter()
    strict_counts = Counter()
    loose_counts = Counter()
    failure_counts = Counter()
    parser_family_counts = Counter()
    parser_failure_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    family_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failure_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        family = classify_family(row["question"], row["answer"])
        strict_family = strict_match(row["question"], row["answer"])
        loose_family = loose_match(row["question"], row["answer"])
        final_expr = final_binary_expression(row["answer"])
        final_value = final_answer_value(row["answer"])

        if family is not None:
            family_counts[family] += 1
            add_example(
                family_examples,
                family,
                row,
                extra={
                    "final_answer": final_value,
                    "final_expression": final_expr[3] if final_expr else None,
                },
            )
            if family in IMPLEMENTED_GSM_FAMILIES:
                implemented_counts[family] += 1
        else:
            reason = family_failure_reason(row["question"], row["answer"])
            failure_counts[reason] += 1
            add_example(
                failure_examples,
                reason,
                row,
                extra={
                    "final_answer": final_value,
                    "final_op": final_op(row["answer"]),
                },
            )

        if strict_family is not None:
            strict_counts[strict_family] += 1
        if loose_family is not None:
            loose_counts[loose_family] += 1

    total = len(rows)
    classified_total = sum(family_counts.values())
    implemented_total = sum(implemented_counts.values())
    strict_total = sum(strict_counts.values())
    loose_total = sum(loose_counts.values())

    parsed_examples, parser_failures = build_gsm8k_examples(
        data_dir=data_dir,
        splits=splits,
        allow_eval_splits=True,
        max_rows=max_rows,
    )
    parsed_trajectories = compile_gsm8k_examples(parsed_examples)
    parser_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    derivation_failures = []
    for example in parsed_examples:
        parser_family_counts[example.family_name] += 1
        add_example(
            parser_examples,
            example.family_name,
            {
                "split": example.metadata["gsm8k_split"],
                "index": example.metadata["gsm8k_index"],
                "question": example.source_text,
            },
            extra={
                "template_name": example.template_name,
                "answer": example.abstract_task.answer,
            },
        )
        try:
            solved = solve_task(example.abstract_task)
            if not values_match(solved, example.abstract_task.answer):
                derivation_failures.append(
                    {
                        "split": example.metadata["gsm8k_split"],
                        "index": example.metadata["gsm8k_index"],
                        "expected": example.abstract_task.answer,
                        "derived": solved,
                    }
                )
        except Exception as exc:  # pragma: no cover - summary should capture failures
            derivation_failures.append(
                {
                    "split": example.metadata["gsm8k_split"],
                    "index": example.metadata["gsm8k_index"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for reason, count in parser_failures.items():
        parser_failure_examples[reason] = []
    for row in rows:
        parsed_example, parsed_failure = parse_gsm8k_row(row)
        if parsed_example is None and parsed_failure is not None:
            add_example(
                parser_failure_examples,
                parsed_failure.reason,
                row,
                extra={
                    "final_answer": final_answer_value(row["answer"]),
                    "details": parsed_failure.details,
                },
            )

    return {
        "num_examples": total,
        "splits": list(splits),
        "family_classifier": {
            "covered_total": classified_total,
            "covered_rate": classified_total / total if total else 0.0,
            "family_counts": dict(family_counts),
            "examples": dict(family_examples),
        },
        "implemented_family_subset": {
            "families": sorted(IMPLEMENTED_GSM_FAMILIES),
            "covered_total": implemented_total,
            "covered_rate": implemented_total / total if total else 0.0,
            "family_counts": dict(implemented_counts),
        },
        "actual_parser": {
            "parsed_total": len(parsed_examples),
            "parsed_rate": len(parsed_examples) / total if total else 0.0,
            "family_counts": dict(parser_family_counts),
            "examples": dict(parser_examples),
            "failure_counts": parser_failures,
            "failure_examples": dict(parser_failure_examples),
            "trajectory_records": len(parsed_trajectories),
            "answer_derivation_failure_count": len(derivation_failures),
            "answer_derivation_failures": derivation_failures[:20],
        },
        "current_template_fit": {
            "strict_total": strict_total,
            "strict_rate": strict_total / total if total else 0.0,
            "strict_counts": dict(strict_counts),
            "loose_total": loose_total,
            "loose_rate": loose_total / total if total else 0.0,
            "loose_counts": dict(loose_counts),
        },
        "unclassified": {
            "total": total - classified_total,
            "rate": (total - classified_total) / total if total else 0.0,
            "failure_counts": dict(failure_counts),
            "examples": dict(failure_examples),
        },
        "notes": {
            "scope": "This evaluates three things: the heuristic family classifier, the supervision-time GSM8K-to-IR parser for compose_total/compose_difference/rate_scale/partition_inverse, and the legacy strict/loose template-fit checker.",
            "implemented_subset": "The parser and canonical trajectory compiler now support compose_total, compose_difference, rate_scale, and partition_inverse. unit_convert remains unsupported as a distinct family today.",
            "parser_mode": "The actual parser is still supervision-time, but it now uses worked-solution tags first and falls back to plain answer-text equations when tags are missing or end in alias-only restatements; it is not a question-only inference parser.",
            "benchmark_hygiene": "Audit coverage is measured on the requested GSM8K splits. Trainable exports should stay on train unless eval splits are explicitly enabled.",
        },
    }


def main() -> None:
    args = parse_args()
    summary = evaluate_gsm8k_parser(data_dir=args.data_dir, splits=args.splits, max_rows=args.max_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
