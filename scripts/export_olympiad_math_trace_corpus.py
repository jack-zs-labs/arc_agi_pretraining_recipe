from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.mixed_reasoning_dataset import serialize_trajectory_records
from arc_trajectory_sampler.olympiad_math_parser import (
    OLYMPIAD_MATH_SUPPORTED_CONFIGS,
    build_olympiad_math_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OlymMATH trajectory traces as a pretraining JSONL corpus."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=list(OLYMPIAD_MATH_SUPPORTED_CONFIGS),
        help="OlymMATH configs to export into the trace corpus.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on total OlymMATH rows before trajectory expansion.",
    )
    parser.add_argument(
        "--include-verifier-targets",
        action="store_true",
        help="Include verifier-side targets in the serialized LM text.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/olympiad_math_trace_corpus/traces.jsonl",
        help="Destination JSONL path for LM-facing trajectory traces.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/olympiad_math_trace_corpus/summary.json",
        help="Destination JSON path for export summary metadata.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    trajectories, failures = build_olympiad_math_trajectories(
        configs=args.configs,
        allow_eval_configs=True,
        max_rows=args.max_rows,
    )
    examples = serialize_trajectory_records(
        trajectories,
        include_verifier_targets=args.include_verifier_targets,
    )

    payload_rows = [
        {
            "benchmark": example.benchmark,
            "trajectory_id": example.trajectory_id,
            "step_index": example.step_index,
            "trace_step": example.trace_step,
            "text": example.text,
            "auxiliary_targets": example.auxiliary_targets,
        }
        for example in examples
    ]
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    write_jsonl(output_path, payload_rows)

    config_counts = Counter(str(record.episode_metadata.get("config", "unknown")) for record in trajectories)
    task_variant_counts = Counter(str(record.episode_metadata.get("task_variant", "unknown")) for record in trajectories)
    benchmark_counts = Counter(example.benchmark for example in examples)
    family_counts = Counter(record.family for record in trajectories)
    summary = {
        "configs": list(args.configs),
        "max_rows": args.max_rows,
        "include_verifier_targets": args.include_verifier_targets,
        "trajectory_record_count": len(trajectories),
        "text_example_count": len(examples),
        "failure_counts": failures,
        "config_counts": dict(sorted(config_counts.items())),
        "task_variant_counts": dict(sorted(task_variant_counts.items())),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "output": str(output_path),
        "notes": [
            "This export is intended for structured reasoning pretraining rather than official benchmark evaluation.",
            "OlymMATH open-answer rows are labeled as olympiad_math_open; Lean theorem/proof rows are labeled as olympiad_math_lean.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
