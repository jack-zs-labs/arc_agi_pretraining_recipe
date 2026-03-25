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

from arc_trajectory_sampler.mixed_reasoning_dataset import build_oscar_graph_reasoning_examples
from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical Oscar graph reasoning tasks as REASONING_STATE_V1 JSONL."
    )
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=OSCAR_GRAPH_REASONING_FAMILIES,
        default=list(OSCAR_GRAPH_REASONING_FAMILIES),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/oscar_graph_reasoning_corpus/traces.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/oscar_graph_reasoning_corpus/summary.json",
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
    examples = build_oscar_graph_reasoning_examples(
        max_examples=args.max_examples,
        families=args.families,
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

    task_kind_counts = Counter(str((example.auxiliary_targets or {}).get("family", "unknown")) for example in examples)
    summary = {
        "example_count": len(examples),
        "trajectory_count": len({example.trajectory_id for example in examples}),
        "benchmark_counts": dict(sorted(Counter(example.benchmark for example in examples).items())),
        "family_counts": dict(sorted(task_kind_counts.items())),
        "requested": {
            "max_examples": args.max_examples,
            "families": list(args.families),
        },
        "output": str(output_path),
        "notes": [
            "These are canonical Oscar graph reasoning tasks serialized as REASONING_STATE_V1 terminal-answer and decision-action records.",
            "They tie the formal process graph, recursive abstraction rules, approval claim DAG, and executor-style frontier updates into one benchmark-facing corpus.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
