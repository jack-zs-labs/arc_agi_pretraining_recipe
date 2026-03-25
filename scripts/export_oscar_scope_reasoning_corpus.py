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

from arc_trajectory_sampler.mixed_reasoning_dataset import build_oscar_scope_reasoning_examples
from arc_trajectory_sampler.oscar_scope_corpus import DEFAULT_OSCAR_SCOPE_VIEWS, OSCAR_SCOPE_VIEW_CHOICES
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export doc-grounded Oscar scope reasoning tasks as REASONING_STATE_V1 JSONL."
    )
    parser.add_argument("--roots", nargs="+", default=[])
    parser.add_argument("--paths", nargs="+", default=[])
    parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-documents", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=OSCAR_SCOPE_VIEW_CHOICES,
        default=list(DEFAULT_OSCAR_SCOPE_VIEWS),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=OSCAR_SCOPE_REASONING_FAMILIES,
        default=list(OSCAR_SCOPE_REASONING_FAMILIES),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/oscar_scope_reasoning_corpus/traces.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/oscar_scope_reasoning_corpus/summary.json",
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
    examples = build_oscar_scope_reasoning_examples(
        roots=args.roots,
        paths=args.paths,
        auto_discover=args.auto_discover,
        max_documents=args.max_documents,
        max_examples=args.max_examples,
        views=args.views,
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

    task_kind_counts = Counter(str((example.auxiliary_targets or {}).get("task_kind", "unknown")) for example in examples)
    family_counts = Counter()
    for example in examples:
        for raw_line in example.text.splitlines():
            if raw_line.startswith("family="):
                family_counts[raw_line.split("=", 1)[1]] += 1
                break
    summary = {
        "example_count": len(examples),
        "trajectory_count": len({example.trajectory_id for example in examples}),
        "benchmark_counts": dict(sorted(Counter(example.benchmark for example in examples).items())),
        "task_kind_counts": dict(sorted(task_kind_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "requested": {
            "roots": list(args.roots),
            "paths": list(args.paths),
            "auto_discover": args.auto_discover,
            "max_documents": args.max_documents,
            "max_examples": args.max_examples,
            "views": list(args.views),
            "families": list(args.families),
        },
        "output": str(output_path),
        "notes": [
            "These are Oscar-scope reasoning tasks serialized as REASONING_STATE_V1 terminal-answer records.",
            "They are intended to complement the native Oscar scope corpus, not replace it.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
