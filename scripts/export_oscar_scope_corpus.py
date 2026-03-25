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

from arc_trajectory_sampler.mixed_reasoning_dataset import build_oscar_scope_examples
from arc_trajectory_sampler.oscar_scope_corpus import (
    DEFAULT_OSCAR_SCOPE_VIEWS,
    OSCAR_SCOPE_VIEW_CHOICES,
    discover_oscar_scope_roots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Oscar process-intelligence spec and meeting notes as a native pretraining corpus."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[],
        help="Optional root directories or Oscar docs root. Defaults to auto-discovery if available.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[],
        help="Optional explicit files to include in addition to discovered roots.",
    )
    parser.add_argument(
        "--auto-discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discover a sibling oscar_design_docs workspace when available.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on documents after source deduplication.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on exported text records.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=OSCAR_SCOPE_VIEW_CHOICES,
        default=list(DEFAULT_OSCAR_SCOPE_VIEWS),
        help="Document views to export.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/oscar_scope_corpus/traces.jsonl",
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/oscar_scope_corpus/summary.json",
        help="Destination JSON summary path.",
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
    examples = build_oscar_scope_examples(
        roots=args.roots,
        paths=args.paths,
        auto_discover=args.auto_discover,
        max_documents=args.max_documents,
        max_chunks=args.max_chunks,
        views=args.views,
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

    benchmark_counts = Counter(example.benchmark for example in examples)
    view_counts = Counter(example.trace_step for example in examples)
    group_counts = Counter(
        str((example.auxiliary_targets or {}).get("doc_group", "unknown"))
        for example in examples
    )
    title_counts = Counter(
        str((example.auxiliary_targets or {}).get("doc_title", "unknown"))
        for example in examples
    )
    summary = {
        "example_count": len(examples),
        "document_count": len({example.trajectory_id for example in examples}),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "view_counts": dict(sorted(view_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "title_counts": dict(sorted(title_counts.items())),
        "requested": {
            "roots": list(args.roots),
            "paths": list(args.paths),
            "auto_discover": args.auto_discover,
            "max_documents": args.max_documents,
            "max_chunks": args.max_chunks,
            "views": list(args.views),
            "discovered_roots": [str(path) for path in discover_oscar_scope_roots()] if args.auto_discover else [],
        },
        "output": str(output_path),
        "notes": [
            "This export is intended for native domain pretraining within the Oscar process-intelligence scope.",
            "It is intentionally separate from the structured REASONING_STATE_V1 Ptraj export path.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
