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
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_WORKFLOW_REASONING_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export workflow-focused Oscar reasoning traces from business case-study and workflow documents."
    )
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=OSCAR_WORKFLOW_REASONING_FAMILIES,
        default=list(OSCAR_WORKFLOW_REASONING_FAMILIES),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/oscar_workflow_reasoning_corpus/traces.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/oscar_workflow_reasoning_corpus/summary.json",
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
        auto_discover=True,
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

    family_counts = Counter(str((example.auxiliary_targets or {}).get("family", "unknown")) for example in examples)
    trace_step_counts = Counter(str(example.trace_step) for example in examples)
    environment_counts = Counter(
        str((example.auxiliary_targets or {}).get("workflow_environment", "unknown"))
        for example in examples
    )
    doc_counts = Counter(
        str((example.auxiliary_targets or {}).get("doc_id", "unknown"))
        for example in examples
    )
    reward_scores = [
        float((example.auxiliary_targets or {}).get("workflow_reward_score", 0.0) or 0.0)
        for example in examples
        if (example.auxiliary_targets or {}).get("family") == "oscar_workflow_intervention_trace"
    ]
    notes = [
        "These traces focus on real business workflow environments rather than on Oscar-self-descriptive specification text.",
        "They include workflow environment anchoring, KPI tags, bottleneck tags, improvement tags, KPI-to-improvement supervision, and multi-step intervention sequencing with projected reward labels.",
    ]
    if len(examples) < int(args.max_examples) and "oscar_workflow_intervention_trace" in args.families:
        notes.append(
            "The export preserves complete intervention trajectories under the cap, so the final example count may be slightly below the requested max_examples."
        )
    if "oscar_workflow_case_analogy" in args.families or "oscar_workflow_transfer" in args.families:
        notes.append(
            "The workflow export now includes cross-case abstraction tasks that pair different business environments through shared workflow motifs and intervention transfer."
        )
    summary = {
        "example_count": len(examples),
        "trajectory_count": len({example.trajectory_id for example in examples}),
        "family_counts": dict(sorted(family_counts.items())),
        "trace_step_counts": dict(sorted(trace_step_counts.items())),
        "environment_counts": dict(sorted(environment_counts.items())),
        "document_counts": dict(sorted(doc_counts.items())),
        "intervention_reward_stats": {
            "count": len(reward_scores),
            "mean": (sum(reward_scores) / len(reward_scores)) if reward_scores else 0.0,
            "min": min(reward_scores) if reward_scores else 0.0,
            "max": max(reward_scores) if reward_scores else 0.0,
        },
        "requested": {
            "max_examples": args.max_examples,
            "families": list(args.families),
        },
        "output": str(output_path),
        "notes": notes,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
