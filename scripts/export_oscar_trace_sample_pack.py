from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.mixed_reasoning_dataset import (  # noqa: E402
    build_oscar_graph_reasoning_examples,
    build_oscar_scope_reasoning_examples,
)


@dataclass(frozen=True)
class SampleSpec:
    sample_name: str
    benchmark: str
    family: str
    trace_step: str | None = None


SAMPLE_SPECS: tuple[SampleSpec, ...] = (
    SampleSpec("oscar_section_anchor", "oscar_scope_reasoning", "oscar_section_anchor"),
    SampleSpec("oscar_outline_next_heading", "oscar_scope_reasoning", "oscar_outline_next_heading"),
    SampleSpec("oscar_concept_tags", "oscar_scope_reasoning", "oscar_concept_tags"),
    SampleSpec("oscar_graph_relation", "oscar_graph_reasoning", "oscar_graph_relation"),
    SampleSpec("oscar_graph_neighbors", "oscar_graph_reasoning", "oscar_graph_neighbors"),
    SampleSpec("oscar_graph_path_completion", "oscar_graph_reasoning", "oscar_graph_path_completion"),
    SampleSpec("oscar_graph_grounding", "oscar_graph_reasoning", "oscar_graph_grounding"),
    SampleSpec("oscar_graph_executor_rollout", "oscar_graph_reasoning", "oscar_graph_executor_rollout"),
    SampleSpec(
        "oscar_graph_executor_trace_select",
        "oscar_graph_reasoning",
        "oscar_graph_executor_trace",
        trace_step="select_motif",
    ),
    SampleSpec(
        "oscar_graph_executor_trace_advance",
        "oscar_graph_reasoning",
        "oscar_graph_executor_trace",
        trace_step="advance_frontier",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a curated Oscar trace sample pack.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/oscar_trace_sample_pack",
        help="Directory for markdown and JSON outputs.",
    )
    parser.add_argument(
        "--oscar-scope-max-documents",
        type=int,
        default=4,
        help="Max Oscar documents to scan for scope reasoning samples.",
    )
    parser.add_argument(
        "--oscar-scope-max-examples",
        type=int,
        default=24,
        help="Max Oscar scope reasoning examples to build before sampling.",
    )
    parser.add_argument(
        "--oscar-graph-max-examples",
        type=int,
        default=48,
        help="Max Oscar graph reasoning examples to build before sampling.",
    )
    return parser.parse_args()


def parse_reasoning_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "REASONING_STATE_V1" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def parse_json_field(fields: dict[str, str], key: str) -> Any:
    value = fields.get(key)
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def excerpt_preview(context: Any, *, limit: int = 280) -> str:
    if isinstance(context, dict):
        for key in ("excerpt", "support_excerpt", "question", "doc_title"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                compact = " ".join(value.split())
                return compact[:limit] + ("..." if len(compact) > limit else "")
    return ""


def compact_payload(example: Any, *, sample_name: str) -> dict[str, Any]:
    fields = parse_reasoning_fields(example.text)
    context = parse_json_field(fields, "context")
    auxiliary_targets = example.auxiliary_targets or {}
    payload = {
        "sample_name": sample_name,
        "benchmark": example.benchmark,
        "family": str(auxiliary_targets.get("family", "")),
        "trajectory_id": example.trajectory_id,
        "step_index": example.step_index,
        "trace_step": example.trace_step,
        "record_type": fields.get("record_type"),
        "question": parse_json_field(fields, "question"),
        "target_answer": parse_json_field(fields, "target_answer"),
        "target_action": parse_json_field(fields, "target_action"),
        "candidate_bucket": fields.get("candidate_bucket"),
        "state_tokens": fields.get("state_tokens"),
        "excerpt_preview": excerpt_preview(context),
        "context": context,
        "auxiliary_targets": auxiliary_targets,
        "text": example.text,
    }
    return payload


def select_example(
    examples: list[Any],
    *,
    family: str,
    trace_step: str | None,
) -> Any:
    for example in examples:
        auxiliary_targets = example.auxiliary_targets or {}
        if str(auxiliary_targets.get("family", "")) != family:
            continue
        if trace_step is not None and example.trace_step != trace_step:
            continue
        return example
    raise ValueError(f"Could not find Oscar sample for family={family!r} trace_step={trace_step!r}")


def render_markdown(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Oscar Trace Sample Pack",
        "",
        "Curated examples from the current Oscar corpora and graph-reasoning builders.",
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"## {index}. {sample['sample_name']}",
                "",
                f"- Benchmark: `{sample['benchmark']}`",
                f"- Family: `{sample['family']}`",
                f"- Trace step: `{sample['trace_step']}`",
                f"- Record type: `{sample['record_type']}`",
                f"- Trajectory: `{sample['trajectory_id']}`",
                f"- Step index: `{sample['step_index']}`",
                f"- Candidate bucket: `{sample['candidate_bucket']}`",
                f"- State tokens: `{sample['state_tokens']}`",
            ]
        )
        if sample["excerpt_preview"]:
            lines.append(f"- Excerpt preview: {sample['excerpt_preview']}")
        lines.extend(
            [
                "",
                "```json",
                json.dumps(
                    {
                        "question": sample["question"],
                        "target_answer": sample["target_answer"],
                        "target_action": sample["target_action"],
                        "auxiliary_targets": sample["auxiliary_targets"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scope_examples = list(
        build_oscar_scope_reasoning_examples(
            auto_discover=True,
            max_documents=args.oscar_scope_max_documents,
            max_examples=args.oscar_scope_max_examples,
        )
    )
    graph_examples = list(
        build_oscar_graph_reasoning_examples(
            max_examples=args.oscar_graph_max_examples,
        )
    )

    selected_samples: list[dict[str, Any]] = []
    for spec in SAMPLE_SPECS:
        example_pool = scope_examples if spec.benchmark == "oscar_scope_reasoning" else graph_examples
        example = select_example(example_pool, family=spec.family, trace_step=spec.trace_step)
        selected_samples.append(compact_payload(example, sample_name=spec.sample_name))

    markdown = render_markdown(selected_samples)
    markdown_path = output_dir / "sample_pack.md"
    json_path = output_dir / "sample_pack.json"
    summary_path = output_dir / "summary.json"

    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(selected_samples, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "sample_count": len(selected_samples),
        "sample_names": [sample["sample_name"] for sample in selected_samples],
        "benchmarks": sorted({sample["benchmark"] for sample in selected_samples}),
        "families": [sample["family"] for sample in selected_samples],
        "output_markdown": str(markdown_path),
        "output_json": str(json_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
