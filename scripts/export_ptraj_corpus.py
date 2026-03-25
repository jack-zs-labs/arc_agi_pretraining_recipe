from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.core_loader import DEFAULT_CORE_DATA_DIR, DEPENDENCY_KIND_CHOICES
from arc_trajectory_sampler.mixed_reasoning_dataset import (
    DEFAULT_GSM8K_DATA_DIR,
    DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS,
    ReasoningTextExample,
    build_ptraj_examples,
)
from arc_trajectory_sampler.mmlu_parser import DEFAULT_DATA_DIR as DEFAULT_MMLU_DATA_DIR
from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES
from arc_trajectory_sampler.olympiad_math_parser import OLYMPIAD_MATH_SUPPORTED_CONFIGS
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a mixed structured-reasoning pretraining corpus as JSONL."
    )
    parser.add_argument("--arc-episodes", type=int, default=16, help="Number of ARC episodes to expand into traces.")
    parser.add_argument(
        "--arc-seed-start",
        type=int,
        default=0,
        help="Starting seed for ARC episode generation.",
    )
    parser.add_argument(
        "--gsm8k-data-dir",
        type=str,
        default=DEFAULT_GSM8K_DATA_DIR,
        help="Local GSM8K cache directory.",
    )
    parser.add_argument(
        "--gsm8k-max-rows",
        type=int,
        default=48,
        help="Maximum GSM8K train rows to include.",
    )
    parser.add_argument(
        "--mmlu-data-dir",
        type=str,
        default=DEFAULT_MMLU_DATA_DIR,
        help="Local MMLU cache directory.",
    )
    parser.add_argument(
        "--mmlu-max-rows",
        type=int,
        default=48,
        help="Maximum benchmark-safe MMLU auxiliary_train rows to include.",
    )
    parser.add_argument(
        "--olympiad-math-configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=list(DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS),
        help="OlymMATH configs to include. Defaults to open-answer English plus lean.",
    )
    parser.add_argument(
        "--olympiad-math-max-rows",
        type=int,
        default=48,
        help="Maximum OlymMATH rows to include before trajectory expansion.",
    )
    parser.add_argument(
        "--core-data-dir",
        type=str,
        default=DEFAULT_CORE_DATA_DIR,
        help="Local CoRe cache directory.",
    )
    parser.add_argument(
        "--core-max-rows",
        type=int,
        default=48,
        help="Maximum CoRe query-level tasks to include.",
    )
    parser.add_argument(
        "--core-graph-backend",
        choices=("auto", "heuristic", "python_ast"),
        default="auto",
        help="CoRe graph extraction backend.",
    )
    parser.add_argument(
        "--core-languages",
        nargs="+",
        default=None,
        help="Optional CoRe language filter.",
    )
    parser.add_argument(
        "--core-categories",
        nargs="+",
        default=None,
        help="Optional CoRe category filter.",
    )
    parser.add_argument(
        "--core-dependency-kinds",
        nargs="+",
        choices=DEPENDENCY_KIND_CHOICES,
        default=None,
        help="Optional CoRe dependency-kind filter.",
    )
    parser.add_argument(
        "--oscar-scope-auto-discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discover a sibling oscar_design_docs workspace when available.",
    )
    parser.add_argument("--oscar-scope-roots", nargs="+", default=[])
    parser.add_argument("--oscar-scope-paths", nargs="+", default=[])
    parser.add_argument(
        "--oscar-scope-max-documents",
        type=int,
        default=0,
        help="Maximum Oscar documents to inspect when deriving structured reasoning tasks.",
    )
    parser.add_argument(
        "--oscar-scope-reasoning-max-examples",
        type=int,
        default=96,
        help="Maximum structured Oscar reasoning traces to include.",
    )
    parser.add_argument(
        "--oscar-scope-reasoning-families",
        nargs="+",
        choices=OSCAR_SCOPE_REASONING_FAMILIES,
        default=list(OSCAR_SCOPE_REASONING_FAMILIES),
        help="Oscar reasoning-task families to include when max-examples is positive.",
    )
    parser.add_argument(
        "--oscar-graph-reasoning-max-examples",
        type=int,
        default=64,
        help="Maximum canonical Oscar graph reasoning traces to include.",
    )
    parser.add_argument(
        "--oscar-graph-reasoning-families",
        nargs="+",
        choices=OSCAR_GRAPH_REASONING_FAMILIES,
        default=list(OSCAR_GRAPH_REASONING_FAMILIES),
        help="Oscar graph reasoning-task families to include when max-examples is positive.",
    )
    parser.add_argument(
        "--include-verifier-targets",
        action="store_true",
        help="Include verifier-side targets in LM text where supported.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional deterministic shuffle applied after source mixing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ptraj_corpus/traces.jsonl",
        help="Destination JSONL path for LM-facing trajectory traces.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/ptraj_corpus/summary.json",
        help="Destination JSON path for export summary metadata.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def trajectory_counts_by_benchmark(examples: tuple[ReasoningTextExample, ...]) -> dict[str, int]:
    ids_by_benchmark: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        ids_by_benchmark[example.benchmark].add(example.trajectory_id)
    return {
        benchmark: len(trajectory_ids)
        for benchmark, trajectory_ids in sorted(ids_by_benchmark.items())
    }


def main() -> None:
    args = parse_args()
    examples = build_ptraj_examples(
        arc_episodes=args.arc_episodes,
        arc_seed_start=args.arc_seed_start,
        gsm8k_data_dir=args.gsm8k_data_dir,
        gsm8k_max_rows=args.gsm8k_max_rows,
        mmlu_data_dir=args.mmlu_data_dir,
        mmlu_max_rows=args.mmlu_max_rows,
        olympiad_math_configs=args.olympiad_math_configs,
        olympiad_math_max_rows=args.olympiad_math_max_rows,
        core_data_dir=args.core_data_dir,
        core_max_rows=args.core_max_rows,
        core_languages=args.core_languages,
        core_categories=args.core_categories,
        core_dependency_kinds=args.core_dependency_kinds,
        core_graph_backend=args.core_graph_backend,
        oscar_scope_roots=args.oscar_scope_roots,
        oscar_scope_paths=args.oscar_scope_paths,
        oscar_scope_auto_discover=args.oscar_scope_auto_discover,
        oscar_scope_max_documents=args.oscar_scope_max_documents,
        oscar_scope_reasoning_max_examples=args.oscar_scope_reasoning_max_examples,
        oscar_scope_reasoning_families=args.oscar_scope_reasoning_families,
        oscar_graph_reasoning_max_examples=args.oscar_graph_reasoning_max_examples,
        oscar_graph_reasoning_families=args.oscar_graph_reasoning_families,
        include_verifier_targets=args.include_verifier_targets,
        shuffle_seed=args.shuffle_seed,
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
    trace_step_counts = Counter(example.trace_step for example in examples)
    source_band_counts = Counter(
        "olympiad_math"
        if example.benchmark.startswith("olympiad_math_")
        else example.benchmark
        for example in examples
    )
    summary = {
        "example_count": len(examples),
        "examples_with_auxiliary_targets": sum(1 for example in examples if example.auxiliary_targets),
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "source_band_counts": dict(sorted(source_band_counts.items())),
        "trajectory_counts": trajectory_counts_by_benchmark(examples),
        "trace_step_counts": dict(sorted(trace_step_counts.items())),
        "requested_sources": {
            "arc_episodes": args.arc_episodes,
            "arc_seed_start": args.arc_seed_start,
            "gsm8k_max_rows": args.gsm8k_max_rows,
            "mmlu_max_rows": args.mmlu_max_rows,
            "olympiad_math_configs": list(args.olympiad_math_configs),
            "olympiad_math_max_rows": args.olympiad_math_max_rows,
            "core_max_rows": args.core_max_rows,
            "core_graph_backend": args.core_graph_backend,
            "core_languages": list(args.core_languages) if args.core_languages is not None else None,
            "core_categories": list(args.core_categories) if args.core_categories is not None else None,
            "core_dependency_kinds": (
                list(args.core_dependency_kinds) if args.core_dependency_kinds is not None else None
            ),
            "oscar_scope_auto_discover": args.oscar_scope_auto_discover,
            "oscar_scope_roots": list(args.oscar_scope_roots),
            "oscar_scope_paths": list(args.oscar_scope_paths),
            "oscar_scope_max_documents": args.oscar_scope_max_documents,
            "oscar_scope_reasoning_max_examples": args.oscar_scope_reasoning_max_examples,
            "oscar_scope_reasoning_families": list(args.oscar_scope_reasoning_families),
            "oscar_graph_reasoning_max_examples": args.oscar_graph_reasoning_max_examples,
            "oscar_graph_reasoning_families": list(args.oscar_graph_reasoning_families),
            "include_verifier_targets": args.include_verifier_targets,
            "shuffle_seed": args.shuffle_seed,
        },
        "output": str(output_path),
        "notes": [
            "This export is intended for structured reasoning pretraining rather than official benchmark evaluation.",
            "MMLU rows are drawn from the benchmark-safe auxiliary_train split only.",
            "OlymMATH rows are split into olympiad_math_open and olympiad_math_lean trace bands.",
            "Oscar-scope reasoning traces are optional and remain separate from the native Oscar domain-text corpus.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
