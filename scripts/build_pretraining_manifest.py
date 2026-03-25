from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.core_loader import DEFAULT_CORE_DATA_DIR, DEPENDENCY_KIND_CHOICES
from arc_trajectory_sampler.mixed_reasoning_dataset import DEFAULT_GSM8K_DATA_DIR, DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS
from arc_trajectory_sampler.dclm_corpus import DEFAULT_DCLM_DATASET_ID, DEFAULT_DCLM_SPLIT, DEFAULT_DCLM_TEXT_FIELD
from arc_trajectory_sampler.mmlu_parser import DEFAULT_MMLU_AUDIT_SPLITS
from arc_trajectory_sampler.mmlu_parser import DEFAULT_DATA_DIR as DEFAULT_MMLU_DATA_DIR
from arc_trajectory_sampler.olympiad_math_parser import OLYMPIAD_MATH_EVAL_CONFIGS, OLYMPIAD_MATH_SUPPORTED_CONFIGS
from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES
from arc_trajectory_sampler.oscar_scope_corpus import DEFAULT_OSCAR_SCOPE_VIEWS, OSCAR_SCOPE_VIEW_CHOICES, discover_oscar_scope_roots
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES
from training.corpus_manifest import write_pretraining_manifest
from training.corpus_sources import (
    build_benchmark_training_pretraining_documents,
    build_dclm_pretraining_documents,
    build_gsm8k_benchmark_documents,
    build_mmlu_benchmark_documents,
    build_mmlu_pro_benchmark_documents,
    build_mmlu_redux_benchmark_documents,
    build_olympiad_math_benchmark_documents,
    build_oscar_graph_reasoning_pretraining_documents,
    build_oscar_scope_pretraining_documents,
    build_oscar_scope_reasoning_pretraining_documents,
    build_ptraj_pretraining_documents,
    repeat_documents_on_train_split,
)


TECH_REASONING_PRESET_SCALES: dict[str, dict[str, int]] = {
    "smoke": {
        "arc_episodes": 8,
        "gsm8k_max_rows": 16,
        "mmlu_max_rows": 16,
        "olympiad_math_max_rows": 16,
        "core_max_rows": 8,
        "oscar_scope_max_documents": 8,
        "oscar_scope_max_chunks": 64,
        "oscar_scope_reasoning_max_examples": 32,
        "oscar_graph_reasoning_max_examples": 32,
        "gsm8k_bench_max_rows": 8,
        "mmlu_bench_max_rows": 8,
        "mmlu_pro_max_rows": 8,
        "mmlu_redux_max_rows": 8,
        "olympiad_math_bench_max_rows": 8,
    },
    "pilot": {
        "arc_episodes": 64,
        "gsm8k_max_rows": 256,
        "mmlu_max_rows": 256,
        "olympiad_math_max_rows": 128,
        "core_max_rows": 256,
        "oscar_scope_max_documents": 128,
        "oscar_scope_max_chunks": 1024,
        "oscar_scope_reasoning_max_examples": 512,
        "oscar_graph_reasoning_max_examples": 512,
        "gsm8k_bench_max_rows": 128,
        "mmlu_bench_max_rows": 128,
        "mmlu_pro_max_rows": 128,
        "mmlu_redux_max_rows": 128,
        "olympiad_math_bench_max_rows": 128,
    },
    "large": {
        "arc_episodes": 256,
        "gsm8k_max_rows": 1024,
        "mmlu_max_rows": 1024,
        "olympiad_math_max_rows": 512,
        "core_max_rows": 1024,
        "oscar_scope_max_documents": 512,
        "oscar_scope_max_chunks": 4096,
        "oscar_scope_reasoning_max_examples": 2048,
        "oscar_graph_reasoning_max_examples": 2048,
        "gsm8k_bench_max_rows": 256,
        "mmlu_bench_max_rows": 256,
        "mmlu_pro_max_rows": 256,
        "mmlu_redux_max_rows": 256,
        "olympiad_math_bench_max_rows": 256,
    },
}


def _fill_nonpositive(args: argparse.Namespace, name: str, value: int) -> None:
    if int(getattr(args, name)) <= 0:
        setattr(args, name, value)


def apply_manifest_preset(args: argparse.Namespace) -> None:
    if args.preset == "manual":
        return
    if args.preset != "tech_reasoning":
        raise ValueError(f"Unsupported manifest preset {args.preset!r}")

    scale = TECH_REASONING_PRESET_SCALES[args.preset_scale]
    args.include_benchmark_train_pgen = True
    args.include_ptraj = True
    args.include_oscar_scope = True
    args.include_oscar_scope_reasoning = True
    args.include_oscar_graph_reasoning = True
    args.include_gsm8k_bench = True
    args.include_mmlu_bench = True
    args.include_mmlu_pro_bench = True
    args.include_mmlu_redux_bench = True
    args.include_olympiad_math_bench = True

    for field_name, field_value in scale.items():
        _fill_nonpositive(args, field_name, field_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified corpus manifest across native text (Pgen), reasoning traces (Ptraj), and held-out benchmark probes (Pbench)."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", choices=("manual", "tech_reasoning"), default="manual")
    parser.add_argument("--preset-scale", choices=tuple(TECH_REASONING_PRESET_SCALES), default="pilot")

    parser.add_argument("--include-dclm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dclm-dataset-id", type=str, default=DEFAULT_DCLM_DATASET_ID)
    parser.add_argument("--dclm-split", type=str, default=DEFAULT_DCLM_SPLIT)
    parser.add_argument("--dclm-text-field", type=str, default=DEFAULT_DCLM_TEXT_FIELD)
    parser.add_argument("--dclm-max-documents", type=int, default=0)
    parser.add_argument("--dclm-shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dclm-shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--dclm-min-text-chars", type=int, default=0)
    parser.add_argument("--dclm-min-language-score", type=float, default=0.0)
    parser.add_argument("--dclm-min-fasttext-score", type=float, default=0.0)
    parser.add_argument("--dclm-language-allowlist", nargs="+", default=["en"])

    parser.add_argument("--include-benchmark-train-pgen", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--benchmark-train-repeat", type=int, default=1)

    parser.add_argument("--include-ptraj", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--arc-episodes", type=int, default=0)
    parser.add_argument("--arc-seed-start", type=int, default=0)
    parser.add_argument("--gsm8k-data-dir", type=str, default=DEFAULT_GSM8K_DATA_DIR)
    parser.add_argument("--gsm8k-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-data-dir", type=str, default=DEFAULT_MMLU_DATA_DIR)
    parser.add_argument("--mmlu-max-rows", type=int, default=0)
    parser.add_argument(
        "--olympiad-math-configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=list(DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS),
    )
    parser.add_argument("--olympiad-math-max-rows", type=int, default=0)
    parser.add_argument("--core-data-dir", type=str, default=DEFAULT_CORE_DATA_DIR)
    parser.add_argument("--core-max-rows", type=int, default=0)
    parser.add_argument("--core-graph-backend", choices=("auto", "heuristic", "python_ast"), default="auto")
    parser.add_argument("--core-languages", nargs="+", default=None)
    parser.add_argument("--core-categories", nargs="+", default=None)
    parser.add_argument(
        "--core-dependency-kinds",
        nargs="+",
        choices=DEPENDENCY_KIND_CHOICES,
        default=None,
    )
    parser.add_argument("--ptraj-repeat", type=int, default=1)
    parser.add_argument("--include-verifier-targets", action="store_true")

    parser.add_argument("--include-oscar-scope", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-oscar-scope-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-oscar-graph-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oscar-scope-roots", nargs="+", default=[])
    parser.add_argument("--oscar-scope-paths", nargs="+", default=[])
    parser.add_argument("--oscar-scope-auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-max-documents", type=int, default=0)
    parser.add_argument("--oscar-scope-max-chunks", type=int, default=0)
    parser.add_argument(
        "--oscar-scope-views",
        nargs="+",
        choices=OSCAR_SCOPE_VIEW_CHOICES,
        default=list(DEFAULT_OSCAR_SCOPE_VIEWS),
    )
    parser.add_argument("--oscar-scope-reasoning-max-examples", type=int, default=0)
    parser.add_argument(
        "--oscar-scope-reasoning-families",
        nargs="+",
        choices=OSCAR_SCOPE_REASONING_FAMILIES,
        default=list(OSCAR_SCOPE_REASONING_FAMILIES),
    )
    parser.add_argument("--oscar-graph-reasoning-max-examples", type=int, default=0)
    parser.add_argument(
        "--oscar-graph-reasoning-families",
        nargs="+",
        choices=OSCAR_GRAPH_REASONING_FAMILIES,
        default=list(OSCAR_GRAPH_REASONING_FAMILIES),
    )
    parser.add_argument("--oscar-native-repeat", type=int, default=1)
    parser.add_argument("--oscar-reasoning-repeat", type=int, default=1)
    parser.add_argument("--oscar-graph-repeat", type=int, default=1)
    parser.add_argument("--include-mmlu-pro-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mmlu-pro-max-rows", type=int, default=0)
    parser.add_argument("--include-mmlu-redux-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mmlu-redux-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-label-mode", type=str, default="corrected_single")
    parser.add_argument("--include-gsm8k-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gsm8k-bench-max-rows", type=int, default=0)
    parser.add_argument("--include-mmlu-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mmlu-bench-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-bench-splits", nargs="+", default=list(DEFAULT_MMLU_AUDIT_SPLITS))
    parser.add_argument("--include-olympiad-math-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--olympiad-math-bench-configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=list(OLYMPIAD_MATH_EVAL_CONFIGS),
    )
    parser.add_argument("--olympiad-math-bench-max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_manifest_preset(args)
    documents = []
    build_summary: dict[str, dict[str, object]] = {}

    if args.include_dclm:
        if args.dclm_max_documents <= 0:
            raise SystemExit("Pass a positive --dclm-max-documents when --include-dclm is enabled.")
        dclm_documents = build_dclm_pretraining_documents(
            dataset_id=args.dclm_dataset_id,
            split=args.dclm_split,
            text_field=args.dclm_text_field,
            max_documents=args.dclm_max_documents,
            shuffle=args.dclm_shuffle,
            shuffle_buffer_size=args.dclm_shuffle_buffer_size,
            seed=args.seed,
            min_text_chars=args.dclm_min_text_chars,
            min_language_score=args.dclm_min_language_score,
            min_fasttext_score=args.dclm_min_fasttext_score,
            language_allowlist=tuple(args.dclm_language_allowlist),
        )
        documents.extend(dclm_documents)
        build_summary["dclm"] = {
            "base_document_count": len(dclm_documents),
            "manifest_document_count": len(dclm_documents),
            "band": "pgen",
        }

    if args.include_benchmark_train_pgen:
        benchmark_train_documents = build_benchmark_training_pretraining_documents(
            gsm8k_data_dir=args.gsm8k_data_dir,
            gsm8k_max_rows=args.gsm8k_max_rows,
            mmlu_data_dir=args.mmlu_data_dir,
            mmlu_max_rows=args.mmlu_max_rows,
            olympiad_math_configs=args.olympiad_math_configs,
            olympiad_math_max_rows=args.olympiad_math_max_rows,
        )
        if not benchmark_train_documents:
            raise SystemExit(
                "Benchmark train Pgen was enabled but no benchmark-safe training rows were selected. "
                "Set one of --gsm8k-max-rows, --mmlu-max-rows, or --olympiad-math-max-rows."
            )
        repeated_benchmark_train_documents = repeat_documents_on_train_split(
            benchmark_train_documents,
            train_repeat=args.benchmark_train_repeat,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        documents.extend(repeated_benchmark_train_documents)
        build_summary["benchmark_train_pgen"] = {
            "base_document_count": len(benchmark_train_documents),
            "manifest_document_count": len(repeated_benchmark_train_documents),
            "train_repeat_factor": max(int(args.benchmark_train_repeat), 1),
            "band": "pgen",
        }

    if args.include_ptraj:
        ptraj_documents = build_ptraj_pretraining_documents(
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
            include_verifier_targets=args.include_verifier_targets,
        )
        if not ptraj_documents:
            raise SystemExit(
                "Ptraj was enabled but no reasoning-trace sources were selected. "
                "Set one of --arc-episodes, --gsm8k-max-rows, --mmlu-max-rows, "
                "--olympiad-math-max-rows, or --core-max-rows."
            )
        repeated_ptraj_documents = repeat_documents_on_train_split(
            ptraj_documents,
            train_repeat=args.ptraj_repeat,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        documents.extend(repeated_ptraj_documents)
        build_summary["ptraj"] = {
            "base_document_count": len(ptraj_documents),
            "manifest_document_count": len(repeated_ptraj_documents),
            "train_repeat_factor": max(int(args.ptraj_repeat), 1),
            "band": "ptraj",
        }

    if args.include_oscar_scope:
        oscar_scope_documents = build_oscar_scope_pretraining_documents(
            roots=args.oscar_scope_roots,
            paths=args.oscar_scope_paths,
            auto_discover=args.oscar_scope_auto_discover,
            max_documents=None if args.oscar_scope_max_documents <= 0 else args.oscar_scope_max_documents,
            max_chunks=None if args.oscar_scope_max_chunks <= 0 else args.oscar_scope_max_chunks,
            views=args.oscar_scope_views,
        )
        repeated_scope_documents = repeat_documents_on_train_split(
            oscar_scope_documents,
            train_repeat=args.oscar_native_repeat,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        documents.extend(repeated_scope_documents)
        build_summary["oscar_scope"] = {
            "base_document_count": len(oscar_scope_documents),
            "manifest_document_count": len(repeated_scope_documents),
            "train_repeat_factor": max(int(args.oscar_native_repeat), 1),
            "band": "pgen",
        }

    if args.include_oscar_scope_reasoning:
        if args.oscar_scope_reasoning_max_examples <= 0:
            raise SystemExit(
                "Pass a positive --oscar-scope-reasoning-max-examples when --include-oscar-scope-reasoning is enabled."
            )
        oscar_reasoning_documents = build_oscar_scope_reasoning_pretraining_documents(
            roots=args.oscar_scope_roots,
            paths=args.oscar_scope_paths,
            auto_discover=args.oscar_scope_auto_discover,
            max_documents=None if args.oscar_scope_max_documents <= 0 else args.oscar_scope_max_documents,
            max_examples=args.oscar_scope_reasoning_max_examples,
            views=args.oscar_scope_views,
            families=args.oscar_scope_reasoning_families,
        )
        repeated_reasoning_documents = repeat_documents_on_train_split(
            oscar_reasoning_documents,
            train_repeat=args.oscar_reasoning_repeat,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        documents.extend(repeated_reasoning_documents)
        build_summary["oscar_scope_reasoning"] = {
            "base_document_count": len(oscar_reasoning_documents),
            "manifest_document_count": len(repeated_reasoning_documents),
            "train_repeat_factor": max(int(args.oscar_reasoning_repeat), 1),
            "band": "ptraj",
        }

    if args.include_oscar_graph_reasoning:
        if args.oscar_graph_reasoning_max_examples <= 0:
            raise SystemExit(
                "Pass a positive --oscar-graph-reasoning-max-examples when --include-oscar-graph-reasoning is enabled."
            )
        oscar_graph_documents = build_oscar_graph_reasoning_pretraining_documents(
            max_examples=args.oscar_graph_reasoning_max_examples,
            families=args.oscar_graph_reasoning_families,
        )
        repeated_graph_documents = repeat_documents_on_train_split(
            oscar_graph_documents,
            train_repeat=args.oscar_graph_repeat,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        documents.extend(repeated_graph_documents)
        build_summary["oscar_graph_reasoning"] = {
            "base_document_count": len(oscar_graph_documents),
            "manifest_document_count": len(repeated_graph_documents),
            "train_repeat_factor": max(int(args.oscar_graph_repeat), 1),
            "band": "ptraj",
        }

    if args.include_mmlu_pro_bench:
        if args.mmlu_pro_max_rows <= 0:
            raise SystemExit("Pass a positive --mmlu-pro-max-rows when --include-mmlu-pro-bench is enabled.")
        mmlu_pro_documents = build_mmlu_pro_benchmark_documents(max_rows=args.mmlu_pro_max_rows)
        documents.extend(mmlu_pro_documents)
        build_summary["mmlu_pro_bench"] = {
            "base_document_count": len(mmlu_pro_documents),
            "manifest_document_count": len(mmlu_pro_documents),
            "band": "pbench",
            "holdout_group": "mmlu_pro",
        }

    if args.include_gsm8k_bench:
        if args.gsm8k_bench_max_rows <= 0:
            raise SystemExit("Pass a positive --gsm8k-bench-max-rows when --include-gsm8k-bench is enabled.")
        gsm8k_bench_documents = build_gsm8k_benchmark_documents(
            data_dir=args.gsm8k_data_dir,
            max_rows=args.gsm8k_bench_max_rows,
        )
        documents.extend(gsm8k_bench_documents)
        build_summary["gsm8k_bench"] = {
            "base_document_count": len(gsm8k_bench_documents),
            "manifest_document_count": len(gsm8k_bench_documents),
            "band": "pbench",
            "holdout_group": "gsm8k_test",
        }

    if args.include_mmlu_bench:
        if args.mmlu_bench_max_rows <= 0:
            raise SystemExit("Pass a positive --mmlu-bench-max-rows when --include-mmlu-bench is enabled.")
        mmlu_bench_documents = build_mmlu_benchmark_documents(
            data_dir=args.mmlu_data_dir,
            max_rows=args.mmlu_bench_max_rows,
            splits=args.mmlu_bench_splits,
        )
        documents.extend(mmlu_bench_documents)
        build_summary["mmlu_bench"] = {
            "base_document_count": len(mmlu_bench_documents),
            "manifest_document_count": len(mmlu_bench_documents),
            "band": "pbench",
            "holdout_group": "mmlu_audit",
            "splits": list(args.mmlu_bench_splits),
        }

    if args.include_mmlu_redux_bench:
        if args.mmlu_redux_max_rows <= 0:
            raise SystemExit("Pass a positive --mmlu-redux-max-rows when --include-mmlu-redux-bench is enabled.")
        mmlu_redux_documents = build_mmlu_redux_benchmark_documents(
            max_rows=args.mmlu_redux_max_rows,
            label_mode=args.mmlu_redux_label_mode,
        )
        documents.extend(mmlu_redux_documents)
        build_summary["mmlu_redux_bench"] = {
            "base_document_count": len(mmlu_redux_documents),
            "manifest_document_count": len(mmlu_redux_documents),
            "band": "pbench",
            "holdout_group": "mmlu_redux",
        }

    if args.include_olympiad_math_bench:
        if args.olympiad_math_bench_max_rows <= 0:
            raise SystemExit(
                "Pass a positive --olympiad-math-bench-max-rows when --include-olympiad-math-bench is enabled."
            )
        olympiad_math_bench_documents = build_olympiad_math_benchmark_documents(
            configs=args.olympiad_math_bench_configs,
            max_rows=args.olympiad_math_bench_max_rows,
        )
        documents.extend(olympiad_math_bench_documents)
        build_summary["olympiad_math_bench"] = {
            "base_document_count": len(olympiad_math_bench_documents),
            "manifest_document_count": len(olympiad_math_bench_documents),
            "band": "pbench",
            "holdout_group": "olympiad_math_eval",
            "configs": list(args.olympiad_math_bench_configs),
        }

    if not documents:
        raise SystemExit("No corpus sources were selected. Enable at least one Pgen, Ptraj, or Pbench source.")
    if not any(document.holdout_group is None for document in documents):
        raise SystemExit(
            "The unified manifest currently requires at least one trainable Pgen or Ptraj source. "
            "Do not build a manifest from Pbench-only inputs."
        )

    manifest = write_pretraining_manifest(
        output_dir=args.output_dir,
        documents=documents,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        corpus_name="unified_pretraining_corpus",
        source_name="pgen_ptraj_pbench_mix",
        selection_parameters={
            "preset": args.preset,
            "preset_scale": args.preset_scale,
            "include_dclm": args.include_dclm,
            "dclm_dataset_id": args.dclm_dataset_id,
            "dclm_split": args.dclm_split,
            "dclm_text_field": args.dclm_text_field,
            "dclm_max_documents": args.dclm_max_documents,
            "dclm_shuffle": args.dclm_shuffle,
            "dclm_shuffle_buffer_size": args.dclm_shuffle_buffer_size,
            "dclm_min_text_chars": args.dclm_min_text_chars,
            "dclm_min_language_score": args.dclm_min_language_score,
            "dclm_min_fasttext_score": args.dclm_min_fasttext_score,
            "dclm_language_allowlist": list(args.dclm_language_allowlist),
            "include_benchmark_train_pgen": args.include_benchmark_train_pgen,
            "benchmark_train_repeat": args.benchmark_train_repeat,
            "include_ptraj": args.include_ptraj,
            "arc_episodes": args.arc_episodes,
            "arc_seed_start": args.arc_seed_start,
            "gsm8k_data_dir": args.gsm8k_data_dir,
            "gsm8k_max_rows": args.gsm8k_max_rows,
            "mmlu_data_dir": args.mmlu_data_dir,
            "mmlu_max_rows": args.mmlu_max_rows,
            "olympiad_math_configs": list(args.olympiad_math_configs),
            "olympiad_math_max_rows": args.olympiad_math_max_rows,
            "core_data_dir": args.core_data_dir,
            "core_max_rows": args.core_max_rows,
            "core_graph_backend": args.core_graph_backend,
            "core_languages": list(args.core_languages) if args.core_languages is not None else None,
            "core_categories": list(args.core_categories) if args.core_categories is not None else None,
            "core_dependency_kinds": list(args.core_dependency_kinds) if args.core_dependency_kinds is not None else None,
            "ptraj_repeat": args.ptraj_repeat,
            "include_verifier_targets": args.include_verifier_targets,
            "include_oscar_scope": args.include_oscar_scope,
            "include_oscar_scope_reasoning": args.include_oscar_scope_reasoning,
            "include_oscar_graph_reasoning": args.include_oscar_graph_reasoning,
            "oscar_scope_roots": list(args.oscar_scope_roots),
            "oscar_scope_paths": list(args.oscar_scope_paths),
            "oscar_scope_auto_discover": args.oscar_scope_auto_discover,
            "oscar_scope_max_documents": args.oscar_scope_max_documents,
            "oscar_scope_max_chunks": args.oscar_scope_max_chunks,
            "oscar_scope_views": list(args.oscar_scope_views),
            "oscar_scope_reasoning_max_examples": args.oscar_scope_reasoning_max_examples,
            "oscar_scope_reasoning_families": list(args.oscar_scope_reasoning_families),
            "oscar_graph_reasoning_max_examples": args.oscar_graph_reasoning_max_examples,
            "oscar_graph_reasoning_families": list(args.oscar_graph_reasoning_families),
            "oscar_native_repeat": args.oscar_native_repeat,
            "oscar_reasoning_repeat": args.oscar_reasoning_repeat,
            "oscar_graph_repeat": args.oscar_graph_repeat,
            "include_mmlu_pro_bench": args.include_mmlu_pro_bench,
            "mmlu_pro_max_rows": args.mmlu_pro_max_rows,
            "include_gsm8k_bench": args.include_gsm8k_bench,
            "gsm8k_bench_max_rows": args.gsm8k_bench_max_rows,
            "include_mmlu_bench": args.include_mmlu_bench,
            "mmlu_bench_max_rows": args.mmlu_bench_max_rows,
            "mmlu_bench_splits": list(args.mmlu_bench_splits),
            "include_mmlu_redux_bench": args.include_mmlu_redux_bench,
            "mmlu_redux_max_rows": args.mmlu_redux_max_rows,
            "mmlu_redux_label_mode": args.mmlu_redux_label_mode,
            "include_olympiad_math_bench": args.include_olympiad_math_bench,
            "olympiad_math_bench_configs": list(args.olympiad_math_bench_configs),
            "olympiad_math_bench_max_rows": args.olympiad_math_bench_max_rows,
        },
        extra_metadata={
            "builder": "scripts/build_pretraining_manifest.py",
            "build_summary": build_summary,
            "bands_present": sorted({str(document.band) for document in documents}),
            "holdout_groups_present": sorted(
                {
                    str(document.holdout_group)
                    for document in documents
                    if document.holdout_group is not None
                }
            ),
            "discovered_oscar_roots": [str(path) for path in discover_oscar_scope_roots()] if args.oscar_scope_auto_discover else [],
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
