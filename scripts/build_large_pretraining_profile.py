from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.core_loader import DEFAULT_CORE_DATA_DIR
from arc_trajectory_sampler.dclm_corpus import DEFAULT_DCLM_DATASET_ID, DEFAULT_DCLM_SPLIT, DEFAULT_DCLM_TEXT_FIELD
from arc_trajectory_sampler.mixed_reasoning_dataset import DEFAULT_GSM8K_DATA_DIR
from arc_trajectory_sampler.mmlu_parser import DEFAULT_MMLU_AUDIT_SPLITS
from arc_trajectory_sampler.mmlu_parser import DEFAULT_DATA_DIR as DEFAULT_MMLU_DATA_DIR
from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES
from arc_trajectory_sampler.oscar_scope_corpus import DEFAULT_OSCAR_SCOPE_VIEWS, discover_oscar_scope_roots
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES
from models.reasoning_tokenizer import add_tokenizer_cli_arguments
from training.corpus_manifest import PretrainingDocument, assign_split_for_doc_id, write_pretraining_manifest
from training.corpus_sources import (
    build_benchmark_training_pretraining_documents,
    build_dclm_pretraining_documents,
    build_gsm8k_benchmark_documents,
    build_mmlu_benchmark_documents,
    build_mmlu_pro_benchmark_documents,
    build_mmlu_redux_benchmark_documents,
    build_oscar_graph_reasoning_pretraining_documents,
    build_oscar_scope_pretraining_documents,
    build_oscar_scope_reasoning_pretraining_documents,
    build_ptraj_pretraining_documents,
    repeat_documents_on_train_split,
)
from training.token_packer import pack_pretraining_document_manifest


DEFAULT_EXPECTED_HOLDOUT_GROUPS = ("gsm8k_test", "mmlu_audit")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_json_command(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        detail = exc.stdout.strip() or exc.stderr.strip() or str(exc)
        raise SystemExit(detail) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deployment-oriented large-pretraining corpus profile with "
            "dominant Pgen, bounded Ptraj, and holdout-only Pbench."
        )
    )
    parser.add_argument("--output-dir", type=str, default="results/large_pretraining_profile")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-preflight", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--include-dclm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dclm-max-documents", type=int, default=0)
    parser.add_argument("--dclm-dataset-id", type=str, default=DEFAULT_DCLM_DATASET_ID)
    parser.add_argument("--dclm-split", type=str, default=DEFAULT_DCLM_SPLIT)
    parser.add_argument("--dclm-text-field", type=str, default=DEFAULT_DCLM_TEXT_FIELD)

    parser.add_argument("--include-benchmark-train-pgen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pgen-gsm8k-max-rows", type=int, default=2048)
    parser.add_argument("--pgen-mmlu-max-rows", type=int, default=2048)
    parser.add_argument("--pgen-olympiad-math-max-rows", type=int, default=0)
    parser.add_argument("--benchmark-train-repeat", type=int, default=2)

    parser.add_argument("--include-oscar-scope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-roots", nargs="+", default=[])
    parser.add_argument("--oscar-scope-paths", nargs="+", default=[])
    parser.add_argument("--oscar-scope-auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-max-documents", type=int, default=32)
    parser.add_argument("--oscar-scope-max-chunks", type=int, default=1024)
    parser.add_argument("--oscar-native-repeat", type=int, default=2)
    parser.add_argument(
        "--oscar-scope-views",
        nargs="+",
        default=list(DEFAULT_OSCAR_SCOPE_VIEWS),
    )

    parser.add_argument("--include-ptraj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arc-episodes", type=int, default=128)
    parser.add_argument("--arc-seed-start", type=int, default=0)
    parser.add_argument("--ptraj-gsm8k-max-rows", type=int, default=128)
    parser.add_argument("--ptraj-mmlu-max-rows", type=int, default=128)
    parser.add_argument("--ptraj-olympiad-math-max-rows", type=int, default=0)
    parser.add_argument("--ptraj-core-max-rows", type=int, default=96)
    parser.add_argument("--ptraj-repeat", type=int, default=1)
    parser.add_argument("--include-verifier-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gsm8k-data-dir", type=str, default=DEFAULT_GSM8K_DATA_DIR)
    parser.add_argument("--mmlu-data-dir", type=str, default=DEFAULT_MMLU_DATA_DIR)
    parser.add_argument("--core-data-dir", type=str, default=DEFAULT_CORE_DATA_DIR)
    parser.add_argument("--core-graph-backend", choices=("auto", "heuristic", "python_ast"), default="auto")

    parser.add_argument("--include-oscar-scope-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-reasoning-max-examples", type=int, default=192)
    parser.add_argument(
        "--oscar-scope-reasoning-families",
        nargs="+",
        choices=OSCAR_SCOPE_REASONING_FAMILIES,
        default=list(OSCAR_SCOPE_REASONING_FAMILIES),
    )
    parser.add_argument("--oscar-reasoning-repeat", type=int, default=1)

    parser.add_argument("--include-oscar-graph-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-graph-reasoning-max-examples", type=int, default=96)
    parser.add_argument(
        "--oscar-graph-reasoning-families",
        nargs="+",
        choices=OSCAR_GRAPH_REASONING_FAMILIES,
        default=list(OSCAR_GRAPH_REASONING_FAMILIES),
    )
    parser.add_argument("--oscar-graph-repeat", type=int, default=1)

    parser.add_argument("--include-gsm8k-bench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gsm8k-bench-max-rows", type=int, default=256)
    parser.add_argument("--include-mmlu-bench", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mmlu-bench-max-rows", type=int, default=256)
    parser.add_argument("--mmlu-bench-splits", nargs="+", default=list(DEFAULT_MMLU_AUDIT_SPLITS))
    parser.add_argument("--include-mmlu-pro-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mmlu-pro-max-rows", type=int, default=0)
    parser.add_argument("--include-mmlu-redux-bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mmlu-redux-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-label-mode", type=str, default="corrected_single")

    parser.add_argument("--min-train-pgen-fraction", type=float, default=0.70)
    parser.add_argument("--min-train-ptraj-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-ptraj-fraction", type=float, default=0.30)
    parser.add_argument("--auto-balance-pgen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-auto-pgen-copies", type=int, default=32)
    parser.add_argument("--require-any-holdout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-holdout-groups", nargs="+", default=list(DEFAULT_EXPECTED_HOLDOUT_GROUPS))

    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--target-shard-sequences", type=int, default=8192)
    parser.add_argument("--pad-final-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tokenizer-fit-max-documents", type=int, default=50_000)
    parser.add_argument("--tokenizer-fit-seed", type=int, default=0)
    add_tokenizer_cli_arguments(
        parser,
        default_kind="epiplex",
        default_vocab_size=4096,
        default_task="generic",
        default_fit_verbose=False,
    )

    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--train-learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-output-root", type=str, default="")
    return parser.parse_args()


def _record_build_summary(
    summary: dict[str, dict[str, object]],
    *,
    name: str,
    base_documents: tuple[PretrainingDocument, ...],
    manifest_documents: tuple[PretrainingDocument, ...],
    band: str,
    holdout_group: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "base_document_count": len(base_documents),
        "manifest_document_count": len(manifest_documents),
        "band": band,
    }
    if holdout_group is not None:
        payload["holdout_group"] = holdout_group
    if extra:
        payload.update(extra)
    summary[name] = payload


def _repeat_train_documents(
    documents: tuple[PretrainingDocument, ...],
    *,
    repeat: int,
    validation_fraction: float,
    seed: int,
) -> tuple[PretrainingDocument, ...]:
    return repeat_documents_on_train_split(
        documents,
        train_repeat=repeat,
        validation_fraction=validation_fraction,
        seed=seed,
    )


def _estimate_split(document: PretrainingDocument, *, validation_fraction: float, seed: int) -> str:
    if document.holdout_group is not None:
        return "holdout"
    if document.preferred_split is not None:
        return str(document.preferred_split)
    return assign_split_for_doc_id(document.doc_id, validation_fraction=validation_fraction, seed=seed)


def _estimate_train_band_bytes(
    documents: list[PretrainingDocument] | tuple[PretrainingDocument, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for document in documents:
        if _estimate_split(document, validation_fraction=validation_fraction, seed=seed) != "train":
            continue
        counts[document.band] += document.byte_length()
    return dict(counts)


def _boost_train_only_documents(
    documents: tuple[PretrainingDocument, ...],
    *,
    extra_copies: int,
    validation_fraction: float,
    seed: int,
    source_tag: str,
) -> tuple[PretrainingDocument, ...]:
    if extra_copies <= 0:
        return ()
    boosted: list[PretrainingDocument] = []
    for document in documents:
        if document.holdout_group is not None:
            continue
        if assign_split_for_doc_id(document.doc_id, validation_fraction=validation_fraction, seed=seed) != "train":
            continue
        base_metadata = dict(document.metadata)
        base_metadata.setdefault("base_doc_id", document.doc_id)
        for copy_index in range(extra_copies):
            boosted.append(
                PretrainingDocument(
                    corpus=document.corpus,
                    doc_id=f"{document.doc_id}::auto_balance_{source_tag}_{copy_index}",
                    text=document.text,
                    band=document.band,
                    source_split=document.source_split,
                    preferred_split="train",
                    metadata={
                        **base_metadata,
                        "auto_balance_copy_index": copy_index,
                        "auto_balance_source": source_tag,
                        "auto_balance_extra_copies": extra_copies,
                    },
                )
            )
    return tuple(boosted)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = output_dir / "manifest"
    packed_dir = output_dir / "packed"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if args.pack:
        packed_dir.mkdir(parents=True, exist_ok=True)

    build_summary: dict[str, dict[str, object]] = {}
    documents: list[PretrainingDocument] = []
    oscar_roots = discover_oscar_scope_roots() if args.oscar_scope_auto_discover else ()
    benchmark_pgen_documents: tuple[PretrainingDocument, ...] = ()
    oscar_scope_documents: tuple[PretrainingDocument, ...] = ()

    if args.include_dclm:
        if args.dclm_max_documents <= 0:
            raise SystemExit("Pass a positive --dclm-max-documents when --include-dclm is enabled.")
        dclm_documents = build_dclm_pretraining_documents(
            dataset_id=args.dclm_dataset_id,
            split=args.dclm_split,
            text_field=args.dclm_text_field,
            max_documents=args.dclm_max_documents,
            seed=args.seed,
        )
        documents.extend(dclm_documents)
        _record_build_summary(
            build_summary,
            name="dclm",
            base_documents=dclm_documents,
            manifest_documents=dclm_documents,
            band="pgen",
        )

    if args.include_benchmark_train_pgen:
        benchmark_pgen_documents = build_benchmark_training_pretraining_documents(
            gsm8k_data_dir=args.gsm8k_data_dir,
            gsm8k_max_rows=args.pgen_gsm8k_max_rows,
            mmlu_data_dir=args.mmlu_data_dir,
            mmlu_max_rows=args.pgen_mmlu_max_rows,
            olympiad_math_max_rows=args.pgen_olympiad_math_max_rows,
        )
        if benchmark_pgen_documents:
            repeated_benchmark_pgen = _repeat_train_documents(
                benchmark_pgen_documents,
                repeat=args.benchmark_train_repeat,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            documents.extend(repeated_benchmark_pgen)
            _record_build_summary(
                build_summary,
                name="benchmark_train_pgen",
                base_documents=benchmark_pgen_documents,
                manifest_documents=repeated_benchmark_pgen,
                band="pgen",
                extra={
                    "train_repeat_factor": max(int(args.benchmark_train_repeat), 1),
                    "gsm8k_max_rows": args.pgen_gsm8k_max_rows,
                    "mmlu_max_rows": args.pgen_mmlu_max_rows,
                    "olympiad_math_max_rows": args.pgen_olympiad_math_max_rows,
                },
            )

    if args.include_oscar_scope:
        oscar_scope_documents = build_oscar_scope_pretraining_documents(
            roots=args.oscar_scope_roots,
            paths=args.oscar_scope_paths,
            auto_discover=args.oscar_scope_auto_discover,
            max_documents=None if args.oscar_scope_max_documents <= 0 else args.oscar_scope_max_documents,
            max_chunks=None if args.oscar_scope_max_chunks <= 0 else args.oscar_scope_max_chunks,
            views=args.oscar_scope_views,
        )
        if oscar_scope_documents:
            repeated_oscar_scope_documents = _repeat_train_documents(
                oscar_scope_documents,
                repeat=args.oscar_native_repeat,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            documents.extend(repeated_oscar_scope_documents)
            _record_build_summary(
                build_summary,
                name="oscar_scope",
                base_documents=oscar_scope_documents,
                manifest_documents=repeated_oscar_scope_documents,
                band="pgen",
                extra={
                    "train_repeat_factor": max(int(args.oscar_native_repeat), 1),
                    "max_documents": args.oscar_scope_max_documents,
                    "max_chunks": args.oscar_scope_max_chunks,
                },
            )

    if args.include_ptraj:
        ptraj_documents = build_ptraj_pretraining_documents(
            arc_episodes=args.arc_episodes,
            arc_seed_start=args.arc_seed_start,
            gsm8k_data_dir=args.gsm8k_data_dir,
            gsm8k_max_rows=args.ptraj_gsm8k_max_rows,
            mmlu_data_dir=args.mmlu_data_dir,
            mmlu_max_rows=args.ptraj_mmlu_max_rows,
            olympiad_math_max_rows=args.ptraj_olympiad_math_max_rows,
            core_data_dir=args.core_data_dir,
            core_max_rows=args.ptraj_core_max_rows,
            core_graph_backend=args.core_graph_backend,
            include_verifier_targets=args.include_verifier_targets,
        )
        if ptraj_documents:
            repeated_ptraj_documents = _repeat_train_documents(
                ptraj_documents,
                repeat=args.ptraj_repeat,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            documents.extend(repeated_ptraj_documents)
            _record_build_summary(
                build_summary,
                name="ptraj",
                base_documents=ptraj_documents,
                manifest_documents=repeated_ptraj_documents,
                band="ptraj",
                extra={
                    "train_repeat_factor": max(int(args.ptraj_repeat), 1),
                    "arc_episodes": args.arc_episodes,
                    "gsm8k_max_rows": args.ptraj_gsm8k_max_rows,
                    "mmlu_max_rows": args.ptraj_mmlu_max_rows,
                    "core_max_rows": args.ptraj_core_max_rows,
                    "include_verifier_targets": bool(args.include_verifier_targets),
                },
            )

    if args.include_oscar_scope_reasoning:
        oscar_scope_reasoning_documents = build_oscar_scope_reasoning_pretraining_documents(
            roots=args.oscar_scope_roots,
            paths=args.oscar_scope_paths,
            auto_discover=args.oscar_scope_auto_discover,
            max_documents=None if args.oscar_scope_max_documents <= 0 else args.oscar_scope_max_documents,
            max_examples=None
            if args.oscar_scope_reasoning_max_examples <= 0
            else args.oscar_scope_reasoning_max_examples,
            views=args.oscar_scope_views,
            families=args.oscar_scope_reasoning_families,
        )
        if oscar_scope_reasoning_documents:
            repeated_oscar_scope_reasoning = _repeat_train_documents(
                oscar_scope_reasoning_documents,
                repeat=args.oscar_reasoning_repeat,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            documents.extend(repeated_oscar_scope_reasoning)
            _record_build_summary(
                build_summary,
                name="oscar_scope_reasoning",
                base_documents=oscar_scope_reasoning_documents,
                manifest_documents=repeated_oscar_scope_reasoning,
                band="ptraj",
                extra={
                    "train_repeat_factor": max(int(args.oscar_reasoning_repeat), 1),
                    "max_examples": args.oscar_scope_reasoning_max_examples,
                },
            )

    if args.include_oscar_graph_reasoning:
        oscar_graph_reasoning_documents = build_oscar_graph_reasoning_pretraining_documents(
            max_examples=None if args.oscar_graph_reasoning_max_examples <= 0 else args.oscar_graph_reasoning_max_examples,
            families=args.oscar_graph_reasoning_families,
        )
        if oscar_graph_reasoning_documents:
            repeated_oscar_graph_reasoning = _repeat_train_documents(
                oscar_graph_reasoning_documents,
                repeat=args.oscar_graph_repeat,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            documents.extend(repeated_oscar_graph_reasoning)
            _record_build_summary(
                build_summary,
                name="oscar_graph_reasoning",
                base_documents=oscar_graph_reasoning_documents,
                manifest_documents=repeated_oscar_graph_reasoning,
                band="ptraj",
                extra={
                    "train_repeat_factor": max(int(args.oscar_graph_repeat), 1),
                    "max_examples": args.oscar_graph_reasoning_max_examples,
                },
            )

    if args.include_gsm8k_bench and args.gsm8k_bench_max_rows > 0:
        gsm8k_bench_documents = build_gsm8k_benchmark_documents(
            data_dir=args.gsm8k_data_dir,
            max_rows=args.gsm8k_bench_max_rows,
        )
        documents.extend(gsm8k_bench_documents)
        _record_build_summary(
            build_summary,
            name="gsm8k_bench",
            base_documents=gsm8k_bench_documents,
            manifest_documents=gsm8k_bench_documents,
            band="pbench",
            holdout_group="gsm8k_test",
        )

    if args.include_mmlu_bench and args.mmlu_bench_max_rows > 0:
        mmlu_bench_documents = build_mmlu_benchmark_documents(
            data_dir=args.mmlu_data_dir,
            max_rows=args.mmlu_bench_max_rows,
            splits=args.mmlu_bench_splits,
        )
        documents.extend(mmlu_bench_documents)
        _record_build_summary(
            build_summary,
            name="mmlu_bench",
            base_documents=mmlu_bench_documents,
            manifest_documents=mmlu_bench_documents,
            band="pbench",
            holdout_group="mmlu_audit",
            extra={"splits": list(args.mmlu_bench_splits)},
        )

    if args.include_mmlu_pro_bench and args.mmlu_pro_max_rows > 0:
        mmlu_pro_documents = build_mmlu_pro_benchmark_documents(max_rows=args.mmlu_pro_max_rows)
        documents.extend(mmlu_pro_documents)
        _record_build_summary(
            build_summary,
            name="mmlu_pro_bench",
            base_documents=mmlu_pro_documents,
            manifest_documents=mmlu_pro_documents,
            band="pbench",
            holdout_group="mmlu_pro",
        )

    if args.include_mmlu_redux_bench and args.mmlu_redux_max_rows > 0:
        mmlu_redux_documents = build_mmlu_redux_benchmark_documents(
            max_rows=args.mmlu_redux_max_rows,
            label_mode=args.mmlu_redux_label_mode,
        )
        documents.extend(mmlu_redux_documents)
        _record_build_summary(
            build_summary,
            name="mmlu_redux_bench",
            base_documents=mmlu_redux_documents,
            manifest_documents=mmlu_redux_documents,
            band="pbench",
            holdout_group="mmlu_redux",
            extra={"label_mode": args.mmlu_redux_label_mode},
        )

    auto_balance_extra_copies = 0
    if args.auto_balance_pgen:
        current_train_band_bytes = _estimate_train_band_bytes(
            documents,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        current_train_pgen_bytes = int(current_train_band_bytes.get("pgen", 0))
        current_train_ptraj_bytes = int(current_train_band_bytes.get("ptraj", 0))
        required_train_pgen_bytes = 0
        if current_train_ptraj_bytes > 0:
            required_train_pgen_bytes = int(
                (float(args.min_train_pgen_fraction) / max(1.0 - float(args.min_train_pgen_fraction), 1e-9))
                * float(current_train_ptraj_bytes)
            )
        if current_train_pgen_bytes < required_train_pgen_bytes:
            eligible_boost_bytes = int(
                _estimate_train_band_bytes(
                    list(benchmark_pgen_documents) + list(oscar_scope_documents),
                    validation_fraction=args.validation_fraction,
                    seed=args.seed,
                ).get("pgen", 0)
            )
            if eligible_boost_bytes <= 0:
                raise SystemExit(
                    "Large-pretraining profile cannot reach the required pgen fraction with the current local sources. "
                    "Enable repeatable native-text sources or pass --include-dclm."
                )
            missing_pgen_bytes = required_train_pgen_bytes - current_train_pgen_bytes
            auto_balance_extra_copies = (missing_pgen_bytes + eligible_boost_bytes - 1) // eligible_boost_bytes
            if auto_balance_extra_copies > int(args.max_auto_pgen_copies):
                raise SystemExit(
                    f"Auto-balancing pgen needs {auto_balance_extra_copies} extra train-only copies, which exceeds "
                    f"--max-auto-pgen-copies {int(args.max_auto_pgen_copies)}. Add more native text or enable DCLM."
                )
            boosted_benchmark_pgen = _boost_train_only_documents(
                benchmark_pgen_documents,
                extra_copies=auto_balance_extra_copies,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                source_tag="benchmark_train_pgen",
            )
            boosted_oscar_scope = _boost_train_only_documents(
                oscar_scope_documents,
                extra_copies=auto_balance_extra_copies,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                source_tag="oscar_scope",
            )
            documents.extend(boosted_benchmark_pgen)
            documents.extend(boosted_oscar_scope)
            build_summary["auto_balance"] = {
                "extra_train_only_copies": auto_balance_extra_copies,
                "estimated_train_pgen_bytes_before": current_train_pgen_bytes,
                "estimated_train_ptraj_bytes_before": current_train_ptraj_bytes,
                "estimated_train_pgen_bytes_after": current_train_pgen_bytes
                + auto_balance_extra_copies * eligible_boost_bytes,
            }
            if "benchmark_train_pgen" in build_summary:
                build_summary["benchmark_train_pgen"]["auto_balance_extra_copies"] = auto_balance_extra_copies
                build_summary["benchmark_train_pgen"]["manifest_document_count"] = int(
                    build_summary["benchmark_train_pgen"]["manifest_document_count"]
                ) + len(boosted_benchmark_pgen)
            if "oscar_scope" in build_summary:
                build_summary["oscar_scope"]["auto_balance_extra_copies"] = auto_balance_extra_copies
                build_summary["oscar_scope"]["manifest_document_count"] = int(
                    build_summary["oscar_scope"]["manifest_document_count"]
                ) + len(boosted_oscar_scope)

    if not documents:
        raise SystemExit("No documents were selected for the large-pretraining profile.")

    manifest = write_pretraining_manifest(
        output_dir=manifest_dir,
        documents=documents,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        corpus_name="large_pretraining_profile",
        source_name="deploy_conservative_v1",
        selection_parameters={
            "include_dclm": bool(args.include_dclm),
            "dclm_max_documents": args.dclm_max_documents,
            "include_benchmark_train_pgen": bool(args.include_benchmark_train_pgen),
            "pgen_gsm8k_max_rows": args.pgen_gsm8k_max_rows,
            "pgen_mmlu_max_rows": args.pgen_mmlu_max_rows,
            "benchmark_train_repeat": args.benchmark_train_repeat,
            "include_oscar_scope": bool(args.include_oscar_scope),
            "oscar_scope_max_documents": args.oscar_scope_max_documents,
            "oscar_scope_max_chunks": args.oscar_scope_max_chunks,
            "oscar_native_repeat": args.oscar_native_repeat,
            "include_ptraj": bool(args.include_ptraj),
            "arc_episodes": args.arc_episodes,
            "ptraj_gsm8k_max_rows": args.ptraj_gsm8k_max_rows,
            "ptraj_mmlu_max_rows": args.ptraj_mmlu_max_rows,
            "ptraj_core_max_rows": args.ptraj_core_max_rows,
            "ptraj_repeat": args.ptraj_repeat,
            "include_verifier_targets": bool(args.include_verifier_targets),
            "include_oscar_scope_reasoning": bool(args.include_oscar_scope_reasoning),
            "oscar_scope_reasoning_max_examples": args.oscar_scope_reasoning_max_examples,
            "oscar_reasoning_repeat": args.oscar_reasoning_repeat,
            "include_oscar_graph_reasoning": bool(args.include_oscar_graph_reasoning),
            "oscar_graph_reasoning_max_examples": args.oscar_graph_reasoning_max_examples,
            "oscar_graph_repeat": args.oscar_graph_repeat,
            "include_gsm8k_bench": bool(args.include_gsm8k_bench),
            "gsm8k_bench_max_rows": args.gsm8k_bench_max_rows,
            "include_mmlu_bench": bool(args.include_mmlu_bench),
            "mmlu_bench_max_rows": args.mmlu_bench_max_rows,
            "auto_balance_pgen": bool(args.auto_balance_pgen),
            "auto_balance_extra_copies": auto_balance_extra_copies,
            "include_mmlu_pro_bench": bool(args.include_mmlu_pro_bench),
            "mmlu_pro_max_rows": args.mmlu_pro_max_rows,
            "include_mmlu_redux_bench": bool(args.include_mmlu_redux_bench),
            "mmlu_redux_max_rows": args.mmlu_redux_max_rows,
        },
        extra_metadata={
            "builder": "scripts/build_large_pretraining_profile.py",
            "build_summary": build_summary,
            "discovered_oscar_roots": [str(path) for path in oscar_roots],
        },
    )
    manifest_path = str(manifest["manifest_path"])

    manifest_preflight_payload: dict[str, object] | None = None
    if args.run_preflight:
        manifest_preflight_output = output_dir / "manifest_preflight.json"
        manifest_preflight_payload = run_json_command(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "preflight_large_pretraining_run.py"),
                "--manifest",
                manifest_path,
                "--min-train-pgen-fraction",
                str(args.min_train_pgen_fraction),
                "--min-train-ptraj-fraction",
                str(args.min_train_ptraj_fraction),
                "--max-train-ptraj-fraction",
                str(args.max_train_ptraj_fraction),
                "--require-any-holdout" if args.require_any_holdout else "--no-require-any-holdout",
                "--expected-holdout-groups",
                *args.expected_holdout_groups,
                "--json-output",
                str(manifest_preflight_output),
            ]
        )

    packed_manifest_payload: dict[str, object] | None = None
    packed_preflight_payload: dict[str, object] | None = None
    packed_manifest_path = ""
    if args.pack:
        if args.tokenizer in {"epiplex", "rust_bpe"} and not args.tokenizer_load and not args.tokenizer_save:
            args.tokenizer_save = str(packed_dir / "reasoning_tokenizer.json")
        packed_manifest_payload = pack_pretraining_document_manifest(
            document_manifest_path=manifest_path,
            output_dir=packed_dir,
            tokenizer_kind=args.tokenizer,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
            tokenizer_task=args.tokenizer_task,
            tokenizer_min_freq=args.tokenizer_min_freq,
            tokenizer_candidate_pool_size=args.tokenizer_candidate_pool_size,
            tokenizer_max_piece_chars=args.tokenizer_max_piece_chars,
            tokenizer_fit_workers=args.tokenizer_fit_workers,
            tokenizer_fit_verbose=args.tokenizer_fit_verbose,
            tokenizer_load_path=args.tokenizer_load,
            tokenizer_save_path=args.tokenizer_save,
            tokenizer_fit_max_documents=args.tokenizer_fit_max_documents,
            tokenizer_fit_seed=args.tokenizer_fit_seed,
            seq_len=args.seq_len,
            target_shard_sequences=args.target_shard_sequences,
            pad_final_window=args.pad_final_window,
        )
        packed_manifest_path = str(packed_manifest_payload["manifest_path"])
        if args.run_preflight:
            packed_preflight_output = output_dir / "packed_preflight.json"
            packed_preflight_payload = run_json_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "preflight_large_pretraining_run.py"),
                    "--manifest",
                    packed_manifest_path,
                    "--min-train-pgen-fraction",
                    str(args.min_train_pgen_fraction),
                    "--min-train-ptraj-fraction",
                    str(args.min_train_ptraj_fraction),
                    "--max-train-ptraj-fraction",
                    str(args.max_train_ptraj_fraction),
                    "--require-any-holdout" if args.require_any_holdout else "--no-require-any-holdout",
                    "--expected-holdout-groups",
                    *args.expected_holdout_groups,
                    "--json-output",
                    str(packed_preflight_output),
                ]
            )

    train_output_root = args.train_output_root or str(output_dir / "train_run")
    launch_env: dict[str, str] = {}
    launch_command_shell = ""
    if packed_manifest_path:
        launch_env = {
            "PACKED_MANIFEST": packed_manifest_path,
            "OUTPUT_ROOT": train_output_root,
            "STEPS": str(args.train_steps),
            "BATCH_SIZE": str(args.train_batch_size),
            "GRAD_ACCUMULATION_STEPS": str(args.train_grad_accumulation_steps),
            "LEARNING_RATE": str(args.train_learning_rate),
        }
        launch_command = [
            "bash",
            str(REPO_ROOT / "scripts" / "launch_pretraining_lm_8h100.sh"),
        ]
        launch_command_shell = " ".join(
            [*(f"{key}={shlex.quote(value)}" for key, value in launch_env.items() if value), shell_join(launch_command)]
        )

    summary = {
        "profile_name": "deploy_conservative_v1",
        "output_dir": str(output_dir),
        "manifest_path": manifest_path,
        "packed_manifest_path": packed_manifest_path,
        "discovered_oscar_roots": [str(path) for path in oscar_roots],
        "manifest_build_summary": {
            "document_counts": manifest.get("document_counts", {}),
            "band_counts": manifest.get("band_counts", {}),
            "corpus_counts": manifest.get("corpus_counts", {}),
            "holdout_counts": manifest.get("holdout_counts", {}),
            "extra_metadata": manifest.get("extra_metadata", {}),
        },
        "manifest_preflight": manifest_preflight_payload,
        "packed_preflight": packed_preflight_payload,
        "pack_summary": packed_manifest_payload,
        "recommended_launch_env": launch_env,
        "recommended_launch_command_shell": launch_command_shell,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
