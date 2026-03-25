from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES
from models.reasoning_tokenizer import add_tokenizer_cli_arguments


WORKFLOW_REASONING_FAMILIES = (
    "oscar_workflow_environment",
    "oscar_workflow_kpi_tags",
    "oscar_workflow_bottleneck_tags",
    "oscar_workflow_improvement_tags",
    "oscar_workflow_kpi_improvement",
    "oscar_workflow_intervention_trace",
    "oscar_workflow_case_analogy",
    "oscar_workflow_transfer",
)

DEFAULT_GRAPH_ADAPTER_FAMILIES = tuple(OSCAR_GRAPH_REASONING_FAMILIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a workflow-heavy Oscar adapter bundle for the packed-LM path. "
            "This is the fastest bridge from the Oscar workflow traces into the larger model stack."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/oscar_workflow_adapter_bundle",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-dclm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dclm-max-documents", type=int, default=0)
    parser.add_argument("--oscar-scope-max-documents", type=int, default=16)
    parser.add_argument("--oscar-scope-max-chunks", type=int, default=256)
    parser.add_argument("--oscar-scope-reasoning-max-examples", type=int, default=768)
    parser.add_argument(
        "--oscar-scope-reasoning-families",
        nargs="+",
        choices=OSCAR_SCOPE_REASONING_FAMILIES,
        default=list(WORKFLOW_REASONING_FAMILIES),
    )
    parser.add_argument("--oscar-graph-reasoning-max-examples", type=int, default=384)
    parser.add_argument(
        "--oscar-graph-reasoning-families",
        nargs="+",
        choices=OSCAR_GRAPH_REASONING_FAMILIES,
        default=list(DEFAULT_GRAPH_ADAPTER_FAMILIES),
    )
    parser.add_argument("--oscar-native-repeat", type=int, default=2)
    parser.add_argument("--oscar-reasoning-repeat", type=int, default=8)
    parser.add_argument("--oscar-graph-repeat", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--target-shard-sequences", type=int, default=8192)
    parser.add_argument("--pad-final-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--train-learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-architecture", choices=("dense", "moe"), default="moe")
    parser.add_argument("--train-attention-preset", choices=("mla_default", "mla_sia_prefill_l1"), default="mla_default")
    parser.add_argument("--train-precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--train-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume-from", type=str, default="")
    add_tokenizer_cli_arguments(
        parser,
        default_kind="epiplex",
        default_vocab_size=4096,
        default_task="generic",
        default_fit_verbose=False,
    )
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = output_dir / "manifest"
    packed_dir = output_dir / "packed"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    packed_dir.mkdir(parents=True, exist_ok=True)

    manifest_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_pretraining_manifest.py"),
        "--output-dir",
        str(manifest_dir),
        "--validation-fraction",
        str(args.validation_fraction),
        "--seed",
        str(args.seed),
        "--include-oscar-scope",
        "--oscar-scope-max-documents",
        str(args.oscar_scope_max_documents),
        "--oscar-scope-max-chunks",
        str(args.oscar_scope_max_chunks),
        "--include-oscar-scope-reasoning",
        "--oscar-scope-reasoning-max-examples",
        str(args.oscar_scope_reasoning_max_examples),
        "--oscar-scope-reasoning-families",
        *args.oscar_scope_reasoning_families,
        "--include-oscar-graph-reasoning",
        "--oscar-graph-reasoning-max-examples",
        str(args.oscar_graph_reasoning_max_examples),
        "--oscar-graph-reasoning-families",
        *args.oscar_graph_reasoning_families,
        "--oscar-native-repeat",
        str(args.oscar_native_repeat),
        "--oscar-reasoning-repeat",
        str(args.oscar_reasoning_repeat),
        "--oscar-graph-repeat",
        str(args.oscar_graph_repeat),
    ]
    if args.include_dclm:
        if args.dclm_max_documents <= 0:
            raise SystemExit("Pass a positive --dclm-max-documents when --include-dclm is enabled.")
        manifest_cmd.extend(
            [
                "--include-dclm",
                "--dclm-max-documents",
                str(args.dclm_max_documents),
            ]
        )

    manifest_result = run_command(manifest_cmd)
    manifest_payload = json.loads(manifest_result.stdout)
    manifest_path = str(manifest_payload["manifest_path"])

    pack_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "pack_pretraining_corpus.py"),
        "--document-manifest",
        manifest_path,
        "--output-dir",
        str(packed_dir),
        "--seq-len",
        str(args.seq_len),
        "--target-shard-sequences",
        str(args.target_shard_sequences),
        "--tokenizer",
        args.tokenizer,
        "--tokenizer-vocab-size",
        str(args.tokenizer_vocab_size),
        "--tokenizer-task",
        args.tokenizer_task,
        "--tokenizer-min-freq",
        str(args.tokenizer_min_freq),
        "--tokenizer-candidate-pool-size",
        str(args.tokenizer_candidate_pool_size),
        "--tokenizer-max-piece-chars",
        str(args.tokenizer_max_piece_chars),
        "--tokenizer-fit-workers",
        str(args.tokenizer_fit_workers),
        "--tokenizer-fit-max-documents",
        "50000",
        "--tokenizer-fit-seed",
        str(args.seed),
    ]
    if args.tokenizer_fit_verbose:
        pack_cmd.append("--tokenizer-fit-verbose")
    if args.tokenizer_load:
        pack_cmd.extend(["--tokenizer-load", args.tokenizer_load])
    if args.tokenizer_save:
        pack_cmd.extend(["--tokenizer-save", args.tokenizer_save])
    if args.pad_final_window:
        pack_cmd.append("--pad-final-window")

    pack_result = run_command(pack_cmd)
    pack_payload = json.loads(pack_result.stdout)
    packed_manifest_path = str(pack_payload["manifest_path"])

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_pretraining_lm.py"),
        "--packed-manifest",
        packed_manifest_path,
        "--steps",
        str(args.train_steps),
        "--batch-size",
        str(args.train_batch_size),
        "--grad-accumulation-steps",
        str(args.train_grad_accumulation_steps),
        "--learning-rate",
        str(args.train_learning_rate),
        "--architecture",
        args.train_architecture,
        "--attention-preset",
        args.train_attention_preset,
        "--precision",
        args.train_precision,
        "--device",
        args.train_device,
        "--checkpoint-dir",
        str(output_dir / "checkpoints"),
        "--output",
        str(output_dir / "train_summary.json"),
        "--progress-output",
        str(output_dir / "train_progress.json"),
        "--csv-output",
        str(output_dir / "train_metrics.csv"),
    ]
    if args.resume_from:
        train_cmd.extend(["--resume-from", args.resume_from])

    summary = {
        "bundle_type": "oscar_workflow_adapter",
        "manifest_dir": str(manifest_dir),
        "manifest_path": manifest_path,
        "packed_dir": str(packed_dir),
        "packed_manifest_path": packed_manifest_path,
        "workflow_reasoning_families": list(args.oscar_scope_reasoning_families),
        "graph_reasoning_families": list(args.oscar_graph_reasoning_families),
        "repeat_factors": {
            "oscar_native_repeat": args.oscar_native_repeat,
            "oscar_reasoning_repeat": args.oscar_reasoning_repeat,
            "oscar_graph_repeat": args.oscar_graph_repeat,
        },
        "recommended_train_command": train_cmd,
        "recommended_train_command_shell": shell_join(train_cmd),
        "manifest_build_summary": {
            "document_counts": manifest_payload.get("document_counts", {}),
            "band_counts": manifest_payload.get("band_counts", {}),
            "corpus_counts": manifest_payload.get("corpus_counts", {}),
            "extra_metadata": manifest_payload.get("extra_metadata", {}),
        },
        "pack_build_summary": {
            "seq_len": pack_payload.get("seq_len"),
            "total_sequence_count": pack_payload.get("total_sequence_count"),
            "total_packed_token_count": pack_payload.get("total_packed_token_count"),
            "tokenizer": pack_payload.get("tokenizer", {}),
            "splits": {
                split_name: {
                    "document_count": split_payload.get("document_count"),
                    "sequence_count": split_payload.get("sequence_count"),
                    "corpus_document_counts": split_payload.get("corpus_document_counts", {}),
                    "band_document_counts": split_payload.get("band_document_counts", {}),
                }
                for split_name, split_payload in dict(pack_payload.get("splits", {})).items()
            },
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
