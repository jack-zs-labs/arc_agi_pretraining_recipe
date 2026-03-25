from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.reasoning_tokenizer import add_tokenizer_cli_arguments
from training.token_packer import pack_pretraining_document_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack a frozen pretraining document manifest into fixed-length token shards.")
    parser.add_argument("--document-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--target-shard-sequences", type=int, default=8192)
    parser.add_argument("--pad-final-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tokenizer-fit-max-documents",
        type=int,
        default=50_000,
        help="Number of train documents to sample when fitting a tokenizer. Ignored for --tokenizer byte or --tokenizer-load.",
    )
    parser.add_argument("--tokenizer-fit-seed", type=int, default=0)
    add_tokenizer_cli_arguments(
        parser,
        default_kind="epiplex",
        default_vocab_size=4096,
        default_task="generic",
        default_fit_verbose=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be positive.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.tokenizer in {"epiplex", "rust_bpe"} and not args.tokenizer_load and not args.tokenizer_save:
        args.tokenizer_save = str(output_dir / "reasoning_tokenizer.json")

    manifest = pack_pretraining_document_manifest(
        document_manifest_path=args.document_manifest,
        output_dir=output_dir,
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
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
