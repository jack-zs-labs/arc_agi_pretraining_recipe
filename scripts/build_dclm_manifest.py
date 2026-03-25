from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.dclm_corpus import (
    DEFAULT_DCLM_DATASET_ID,
    DEFAULT_DCLM_SPLIT,
    DEFAULT_DCLM_TEXT_FIELD,
    DCLM_HF_DATASET_URL,
    DCLM_MAINTAINED_REPO_URL,
    DCLM_SUBMISSION_REPO_URL,
    iter_dclm_documents,
)
from training.corpus_manifest import PretrainingDocument, write_pretraining_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen pretraining document manifest from DCLM.")
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DCLM_DATASET_ID)
    parser.add_argument("--split", type=str, default=DEFAULT_DCLM_SPLIT)
    parser.add_argument("--text-field", type=str, default=DEFAULT_DCLM_TEXT_FIELD)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-documents", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--min-text-chars", type=int, default=0)
    parser.add_argument("--min-language-score", type=float, default=0.0)
    parser.add_argument("--min-fasttext-score", type=float, default=0.0)
    parser.add_argument("--language-allowlist", nargs="+", default=["en"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_documents <= 0:
        raise SystemExit("--max-documents must be positive.")
    documents = (
        PretrainingDocument(
            corpus="dclm",
            doc_id=document.row_id,
            text=document.text,
            source_split=args.split,
            metadata={
                "url": document.url,
                "language": document.language,
                "language_score": document.language_score,
                "fasttext_score": document.fasttext_score,
                "dataset_id": args.dataset_id,
            },
        )
        for document in iter_dclm_documents(
            dataset_id=args.dataset_id,
            split=args.split,
            text_field=args.text_field,
            shuffle=args.shuffle,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
            max_documents=args.max_documents,
            min_text_chars=args.min_text_chars,
            min_language_score=args.min_language_score,
            min_fasttext_score=args.min_fasttext_score,
            language_allowlist=tuple(args.language_allowlist),
        )
    )
    manifest = write_pretraining_manifest(
        output_dir=args.output_dir,
        documents=documents,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        corpus_name="dclm",
        source_name=args.dataset_id,
        selection_parameters={
            "split": args.split,
            "text_field": args.text_field,
            "max_documents": args.max_documents,
            "shuffle": args.shuffle,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "min_text_chars": args.min_text_chars,
            "min_language_score": args.min_language_score,
            "min_fasttext_score": args.min_fasttext_score,
            "language_allowlist": list(args.language_allowlist),
        },
        extra_metadata={
            "builder": "scripts/build_dclm_manifest.py",
            "dclm_source": {
                "dataset_id": args.dataset_id,
                "dataset_url": (
                    DCLM_HF_DATASET_URL
                    if args.dataset_id == DEFAULT_DCLM_DATASET_ID
                    else f"https://huggingface.co/datasets/{args.dataset_id}"
                ),
                "submission_repo_url": DCLM_SUBMISSION_REPO_URL,
                "maintained_repo_url": DCLM_MAINTAINED_REPO_URL,
            },
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
