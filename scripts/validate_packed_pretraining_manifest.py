from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a packed pretraining manifest before launch.")
    parser.add_argument("--packed-manifest", type=str, required=True)
    parser.add_argument("--expect-tokenizer-kind", type=str, default="epiplex")
    parser.add_argument("--expect-tokenizer-task", type=str, default="generic")
    parser.add_argument("--expect-seq-len", type=int, default=2048)
    parser.add_argument("--min-train-sequences", type=int, default=100000)
    parser.add_argument("--require-corpus", action="append", default=[])
    parser.add_argument("--require-corpus-prefix", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.packed_manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Packed manifest does not exist: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = dict(data.get("tokenizer", {}))
    train_split = dict(data.get("splits", {}).get("train", {}))
    corpus_counts = dict(data.get("document_manifest_corpus_counts", {}).get("train", {}))

    tokenizer_kind = str(tokenizer.get("kind", ""))
    tokenizer_task = tokenizer.get("task")
    seq_len = int(data.get("seq_len", -1))
    train_sequences = int(train_split.get("sequence_count", 0))

    if tokenizer_kind != args.expect_tokenizer_kind:
        raise SystemExit(
            f"Unexpected tokenizer kind {tokenizer_kind!r}; expected {args.expect_tokenizer_kind!r}."
        )
    if tokenizer_task != args.expect_tokenizer_task:
        raise SystemExit(
            f"Unexpected tokenizer task {tokenizer_task!r}; expected {args.expect_tokenizer_task!r}."
        )
    if seq_len != args.expect_seq_len:
        raise SystemExit(f"Unexpected seq_len {seq_len}; expected {args.expect_seq_len}.")
    if train_sequences < int(args.min_train_sequences):
        raise SystemExit(
            f"Train sequence count {train_sequences} is below required minimum {args.min_train_sequences}."
        )

    missing_corpora = [name for name in args.require_corpus if int(corpus_counts.get(name, 0)) <= 0]
    if missing_corpora:
        raise SystemExit(f"Missing required train corpora: {', '.join(missing_corpora)}")

    for prefix in args.require_corpus_prefix:
        if not any(name.startswith(prefix) and int(count) > 0 for name, count in corpus_counts.items()):
            raise SystemExit(f"Missing required train corpus prefix {prefix!r}.")

    summary = {
        "packed_manifest": str(manifest_path),
        "tokenizer": {
            "kind": tokenizer_kind,
            "task": tokenizer_task,
            "vocab_size": tokenizer.get("vocab_size"),
        },
        "seq_len": seq_len,
        "train_document_count": int(train_split.get("document_count", 0)),
        "train_sequence_count": train_sequences,
        "train_corpora": corpus_counts,
        "status": "READY",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
