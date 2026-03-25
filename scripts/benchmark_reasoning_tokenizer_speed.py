from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.reasoning_tokenizer import build_reasoning_tokenizer
from training.corpus_manifest import iter_pretraining_document_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark reasoning tokenizer fit and encode throughput.")
    parser.add_argument("--document-file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fit-target-chars", type=int, default=1_500_000)
    parser.add_argument("--encode-target-chars", type=int, default=20_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--task", type=str, default="generic")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=2048)
    parser.add_argument("--max-piece-chars", type=int, default=24)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=("epiplex:8", "rust_bpe:8"),
        help="Tokenizer configs in kind:fit_workers form, for example epiplex:8 rust_bpe:8.",
    )
    return parser.parse_args()


def parse_config(raw: str) -> dict[str, object]:
    try:
        kind, workers = raw.split(":", 1)
    except ValueError as exc:
        raise SystemExit(f"Invalid --configs entry {raw!r}; expected kind:fit_workers.") from exc
    return {
        "name": f"{kind}_w{int(workers)}",
        "kind": kind,
        "fit_workers": int(workers),
    }


def main() -> None:
    args = parse_args()
    document_file = Path(args.document_file).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_path.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    texts = [str(row["text"]) for row in iter_pretraining_document_rows(document_file)]
    if not texts:
        raise SystemExit(f"No documents found in {document_file}.")
    source_char_count = sum(len(text) for text in texts)
    fit_repeat = max(1, math.ceil(args.fit_target_chars / max(source_char_count, 1)))
    encode_repeat = max(1, math.ceil(args.encode_target_chars / max(source_char_count, 1)))
    fit_texts = texts * fit_repeat
    encode_texts = texts * encode_repeat
    fit_chars = sum(len(text) for text in fit_texts)
    encode_chars = sum(len(text) for text in encode_texts)

    results: list[dict[str, object]] = []
    for raw_config in args.configs:
        config = parse_config(str(raw_config))
        tokenizer_path = artifact_dir / f"{config['name']}.json"
        if tokenizer_path.exists():
            tokenizer_path.unlink()
        fit_started = time.perf_counter()
        tokenizer = build_reasoning_tokenizer(
            fit_texts,
            kind=str(config["kind"]),
            vocab_size=args.vocab_size,
            task=args.task,
            min_freq=args.min_freq,
            candidate_pool_size=args.candidate_pool_size,
            max_piece_chars=args.max_piece_chars,
            fit_workers=int(config["fit_workers"]),
            fit_verbose=False,
            load_path="",
            save_path=str(tokenizer_path),
        )
        fit_seconds = time.perf_counter() - fit_started

        warmup = encode_texts[: args.batch_size]
        if warmup:
            tokenizer.encode_batch(warmup, add_eos=True)

        encode_tokens = 0
        encode_started = time.perf_counter()
        for index in range(0, len(encode_texts), args.batch_size):
            token_batches = tokenizer.encode_batch(
                encode_texts[index:index + args.batch_size],
                add_eos=True,
            )
            encode_tokens += sum(len(batch) for batch in token_batches)
        encode_seconds = time.perf_counter() - encode_started

        results.append(
            {
                "name": config["name"],
                "kind": config["kind"],
                "fit_workers": config["fit_workers"],
                "fit_seconds": fit_seconds,
                "fit_chars": fit_chars,
                "fit_chars_per_s": fit_chars / max(fit_seconds, 1e-9),
                "encode_seconds": encode_seconds,
                "encode_chars": encode_chars,
                "encode_chars_per_s": encode_chars / max(encode_seconds, 1e-9),
                "encode_tokens": encode_tokens,
                "encode_tokens_per_s": encode_tokens / max(encode_seconds, 1e-9),
                "tokenizer_summary": tokenizer.summary(),
                "tokenizer_path": str(tokenizer_path),
            }
        )

    payload = {
        "mode": "reasoning_tokenizer_speed_benchmark",
        "created_at_unix_s": time.time(),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "document_file": str(document_file),
        "source_document_count": len(texts),
        "source_char_count": source_char_count,
        "fit_repeat": fit_repeat,
        "fit_chars": fit_chars,
        "encode_repeat": encode_repeat,
        "encode_chars": encode_chars,
        "encode_docs": len(encode_texts),
        "batch_size": args.batch_size,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
