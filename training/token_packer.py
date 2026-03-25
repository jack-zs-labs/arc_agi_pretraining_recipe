from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from models.reasoning_tokenizer import (
    ReasoningTokenizer,
    build_reasoning_tokenizer,
    resolved_tokenizer_load_path,
)
from training.corpus_manifest import iter_pretraining_document_rows


PACKED_MANIFEST_FORMAT_VERSION = 1
SUPPORTED_PACKED_SPLITS: tuple[str, ...] = ("train", "val")
UINT16_MAX = int(np.iinfo(np.uint16).max)
UINT32_MAX = int(np.iinfo(np.uint32).max)
PACK_TOKENIZER_BATCH_SIZE = 256
PACK_TOKENIZER_CACHE_SIZE = 4096


@dataclass(frozen=True)
class PackedShardMetadata:
    split: str
    shard_index: int
    sequence_count: int
    token_count: int
    start_sequence_index: int
    bin_path: str
    idx_path: str


def read_document_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    document_files = manifest.get("document_files")
    if not isinstance(document_files, dict):
        raise ValueError(f"{manifest_path} does not look like a pretraining document manifest.")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def document_file_for_split(manifest: Mapping[str, object], split: str) -> Path:
    if split not in SUPPORTED_PACKED_SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {SUPPORTED_PACKED_SPLITS}.")
    document_files = manifest.get("document_files", {})
    if not isinstance(document_files, Mapping) or split not in document_files:
        raise KeyError(f"Document manifest is missing a file for split {split!r}.")
    return Path(str(document_files[split])).resolve()


def iter_manifest_split_rows(manifest: Mapping[str, object], split: str) -> Iterable[dict[str, Any]]:
    return iter_pretraining_document_rows(document_file_for_split(manifest, split))


def token_dtype_name_for_vocab_size(vocab_size: int) -> str:
    if vocab_size <= UINT16_MAX:
        return np.dtype(np.uint16).name
    if vocab_size <= UINT32_MAX:
        return np.dtype(np.uint32).name
    raise ValueError(f"Unsupported vocab size {vocab_size}; token ids exceed uint32.")


def token_dtype_from_name(name: str) -> np.dtype:
    normalized = np.dtype(name).name
    if normalized not in {np.dtype(np.uint16).name, np.dtype(np.uint32).name}:
        raise ValueError(f"Unsupported packed token dtype {name!r}.")
    return np.dtype(normalized)


def _stable_doc_score(doc_id: str, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_tokenizer_fit_rows(
    manifest: Mapping[str, object],
    *,
    split: str = "train",
    max_documents: int | None,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    rows = iter_manifest_split_rows(manifest, split)
    if max_documents is None:
        return tuple(rows)
    if max_documents <= 0:
        raise ValueError("max_documents must be positive when provided.")
    heap: list[tuple[int, str, int, dict[str, Any]]] = []
    for ordinal, row in enumerate(rows):
        doc_id = str(row["doc_id"])
        score = _stable_doc_score(doc_id, seed=seed)
        candidate = (-score, doc_id, ordinal, row)
        if len(heap) < max_documents:
            heapq.heappush(heap, candidate)
            continue
        worst = heap[0]
        if candidate > worst:
            heapq.heapreplace(heap, candidate)
    selected = [(-neg_score, doc_id, ordinal, row) for neg_score, doc_id, ordinal, row in heap]
    selected.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(row for _score, _doc_id, _ordinal, row in selected)


def build_tokenizer_for_packing(
    *,
    manifest: Mapping[str, object],
    kind: str,
    vocab_size: int,
    task: str,
    min_freq: int,
    candidate_pool_size: int,
    max_piece_chars: int,
    fit_workers: int,
    fit_verbose: bool,
    load_path: str = "",
    save_path: str = "",
    fit_max_documents: int | None = None,
    fit_seed: int = 0,
) -> tuple[ReasoningTokenizer, dict[str, object]]:
    if kind == "byte":
        tokenizer = build_reasoning_tokenizer(
            (),
            kind=kind,
            vocab_size=vocab_size,
            task=task,
            min_freq=min_freq,
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            fit_workers=fit_workers,
            fit_verbose=fit_verbose,
            load_path="",
            save_path="",
        )
        return tokenizer, {
            "mode": "builtin",
            "fit_document_count": 0,
            "fit_text_count": 0,
            "fit_max_documents": fit_max_documents,
            "fit_seed": fit_seed,
        }

    if load_path:
        tokenizer = build_reasoning_tokenizer(
            (),
            kind=kind,
            vocab_size=vocab_size,
            task=task,
            min_freq=min_freq,
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            fit_workers=fit_workers,
            fit_verbose=fit_verbose,
            load_path=load_path,
            save_path="",
        )
        return tokenizer, {
            "mode": "loaded",
            "fit_document_count": 0,
            "fit_text_count": 0,
            "fit_max_documents": fit_max_documents,
            "fit_seed": fit_seed,
        }

    fit_rows = sample_tokenizer_fit_rows(
        manifest,
        split="train",
        max_documents=fit_max_documents,
        seed=fit_seed,
    )
    fit_texts: list[str] = []
    seen_fit_texts: set[str] = set()
    duplicate_fit_text_count = 0
    for row in fit_rows:
        text = str(row["text"])
        if text in seen_fit_texts:
            duplicate_fit_text_count += 1
            continue
        seen_fit_texts.add(text)
        fit_texts.append(text)
    tokenizer = build_reasoning_tokenizer(
        fit_texts,
        kind=kind,
        vocab_size=vocab_size,
        task=task,
        min_freq=min_freq,
        candidate_pool_size=candidate_pool_size,
        max_piece_chars=max_piece_chars,
        fit_workers=fit_workers,
        fit_verbose=fit_verbose,
        load_path="",
        save_path=save_path,
    )
    return tokenizer, {
        "mode": "fit",
        "fit_document_count": len(fit_rows),
        "fit_text_count": len(fit_texts),
        "fit_unique_text_count": len(fit_texts),
        "fit_duplicate_text_count": duplicate_fit_text_count,
        "fit_max_documents": fit_max_documents,
        "fit_seed": fit_seed,
        "fit_preview_doc_ids": [str(row["doc_id"]) for row in fit_rows[:5]],
    }


def _flush_packed_shard(
    *,
    split: str,
    split_dir: Path,
    shard_index: int,
    sample_length: int,
    token_dtype: np.dtype,
    sequences: Sequence[Sequence[int]],
    start_sequence_index: int,
) -> PackedShardMetadata:
    if not sequences:
        raise ValueError("Cannot flush an empty shard.")
    split_dir.mkdir(parents=True, exist_ok=True)
    bin_path = split_dir / f"shard_{shard_index:05d}.bin"
    idx_path = split_dir / f"shard_{shard_index:05d}.idx"
    token_array = np.asarray(sequences, dtype=token_dtype)
    if token_array.ndim != 2 or token_array.shape[1] != sample_length:
        raise ValueError(
            f"Expected packed token array of shape [N, {sample_length}], got {token_array.shape}."
        )
    token_array.tofile(bin_path)
    offsets = np.arange(
        0,
        (token_array.shape[0] + 1) * sample_length,
        sample_length,
        dtype=np.uint64,
    )
    offsets.tofile(idx_path)
    return PackedShardMetadata(
        split=split,
        shard_index=shard_index,
        sequence_count=int(token_array.shape[0]),
        token_count=int(token_array.size),
        start_sequence_index=start_sequence_index,
        bin_path=str(bin_path.resolve()),
        idx_path=str(idx_path.resolve()),
    )


def pack_manifest_split(
    *,
    manifest: Mapping[str, object],
    split: str,
    output_dir: str | Path,
    tokenizer: ReasoningTokenizer,
    seq_len: int,
    target_shard_sequences: int,
    pad_final_window: bool,
) -> dict[str, object]:
    if target_shard_sequences <= 0:
        raise ValueError("target_shard_sequences must be positive.")
    sample_length = seq_len + 1
    token_dtype = token_dtype_from_name(token_dtype_name_for_vocab_size(int(tokenizer.vocab_size)))
    split_dir = Path(output_dir) / split
    sequences: list[list[int]] = []
    shard_payloads: list[dict[str, object]] = []
    token_stream: list[int] = []
    token_stream_start = 0
    document_count = 0
    document_token_count = 0
    corpus_document_counts: Counter[str] = Counter()
    corpus_document_token_counts: Counter[str] = Counter()
    band_document_counts: Counter[str] = Counter()
    band_document_token_counts: Counter[str] = Counter()
    tokenizer_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    tokenizer_cache_hits = 0
    tokenizer_unique_encode_count = 0
    tokenizer_reused_document_count = 0
    tokenizer_cache_peak_size = 0
    emitted_sequences = 0
    shard_index = 0
    row_batch: list[dict[str, Any]] = []

    def process_row_batch(batch: Sequence[dict[str, Any]]) -> None:
        nonlocal document_count, document_token_count, emitted_sequences, shard_index
        nonlocal token_stream, token_stream_start, sequences
        nonlocal tokenizer_cache_hits, tokenizer_unique_encode_count, tokenizer_reused_document_count, tokenizer_cache_peak_size
        if not batch:
            return
        missing_texts: list[str] = []
        pending_texts: set[str] = set()
        for row in batch:
            text = str(row["text"])
            if text in tokenizer_cache:
                tokenizer_cache_hits += 1
                tokenizer_reused_document_count += 1
                tokenizer_cache.move_to_end(text)
                continue
            if text in pending_texts:
                tokenizer_reused_document_count += 1
                continue
            pending_texts.add(text)
            missing_texts.append(text)
        if missing_texts:
            encoded_missing_batches = tokenizer.encode_batch(
                missing_texts,
                add_eos=True,
            )
            tokenizer_unique_encode_count += len(missing_texts)
            for text, tokens in zip(missing_texts, encoded_missing_batches, strict=True):
                tokenizer_cache[text] = tuple(int(token_id) for token_id in tokens)
                tokenizer_cache_peak_size = max(tokenizer_cache_peak_size, len(tokenizer_cache))
                while len(tokenizer_cache) > PACK_TOKENIZER_CACHE_SIZE:
                    tokenizer_cache.popitem(last=False)
        for row in batch:
            text = str(row["text"])
            corpus = str(row.get("corpus", "unknown"))
            band = str(row.get("band", "pgen"))
            resolved_tokens = tokenizer_cache[text]
            document_count += 1
            document_token_count += len(resolved_tokens)
            corpus_document_counts[corpus] += 1
            corpus_document_token_counts[corpus] += len(resolved_tokens)
            band_document_counts[band] += 1
            band_document_token_counts[band] += len(resolved_tokens)
            token_stream.extend(resolved_tokens)
            while len(token_stream) - token_stream_start >= sample_length:
                sequences.append(token_stream[token_stream_start:token_stream_start + sample_length])
                token_stream_start += sample_length
                if len(sequences) >= target_shard_sequences:
                    shard = _flush_packed_shard(
                        split=split,
                        split_dir=split_dir,
                        shard_index=shard_index,
                        sample_length=sample_length,
                        token_dtype=token_dtype,
                        sequences=sequences,
                        start_sequence_index=emitted_sequences,
                    )
                    shard_payloads.append({
                        "split": shard.split,
                        "shard_index": shard.shard_index,
                        "sequence_count": shard.sequence_count,
                        "token_count": shard.token_count,
                        "start_sequence_index": shard.start_sequence_index,
                        "bin_path": shard.bin_path,
                        "idx_path": shard.idx_path,
                    })
                    emitted_sequences += shard.sequence_count
                    shard_index += 1
                    sequences = []
            if token_stream_start >= max(sample_length * 4, 65_536):
                token_stream = token_stream[token_stream_start:]
                token_stream_start = 0

    for row in iter_manifest_split_rows(manifest, split):
        row_batch.append(row)
        if len(row_batch) >= PACK_TOKENIZER_BATCH_SIZE:
            process_row_batch(row_batch)
            row_batch = []
    process_row_batch(row_batch)

    remainder_tokens = len(token_stream) - token_stream_start
    if remainder_tokens > 0 and pad_final_window:
        pad_token_id = int(tokenizer.window_pad_token_id)
        final_sequence = list(token_stream[token_stream_start:]) + [pad_token_id] * (sample_length - remainder_tokens)
        sequences.append(final_sequence)
        remainder_tokens = 0

    if sequences:
        shard = _flush_packed_shard(
            split=split,
            split_dir=split_dir,
            shard_index=shard_index,
            sample_length=sample_length,
            token_dtype=token_dtype,
            sequences=sequences,
            start_sequence_index=emitted_sequences,
        )
        shard_payloads.append({
            "split": shard.split,
            "shard_index": shard.shard_index,
            "sequence_count": shard.sequence_count,
            "token_count": shard.token_count,
            "start_sequence_index": shard.start_sequence_index,
            "bin_path": shard.bin_path,
            "idx_path": shard.idx_path,
        })
        emitted_sequences += shard.sequence_count

    return {
        "split": split,
        "document_file": str(document_file_for_split(manifest, split)),
        "document_count": document_count,
        "document_token_count": document_token_count,
        "corpus_document_counts": dict(sorted(corpus_document_counts.items())),
        "corpus_document_token_counts": dict(sorted(corpus_document_token_counts.items())),
        "band_document_counts": dict(sorted(band_document_counts.items())),
        "band_document_token_counts": dict(sorted(band_document_token_counts.items())),
        "tokenizer_cache_hits": tokenizer_cache_hits,
        "tokenizer_unique_encode_count": tokenizer_unique_encode_count,
        "tokenizer_reused_document_count": tokenizer_reused_document_count,
        "tokenizer_cache_peak_size": tokenizer_cache_peak_size,
        "sequence_count": emitted_sequences,
        "packed_token_count": emitted_sequences * sample_length,
        "remainder_tokens_dropped": remainder_tokens,
        "shards": shard_payloads,
    }


def pack_pretraining_document_manifest(
    *,
    document_manifest_path: str | Path,
    output_dir: str | Path,
    tokenizer_kind: str,
    tokenizer_vocab_size: int,
    tokenizer_task: str,
    tokenizer_min_freq: int,
    tokenizer_candidate_pool_size: int,
    tokenizer_max_piece_chars: int,
    tokenizer_fit_workers: int,
    tokenizer_fit_verbose: bool,
    tokenizer_load_path: str = "",
    tokenizer_save_path: str = "",
    tokenizer_fit_max_documents: int | None = None,
    tokenizer_fit_seed: int = 0,
    seq_len: int = 2048,
    target_shard_sequences: int = 8192,
    pad_final_window: bool = False,
) -> dict[str, object]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    document_manifest = read_document_manifest(document_manifest_path)
    tokenizer, tokenizer_fit = build_tokenizer_for_packing(
        manifest=document_manifest,
        kind=tokenizer_kind,
        vocab_size=tokenizer_vocab_size,
        task=tokenizer_task,
        min_freq=tokenizer_min_freq,
        candidate_pool_size=tokenizer_candidate_pool_size,
        max_piece_chars=tokenizer_max_piece_chars,
        fit_workers=tokenizer_fit_workers,
        fit_verbose=tokenizer_fit_verbose,
        load_path=tokenizer_load_path,
        save_path=tokenizer_save_path,
        fit_max_documents=tokenizer_fit_max_documents,
        fit_seed=tokenizer_fit_seed,
    )
    tokenizer_summary = tokenizer.summary()
    packed_splits = {
        split: pack_manifest_split(
            manifest=document_manifest,
            split=split,
            output_dir=output_path / "shards",
            tokenizer=tokenizer,
            seq_len=seq_len,
            target_shard_sequences=target_shard_sequences,
            pad_final_window=pad_final_window,
        )
        for split in SUPPORTED_PACKED_SPLITS
    }
    token_dtype_name = token_dtype_name_for_vocab_size(int(tokenizer.vocab_size))
    sample_length = seq_len + 1
    manifest = {
        "format_version": PACKED_MANIFEST_FORMAT_VERSION,
        "created_at_unix_s": time.time(),
        "document_manifest_path": str(Path(document_manifest["manifest_path"]).resolve()),
        "document_manifest_source_name": document_manifest.get("source_name"),
        "document_manifest_corpus_name": document_manifest.get("corpus_name"),
        "document_manifest_document_counts": dict(document_manifest.get("document_counts", {})),
        "document_manifest_corpus_counts": dict(document_manifest.get("corpus_counts", {})),
        "document_manifest_band_counts": dict(document_manifest.get("band_counts", {})),
        "document_manifest_holdout_counts": dict(document_manifest.get("holdout_counts", {})),
        "document_manifest_selection_parameters": dict(document_manifest.get("selection_parameters", {})),
        "document_manifest_extra_metadata": dict(document_manifest.get("extra_metadata", {})),
        "seq_len": seq_len,
        "sample_length": sample_length,
        "target_shard_sequences": target_shard_sequences,
        "pad_final_window": pad_final_window,
        "token_dtype": token_dtype_name,
        "tokenizer": tokenizer_summary,
        "tokenizer_path": resolved_tokenizer_load_path(tokenizer_summary, manifest_dir=output_path),
        "tokenizer_fit": tokenizer_fit,
        "splits": packed_splits,
        "total_sequence_count": sum(int(split_payload["sequence_count"]) for split_payload in packed_splits.values()),
        "total_packed_token_count": sum(int(split_payload["packed_token_count"]) for split_payload in packed_splits.values()),
    }
    manifest_path = output_path / "packed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest
