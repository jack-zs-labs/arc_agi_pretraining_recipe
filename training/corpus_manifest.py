from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping


MANIFEST_FORMAT_VERSION = 2
_SPLITS: tuple[str, ...] = ("train", "val")


@dataclass(frozen=True)
class PretrainingDocument:
    corpus: str
    doc_id: str
    text: str
    band: str = "pgen"
    source_split: str = "train"
    preferred_split: str | None = None
    holdout_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def char_length(self) -> int:
        return len(self.text)

    def byte_length(self) -> int:
        return len(self.text.encode("utf-8"))

    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PretrainingDocumentRecord:
    corpus: str
    split: str
    doc_id: str
    band: str
    source_split: str
    char_length: int
    byte_length: int
    text_sha256: str
    holdout_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def assign_split_for_doc_id(
    doc_id: str,
    *,
    validation_fraction: float,
    seed: int = 0,
) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the interval [0, 1).")
    if validation_fraction <= 0.0:
        return "train"
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)
    return "val" if sample < validation_fraction else "train"


def build_document_record(
    document: PretrainingDocument,
    *,
    split: str,
) -> PretrainingDocumentRecord:
    if split not in (*_SPLITS, "holdout"):
        raise ValueError(f"Unsupported split {split!r}; expected one of {_SPLITS} or 'holdout'.")
    return PretrainingDocumentRecord(
        corpus=document.corpus,
        split=split,
        doc_id=document.doc_id,
        band=document.band,
        source_split=document.source_split,
        char_length=document.char_length(),
        byte_length=document.byte_length(),
        text_sha256=document.text_sha256(),
        holdout_group=document.holdout_group,
        metadata=dict(document.metadata),
    )


def _document_row(
    document: PretrainingDocument,
    *,
    split: str,
) -> dict[str, Any]:
    record = build_document_record(document, split=split)
    payload = asdict(record)
    payload["text"] = document.text
    return payload


def write_pretraining_manifest(
    *,
    output_dir: str | Path,
    documents: Iterable[PretrainingDocument],
    validation_fraction: float,
    seed: int = 0,
    corpus_name: str = "",
    source_name: str = "",
    selection_parameters: Mapping[str, object] | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    documents_dir = output_path / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    document_paths = {
        split: documents_dir / f"{split}_documents.jsonl"
        for split in _SPLITS
    }
    handles = {
        split: document_paths[split].open("w", encoding="utf-8")
        for split in _SPLITS
    }
    holdout_paths: dict[str, Path] = {}
    holdout_handles: dict[str, Any] = {}
    counts = Counter()
    corpus_counts: dict[str, Counter[str]] = defaultdict(Counter)
    band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    char_counts = Counter()
    byte_counts = Counter()
    preview_ids: dict[str, list[str]] = {split: [] for split in _SPLITS}
    holdout_counts = Counter()
    holdout_corpus_counts: dict[str, Counter[str]] = defaultdict(Counter)
    holdout_band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    holdout_char_counts = Counter()
    holdout_byte_counts = Counter()
    holdout_preview_ids: dict[str, list[str]] = defaultdict(list)
    try:
        for document in documents:
            if document.holdout_group is not None:
                holdout_group = str(document.holdout_group)
                if holdout_group not in holdout_handles:
                    holdout_path = documents_dir / f"holdout_{holdout_group}.jsonl"
                    holdout_paths[holdout_group] = holdout_path
                    holdout_handles[holdout_group] = holdout_path.open("w", encoding="utf-8")
                row = _document_row(document, split="holdout")
                holdout_handles[holdout_group].write(json.dumps(row, sort_keys=True))
                holdout_handles[holdout_group].write("\n")
                holdout_counts[holdout_group] += 1
                holdout_counts["total"] += 1
                holdout_corpus_counts[holdout_group][document.corpus] += 1
                holdout_band_counts[holdout_group][document.band] += 1
                holdout_char_counts[holdout_group] += int(row["char_length"])
                holdout_char_counts["total"] += int(row["char_length"])
                holdout_byte_counts[holdout_group] += int(row["byte_length"])
                holdout_byte_counts["total"] += int(row["byte_length"])
                if len(holdout_preview_ids[holdout_group]) < 5:
                    holdout_preview_ids[holdout_group].append(document.doc_id)
                continue
            split = assign_split_for_doc_id(
                document.doc_id,
                validation_fraction=validation_fraction,
                seed=seed,
            )
            if document.preferred_split is not None:
                if document.preferred_split not in _SPLITS:
                    raise ValueError(
                        f"Unsupported preferred_split {document.preferred_split!r}; expected one of {_SPLITS}."
                    )
                split = document.preferred_split
            row = _document_row(document, split=split)
            handles[split].write(json.dumps(row, sort_keys=True))
            handles[split].write("\n")
            counts[split] += 1
            counts["total"] += 1
            corpus_counts[split][document.corpus] += 1
            band_counts[split][document.band] += 1
            band_counts["total"][document.band] += 1
            char_counts[split] += int(row["char_length"])
            char_counts["total"] += int(row["char_length"])
            byte_counts[split] += int(row["byte_length"])
            byte_counts["total"] += int(row["byte_length"])
            if len(preview_ids[split]) < 5:
                preview_ids[split].append(document.doc_id)
    finally:
        for handle in handles.values():
            handle.close()
        for handle in holdout_handles.values():
            handle.close()

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "created_at_unix_s": time.time(),
        "corpus_name": corpus_name or "pretraining_corpus",
        "source_name": source_name or "custom",
        "validation_fraction": validation_fraction,
        "seed": seed,
        "document_files": {
            split: str(path.resolve())
            for split, path in document_paths.items()
        },
        "document_counts": {split: counts.get(split, 0) for split in (*_SPLITS, "total")},
        "band_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(band_counts.items())
        },
        "char_counts": {split: char_counts.get(split, 0) for split in (*_SPLITS, "total")},
        "byte_counts": {split: byte_counts.get(split, 0) for split in (*_SPLITS, "total")},
        "corpus_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in corpus_counts.items()
        },
        "preview_doc_ids": preview_ids,
        "holdout_files": {
            group: str(path.resolve())
            for group, path in sorted(holdout_paths.items())
        },
        "holdout_counts": {
            group: holdout_counts.get(group, 0)
            for group in (*sorted(holdout_paths.keys()), "total")
        },
        "holdout_band_counts": {
            group: dict(sorted(counter.items()))
            for group, counter in sorted(holdout_band_counts.items())
        },
        "holdout_char_counts": {
            group: holdout_char_counts.get(group, 0)
            for group in (*sorted(holdout_paths.keys()), "total")
        },
        "holdout_byte_counts": {
            group: holdout_byte_counts.get(group, 0)
            for group in (*sorted(holdout_paths.keys()), "total")
        },
        "holdout_corpus_counts": {
            group: dict(sorted(counter.items()))
            for group, counter in sorted(holdout_corpus_counts.items())
        },
        "holdout_preview_doc_ids": {
            group: list(ids)
            for group, ids in sorted(holdout_preview_ids.items())
        },
        "selection_parameters": dict(selection_parameters or {}),
        "extra_metadata": dict(extra_metadata or {}),
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def iter_pretraining_document_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_pretraining_document_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    return tuple(iter_pretraining_document_rows(path))


def read_pretraining_document_records(path: str | Path) -> tuple[PretrainingDocumentRecord, ...]:
    rows = read_pretraining_document_rows(path)
    return tuple(
        PretrainingDocumentRecord(
            corpus=str(row["corpus"]),
            split=str(row["split"]),
            doc_id=str(row["doc_id"]),
            band=str(row.get("band", "pgen")),
            source_split=str(row["source_split"]),
            char_length=int(row["char_length"]),
            byte_length=int(row["byte_length"]),
            text_sha256=str(row["text_sha256"]),
            holdout_group=(
                None
                if row.get("holdout_group") in {None, ""}
                else str(row.get("holdout_group"))
            ),
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    )
