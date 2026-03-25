from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


DEFAULT_DCLM_DATASET_ID = "mlfoundations/dclm-baseline-1.0-parquet"
DEFAULT_DCLM_SPLIT = "train"
DEFAULT_DCLM_TEXT_FIELD = "text"


@dataclass(frozen=True)
class DCLMDocument:
    text: str
    row_id: str
    url: str
    language: str
    language_score: float | None
    fasttext_score: float | None


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "Hugging Face dataset streaming requires the `datasets` package. "
            "Install requirements.txt or requirements-models.txt."
        ) from exc
    return load_dataset


def _extract_text(row: dict[str, Any], *, text_field: str) -> str:
    if text_field in row and isinstance(row[text_field], str):
        return row[text_field]
    for candidate in ("text", "content", "raw_content", "document"):
        value = row.get(candidate)
        if isinstance(value, str):
            return value
    raise KeyError(
        f"Could not find a text field in DCLM row. Requested {text_field!r}; "
        f"available columns were {sorted(row.keys())}."
    )


def iter_dclm_documents(
    *,
    dataset_id: str = DEFAULT_DCLM_DATASET_ID,
    split: str = DEFAULT_DCLM_SPLIT,
    text_field: str = DEFAULT_DCLM_TEXT_FIELD,
    streaming: bool = True,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10_000,
    seed: int = 0,
    max_documents: int | None = None,
    min_text_chars: int = 0,
    min_language_score: float = 0.0,
    min_fasttext_score: float = 0.0,
    language_allowlist: tuple[str, ...] = ("en",),
) -> Iterable[DCLMDocument]:
    load_dataset = _require_datasets()
    dataset = load_dataset(dataset_id, split=split, streaming=streaming)
    if streaming and shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=max(1, shuffle_buffer_size))
    emitted = 0
    for row in dataset:
        text = _extract_text(row, text_field=text_field).strip()
        if min_text_chars > 0 and len(text) < min_text_chars:
            continue
        language = str(row.get("language", "") or "")
        if language_allowlist and language not in language_allowlist:
            continue
        language_score = row.get("language_score")
        if language_score is not None and float(language_score) < min_language_score:
            continue
        fasttext_score = row.get("fasttext_score")
        if fasttext_score is not None and float(fasttext_score) < min_fasttext_score:
            continue
        yield DCLMDocument(
            text=text,
            row_id=str(row.get("id", f"{dataset_id}:{split}:{emitted}")),
            url=str(row.get("url", "")),
            language=language,
            language_score=None if language_score is None else float(language_score),
            fasttext_score=None if fasttext_score is None else float(fasttext_score),
        )
        emitted += 1
        if max_documents is not None and emitted >= max_documents:
            break
