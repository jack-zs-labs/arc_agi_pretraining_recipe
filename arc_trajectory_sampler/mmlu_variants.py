from __future__ import annotations

"""Load MMLU benchmark variants from official Hugging Face sources."""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

try:
    from .mmlu_parser import (
        MMLUExample,
        MMLURow,
        choice_ids_for_count,
        compile_mmlu_examples,
        parse_mmlu_rows,
        slug,
    )
    from .stage4_trajectory_dataset import TrajectoryRecord
except ImportError:  # pragma: no cover - direct script execution
    from mmlu_parser import (  # type: ignore
        MMLUExample,
        MMLURow,
        choice_ids_for_count,
        compile_mmlu_examples,
        parse_mmlu_rows,
        slug,
    )
    from stage4_trajectory_dataset import TrajectoryRecord  # type: ignore


HF_DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASETS_SERVER_SPLITS_URL = "https://datasets-server.huggingface.co/splits"

MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
ALL_MMLU_PRO_SPLITS = ("validation", "test")
DEFAULT_MMLU_PRO_SPLITS = ("validation",)
MMLU_PRO_EVAL_SPLITS = frozenset(ALL_MMLU_PRO_SPLITS)

MMLU_REDUX_DATASET = "edinburgh-dawg/mmlu-redux-2.0"
DEFAULT_MMLU_REDUX_SPLIT = "test"
MMLU_REDUX_EVAL_SPLITS = frozenset({DEFAULT_MMLU_REDUX_SPLIT})
REDUX_LABEL_MODE_CHOICES = ("original", "corrected_single")

_MMLU_PRO_SUBJECT_RE = re.compile(r"(?:cot_lib[-_]|mmlu[-_/])([a-z0-9_]+)$")


def load_hf_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    max_rows: int | None = None,
    page_size: int = 100,
) -> Tuple[Dict[str, Any], ...]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    remaining = max_rows
    while remaining is None or remaining > 0:
        length = page_size if remaining is None else min(page_size, remaining)
        params = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        with urlopen(f"{HF_DATASETS_SERVER_ROWS_URL}?{params}") as response:
            payload = json.load(response)
        batch = [item["row"] for item in payload.get("rows", ())]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if remaining is not None:
            remaining -= len(batch)
        if len(batch) < length or payload.get("partial", False) is False and len(rows) >= payload.get("num_rows_total", 0):
            break
    return tuple(rows)


def load_hf_split_configs(dataset: str) -> Tuple[str, ...]:
    params = urlencode({"dataset": dataset})
    with urlopen(f"{HF_DATASETS_SERVER_SPLITS_URL}?{params}") as response:
        payload = json.load(response)
    configs = sorted({item["config"] for item in payload.get("splits", ())})
    return tuple(configs)


def validate_mmlu_pro_splits(
    splits: Sequence[str],
    *,
    allow_eval_splits: bool,
) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(split) for split in splits))
    blocked = sorted(split for split in normalized if split in MMLU_PRO_EVAL_SPLITS)
    if blocked and not allow_eval_splits:
        blocked_text = ", ".join(blocked)
        raise ValueError(
            f"MMLU-Pro eval split(s) requested without explicit opt-in: {blocked_text}. "
            "Pass allow_eval_splits=True only for audit/evaluation paths."
        )
    return normalized


def validate_mmlu_redux_split(
    split: str,
    *,
    allow_eval_splits: bool,
) -> str:
    normalized = str(split)
    if normalized in MMLU_REDUX_EVAL_SPLITS and not allow_eval_splits:
        raise ValueError(
            "MMLU-Redux rows are benchmark material. Pass allow_eval_splits=True only for audit/evaluation paths."
        )
    return normalized


def subject_from_mmlu_pro_row(row: Dict[str, Any]) -> str:
    src = str(row.get("src", "")).strip()
    match = _MMLU_PRO_SUBJECT_RE.search(src)
    if match:
        return match.group(1)
    return slug(str(row.get("category", "miscellaneous"))) or "miscellaneous"


def normalize_choice_answer(answer: Any, *, choice_count: int) -> str | None:
    choice_ids = choice_ids_for_count(choice_count)
    if isinstance(answer, int):
        if 0 <= answer < choice_count:
            return choice_ids[answer]
        return None
    text = str(answer).strip().upper()
    if text in choice_ids:
        return text
    if text.isdigit():
        index = int(text)
        if 0 <= index < choice_count:
            return choice_ids[index]
    return None


def convert_mmlu_pro_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    split: str,
) -> Tuple[MMLURow, ...]:
    converted: List[MMLURow] = []
    for local_index, row in enumerate(rows):
        choices = tuple(str(option) for option in row.get("options", ()))
        if len(choices) < 2:
            continue
        answer_label = normalize_choice_answer(row.get("answer"), choice_count=len(choices))
        if answer_label is None:
            answer_label = normalize_choice_answer(row.get("answer_index"), choice_count=len(choices))
        if answer_label is None:
            continue
        converted.append(
            MMLURow(
                split=split,
                subject=subject_from_mmlu_pro_row(row),
                index=int(row.get("question_id", local_index)),
                question=str(row.get("question", "")),
                choices=choices,
                answer=answer_label,
                dataset_name="mmlu_pro",
                metadata={
                    "source_dataset": "mmlu_pro",
                    "mmlu_pro_category": row.get("category"),
                    "mmlu_pro_question_id": row.get("question_id"),
                    "mmlu_pro_src": row.get("src"),
                    "mmlu_pro_answer_index": row.get("answer_index"),
                },
            )
        )
    return tuple(converted)


def build_mmlu_pro_examples(
    *,
    splits: Sequence[str] = DEFAULT_MMLU_PRO_SPLITS,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[MMLUExample, ...], Dict[str, int]]:
    resolved_splits = validate_mmlu_pro_splits(splits, allow_eval_splits=allow_eval_splits)
    rows: List[MMLURow] = []
    remaining = max_rows
    for split in resolved_splits:
        split_max_rows = remaining
        fetched = load_hf_rows(dataset=MMLU_PRO_DATASET, config="default", split=split, max_rows=split_max_rows)
        converted = convert_mmlu_pro_rows(fetched, split=split)
        rows.extend(converted)
        if remaining is not None:
            remaining -= len(converted)
            if remaining <= 0:
                break
    return parse_mmlu_rows(rows)


def build_mmlu_pro_trajectories(
    *,
    splits: Sequence[str] = DEFAULT_MMLU_PRO_SPLITS,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[TrajectoryRecord, ...], Dict[str, int]]:
    examples, failures = build_mmlu_pro_examples(
        splits=splits,
        allow_eval_splits=allow_eval_splits,
        max_rows=max_rows,
    )
    return compile_mmlu_examples(examples), failures


def convert_mmlu_redux_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    subject: str,
    split: str,
    label_mode: str,
) -> Tuple[Tuple[MMLURow, ...], Dict[str, int]]:
    converted: List[MMLURow] = []
    failure_counts: Counter[str] = Counter()
    for local_index, row in enumerate(rows):
        choices = tuple(str(option) for option in row.get("choices", ()))
        if len(choices) < 2:
            failure_counts["invalid_choice_count"] += 1
            continue
        answer_label = normalize_choice_answer(row.get("answer"), choice_count=len(choices))
        if answer_label is None:
            failure_counts["invalid_original_answer"] += 1
            continue

        error_type = str(row.get("error_type", "ok"))
        corrected_choice = normalize_choice_answer(row.get("correct_answer"), choice_count=len(choices))
        resolved_answer = answer_label
        if label_mode == "corrected_single":
            if error_type in {"no_correct_answer", "multiple_correct_answers"}:
                failure_counts[f"unsupported_{error_type}"] += 1
                continue
            if error_type == "wrong_groundtruth" and corrected_choice is not None:
                resolved_answer = corrected_choice

        converted.append(
            MMLURow(
                split=split,
                subject=subject,
                index=local_index,
                question=str(row.get("question", "")),
                choices=choices,
                answer=resolved_answer,
                dataset_name="mmlu_redux",
                metadata={
                    "source_dataset": "mmlu_redux",
                    "mmlu_redux_error_type": error_type,
                    "mmlu_redux_source": row.get("source"),
                    "mmlu_redux_correct_answer": row.get("correct_answer"),
                    "mmlu_redux_potential_reason": row.get("potential_reason"),
                    "mmlu_redux_original_answer": answer_label,
                    "mmlu_redux_label_mode": label_mode,
                },
            )
        )
    return tuple(converted), dict(failure_counts)


def build_mmlu_redux_examples(
    *,
    configs: Sequence[str] | None = None,
    split: str = DEFAULT_MMLU_REDUX_SPLIT,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
    label_mode: str = "corrected_single",
) -> Tuple[Tuple[MMLUExample, ...], Dict[str, int]]:
    if label_mode not in REDUX_LABEL_MODE_CHOICES:
        raise ValueError(f"Unsupported MMLU-Redux label_mode: {label_mode}")
    resolved_split = validate_mmlu_redux_split(split, allow_eval_splits=allow_eval_splits)
    resolved_configs = tuple(configs) if configs is not None else load_hf_split_configs(MMLU_REDUX_DATASET)
    remaining = max_rows
    rows: List[MMLURow] = []
    failure_counts: Counter[str] = Counter()
    for subject in resolved_configs:
        subject_rows = load_hf_rows(
            dataset=MMLU_REDUX_DATASET,
            config=subject,
            split=resolved_split,
            max_rows=remaining,
        )
        converted, subject_failures = convert_mmlu_redux_rows(
            subject_rows,
            subject=subject,
            split=resolved_split,
            label_mode=label_mode,
        )
        rows.extend(converted)
        failure_counts.update(subject_failures)
        if remaining is not None:
            remaining -= len(converted)
            if remaining <= 0:
                break
    examples, parser_failures = parse_mmlu_rows(rows)
    failure_counts.update(parser_failures)
    return examples, dict(failure_counts)


def build_mmlu_redux_trajectories(
    *,
    configs: Sequence[str] | None = None,
    split: str = DEFAULT_MMLU_REDUX_SPLIT,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
    label_mode: str = "corrected_single",
) -> Tuple[Tuple[TrajectoryRecord, ...], Dict[str, int]]:
    examples, failures = build_mmlu_redux_examples(
        configs=configs,
        split=split,
        allow_eval_splits=allow_eval_splits,
        max_rows=max_rows,
        label_mode=label_mode,
    )
    return compile_mmlu_examples(examples), failures
