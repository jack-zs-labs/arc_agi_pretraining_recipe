from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from arc_trajectory_sampler.dclm_corpus import (
    DEFAULT_DCLM_DATASET_ID,
    DEFAULT_DCLM_SPLIT,
    DEFAULT_DCLM_TEXT_FIELD,
    iter_dclm_documents,
)
from arc_trajectory_sampler.mixed_reasoning_dataset import (
    DEFAULT_GSM8K_DATA_DIR,
    DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS,
    ReasoningTextExample,
    build_arc_reasoning_examples,
    build_core_reasoning_examples,
    build_gsm8k_reasoning_examples,
    build_mmlu_pro_reasoning_examples,
    build_mmlu_reasoning_examples,
    build_mmlu_redux_reasoning_examples,
    build_olympiad_math_pretraining_examples,
    build_oscar_graph_reasoning_examples,
    build_oscar_scope_examples,
    build_oscar_scope_reasoning_examples,
    serialize_trajectory_records,
)
from arc_trajectory_sampler.core_loader import DEFAULT_CORE_DATA_DIR
from arc_trajectory_sampler.gsm8k_reasoning_parser import build_gsm8k_examples, build_gsm8k_trajectories
from arc_trajectory_sampler.mmlu_parser import DEFAULT_MMLU_AUDIT_SPLITS
from arc_trajectory_sampler.mmlu_parser import DEFAULT_DATA_DIR as DEFAULT_MMLU_DATA_DIR
from arc_trajectory_sampler.mmlu_parser import build_mmlu_examples, build_mmlu_trajectories
from arc_trajectory_sampler.olympiad_math_parser import OLYMPIAD_MATH_EVAL_CONFIGS
from arc_trajectory_sampler.olympiad_math_parser import build_olympiad_math_examples, build_olympiad_math_trajectories
from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES
from arc_trajectory_sampler.oscar_scope_corpus import DEFAULT_OSCAR_SCOPE_VIEWS
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES
from training.corpus_manifest import PretrainingDocument, assign_split_for_doc_id


def build_dclm_pretraining_documents(
    *,
    dataset_id: str = DEFAULT_DCLM_DATASET_ID,
    split: str = DEFAULT_DCLM_SPLIT,
    text_field: str = DEFAULT_DCLM_TEXT_FIELD,
    max_documents: int | None = None,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10_000,
    seed: int = 0,
    min_text_chars: int = 0,
    min_language_score: float = 0.0,
    min_fasttext_score: float = 0.0,
    language_allowlist: Sequence[str] = ("en",),
) -> tuple[PretrainingDocument, ...]:
    return tuple(
        PretrainingDocument(
            corpus="dclm",
            doc_id=document.row_id,
            text=document.text,
            band="pgen",
            source_split=split,
            metadata={
                "dataset_id": dataset_id,
                "url": document.url,
                "language": document.language,
                "language_score": document.language_score,
                "fasttext_score": document.fasttext_score,
                "text_field": text_field,
            },
        )
        for document in iter_dclm_documents(
            dataset_id=dataset_id,
            split=split,
            text_field=text_field,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            max_documents=max_documents,
            min_text_chars=min_text_chars,
            min_language_score=min_language_score,
            min_fasttext_score=min_fasttext_score,
            language_allowlist=language_allowlist,
        )
    )


def _reasoning_examples_to_documents(
    examples: Sequence[ReasoningTextExample],
    *,
    source_split: str,
    band: str,
    holdout_group: str | None = None,
) -> tuple[PretrainingDocument, ...]:
    return tuple(
        PretrainingDocument(
            corpus=str(example.benchmark),
            doc_id=f"{example.benchmark}:{example.trajectory_id}:{example.trace_step}:{example.step_index}",
            text=example.text,
            band=band,
            source_split=source_split,
            holdout_group=holdout_group,
            metadata={
                "benchmark": example.benchmark,
                "trajectory_id": example.trajectory_id,
                "trace_step": example.trace_step,
                "step_index": example.step_index,
                **dict(example.auxiliary_targets or {}),
            },
        )
        for example in examples
    )


def build_oscar_scope_pretraining_documents(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
    max_documents: int | None = None,
    max_chunks: int | None = None,
    views: Sequence[str] = DEFAULT_OSCAR_SCOPE_VIEWS,
) -> tuple[PretrainingDocument, ...]:
    examples = build_oscar_scope_examples(
        roots=roots,
        paths=paths,
        auto_discover=auto_discover,
        max_documents=max_documents,
        max_chunks=max_chunks,
        views=views,
    )
    return _reasoning_examples_to_documents(examples, source_split="local_oscar", band="pgen")


def build_oscar_scope_reasoning_pretraining_documents(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
    max_documents: int | None = None,
    max_examples: int | None = None,
    views: Sequence[str] = DEFAULT_OSCAR_SCOPE_VIEWS,
    families: Sequence[str] = OSCAR_SCOPE_REASONING_FAMILIES,
) -> tuple[PretrainingDocument, ...]:
    examples = build_oscar_scope_reasoning_examples(
        roots=roots,
        paths=paths,
        auto_discover=auto_discover,
        max_documents=max_documents,
        max_examples=max_examples,
        views=views,
        families=families,
    )
    return _reasoning_examples_to_documents(examples, source_split="local_oscar_reasoning", band="ptraj")


def build_oscar_graph_reasoning_pretraining_documents(
    *,
    max_examples: int | None = None,
    families: Sequence[str] = OSCAR_GRAPH_REASONING_FAMILIES,
) -> tuple[PretrainingDocument, ...]:
    examples = build_oscar_graph_reasoning_examples(
        max_examples=max_examples,
        families=families,
    )
    return _reasoning_examples_to_documents(examples, source_split="local_oscar_graph", band="ptraj")


def _format_gsm8k_training_document(example: object) -> str:
    question = str(getattr(example, "source_text", "")).strip()
    task = getattr(example, "abstract_task", None)
    task_metadata = dict(getattr(task, "metadata", {}) or {})
    worked_answer = str(task_metadata.get("gsm8k_answer", "")).strip()
    final_answer = str(getattr(task, "answer", "")).strip()
    sections = [
        "Benchmark: GSM8K",
        "Split: train",
        "",
        "Question:",
        question,
    ]
    if worked_answer:
        sections.extend(["", "Worked Answer:", worked_answer])
    if final_answer:
        sections.extend(["", "Canonical Final Answer:", final_answer])
    return "\n".join(sections).strip()


def _format_mmlu_training_document(example: object) -> str:
    row = getattr(example, "row", None)
    if row is None:
        raise ValueError("Expected MMLU example with attached row.")
    question = str(getattr(row, "question", "")).strip()
    choices = tuple(str(choice) for choice in getattr(row, "choices", ()))
    answer = str(getattr(row, "answer", "")).strip()
    answer_index = ord(answer) - ord("A") if len(answer) == 1 else -1
    answer_text = choices[answer_index] if 0 <= answer_index < len(choices) else ""
    sections = [
        "Benchmark: MMLU",
        f"Split: {getattr(row, 'split', 'auxiliary_train')}",
        f"Subject: {getattr(row, 'subject', '')}",
        "",
        "Question:",
        question,
        "",
        "Choices:",
    ]
    for index, choice in enumerate(choices):
        label = chr(ord("A") + index)
        sections.append(f"{label}. {choice}")
    if answer:
        sections.extend(["", "Correct Choice:", answer])
    if answer_text:
        sections.extend(["", "Correct Choice Text:", answer_text])
    return "\n".join(sections).strip()


def _format_olympiad_math_training_document(example: object) -> str:
    row = getattr(example, "row", None)
    if row is None:
        raise ValueError("Expected OlymMATH example with attached row.")
    sections = [
        "Benchmark: Olympiad Math",
        f"Config: {getattr(row, 'config', '')}",
        f"Subject: {getattr(row, 'subject', '')}",
        f"Difficulty: {getattr(row, 'difficulty_tier', '')}",
        f"Variant: {getattr(row, 'task_variant', '')}",
        "",
        "Problem:",
        str(getattr(row, "problem", "")).strip(),
        "",
        "Answer:",
        str(getattr(row, "answer", "")).strip(),
    ]
    formal_statement = str(getattr(row, "formal_statement", "") or "").strip()
    formal_proof = str(getattr(row, "formal_proof", "") or "").strip()
    if formal_statement:
        sections.extend(["", "Formal Statement:", formal_statement])
    if formal_proof:
        sections.extend(["", "Formal Proof:", formal_proof])
    return "\n".join(sections).strip()


def build_benchmark_training_pretraining_documents(
    *,
    gsm8k_data_dir: str | Path = DEFAULT_GSM8K_DATA_DIR,
    gsm8k_max_rows: int = 0,
    mmlu_data_dir: str | Path = DEFAULT_MMLU_DATA_DIR,
    mmlu_max_rows: int = 0,
    olympiad_math_configs: Sequence[str] = DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS,
    olympiad_math_max_rows: int = 0,
) -> tuple[PretrainingDocument, ...]:
    documents: list[PretrainingDocument] = []

    if gsm8k_max_rows > 0:
        examples, _failures = build_gsm8k_examples(
            data_dir=gsm8k_data_dir,
            splits=("train",),
            allow_eval_splits=False,
            max_rows=gsm8k_max_rows,
        )
        for example in examples:
            task = getattr(example, "abstract_task", None)
            task_metadata = dict(getattr(task, "metadata", {}) or {})
            split = str(task_metadata.get("gsm8k_split", "train"))
            index = int(task_metadata.get("gsm8k_index", 0))
            example_id = str(getattr(example, "example_id", f"gsm8k_train_{index}"))
            documents.append(
                PretrainingDocument(
                    corpus="gsm8k",
                    doc_id=f"gsm8k_train_native:{example_id}",
                    text=_format_gsm8k_training_document(example),
                    band="pgen",
                    source_split=split,
                    metadata={
                        "benchmark": "gsm8k",
                        "benchmark_doc_style": "native_train",
                        "gsm8k_index": index,
                        "example_id": example_id,
                    },
                )
            )

    if mmlu_max_rows > 0:
        examples, _failures = build_mmlu_examples(
            data_dir=mmlu_data_dir,
            splits=("auxiliary_train",),
            allow_eval_splits=False,
            max_rows=mmlu_max_rows,
        )
        for example in examples:
            row = getattr(example, "row", None)
            if row is None:
                continue
            example_id = str(getattr(example, "example_id", ""))
            documents.append(
                PretrainingDocument(
                    corpus="mmlu",
                    doc_id=f"mmlu_train_native:{example_id}",
                    text=_format_mmlu_training_document(example),
                    band="pgen",
                    source_split=str(getattr(row, "split", "auxiliary_train")),
                    metadata={
                        "benchmark": "mmlu",
                        "benchmark_doc_style": "native_train",
                        "subject": str(getattr(row, "subject", "")),
                        "row_index": int(getattr(row, "index", 0)),
                        "dataset_name": str(getattr(row, "dataset_name", "mmlu")),
                        "example_id": example_id,
                    },
                )
            )

    if olympiad_math_max_rows > 0:
        examples, _failures = build_olympiad_math_examples(
            configs=olympiad_math_configs,
            allow_eval_configs=True,
            max_rows=olympiad_math_max_rows,
        )
        for example in examples:
            row = getattr(example, "row", None)
            if row is None:
                continue
            example_id = str(getattr(example, "example_id", ""))
            documents.append(
                PretrainingDocument(
                    corpus="olympiad_math",
                    doc_id=f"olympiad_math_train_native:{example_id}",
                    text=_format_olympiad_math_training_document(example),
                    band="pgen",
                    source_split="train",
                    metadata={
                        "benchmark": "olympiad_math",
                        "benchmark_doc_style": "native_train",
                        "config": str(getattr(row, "config", "")),
                        "subject": str(getattr(row, "subject", "")),
                        "difficulty_tier": str(getattr(row, "difficulty_tier", "")),
                        "task_variant": str(getattr(row, "task_variant", "")),
                        "unique_id": str(getattr(row, "unique_id", "")),
                        "example_id": example_id,
                    },
                )
            )

    return tuple(documents)


def build_ptraj_pretraining_documents(
    *,
    arc_episodes: int = 0,
    arc_seed_start: int = 0,
    gsm8k_data_dir: str | Path = DEFAULT_GSM8K_DATA_DIR,
    gsm8k_max_rows: int = 0,
    mmlu_data_dir: str | Path = DEFAULT_MMLU_DATA_DIR,
    mmlu_max_rows: int = 0,
    olympiad_math_configs: Sequence[str] = DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS,
    olympiad_math_max_rows: int = 0,
    core_data_dir: str | Path = DEFAULT_CORE_DATA_DIR,
    core_max_rows: int = 0,
    core_languages: Sequence[str] | None = None,
    core_categories: Sequence[str] | None = None,
    core_dependency_kinds: Sequence[str] | None = None,
    core_graph_backend: str = "auto",
    include_verifier_targets: bool = False,
) -> tuple[PretrainingDocument, ...]:
    documents: list[PretrainingDocument] = []

    if arc_episodes > 0:
        documents.extend(
            _reasoning_examples_to_documents(
                build_arc_reasoning_examples(
                    num_episodes=arc_episodes,
                    seed_start=arc_seed_start,
                    include_verifier_targets=include_verifier_targets,
                ),
                source_split="train",
                band="ptraj",
            )
        )
    if gsm8k_max_rows > 0:
        documents.extend(
            _reasoning_examples_to_documents(
                build_gsm8k_reasoning_examples(
                    data_dir=gsm8k_data_dir,
                    max_rows=gsm8k_max_rows,
                    include_verifier_targets=include_verifier_targets,
                ),
                source_split="train",
                band="ptraj",
            )
        )
    if mmlu_max_rows > 0:
        documents.extend(
            _reasoning_examples_to_documents(
                build_mmlu_reasoning_examples(
                    data_dir=mmlu_data_dir,
                    max_rows=mmlu_max_rows,
                    include_verifier_targets=include_verifier_targets,
                ),
                source_split="auxiliary_train",
                band="ptraj",
            )
        )
    if olympiad_math_max_rows > 0:
        documents.extend(
            _reasoning_examples_to_documents(
                build_olympiad_math_pretraining_examples(
                    configs=olympiad_math_configs,
                    max_rows=olympiad_math_max_rows,
                    include_verifier_targets=include_verifier_targets,
                ),
                source_split="train",
                band="ptraj",
            )
        )
    if core_max_rows > 0:
        documents.extend(
            _reasoning_examples_to_documents(
                build_core_reasoning_examples(
                    data_dir=core_data_dir,
                    max_rows=core_max_rows,
                    languages=core_languages,
                    categories=core_categories,
                    dependency_kinds=core_dependency_kinds,
                    graph_backend=core_graph_backend,
                ),
                source_split="train",
                band="ptraj",
            )
        )
    return tuple(documents)


def build_mmlu_pro_benchmark_documents(
    *,
    max_rows: int,
) -> tuple[PretrainingDocument, ...]:
    if max_rows <= 0:
        return ()
    return _reasoning_examples_to_documents(
        build_mmlu_pro_reasoning_examples(max_rows=max_rows, include_verifier_targets=False),
        source_split="validation",
        band="pbench",
        holdout_group="mmlu_pro",
    )


def build_mmlu_redux_benchmark_documents(
    *,
    max_rows: int,
    label_mode: str = "corrected_single",
) -> tuple[PretrainingDocument, ...]:
    if max_rows <= 0:
        return ()
    return _reasoning_examples_to_documents(
        build_mmlu_redux_reasoning_examples(
            max_rows=max_rows,
            include_verifier_targets=False,
            label_mode=label_mode,
        ),
        source_split="test",
        band="pbench",
        holdout_group="mmlu_redux",
    )


def build_gsm8k_benchmark_documents(
    *,
    data_dir: str | Path = DEFAULT_GSM8K_DATA_DIR,
    max_rows: int,
) -> tuple[PretrainingDocument, ...]:
    if max_rows <= 0:
        return ()
    records, _failures = build_gsm8k_trajectories(
        data_dir=data_dir,
        splits=("test",),
        allow_eval_splits=True,
        max_rows=max_rows,
    )
    return _reasoning_examples_to_documents(
        serialize_trajectory_records(records, include_verifier_targets=False),
        source_split="test",
        band="pbench",
        holdout_group="gsm8k_test",
    )


def build_mmlu_benchmark_documents(
    *,
    data_dir: str | Path = DEFAULT_MMLU_DATA_DIR,
    max_rows: int,
    splits: Sequence[str] = DEFAULT_MMLU_AUDIT_SPLITS,
) -> tuple[PretrainingDocument, ...]:
    if max_rows <= 0:
        return ()
    records, _failures = build_mmlu_trajectories(
        data_dir=data_dir,
        splits=splits,
        allow_eval_splits=True,
        max_rows=max_rows,
    )
    return _reasoning_examples_to_documents(
        serialize_trajectory_records(records, include_verifier_targets=False),
        source_split="audit",
        band="pbench",
        holdout_group="mmlu_audit",
    )


def build_olympiad_math_benchmark_documents(
    *,
    configs: Sequence[str] = OLYMPIAD_MATH_EVAL_CONFIGS,
    max_rows: int,
) -> tuple[PretrainingDocument, ...]:
    if max_rows <= 0:
        return ()
    records, _failures = build_olympiad_math_trajectories(
        configs=configs,
        allow_eval_configs=True,
        max_rows=max_rows,
    )
    return _reasoning_examples_to_documents(
        serialize_trajectory_records(records, include_verifier_targets=False),
        source_split="test",
        band="pbench",
        holdout_group="olympiad_math_eval",
    )


def repeat_documents_on_train_split(
    documents: Sequence[PretrainingDocument],
    *,
    train_repeat: int,
    validation_fraction: float,
    seed: int,
) -> tuple[PretrainingDocument, ...]:
    resolved_repeat = max(int(train_repeat), 1)
    repeated: list[PretrainingDocument] = []
    for document in documents:
        if document.holdout_group is not None:
            repeated.append(document)
            continue
        split = assign_split_for_doc_id(
            document.doc_id,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        base_metadata = dict(document.metadata)
        base_metadata.setdefault("base_doc_id", document.doc_id)
        if split == "val":
            repeated.append(
                replace(
                    document,
                    preferred_split="val",
                    metadata={
                        **base_metadata,
                        "repeat_index": 0,
                        "train_repeat_factor": resolved_repeat,
                    },
                )
            )
            continue
        for repeat_index in range(resolved_repeat):
            doc_id = document.doc_id if repeat_index == 0 else f"{document.doc_id}::repeat_{repeat_index}"
            repeated.append(
                PretrainingDocument(
                    corpus=document.corpus,
                    doc_id=doc_id,
                    text=document.text,
                    band=document.band,
                    source_split=document.source_split,
                    preferred_split="train",
                    holdout_group=document.holdout_group,
                    metadata={
                        **base_metadata,
                        "repeat_index": repeat_index,
                        "train_repeat_factor": resolved_repeat,
                    },
                )
            )
    return tuple(repeated)
