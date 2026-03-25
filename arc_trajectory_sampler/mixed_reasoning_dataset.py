from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from .core_loader import DEFAULT_CORE_DATA_DIR
from .core_reasoning_adapter import build_core_reasoning_tasks, serialize_core_reasoning_task
from .dclm_corpus import DEFAULT_DCLM_DATASET_ID, DEFAULT_DCLM_SPLIT, DEFAULT_DCLM_TEXT_FIELD, iter_dclm_documents
from .gsm8k_reasoning_parser import build_gsm8k_trajectories
from .mmlu_parser import DEFAULT_DATA_DIR as DEFAULT_MMLU_DATA_DIR
from .mmlu_parser import build_mmlu_trajectories
from .mmlu_variants import build_mmlu_pro_trajectories, build_mmlu_redux_trajectories
from .olympiad_math_parser import DEFAULT_OLYMPIAD_MATH_CONFIGS, build_olympiad_math_trajectories
from .oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES, build_oscar_graph_reasoning_tasks
from .oscar_scope_corpus import DEFAULT_OSCAR_SCOPE_VIEWS, build_oscar_scope_records
from .oscar_scope_reasoning import OSCAR_SCOPE_REASONING_FAMILIES, build_oscar_scope_reasoning_tasks
from .stage1_latent_sampler import sample_latent_rule
from .stage2_episode_sampler import sample_episode
from .stage4_trajectory_dataset import TrajectoryRecord, build_trajectories
from .state_adapter import encode_workspace, serialize_workspace_text, verifier_targets

DEFAULT_GSM8K_DATA_DIR = "arc_trajectory_sampler/data/gsm8k"
DEFAULT_PTRAJ_OLYMPIAD_MATH_CONFIGS = (*DEFAULT_OLYMPIAD_MATH_CONFIGS, "lean")


@dataclass(frozen=True)
class ReasoningTextExample:
    benchmark: str
    text: str
    trajectory_id: str
    step_index: int
    trace_step: str
    auxiliary_targets: dict[str, object] | None = None


def serialize_action_target(name: str, action: dict[str, object]) -> str:
    return json.dumps(
        {
            "name": name,
            "action": action,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def serialize_trajectory_records(
    records: Sequence[TrajectoryRecord],
    *,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    examples: list[ReasoningTextExample] = []
    for record in records:
        previous_action = "<start>"
        for step_index, step in enumerate(record.steps):
            encoded = encode_workspace(record, step_index, include_verifier=False)
            target_action = serialize_action_target(step.name, step.action)
            text = serialize_workspace_text(
                encoded,
                previous_action=previous_action,
                target_action=target_action,
                verifier_state=verifier_targets(record, step_index) if include_verifier_targets else None,
            )
            examples.append(
                ReasoningTextExample(
                    benchmark=str(encoded.metadata["dataset"]),
                    text=text,
                    trajectory_id=str(record.trajectory_id),
                    step_index=step_index,
                    trace_step=str(step.name),
                    auxiliary_targets=None,
                )
            )
            previous_action = target_action
    return tuple(examples)


def build_arc_reasoning_examples(
    *,
    num_episodes: int,
    seed_start: int = 0,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    records: list[TrajectoryRecord] = []
    for offset in range(num_episodes):
        seed = seed_start + offset
        latent = sample_latent_rule(seed=seed)
        episode = sample_episode(latent, seed=seed)
        records.extend(build_trajectories(episode, include_test=True))
    return serialize_trajectory_records(records, include_verifier_targets=include_verifier_targets)


def build_gsm8k_reasoning_examples(
    *,
    data_dir: str | Path,
    max_rows: int | None = None,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    records, _ = build_gsm8k_trajectories(
        data_dir=data_dir,
        splits=("train",),
        allow_eval_splits=False,
        max_rows=max_rows,
    )
    return serialize_trajectory_records(records, include_verifier_targets=include_verifier_targets)


def build_mmlu_reasoning_examples(
    *,
    data_dir: str | Path,
    max_rows: int | None = None,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    records, _ = build_mmlu_trajectories(
        data_dir=data_dir,
        splits=("auxiliary_train",),
        allow_eval_splits=False,
        max_rows=max_rows,
    )
    return serialize_trajectory_records(records, include_verifier_targets=include_verifier_targets)


def build_mmlu_pro_reasoning_examples(
    *,
    max_rows: int | None = None,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    records, _ = build_mmlu_pro_trajectories(
        splits=("validation",),
        allow_eval_splits=True,
        max_rows=max_rows,
    )
    return serialize_trajectory_records(records, include_verifier_targets=include_verifier_targets)


def build_mmlu_redux_reasoning_examples(
    *,
    max_rows: int | None = None,
    include_verifier_targets: bool = False,
    label_mode: str = "corrected_single",
) -> tuple[ReasoningTextExample, ...]:
    records, _ = build_mmlu_redux_trajectories(
        split="test",
        allow_eval_splits=True,
        max_rows=max_rows,
        label_mode=label_mode,
    )
    return serialize_trajectory_records(records, include_verifier_targets=include_verifier_targets)


def build_olympiad_math_reasoning_examples(
    *,
    configs: Sequence[str],
    max_rows: int | None = None,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    records, _ = build_olympiad_math_trajectories(
        configs=configs,
        allow_eval_configs=True,
        max_rows=max_rows,
    )
    return serialize_trajectory_records(records, include_verifier_targets=include_verifier_targets)


def build_olympiad_math_pretraining_examples(
    *,
    configs: Sequence[str],
    max_rows: int | None = None,
    include_verifier_targets: bool = False,
) -> tuple[ReasoningTextExample, ...]:
    return build_olympiad_math_reasoning_examples(
        configs=configs,
        max_rows=max_rows,
        include_verifier_targets=include_verifier_targets,
    )


def build_core_reasoning_examples(
    *,
    data_dir: str | Path,
    max_rows: int | None = None,
    languages: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    dependency_kinds: Sequence[str] | None = None,
    graph_backend: str = "auto",
) -> tuple[ReasoningTextExample, ...]:
    tasks = build_core_reasoning_tasks(
        data_dir=data_dir,
        max_examples=max_rows,
        languages=languages,
        categories=categories,
        dependency_kinds=dependency_kinds,
        graph_backend=graph_backend,
    )
    return tuple(
        ReasoningTextExample(
            benchmark="core",
            text=serialize_core_reasoning_task(task),
            trajectory_id=task.task_id,
            step_index=0,
            trace_step=task.trace_step,
            auxiliary_targets=task.auxiliary_targets,
        )
        for task in tasks
    )


def build_oscar_scope_examples(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
    max_documents: int | None = None,
    max_chunks: int | None = None,
    views: Sequence[str] = DEFAULT_OSCAR_SCOPE_VIEWS,
) -> tuple[ReasoningTextExample, ...]:
    records = build_oscar_scope_records(
        roots=roots,
        paths=paths,
        auto_discover=auto_discover,
        max_documents=max_documents,
        max_chunks=max_chunks,
        views=views,
    )
    return tuple(
        ReasoningTextExample(
            benchmark=record.benchmark,
            text=record.text,
            trajectory_id=record.document_id,
            step_index=record.chunk_index,
            trace_step=record.view,
            auxiliary_targets=record.metadata,
        )
        for record in records
    )


def build_oscar_scope_reasoning_examples(
    *,
    roots: Sequence[str | Path] = (),
    paths: Sequence[str | Path] = (),
    auto_discover: bool = True,
    max_documents: int | None = None,
    max_examples: int | None = None,
    views: Sequence[str] = DEFAULT_OSCAR_SCOPE_VIEWS,
    families: Sequence[str] = OSCAR_SCOPE_REASONING_FAMILIES,
) -> tuple[ReasoningTextExample, ...]:
    tasks = build_oscar_scope_reasoning_tasks(
        roots=roots,
        paths=paths,
        auto_discover=auto_discover,
        max_documents=max_documents,
        max_examples=max_examples,
        views=views,
        families=families,
    )
    return tuple(
        ReasoningTextExample(
            benchmark=task.benchmark,
            text=task.text,
            trajectory_id=str(task.metadata.get("trajectory_id", task.task_id)),
            step_index=int(task.metadata.get("step_index", 0)),
            trace_step=task.trace_step,
            auxiliary_targets=task.metadata,
        )
        for task in tasks
    )


def build_oscar_graph_reasoning_examples(
    *,
    max_examples: int | None = None,
    families: Sequence[str] = OSCAR_GRAPH_REASONING_FAMILIES,
) -> tuple[ReasoningTextExample, ...]:
    tasks = build_oscar_graph_reasoning_tasks(
        max_examples=max_examples,
        families=families,
    )
    return tuple(
        ReasoningTextExample(
            benchmark=task.benchmark,
            text=task.text,
            trajectory_id=str(task.metadata.get("trajectory_id", task.task_id)),
            step_index=int(task.metadata.get("step_index", 0)),
            trace_step=task.trace_step,
            auxiliary_targets=task.metadata,
        )
        for task in tasks
    )


def build_dclm_text_examples(
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
) -> tuple[ReasoningTextExample, ...]:
    return tuple(
        ReasoningTextExample(
            benchmark="dclm",
            text=document.text,
            trajectory_id=document.row_id,
            step_index=0,
            trace_step="document",
            auxiliary_targets={
                "dataset_id": dataset_id,
                "split": split,
                "url": document.url,
                "language": document.language,
                "language_score": document.language_score,
                "fasttext_score": document.fasttext_score,
            },
        )
        for document in iter_dclm_documents(
            dataset_id=dataset_id,
            split=split,
            text_field=text_field,
            max_documents=max_documents,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            min_text_chars=min_text_chars,
            min_language_score=min_language_score,
            min_fasttext_score=min_fasttext_score,
            language_allowlist=tuple(language_allowlist),
        )
    )


def build_ptraj_examples(
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
    oscar_scope_roots: Sequence[str | Path] = (),
    oscar_scope_paths: Sequence[str | Path] = (),
    oscar_scope_auto_discover: bool = True,
    oscar_scope_max_documents: int = 0,
    oscar_scope_max_chunks: int = 0,
    oscar_scope_views: Sequence[str] = DEFAULT_OSCAR_SCOPE_VIEWS,
    oscar_scope_reasoning_max_examples: int = 96,
    oscar_scope_reasoning_families: Sequence[str] = OSCAR_SCOPE_REASONING_FAMILIES,
    oscar_graph_reasoning_max_examples: int = 64,
    oscar_graph_reasoning_families: Sequence[str] = OSCAR_GRAPH_REASONING_FAMILIES,
    include_verifier_targets: bool = False,
    shuffle_seed: int | None = None,
) -> tuple[ReasoningTextExample, ...]:
    examples: list[ReasoningTextExample] = []

    if arc_episodes > 0:
        examples.extend(
            build_arc_reasoning_examples(
                num_episodes=arc_episodes,
                seed_start=arc_seed_start,
                include_verifier_targets=include_verifier_targets,
            )
        )
    if gsm8k_max_rows > 0:
        examples.extend(
            build_gsm8k_reasoning_examples(
                data_dir=gsm8k_data_dir,
                max_rows=gsm8k_max_rows,
                include_verifier_targets=include_verifier_targets,
            )
        )
    if mmlu_max_rows > 0:
        examples.extend(
            build_mmlu_reasoning_examples(
                data_dir=mmlu_data_dir,
                max_rows=mmlu_max_rows,
                include_verifier_targets=include_verifier_targets,
            )
        )
    if olympiad_math_max_rows > 0:
        examples.extend(
            build_olympiad_math_pretraining_examples(
                configs=olympiad_math_configs,
                max_rows=olympiad_math_max_rows,
                include_verifier_targets=include_verifier_targets,
            )
        )
    if core_max_rows > 0:
        examples.extend(
            build_core_reasoning_examples(
                data_dir=core_data_dir,
                max_rows=core_max_rows,
                languages=core_languages,
                categories=core_categories,
                dependency_kinds=core_dependency_kinds,
                graph_backend=core_graph_backend,
            )
        )
    if oscar_scope_max_documents > 0 or oscar_scope_max_chunks > 0:
        examples.extend(
            build_oscar_scope_examples(
                roots=oscar_scope_roots,
                paths=oscar_scope_paths,
                auto_discover=oscar_scope_auto_discover,
                max_documents=None if oscar_scope_max_documents <= 0 else oscar_scope_max_documents,
                max_chunks=None if oscar_scope_max_chunks <= 0 else oscar_scope_max_chunks,
                views=oscar_scope_views,
            )
        )
    if oscar_scope_reasoning_max_examples > 0:
        examples.extend(
            build_oscar_scope_reasoning_examples(
                roots=oscar_scope_roots,
                paths=oscar_scope_paths,
                auto_discover=oscar_scope_auto_discover,
                max_documents=None if oscar_scope_max_documents <= 0 else oscar_scope_max_documents,
                max_examples=oscar_scope_reasoning_max_examples,
                views=oscar_scope_views,
                families=oscar_scope_reasoning_families,
            )
        )
    if oscar_graph_reasoning_max_examples > 0:
        examples.extend(
            build_oscar_graph_reasoning_examples(
                max_examples=oscar_graph_reasoning_max_examples,
                families=oscar_graph_reasoning_families,
            )
        )

    if shuffle_seed is not None and len(examples) > 1:
        rng = random.Random(shuffle_seed)
        rng.shuffle(examples)

    return tuple(examples)


def split_examples(
    examples: Sequence[ReasoningTextExample],
    *,
    validation_fraction: float = 0.2,
) -> tuple[tuple[ReasoningTextExample, ...], tuple[ReasoningTextExample, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the open interval (0, 1).")
    if len(examples) < 2:
        return tuple(examples), ()
    split_index = max(1, min(len(examples) - 1, int(round(len(examples) * (1.0 - validation_fraction)))))
    return tuple(examples[:split_index]), tuple(examples[split_index:])


def texts_from_examples(examples: Iterable[ReasoningTextExample]) -> list[str]:
    return [example.text for example in examples]
