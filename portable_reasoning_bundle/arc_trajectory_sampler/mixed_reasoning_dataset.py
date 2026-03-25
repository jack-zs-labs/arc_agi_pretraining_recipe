from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .gsm8k_reasoning_parser import build_gsm8k_trajectories
from .mmlu_parser import build_mmlu_trajectories
from .stage1_latent_sampler import sample_latent_rule
from .stage2_episode_sampler import sample_episode
from .stage4_trajectory_dataset import TrajectoryRecord, build_trajectories
from .state_adapter import encode_workspace, serialize_workspace_text, verifier_targets


@dataclass(frozen=True)
class ReasoningTextExample:
    benchmark: str
    text: str
    trajectory_id: str
    step_index: int
    trace_step: str


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
