from __future__ import annotations

"""Modality-neutral intermediate representation for abstract reasoning tasks."""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Tuple

try:
    from .stage1_latent_sampler import Program, RoleVar, TraceStep
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import Program, RoleVar, TraceStep  # type: ignore


@dataclass(frozen=True)
class EntitySpec:
    entity_id: str
    label: str
    kind: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantitySpec:
    quantity_id: str
    value: int | float
    unit: str
    owner_id: str | None = None
    role: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalSpec:
    target_id: str
    query: str
    unit: str


@dataclass(frozen=True)
class ChoiceSpec:
    choice_id: str
    text: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AbstractReasoningTask:
    task_id: str
    source_modality: str
    source_text: str
    entities: Tuple[EntitySpec, ...]
    quantities: Tuple[QuantitySpec, ...]
    relations: Tuple[Dict[str, Any], ...]
    goal: GoalSpec
    program: Program
    trace_template: Tuple[TraceStep, ...]
    answer: int | float | str
    concept_tags: Tuple[str, ...]
    difficulty: int
    answer_format: str = "open"
    choices: Tuple[ChoiceSpec, ...] = ()
    correct_choice: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)

    def to_state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_modality": self.source_modality,
            "source_text": self.source_text,
            "entities": _encode(self.entities),
            "quantities": _encode(self.quantities),
            "relations": _encode(self.relations),
            "goal": _encode(self.goal),
            "program": _encode(self.program),
            "answer": self.answer,
            "answer_format": self.answer_format,
            "choices": _encode(self.choices),
            "correct_choice": self.correct_choice,
            "concept_tags": list(self.concept_tags),
            "difficulty": self.difficulty,
            "metadata": _encode(self.metadata),
        }


def _encode(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, RoleVar):
        return asdict(obj)
    if isinstance(obj, tuple):
        return [_encode(item) for item in obj]
    if isinstance(obj, list):
        return [_encode(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if is_dataclass(obj):
        return {key: _encode(value) for key, value in asdict(obj).items()}
    return obj
