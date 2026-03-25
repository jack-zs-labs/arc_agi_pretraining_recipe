from __future__ import annotations

"""Parse official MMLU multiple-choice rows into the canonical reasoning IR."""

import argparse
import csv
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path
import re
import string
import shutil
import tarfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.request import urlretrieve

try:
    from .reasoning_ir import AbstractReasoningTask, ChoiceSpec, EntitySpec, GoalSpec, QuantitySpec
    from .stage1_latent_sampler import Program, TraceStep
    from .stage4_trajectory_dataset import (
        STEP_WEIGHTS,
        State,
        TrajectoryRecord,
        TrajectoryStep,
        reward_from_terms,
        symbolic_step_reward_terms,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution
    from reasoning_ir import AbstractReasoningTask, ChoiceSpec, EntitySpec, GoalSpec, QuantitySpec  # type: ignore
    from stage1_latent_sampler import Program, TraceStep  # type: ignore
    from stage4_trajectory_dataset import (  # type: ignore
        STEP_WEIGHTS,
        State,
        TrajectoryRecord,
        TrajectoryStep,
        reward_from_terms,
        symbolic_step_reward_terms,
        write_jsonl,
    )


MMLU_DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
DEFAULT_DATA_DIR = "arc_trajectory_sampler/data/mmlu"
DEFAULT_MMLU_TRAINING_SPLITS = ("auxiliary_train",)
DEFAULT_MMLU_AUDIT_SPLITS = ("dev", "val", "test")
ALL_MMLU_SPLITS = DEFAULT_MMLU_TRAINING_SPLITS + DEFAULT_MMLU_AUDIT_SPLITS
MMLU_EVAL_SPLITS = frozenset(DEFAULT_MMLU_AUDIT_SPLITS)
CHOICE_LABELS = tuple(string.ascii_uppercase)
MMLU_STEP_WEIGHTS = {
    "segment": 0.12,
    "bind": 0.18,
    "match": 0.18,
    "eliminate": 0.18,
    "select": 0.18,
    "render": 0.22,
}
ELIMINATION_FAMILIES = {
    "case_application_choice",
    "passage_reference_choice",
    "negation_exception_choice",
    "rule_application_choice",
    "statement_evaluation_choice",
    "comparative_inference_choice",
}
MODAL_PREFIXES = ("would ", "should ", "will ", "did ", "can ", "may ", "could ")
RULE_APPLICATION_CUES = (
    "court will most likely",
    "strongest constitutional argument",
    "most persuasive argument",
    "the court",
    "is liable",
    "did the search",
    "under the terms",
    "under these facts",
    "under these circumstances",
    "acceptable in contemporary practice",
    "reservation to the definition",
    "should the",
    "would the",
    "will the",
    "may the",
    "can the",
)
STATEMENT_EVALUATION_CUES = (
    "statement 1",
    "statement i",
    "which statement",
    "which of the following is true",
    "which of the following is correct",
    "is true of",
    "accurate regarding",
    "true of a",
    "true of an",
    "true of the",
    "describes the structural level",
)
DESCRIPTOR_MATCH_CUES = (
    "best describes",
    "best described",
    "best explains",
    "best reflects",
    "best exemplifies",
    "would best be classified",
    "best be classified",
    "what type of",
    "how can ",
    "relationship between",
    "which position",
    "what was his perspective",
    "what is wrong with",
    "according to ",
    "is known as",
    "fallacy of",
    "consists of",
    "the study of",
    "primarily involves",
)
COMPARATIVE_INFERENCE_CUES = (
    "most likely",
    "least likely",
    "most plausible",
    "least affected",
    "most accurate",
    "smallest",
    "largest",
    "lowest",
    "highest",
    "minimum number",
    "maximum number",
    "increase the",
    "decrease the",
    "in order of",
    "order of thermal stability",
    "would lead to",
    "must exist",
    "what will happen",
    "what is happening",
    "greatest positive impact",
    "most efficient method",
    "strongest connection",
    "which stage",
    "protective effect",
    "if the",
    "would have the",
    "guaranteed to incur the minimum",
    "characterized by",
    "would increase",
    "would decrease",
)
QUANTITATIVE_SUBJECTS = {
    "abstract_algebra",
    "astronomy",
    "college_mathematics",
    "college_physics",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
}
FACTUAL_SUBJECTS = {
    "anatomy",
    "global_facts",
    "high_school_european_history",
    "high_school_geography",
    "high_school_us_history",
    "high_school_world_history",
    "prehistory",
    "us_foreign_policy",
    "virology",
    "world_religions",
}
QUANTITATIVE_CUES = (
    "how many",
    "what is the value",
    "calculate",
    "probability",
    "percentage",
    "percent",
    "ratio",
    "sum of",
    "difference between",
)
FACTUAL_CUES = (
    "which of the following",
    "what is the",
    "who was",
    "where is",
    "when did",
    "which statement",
)
SCENARIO_CUES = (
    "this question refers to the following information",
    "read the following excerpt",
    "scenario 1",
    "comes to the office",
    "comes to the physician",
    "most likely",
    "strongest",
    "best describes",
    "best explains",
    "which of the following is not",
    "all of the following",
    "except",
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+/\d+|\d+)(?:%|x)?")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]*\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class MMLURow:
    split: str
    subject: str
    index: int
    question: str
    choices: Tuple[str, ...]
    answer: str
    dataset_name: str = "mmlu"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MMLUParserFailure:
    reason: str
    details: str = ""


@dataclass(frozen=True)
class MMLUExample:
    example_id: str
    row: MMLURow
    abstract_task: AbstractReasoningTask
    family_name: str
    template_name: str
    notes: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


def _encode(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_encode(item) for item in obj]
    if isinstance(obj, list):
        return [_encode(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if is_dataclass(obj):
        return {key: _encode(value) for key, value in asdict(obj).items()}
    return obj


def slug(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def choice_ids_for_count(choice_count: int) -> Tuple[str, ...]:
    if choice_count <= 0:
        raise ValueError("choice_count must be positive.")
    if choice_count > len(CHOICE_LABELS):
        raise ValueError(f"Unsupported choice count {choice_count}; max supported is {len(CHOICE_LABELS)}.")
    return CHOICE_LABELS[:choice_count]


def choice_ids_for_row(row: MMLURow) -> Tuple[str, ...]:
    return choice_ids_for_count(len(row.choices))


def choice_text_for_answer(choice_id: str, choices: Sequence[ChoiceSpec]) -> str:
    for choice in choices:
        if choice.choice_id == choice_id:
            return choice.text
    raise KeyError(f"Choice {choice_id!r} not present in row choices.")


def dataset_slug_for_row(row: MMLURow) -> str:
    return slug(row.dataset_name or "mmlu")


def choice_trace(match_description: str) -> Tuple[TraceStep, ...]:
    return (
        TraceStep("segment", "Parse the multiple-choice prompt, subject, and answer candidates."),
        TraceStep("bind", "Bind typed question features and candidate option signatures."),
        TraceStep("match", match_description),
        TraceStep("select", "Choose the best-supported answer option."),
        TraceStep("render", "Emit the full typed reasoning IR."),
    )


def elimination_trace(eliminate_description: str) -> Tuple[TraceStep, ...]:
    return (
        TraceStep("segment", "Parse the scenario, passage, or question stem into a decision context."),
        TraceStep("bind", "Bind the comparison cue, candidate choices, and subject frame."),
        TraceStep("eliminate", eliminate_description),
        TraceStep("select", "Choose the remaining best-supported answer option."),
        TraceStep("render", "Emit the full typed reasoning IR."),
    )


def ensure_mmlu_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    root = Path(data_dir)
    if (root / "dev").is_dir() and (root / "test").is_dir():
        return root
    if (root / "data" / "dev").is_dir() and (root / "data" / "test").is_dir():
        return root / "data"

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "data.tar"
    if not archive_path.exists():
        urlretrieve(MMLU_DATA_URL, archive_path)

    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.name.startswith("data/"):
                continue
            relative = Path(member.name).relative_to("data")
            destination = root / relative
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("wb") as target:
                if source is None:
                    continue
                shutil.copyfileobj(source, target)
    return root


def iter_split_rows(data_dir: Path, split: str) -> Iterable[MMLURow]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        return
    if split == "auxiliary_train":
        subject_paths = sorted(split_dir.glob("*.csv"))
    else:
        subject_paths = sorted(split_dir.glob(f"*_{split}.csv"))
    for subject_path in subject_paths:
        if split == "auxiliary_train":
            subject = subject_path.stem
        else:
            subject = subject_path.name[: -len(f"_{split}.csv")]
        with subject_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if len(row) < 6:
                    continue
                choices = tuple(cell.strip() for cell in row[1:5])
                answer = row[5].strip().upper()
                if answer not in choice_ids_for_count(len(choices)):
                    continue
                yield MMLURow(
                    split=split,
                    subject=subject,
                    index=index,
                    question=row[0].strip(),
                    choices=choices,
                    answer=answer,
                )


def load_mmlu_rows(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    splits: Sequence[str] = DEFAULT_MMLU_AUDIT_SPLITS,
    max_rows: int | None = None,
) -> Tuple[MMLURow, ...]:
    resolved = ensure_mmlu_files(data_dir)
    rows: List[MMLURow] = []
    row_limit = max_rows if max_rows is not None and max_rows > 0 else None
    for split in splits:
        for row in iter_split_rows(resolved, split):
            rows.append(row)
            if row_limit is not None and len(rows) >= row_limit:
                return tuple(rows)
    return tuple(rows)


def validate_mmlu_splits(
    splits: Sequence[str],
    *,
    allow_eval_splits: bool,
) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(split) for split in splits))
    blocked = sorted(split for split in normalized if split in MMLU_EVAL_SPLITS)
    if blocked and not allow_eval_splits:
        blocked_text = ", ".join(blocked)
        raise ValueError(
            f"MMLU eval split(s) requested without explicit opt-in: {blocked_text}. "
            "Pass allow_eval_splits=True or --allow-eval-splits only for audit/evaluation paths."
        )
    return normalized


def extract_numeric_fragments(text: str) -> Tuple[str, ...]:
    return tuple(match.group(0) for match in NUMBER_RE.finditer(text.replace(",", "")))


def parse_numeric_fragment(fragment: str) -> int | float | None:
    cleaned = fragment.strip().rstrip("xX")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    try:
        if "/" in cleaned and cleaned.count("/") == 1:
            numerator, denominator = cleaned.split("/")
            value = float(numerator) / float(denominator)
        else:
            value = float(cleaned)
    except ValueError:
        return None
    if percent:
        value = value / 100.0
    return int(value) if value.is_integer() else value


def normalize_tokens(text: str) -> Tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_RE.findall(text))


def focus_tokens(text: str) -> Tuple[str, ...]:
    tokens = [token for token in normalize_tokens(text) if token not in STOPWORDS and len(token) > 2]
    return tuple(tokens[:8])


def has_fill_blank(text: str) -> bool:
    return "_______" in text or "____" in text


def question_sentence_count(text: str) -> int:
    return len([segment for segment in re.split(r"[.!?]+", text) if segment.strip()])


def infer_family(row: MMLURow) -> str | None:
    lowered_question = row.question.lower()
    numeric_choice_count = sum(1 for choice in row.choices if extract_numeric_fragments(choice))
    has_numeric_cue = bool(extract_numeric_fragments(row.question)) or any(cue in lowered_question for cue in QUANTITATIVE_CUES)
    if row.subject in QUANTITATIVE_SUBJECTS or (has_numeric_cue and numeric_choice_count >= 2):
        return "quantitative_choice"

    option_token_lengths = [len(normalize_tokens(choice)) for choice in row.choices]
    short_options = option_token_lengths and max(option_token_lengths) <= 10
    low_numeric_density = numeric_choice_count == 0 and not extract_numeric_fragments(row.question)
    if row.subject in FACTUAL_SUBJECTS and short_options and low_numeric_density:
        return "factual_choice"
    if low_numeric_density and short_options and any(cue in lowered_question for cue in FACTUAL_CUES):
        return "factual_choice"
    if any(cue in lowered_question for cue in SCENARIO_CUES):
        return infer_elimination_family(row)
    if len(row.question) >= 180:
        return infer_elimination_family(row)
    if max(option_token_lengths or [0]) > 10:
        return infer_elimination_family(row)
    return infer_elimination_family(row)


def infer_entities(row: MMLURow) -> Tuple[EntitySpec, ...]:
    entities: List[EntitySpec] = [
        EntitySpec(
            entity_id=f"subject_{row.subject}",
            label=row.subject.replace("_", " "),
            kind="subject",
            attributes={"source": "mmlu_subject"},
        )
    ]
    seen_ids = {entities[0].entity_id}
    for token in CAPITALIZED_RE.findall(" ".join((row.question, *row.choices))):
        entity_id = f"term_{slug(token)}"
        if entity_id in seen_ids:
            continue
        entities.append(EntitySpec(entity_id=entity_id, label=token, kind="concept"))
        seen_ids.add(entity_id)
        if len(entities) >= 8:
            break
    return tuple(entities)


def infer_quantities(row: MMLURow) -> Tuple[QuantitySpec, ...]:
    quantities: List[QuantitySpec] = []
    for index, fragment in enumerate(extract_numeric_fragments(row.question)):
        parsed = parse_numeric_fragment(fragment)
        if parsed is None:
            continue
        quantities.append(
            QuantitySpec(
                quantity_id=f"q_stem_{index}",
                value=parsed,
                unit="scalar",
                role="question_numeric",
                attributes={"exact_text": fragment},
            )
        )
    for choice_id, choice_text in zip(choice_ids_for_row(row), row.choices):
        for index, fragment in enumerate(extract_numeric_fragments(choice_text)):
            parsed = parse_numeric_fragment(fragment)
            if parsed is None:
                continue
            quantities.append(
                QuantitySpec(
                    quantity_id=f"q_choice_{choice_id.lower()}_{index}",
                    value=parsed,
                    unit="scalar",
                    role=f"choice_{choice_id.lower()}_numeric",
                    attributes={"exact_text": fragment, "choice_id": choice_id},
                )
            )
    return tuple(quantities)


def build_choices(row: MMLURow) -> Tuple[ChoiceSpec, ...]:
    choices: List[ChoiceSpec] = []
    for choice_id, choice_text in zip(choice_ids_for_row(row), row.choices):
        choices.append(
            ChoiceSpec(
                choice_id=choice_id,
                text=choice_text,
                attributes={
                    "token_count": len(normalize_tokens(choice_text)),
                    "numeric_fragments": list(extract_numeric_fragments(choice_text)),
                    "keyword_tokens": list(focus_tokens(choice_text)),
                },
            )
        )
    return tuple(choices)


def quantitative_metadata(row: MMLURow, choices: Sequence[ChoiceSpec], quantities: Sequence[QuantitySpec]) -> Dict[str, Any]:
    dataset_slug = dataset_slug_for_row(row)
    choice_signatures = {
        choice.choice_id: list(choice.attributes.get("numeric_fragments", []))
        for choice in choices
    }
    return {
        "template": f"{dataset_slug}_quantitative_choice",
        "subject": row.subject,
        "mmlu_split": row.split,
        "mmlu_index": row.index,
        "source_dataset": row.dataset_name,
        "family_name": "quantitative_choice",
        "question_numeric_fragments": list(extract_numeric_fragments(row.question)),
        "choice_signatures": choice_signatures,
        "primitive_quantity_ids": tuple(quantity.quantity_id for quantity in quantities),
        "supported_choice_ids": (row.answer,),
        "correct_choice_text": choice_text_for_answer(row.answer, choices),
    }


def factual_metadata(row: MMLURow, choices: Sequence[ChoiceSpec]) -> Dict[str, Any]:
    dataset_slug = dataset_slug_for_row(row)
    choice_keywords = {
        choice.choice_id: list(choice.attributes.get("keyword_tokens", []))
        for choice in choices
    }
    return {
        "template": f"{dataset_slug}_factual_choice",
        "subject": row.subject,
        "mmlu_split": row.split,
        "mmlu_index": row.index,
        "source_dataset": row.dataset_name,
        "family_name": "factual_choice",
        "question_focus_tokens": list(focus_tokens(row.question)),
        "choice_signatures": choice_keywords,
        "supported_choice_ids": (row.answer,),
        "correct_choice_text": choice_text_for_answer(row.answer, choices),
    }


def infer_selection_cue(row: MMLURow) -> str:
    lowered = row.question.lower()
    if "scenario 1" in lowered:
        return "scenario_pairing"
    if "refers to the following information" in lowered or "excerpt" in lowered:
        return "passage_reference"
    if "comes to the office" in lowered or "comes to the physician" in lowered or "presents to the office" in lowered:
        return "clinical_case"
    if has_fill_blank(row.question):
        return "completion"
    if " not " in f" {lowered} " or " except" in lowered or "all of the following" in lowered:
        return "negation"
    if any(cue in lowered for cue in STATEMENT_EVALUATION_CUES):
        return "statement_evaluation"
    if any(cue in lowered for cue in DESCRIPTOR_MATCH_CUES):
        return "descriptor_match"
    if question_sentence_count(row.question) >= 3 or lowered.startswith(MODAL_PREFIXES) or any(
        cue in lowered for cue in RULE_APPLICATION_CUES
    ):
        return "rule_application"
    if any(cue in lowered for cue in COMPARATIVE_INFERENCE_CUES):
        return "comparative_inference"
    return "concept_identification"


def infer_prompt_structure(row: MMLURow) -> Dict[str, Any]:
    question = row.question
    lowered = question.lower()
    structure = {
        "cue_type": infer_selection_cue(row),
        "question_length": len(question),
        "question_token_count": len(normalize_tokens(question)),
        "sentence_count": question_sentence_count(question),
        "has_fill_blank": has_fill_blank(question),
        "has_question_mark": "?" in question,
        "has_passage_prefix": "refers to the following information" in lowered or "excerpt" in lowered,
        "has_multiple_scenarios": "scenario 1" in lowered and "scenario 2" in lowered,
        "has_case_vignette": any(
            phrase in lowered
            for phrase in ("comes to the office", "comes to the physician", "presents to the office", "history of")
        ),
        "question_starts_with_modal": lowered.startswith(MODAL_PREFIXES),
    }
    parts = [part.strip() for part in re.split(r"\bScenario \d+\s*\||\bRead the following excerpt\.|\bThis question refers to the following information\.", question) if part.strip()]
    structure["prompt_segments"] = parts[:4]
    return structure


def infer_elimination_family(row: MMLURow) -> str:
    structure = infer_prompt_structure(row)
    if structure["has_passage_prefix"]:
        return "passage_reference_choice"
    if structure["has_case_vignette"] or structure["has_multiple_scenarios"]:
        return "case_application_choice"
    if structure["cue_type"] == "negation":
        return "negation_exception_choice"
    if structure["cue_type"] == "completion":
        return "completion_choice"
    if structure["cue_type"] == "statement_evaluation":
        return "statement_evaluation_choice"
    if structure["cue_type"] == "descriptor_match":
        return "descriptor_match_choice"
    if structure["cue_type"] == "rule_application":
        return "rule_application_choice"
    if structure["cue_type"] == "comparative_inference":
        return "comparative_inference_choice"
    return "concept_identification_choice"


def structured_choice_metadata(row: MMLURow, choices: Sequence[ChoiceSpec], family: str) -> Dict[str, Any]:
    dataset_slug = dataset_slug_for_row(row)
    prompt_structure = infer_prompt_structure(row)
    choice_keywords = {
        choice.choice_id: list(choice.attributes.get("keyword_tokens", []))
        for choice in choices
    }
    return {
        "template": f"{dataset_slug}_{family}",
        "subject": row.subject,
        "mmlu_split": row.split,
        "mmlu_index": row.index,
        "source_dataset": row.dataset_name,
        "family_name": family,
        "prompt_structure": prompt_structure,
        "question_focus_tokens": list(focus_tokens(row.question)),
        "choice_signatures": choice_keywords,
        "supported_choice_ids": (row.answer,),
        "eliminated_choice_ids": tuple(choice.choice_id for choice in choices if choice.choice_id != row.answer),
        "correct_choice_text": choice_text_for_answer(row.answer, choices),
    }


def parse_mmlu_row(row: MMLURow) -> Tuple[MMLUExample | None, MMLUParserFailure | None]:
    family = infer_family(row)
    if family is None:
        return None, MMLUParserFailure("unsupported_row_type", row.subject)
    if row.answer not in choice_ids_for_row(row):
        return None, MMLUParserFailure("unsupported_answer_label", row.answer)

    entities = infer_entities(row)
    choices = build_choices(row)
    quantities = infer_quantities(row) if family == "quantitative_choice" else ()
    dataset_slug = dataset_slug_for_row(row)
    if dataset_slug == "mmlu":
        notes = ("parsed from official MMLU multiple-choice CSV row",)
    elif dataset_slug == "mmlu_pro":
        notes = ("parsed from official MMLU-Pro Hugging Face dataset",)
    elif dataset_slug == "mmlu_redux":
        notes = ("parsed from official MMLU-Redux Hugging Face dataset",)
    else:
        notes = (f"parsed from {row.dataset_name} multiple-choice benchmark row",)

    if family == "quantitative_choice":
        trace_template = choice_trace("Match quantitative cues and option signatures to the compatible choice.")
        concept_tags = (dataset_slug, "multiple_choice", "quantitative", "selection")
        metadata = quantitative_metadata(row, choices, quantities)
        relations = (
            {
                "type": f"{dataset_slug}_source",
                "subject": row.subject,
                "split": row.split,
                "index": row.index,
                "dataset": row.dataset_name,
            },
            *(
                {
                    "type": "choice_candidate",
                    "choice_id": choice.choice_id,
                    "numeric_fragments": choice.attributes.get("numeric_fragments", []),
                }
                for choice in choices
            ),
        )
    elif family == "factual_choice":
        trace_template = choice_trace("Match factual cues and subject keywords to the compatible choice.")
        concept_tags = (dataset_slug, "multiple_choice", "factual", "selection")
        metadata = factual_metadata(row, choices)
        relations = (
            {
                "type": f"{dataset_slug}_source",
                "subject": row.subject,
                "split": row.split,
                "index": row.index,
                "dataset": row.dataset_name,
            },
            {"type": "question_focus", "tokens": metadata["question_focus_tokens"]},
            *(
                {
                    "type": "choice_candidate",
                    "choice_id": choice.choice_id,
                    "keyword_tokens": choice.attributes.get("keyword_tokens", []),
                }
                for choice in choices
            ),
        )
    else:
        choice_match_descriptions = {
            "completion_choice": "Match the partial stem and blank slots to the compatible completion choice.",
            "descriptor_match_choice": "Match the prompt descriptor, classification, or stance to the compatible choice.",
            "concept_identification_choice": "Match the stem concept or term to the compatible choice.",
        }
        elimination_descriptions = {
            "case_application_choice": "Eliminate choices that do not fit the case facts or scenario constraints.",
            "passage_reference_choice": "Eliminate choices that are not supported by the referenced passage or excerpt.",
            "negation_exception_choice": "Eliminate choices until only the NOT/EXCEPT exception remains.",
            "rule_application_choice": "Eliminate choices that do not follow from the applied rule or scenario facts.",
            "statement_evaluation_choice": "Eliminate choices that fail the true/correct statement test in the prompt.",
            "comparative_inference_choice": "Eliminate choices that violate the ordering, trend, or if-then constraint.",
        }
        if family in ELIMINATION_FAMILIES:
            trace_template = elimination_trace(elimination_descriptions[family])
            concept_tags = (dataset_slug, "multiple_choice", "elimination", family.replace("_choice", ""))
        else:
            trace_template = choice_trace(choice_match_descriptions[family])
            concept_tags = (dataset_slug, "multiple_choice", "matching", family.replace("_choice", ""))
        metadata = structured_choice_metadata(row, choices, family)
        relations = (
            {
                "type": f"{dataset_slug}_source",
                "subject": row.subject,
                "split": row.split,
                "index": row.index,
                "dataset": row.dataset_name,
            },
            {"type": "prompt_structure", **metadata["prompt_structure"]},
            {"type": "question_focus", "tokens": metadata["question_focus_tokens"]},
            *(
                {
                    "type": "choice_candidate",
                    "choice_id": choice.choice_id,
                    "keyword_tokens": choice.attributes.get("keyword_tokens", []),
                    "token_count": choice.attributes.get("token_count"),
                }
                for choice in choices
            ),
        )

    program = Program(
        op="SelectChoice",
        args=(tuple(choice.choice_id for choice in choices), {"mode": family, "subject": row.subject}),
    )
    task = AbstractReasoningTask(
        task_id=f"{dataset_slug}_{row.split}_{row.subject}_{row.index}_task",
        source_modality="text_mcq",
        source_text=row.question,
        entities=entities,
        quantities=tuple(quantities),
        relations=tuple(relations),
        goal=GoalSpec(
            target_id="answer",
            query=f"{dataset_slug}_{family}_{row.split}_{row.subject}_{row.index}",
            unit="choice_id",
        ),
        program=program,
        trace_template=trace_template,
        answer=row.answer,
        concept_tags=concept_tags,
        difficulty=4,
        answer_format="multiple_choice",
        choices=tuple(choices),
        correct_choice=row.answer,
        metadata=metadata,
    )
    example = MMLUExample(
        example_id=f"{dataset_slug}_{row.split}_{row.subject}_{row.index}",
        row=row,
        abstract_task=task,
        family_name=family,
        template_name=metadata["template"],
        notes=notes,
        metadata={
            "subject": row.subject,
            "split": row.split,
            "mmlu_index": row.index,
            "answer_format": "multiple_choice",
            "family_name": family,
            "source_dataset": row.dataset_name,
            **dict(row.metadata),
        },
    )
    return example, None


def parse_mmlu_rows(rows: Sequence[MMLURow]) -> Tuple[Tuple[MMLUExample, ...], Dict[str, int]]:
    examples: List[MMLUExample] = []
    failure_counts: Dict[str, int] = {}
    for row in rows:
        example, failure = parse_mmlu_row(row)
        if example is not None:
            examples.append(example)
            continue
        reason = failure.reason if failure else "unknown_failure"
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return tuple(examples), failure_counts


def input_state_for_example(example: MMLUExample) -> State:
    task = example.abstract_task
    return {
        "source_modality": task.source_modality,
        "source_text": example.row.question,
        "answer_format": task.answer_format,
        "subject": example.row.subject,
        "choices": _encode(task.choices),
        "correct_choice": None,
    }


def example_step_states(example: MMLUExample) -> Dict[str, State]:
    task = example.abstract_task
    final_state = task.to_state()
    prefix = input_state_for_example(example)
    segment_state = {
        **prefix,
        "goal": final_state["goal"],
    }
    bind_state = {
        **segment_state,
        "entities": final_state["entities"],
        "quantities": final_state["quantities"],
        "relations": final_state["relations"],
        "program": final_state["program"],
    }
    match_state = {
        **bind_state,
        "supported_choice_ids": list(task.metadata.get("supported_choice_ids", ())),
        "choice_signatures": _encode(task.metadata.get("choice_signatures", {})),
    }
    if example.family_name == "factual_choice":
        match_state["question_focus_tokens"] = _encode(task.metadata.get("question_focus_tokens", ()))
    elif example.family_name == "quantitative_choice":
        match_state["question_numeric_fragments"] = _encode(task.metadata.get("question_numeric_fragments", ()))
    else:
        match_state["prompt_structure"] = _encode(task.metadata.get("prompt_structure", {}))
        match_state["question_focus_tokens"] = _encode(task.metadata.get("question_focus_tokens", ()))
    eliminate_state = {
        **match_state,
        "eliminated_choice_ids": _encode(task.metadata.get("eliminated_choice_ids", ())),
        "remaining_choice_ids": list(task.metadata.get("supported_choice_ids", ())),
    }
    select_state = {
        **(eliminate_state if example.family_name in ELIMINATION_FAMILIES else match_state),
        "selected_choice_id": task.correct_choice,
        "selected_choice_text": task.metadata.get("correct_choice_text"),
        "answer": task.answer,
        "eliminated_choice_ids": [choice.choice_id for choice in task.choices if choice.choice_id != task.correct_choice],
    }
    states = {
        "segment": segment_state,
        "bind": bind_state,
        "select": select_state,
        "render": final_state,
    }
    if example.family_name in ELIMINATION_FAMILIES:
        states["eliminate"] = eliminate_state
    else:
        states["match"] = match_state
    return states


def step_action(step_name: str, example: MMLUExample) -> Dict[str, Any]:
    task = example.abstract_task
    if step_name == "segment":
        return {
            "subject": example.row.subject,
            "choice_ids": [choice.choice_id for choice in task.choices],
            "goal_query": task.goal.query,
        }
    if step_name == "bind":
        return {
            "entity_ids": [entity.entity_id for entity in task.entities],
            "quantity_ids": [quantity.quantity_id for quantity in task.quantities],
            "program_op": task.program.op,
        }
    if step_name == "match":
        action = {
            "family_name": example.family_name,
            "supported_choice_ids": list(task.metadata.get("supported_choice_ids", ())),
        }
        if task.metadata.get("question_focus_tokens"):
            action["question_focus_tokens"] = _encode(task.metadata.get("question_focus_tokens", ()))
        return action
    if step_name == "eliminate":
        return {
            "cue_type": task.metadata.get("prompt_structure", {}).get("cue_type"),
            "eliminated_choice_ids": _encode(task.metadata.get("eliminated_choice_ids", ())),
            "remaining_choice_ids": list(task.metadata.get("supported_choice_ids", ())),
        }
    if step_name == "select":
        return {
            "selected_choice_id": task.correct_choice,
            "selected_choice_text": task.metadata.get("correct_choice_text"),
        }
    return {
        "ir_keys": sorted(task.to_state()),
        "template_name": example.template_name,
    }


def compile_mmlu_trajectory(
    example: MMLUExample,
    *,
    trajectory_index: int,
) -> TrajectoryRecord:
    final_state = example.abstract_task.to_state()
    step_states = example_step_states(example)
    total_possible_reward = sum(
        MMLU_STEP_WEIGHTS.get(step.name, STEP_WEIGHTS.get(step.name, 0.15))
        for step in example.abstract_task.trace_template
    )
    previous_state = input_state_for_example(example)
    initial_input_state = previous_state
    cumulative = 0.0
    steps: List[TrajectoryStep] = []

    for index, trace_step in enumerate(example.abstract_task.trace_template):
        current_state = step_states[trace_step.name]
        reward_terms = symbolic_step_reward_terms(
            current_state=current_state,
            target_state=current_state,
            previous_state=previous_state,
            final_state=final_state,
        )
        weight = MMLU_STEP_WEIGHTS.get(trace_step.name, STEP_WEIGHTS.get(trace_step.name, 0.15))
        reward = reward_from_terms(weight, reward_terms)
        cumulative += reward
        steps.append(
            TrajectoryStep(
                index=index,
                name=trace_step.name,
                description=trace_step.description,
                action=step_action(trace_step.name, example),
                reward=reward,
                reward_terms=reward_terms,
                cumulative_reward=cumulative,
                progress=(index + 1) / max(1, len(example.abstract_task.trace_template)),
                stop_target=index == len(example.abstract_task.trace_template) - 1,
                workspace_state=current_state,
                verifier={
                    "exact_match": index == len(example.abstract_task.trace_template) - 1,
                    "should_stop": index == len(example.abstract_task.trace_template) - 1,
                    "resolved_subgoal_count": index + 1,
                    "unresolved_subgoal_count": len(example.abstract_task.trace_template) - index - 1,
                    "next_subgoal": (
                        example.abstract_task.trace_template[index + 1].name
                        if index + 1 < len(example.abstract_task.trace_template)
                        else None
                    ),
                    "non_terminal_reason": (
                        None
                        if index == len(example.abstract_task.trace_template) - 1
                        else f"remaining_subgoal:{example.abstract_task.trace_template[index + 1].name}"
                    ),
                },
                done=index == len(example.abstract_task.trace_template) - 1,
            )
        )
        previous_state = current_state

    return TrajectoryRecord(
        trajectory_id=f"mmlu:{example.row.split}:{example.example_id}:{trajectory_index}",
        split=example.row.split,
        family=example.family_name,
        difficulty=example.abstract_task.difficulty,
        source_modality=example.abstract_task.source_modality,
        concept_tags=example.abstract_task.concept_tags,
        trace_template=tuple(step.name for step in example.abstract_task.trace_template),
        role_bindings={},
        episode_metadata=example.metadata,
        shortcut_checks=(
            "question subject must match the supported family inventory",
            "choice signatures must preserve the original answer candidates",
            "selected choice must agree with the official answer key",
        ),
        example=example,
        input_state=initial_input_state,
        output_state=final_state,
        steps=tuple(steps),
        total_reward=cumulative,
        total_possible_reward=total_possible_reward,
    )


def build_mmlu_examples(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    splits: Sequence[str] = DEFAULT_MMLU_TRAINING_SPLITS,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[MMLUExample, ...], Dict[str, int]]:
    resolved_splits = validate_mmlu_splits(splits, allow_eval_splits=allow_eval_splits)
    rows = load_mmlu_rows(data_dir, splits=resolved_splits, max_rows=max_rows)
    return parse_mmlu_rows(rows)


def compile_mmlu_examples(examples: Sequence[MMLUExample]) -> Tuple[TrajectoryRecord, ...]:
    records: List[TrajectoryRecord] = []
    for index, example in enumerate(examples):
        records.append(compile_mmlu_trajectory(example, trajectory_index=index))
    return tuple(records)


def build_mmlu_trajectories(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    splits: Sequence[str] = DEFAULT_MMLU_TRAINING_SPLITS,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[TrajectoryRecord, ...], Dict[str, int]]:
    examples, failures = build_mmlu_examples(
        data_dir=data_dir,
        splits=splits,
        allow_eval_splits=allow_eval_splits,
        max_rows=max_rows,
    )
    return compile_mmlu_examples(examples), failures


def write_translation_jsonl(path: str | Path, examples: Sequence[MMLUExample]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_jsonable()))
            handle.write("\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse official MMLU rows into canonical reasoning IR and trajectories.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the official MMLU CSV folders and optional auxiliary training rows.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_MMLU_SPLITS,
        default=DEFAULT_MMLU_TRAINING_SPLITS,
        help="MMLU splits to parse. Defaults to auxiliary_train only for benchmark-safe exports.",
    )
    parser.add_argument(
        "--translation-output",
        type=str,
        default="arc_trajectory_sampler/results/mmlu_reasoning_examples.jsonl",
        help="Destination JSONL path for MMLU rows mapped to the reasoning IR.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default="arc_trajectory_sampler/results/mmlu_reasoning_trajectories.jsonl",
        help="Destination JSONL path for canonical MMLU trajectory records.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of MMLU rows to parse for a quick smoke run.",
    )
    parser.add_argument(
        "--allow-eval-splits",
        action="store_true",
        help="Allow official MMLU benchmark splits such as `dev`, `val`, or `test`. Intended for audit/evaluation only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, failure_counts = build_mmlu_examples(
        data_dir=args.data_dir,
        splits=args.splits,
        allow_eval_splits=args.allow_eval_splits,
        max_rows=args.max_rows,
    )
    trajectories = compile_mmlu_examples(examples)
    write_translation_jsonl(args.translation_output, examples)
    write_jsonl(args.trajectory_output, trajectories)
    print(
        json.dumps(
            {
                "example_count": len(examples),
                "trajectory_count": len(trajectories),
                "failure_counts": failure_counts,
                "splits": list(args.splits),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
