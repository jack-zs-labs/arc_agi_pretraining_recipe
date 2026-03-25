from __future__ import annotations

"""Parse the official OlymMATH benchmark into canonical reasoning IR and trajectories."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import sympy
from sympy import Interval
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from .reasoning_ir import AbstractReasoningTask, EntitySpec, GoalSpec, QuantitySpec
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
    from reasoning_ir import AbstractReasoningTask, EntitySpec, GoalSpec, QuantitySpec  # type: ignore
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


HF_DATASETS_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASETS_SERVER_SPLITS_URL = "https://datasets-server.huggingface.co/splits"
OLYMPIAD_MATH_DATASET = "RUC-AIBOX/OlymMATH"
OLYMPIAD_MATH_SUPPORTED_CONFIGS = ("en-easy", "en-hard", "zh-easy", "zh-hard", "lean")
DEFAULT_OLYMPIAD_MATH_CONFIGS = ("en-easy", "en-hard")
OLYMPIAD_MATH_EVAL_CONFIGS = frozenset(OLYMPIAD_MATH_SUPPORTED_CONFIGS)

SUBJECT_CANONICAL = {
    "algebra": "Algebra",
    "geometry": "Geometry",
    "number theory": "Number Theory",
    "combinatorics": "Combinatorics",
    "代数": "Algebra",
    "几何": "Geometry",
    "数论": "Number Theory",
    "组合": "Combinatorics",
}
LANGUAGE_BY_CONFIG = {
    "en-easy": "en",
    "en-hard": "en",
    "zh-easy": "zh",
    "zh-hard": "zh",
    "lean": "multilingual",
}
DIFFICULTY_BY_CONFIG = {
    "en-easy": "easy",
    "en-hard": "hard",
    "zh-easy": "easy",
    "zh-hard": "hard",
    "lean": "formal",
}
OLYMPIAD_STEP_WEIGHTS = {
    "segment": 0.10,
    "bind": 0.16,
    "plan": 0.16,
    "derive": 0.22,
    "verify": 0.16,
    "render": 0.20,
}
INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+/\d+|\d+)(?![A-Za-z])")
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
LATEX_SYMBOL_RE = re.compile(r"(?<!\\)\b[A-Za-z]+(?:_[A-Za-z0-9]+)?\b")
GEOMETRY_OBJECT_RE = re.compile(r"\b(?:triangle|circle|ellipse|parabola|line|segment|focus|diameter|polygon)\b", re.I)
LEAN_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)", re.MULTILINE)
LEAN_THEOREM_RE = re.compile(r"\btheorem\s+([A-Za-z0-9_']+)")
LEAN_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_']*\b")
STOPWORD_TOKENS = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "find",
    "for",
    "from",
    "given",
    "if",
    "in",
    "is",
    "it",
    "let",
    "of",
    "on",
    "that",
    "the",
    "there",
    "to",
    "value",
    "with",
}


@dataclass(frozen=True)
class OlympiadMathRow:
    config: str
    split: str
    index: int
    problem: str
    answer: str
    subject: str
    unique_id: str
    language: str
    difficulty_tier: str
    dataset_name: str = "olympiad_math"
    task_variant: str = "open_answer"
    formal_statement: str = ""
    formal_proof: str = ""
    formal_statement_raw: str = ""
    en_informal: str = ""
    en_nl_proof: str = ""
    zh_informal: str = ""
    zh_nl_proof: str = ""


@dataclass(frozen=True)
class OlympiadMathParserFailure:
    reason: str
    details: str = ""


@dataclass(frozen=True)
class OlympiadMathExample:
    example_id: str
    row: OlympiadMathRow
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
        if len(batch) < length:
            break
    return tuple(rows)


def load_hf_split_configs(dataset: str) -> Tuple[str, ...]:
    params = urlencode({"dataset": dataset})
    with urlopen(f"{HF_DATASETS_SERVER_SPLITS_URL}?{params}") as response:
        payload = json.load(response)
    return tuple(sorted({item["config"] for item in payload.get("splits", ())}))


def validate_olympiad_math_configs(
    configs: Sequence[str],
    *,
    allow_eval_configs: bool,
) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(config) for config in configs))
    unsupported = sorted(config for config in normalized if config not in OLYMPIAD_MATH_SUPPORTED_CONFIGS)
    if unsupported:
        unsupported_text = ", ".join(unsupported)
        raise ValueError(f"Unsupported OlymMATH config(s): {unsupported_text}")
    blocked = sorted(config for config in normalized if config in OLYMPIAD_MATH_EVAL_CONFIGS)
    if blocked and not allow_eval_configs:
        blocked_text = ", ".join(blocked)
        raise ValueError(
            f"OlymMATH benchmark config(s) requested without explicit opt-in: {blocked_text}. "
            "Pass allow_eval_configs=True only for audit/evaluation paths."
        )
    return normalized


def canonical_subject(subject: str) -> str:
    normalized = subject.strip()
    return SUBJECT_CANONICAL.get(normalized, normalized)


def interleave_rows(
    by_config: Dict[str, Sequence[OlympiadMathRow]],
    *,
    max_rows: int | None,
) -> Tuple[OlympiadMathRow, ...]:
    ordered_configs = tuple(by_config)
    interleaved: List[OlympiadMathRow] = []
    cursor = 0
    while True:
        added = False
        for config in ordered_configs:
            rows = by_config[config]
            if cursor < len(rows):
                interleaved.append(rows[cursor])
                added = True
                if max_rows is not None and len(interleaved) >= max_rows:
                    return tuple(interleaved)
        if not added:
            break
        cursor += 1
    return tuple(interleaved)


def load_olympiad_math_rows(
    *,
    configs: Sequence[str] = DEFAULT_OLYMPIAD_MATH_CONFIGS,
    allow_eval_configs: bool = False,
    max_rows: int | None = None,
) -> Tuple[OlympiadMathRow, ...]:
    resolved_configs = validate_olympiad_math_configs(configs, allow_eval_configs=allow_eval_configs)
    per_config_limit = None
    if max_rows is not None:
        per_config_limit = max(1, math.ceil(max_rows / max(1, len(resolved_configs))))
    rows_by_config: Dict[str, Sequence[OlympiadMathRow]] = {}
    for config in resolved_configs:
        fetched = load_hf_rows(
            dataset=OLYMPIAD_MATH_DATASET,
            config=config,
            split="test",
            max_rows=per_config_limit,
        )
        if config == "lean":
            rows_by_config[config] = tuple(
                OlympiadMathRow(
                    config=config,
                    split="test",
                    index=index,
                    problem=str(row.get("en_informal", "") or row.get("zh_informal", "") or row.get("formal_statement_raw", "")),
                    answer=str(row.get("formal_statement_raw", "") or row.get("formal_statement", "")),
                    subject=canonical_subject(str(row.get("subject", ""))),
                    unique_id=str(row.get("unique_id", f"{config}_{index}")),
                    language=LANGUAGE_BY_CONFIG[config],
                    difficulty_tier=DIFFICULTY_BY_CONFIG[config],
                    task_variant="lean",
                    formal_statement=str(row.get("formal_statement", "")),
                    formal_proof=str(row.get("formal_proof", "")),
                    formal_statement_raw=str(row.get("formal_statement_raw", "")),
                    en_informal=str(row.get("en_informal", "")),
                    en_nl_proof=str(row.get("en_nl_proof", "")),
                    zh_informal=str(row.get("zh_informal", "")),
                    zh_nl_proof=str(row.get("zh_nl_proof", "")),
                )
                for index, row in enumerate(fetched)
            )
        else:
            rows_by_config[config] = tuple(
                OlympiadMathRow(
                    config=config,
                    split="test",
                    index=index,
                    problem=str(row.get("problem", "")),
                    answer=str(row.get("answer", "")),
                    subject=canonical_subject(str(row.get("subject", ""))),
                    unique_id=str(row.get("unique_id", f"{config}_{index}")),
                    language=LANGUAGE_BY_CONFIG[config],
                    difficulty_tier=DIFFICULTY_BY_CONFIG[config],
                )
                for index, row in enumerate(fetched)
            )
    return interleave_rows(rows_by_config, max_rows=max_rows)


def strip_outer_delimiters(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        if stripped.startswith("$") and stripped.endswith("$"):
            stripped = stripped[1:-1].strip()
            changed = True
            continue
        if stripped.startswith("\\boxed{") and stripped.endswith("}"):
            stripped = stripped[len("\\boxed{") : -1].strip()
            changed = True
    return stripped


def find_matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        raise ValueError("find_matching_brace requires an opening brace index.")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced braces in LaTeX expression.")


def split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth_round = 0
    depth_square = 0
    depth_curly = 0
    for index, char in enumerate(text):
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round = max(0, depth_round - 1)
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square = max(0, depth_square - 1)
        elif char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly = max(0, depth_curly - 1)
        if char == "," and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            previous_char = text[index - 1] if index > 0 else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if previous_char.isdigit() and next_char.isdigit():
                current.append(char)
                continue
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def replace_latex_commands(text: str) -> str:
    replaced = text
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\pi": "pi",
        "\\arccos": "acos",
        "\\arcsin": "asin",
        "\\arctan": "atan",
        "\\cos": "cos",
        "\\sin": "sin",
        "\\tan": "tan",
        "\\ln": "log",
    }
    for source, target in replacements.items():
        replaced = replaced.replace(source, target)
    replaced = replaced.replace("−", "-").replace("–", "-").replace("×", "*")
    replaced = replaced.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    return replaced


def strip_numeric_grouping_commas(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def latex_atom_to_sympy_text(text: str) -> str:
    stripped = strip_outer_delimiters(text)
    stripped = replace_latex_commands(stripped)
    stripped = strip_numeric_grouping_commas(stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1]
    index = 0
    converted: List[str] = []
    while index < len(stripped):
        if stripped.startswith("\\frac{", index):
            numerator_start = index + len("\\frac")
            numerator_left = numerator_start
            numerator_right = find_matching_brace(stripped, numerator_left)
            denominator_left = numerator_right + 1
            if denominator_left >= len(stripped) or stripped[denominator_left] != "{":
                raise ValueError(f"Malformed fraction: {text!r}")
            denominator_right = find_matching_brace(stripped, denominator_left)
            numerator = stripped[numerator_left + 1 : numerator_right]
            denominator = stripped[denominator_left + 1 : denominator_right]
            converted.append(
                f"(({latex_atom_to_sympy_text(numerator)})/({latex_atom_to_sympy_text(denominator)}))"
            )
            index = denominator_right + 1
            continue
        if stripped.startswith("\\sqrt[", index):
            degree_left = index + len("\\sqrt")
            degree_right = stripped.find("]", degree_left)
            if degree_right == -1:
                raise ValueError(f"Malformed rooted expression: {text!r}")
            degree = stripped[degree_left + 1 : degree_right]
            body_left = degree_right + 1
            if body_left >= len(stripped) or stripped[body_left] != "{":
                raise ValueError(f"Malformed rooted expression: {text!r}")
            body_right = find_matching_brace(stripped, body_left)
            body = stripped[body_left + 1 : body_right]
            converted.append(
                f"(({latex_atom_to_sympy_text(body)})**(1/({latex_atom_to_sympy_text(degree)})))"
            )
            index = body_right + 1
            continue
        if stripped.startswith("\\sqrt{", index):
            body_left = index + len("\\sqrt")
            body_right = find_matching_brace(stripped, body_left)
            body = stripped[body_left + 1 : body_right]
            converted.append(f"sqrt({latex_atom_to_sympy_text(body)})")
            index = body_right + 1
            continue
        converted.append(stripped[index])
        index += 1
    expr = "".join(converted)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<![A-Za-z0-9_])acos\s+(\([^)]*\)|sqrt\([^)]*\)|pi|[A-Za-z0-9_\.]+)", r"acos(\1)", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])asin\s+(\([^)]*\)|sqrt\([^)]*\)|pi|[A-Za-z0-9_\.]+)", r"asin(\1)", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])atan\s+(\([^)]*\)|sqrt\([^)]*\)|pi|[A-Za-z0-9_\.]+)", r"atan(\1)", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])cos\s+(\([^)]*\)|sqrt\([^)]*\)|pi|[A-Za-z0-9_\.]+)", r"cos(\1)", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])sin\s+(\([^)]*\)|sqrt\([^)]*\)|pi|[A-Za-z0-9_\.]+)", r"sin(\1)", expr)
    expr = re.sub(r"(?<![A-Za-z0-9_])tan\s+(\([^)]*\)|sqrt\([^)]*\)|pi|[A-Za-z0-9_\.]+)", r"tan(\1)", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def parse_scalar_answer_expression(text: str) -> tuple[sympy.Expr, str, str | None]:
    stripped = strip_outer_delimiters(text)
    unit = None
    if stripped.endswith("^{\\circ}"):
        stripped = stripped[: -len("^{\\circ}")].strip()
        unit = "degrees"
    sympy_text = latex_atom_to_sympy_text(stripped)
    expr = parse_expr(
        sympy_text,
        local_dict={
            "pi": sympy.pi,
            "sqrt": sympy.sqrt,
            "acos": sympy.acos,
            "asin": sympy.asin,
            "atan": sympy.atan,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "log": sympy.log,
        },
        transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
        evaluate=True,
    )
    return sympy.simplify(expr), sympy_text, unit


def canonicalize_olympiad_answer(answer_text: str) -> dict[str, object]:
    stripped = strip_outer_delimiters(answer_text)
    if not stripped:
        raise ValueError("Empty OlymMATH answer.")
    interval_surface = stripped.replace("\\left", "").replace("\\right", "").strip()
    if interval_surface and interval_surface[0] in "[(" and interval_surface[-1] in "])":
        content = interval_surface[1:-1].strip()
        parts = split_top_level_commas(content)
        if len(parts) == 1:
            expr, sympy_text, unit = parse_scalar_answer_expression(parts[0])
            return {
                "kind": "bracketed_scalar",
                "canonical": str(expr),
                "sympy_text": sympy_text,
                "unit": unit,
            }
        left_expr, left_text, left_unit = parse_scalar_answer_expression(parts[0])
        right_expr, right_text, right_unit = parse_scalar_answer_expression(parts[1])
        interval = Interval(
            left_expr,
            right_expr,
            left_open=stripped[0] == "(",
            right_open=stripped[-1] == ")",
        )
        return {
            "kind": "interval",
            "canonical": str(interval),
            "sympy_text": f"{interval_surface[0]}{left_text},{right_text}{interval_surface[-1]}",
            "unit": left_unit or right_unit,
        }
    expr, sympy_text, unit = parse_scalar_answer_expression(stripped)
    return {
        "kind": "expression",
        "canonical": str(expr),
        "sympy_text": sympy_text,
        "unit": unit,
    }


def normalize_multiline_text(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines).strip()


def extract_lean_imports(*texts: str) -> Tuple[str, ...]:
    imports: List[str] = []
    for text in texts:
        for match in LEAN_IMPORT_RE.finditer(text):
            candidate = match.group(1)
            if candidate and candidate not in imports:
                imports.append(candidate)
    return tuple(imports)


def extract_lean_theorem_name(*texts: str) -> str | None:
    for text in texts:
        match = LEAN_THEOREM_RE.search(text)
        if match:
            return match.group(1)
    return None


def lean_identifiers(*texts: str) -> Tuple[str, ...]:
    identifiers: List[str] = []
    for text in texts:
        for candidate in LEAN_IDENTIFIER_RE.findall(text):
            if candidate in STOPWORD_TOKENS or candidate in identifiers:
                continue
            identifiers.append(candidate)
            if len(identifiers) >= 64:
                return tuple(identifiers)
    return tuple(identifiers)


def canonicalize_lean_formal_payload(row: OlympiadMathRow) -> dict[str, object]:
    formal_statement = normalize_multiline_text(row.formal_statement_raw or row.formal_statement)
    formal_proof = normalize_multiline_text(row.formal_proof)
    theorem_name = extract_lean_theorem_name(formal_statement, formal_proof)
    imports = extract_lean_imports(formal_statement, formal_proof)
    informal_symbols = set(math_symbols(row.problem))
    formal_identifiers = set(lean_identifiers(formal_statement, formal_proof))
    concordance_profile = {
        "has_en_informal": bool(row.en_informal.strip()),
        "has_zh_informal": bool(row.zh_informal.strip()),
        "has_en_nl_proof": bool(row.en_nl_proof.strip()),
        "has_zh_nl_proof": bool(row.zh_nl_proof.strip()),
        "shared_symbol_count": len(informal_symbols & formal_identifiers),
        "formal_identifier_count": len(formal_identifiers),
        "import_count": len(imports),
    }
    return {
        "kind": "lean_statement",
        "canonical": formal_statement,
        "sympy_text": None,
        "unit": "lean_theorem",
        "theorem_name": theorem_name,
        "imports": imports,
        "formal_statement_line_count": len([line for line in formal_statement.splitlines() if line.strip()]),
        "formal_proof_line_count": len([line for line in formal_proof.splitlines() if line.strip()]),
        "formal_statement_has_sorry": "sorry" in formal_statement,
        "formal_proof_has_sorry": "sorry" in formal_proof,
        "concordance_profile": concordance_profile,
    }


def english_focus_tokens(problem: str) -> Tuple[str, ...]:
    tokens: List[str] = []
    for token in ENGLISH_TOKEN_RE.findall(problem.lower()):
        if token in STOPWORD_TOKENS or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= 16:
            break
    return tuple(tokens)


def math_symbols(problem: str) -> Tuple[str, ...]:
    symbols: List[str] = []
    for span in INLINE_MATH_RE.findall(problem):
        for token in LATEX_SYMBOL_RE.findall(span):
            if token in STOPWORD_TOKENS or token in symbols:
                continue
            symbols.append(token)
            if len(symbols) >= 16:
                return tuple(symbols)
    return tuple(symbols)


def extract_quantities(problem: str) -> Tuple[QuantitySpec, ...]:
    quantities: List[QuantitySpec] = []
    for index, match in enumerate(NUMBER_RE.finditer(problem)):
        fragment = match.group(0)
        try:
            value = sympy.Rational(fragment)
        except Exception:
            continue
        quantities.append(
            QuantitySpec(
                quantity_id=f"q_given_{index}",
                value=int(value) if value.q == 1 else float(value),
                unit="",
                role="given",
                attributes={"surface": fragment},
            )
        )
        if len(quantities) >= 16:
            break
    return tuple(quantities)


def extract_entities(row: OlympiadMathRow) -> Tuple[EntitySpec, ...]:
    entities: List[EntitySpec] = [
        EntitySpec(
            entity_id=f"subject_{slug(row.subject)}",
            label=row.subject,
            kind="math_subject",
            attributes={"language": row.language, "difficulty_tier": row.difficulty_tier},
        )
    ]
    for symbol in math_symbols(row.problem):
        kind = "symbol"
        if row.subject == "Geometry" and (len(symbol) == 1 and symbol.isupper()):
            kind = "point"
        elif row.subject == "Geometry" and len(symbol) == 2 and symbol.isupper():
            kind = "segment"
        entities.append(
            EntitySpec(
                entity_id=f"sym_{slug(symbol)}",
                label=symbol,
                kind=kind,
                attributes={"source": "latex_span"},
            )
        )
        if len(entities) >= 17:
            break
    return tuple(entities)


def family_for_row(row: OlympiadMathRow) -> str:
    problem_lower = row.problem.lower()
    if row.subject == "Geometry":
        if any(token in problem_lower for token in ("minimum", "maximum", "min", "max")):
            base_family = "geometric_optimization"
        else:
            base_family = "geometric_construction"
    elif row.subject == "Number Theory":
        if any(token in problem_lower for token in ("mod", "prime", "remainder", "residue", "divisibility")):
            base_family = "modular_number_theory"
        else:
            base_family = "diophantine_number_theory"
    elif row.subject == "Combinatorics":
        if any(token in problem_lower for token in ("probability", "random", "grid", "color", "graph")):
            base_family = "enumerative_combinatorics"
        else:
            base_family = "counting_combinatorics"
    elif any(token in problem_lower for token in ("sequence", "function", "polynomial", "equation")):
        base_family = "algebraic_transformation"
    else:
        base_family = "symbolic_algebra"
    if row.task_variant == "lean":
        return f"formal_{base_family}"
    return base_family


def trace_template_for_family(family: str) -> Tuple[TraceStep, ...]:
    if family.startswith("formal_"):
        return (
            TraceStep("segment", "Parse the informal Olympiad theorem statement and identify the target formal claim."),
            TraceStep("bind", "Bind informal givens, theorem variables, and Lean hypotheses into typed theorem state."),
            TraceStep("plan", "Align the informal proof sketch with the Lean theorem statement and expected proof strategy."),
            TraceStep("derive", "Derive the formal theorem structure, key lemmas, and proof obligations needed for Lean."),
            TraceStep("verify", "Check concordance between the informal theorem, the formal statement, and the supplied Lean proof artifact."),
            TraceStep("render", "Emit the final typed reasoning IR with the canonical Lean statement and proof metadata."),
        )
    plan_descriptions = {
        "geometric_optimization": "Choose the auxiliary construction or extremal relation that controls the target quantity.",
        "geometric_construction": "Choose the auxiliary construction, symmetry, or metric relation that exposes the target quantity.",
        "modular_number_theory": "Choose the modular, divisibility, or factorization lens that constrains the answer.",
        "diophantine_number_theory": "Choose the integer-structure or divisibility argument that constrains the answer.",
        "counting_combinatorics": "Choose the counting structure, invariant, or case split that matches the constraints.",
        "enumerative_combinatorics": "Choose the counting structure, probabilistic argument, or extremal split that matches the constraints.",
        "algebraic_transformation": "Choose the algebraic normalization, substitution, or invariant that reduces the problem.",
        "symbolic_algebra": "Choose the algebraic identity, inequality, or transformation that reduces the target expression.",
    }
    return (
        TraceStep("segment", "Parse the Olympiad problem statement into givens, symbolic objects, and the final target."),
        TraceStep("bind", "Bind extracted symbols, numeric givens, and structural constraints into typed math state."),
        TraceStep("plan", plan_descriptions.get(family, "Choose the solution strategy family that matches the constraints.")),
        TraceStep("derive", "Derive the key intermediate expressions, invariants, or case reductions that determine the answer."),
        TraceStep("verify", "Check that the candidate answer satisfies the original constraints and final target form."),
        TraceStep("render", "Emit the final typed reasoning IR and canonical answer."),
    )


def strategy_hint_for_family(family: str) -> str:
    if family.startswith("formal_"):
        return "formalize_and_verify"
    hints = {
        "geometric_optimization": "auxiliary_construction_extremal",
        "geometric_construction": "auxiliary_construction_relation",
        "modular_number_theory": "modular_factorization",
        "diophantine_number_theory": "integer_structure",
        "counting_combinatorics": "casework_invariant",
        "enumerative_combinatorics": "counting_or_probability",
        "algebraic_transformation": "normalize_substitute_solve",
        "symbolic_algebra": "transform_expression",
    }
    return hints.get(family, "general_olympiad_strategy")


def row_notes(row: OlympiadMathRow, answer_metadata: dict[str, object]) -> Tuple[str, ...]:
    notes = ["parsed from the official OlymMATH Hugging Face benchmark row"]
    if row.task_variant == "lean":
        notes.append("row contains bilingual informal theorem text plus Lean formal statement and proof")
        notes.append("canonical output is the normalized Lean theorem statement, with proof concordance stored in metadata")
    else:
        notes.append("answers are canonicalized with a SymPy-backed LaTeX-to-expression adapter")
        if answer_metadata.get("kind") == "interval":
            notes.append("answer is interval-valued rather than a scalar expression")
    if row.language == "zh":
        notes.append("problem text is Chinese; current lexical focus extraction is language-light")
    return tuple(notes)


def parse_olympiad_math_row(
    row: OlympiadMathRow,
) -> Tuple[OlympiadMathExample | None, OlympiadMathParserFailure | None]:
    if not row.problem.strip():
        return None, OlympiadMathParserFailure("empty_problem")
    if not row.answer.strip():
        return None, OlympiadMathParserFailure("empty_answer")
    try:
        answer_metadata = (
            canonicalize_lean_formal_payload(row)
            if row.task_variant == "lean"
            else canonicalize_olympiad_answer(row.answer)
        )
    except Exception as exc:
        return None, OlympiadMathParserFailure("answer_parse_failure", f"{type(exc).__name__}: {exc}")

    entities = extract_entities(row)
    quantities = extract_quantities(row.problem)
    family = family_for_row(row)
    strategy_hint = strategy_hint_for_family(family)
    uses_english_source = row.language == "en" or row.task_variant == "lean"
    focus_tokens = english_focus_tokens(row.problem) if uses_english_source else ()
    problem_profile = {
        "has_geometry_objects": bool(GEOMETRY_OBJECT_RE.search(row.problem)),
        "has_inline_math": bool(INLINE_MATH_RE.search(row.problem)),
        "has_optimization_cue": any(
            cue in row.problem.lower() for cue in ("minimum", "maximum", "min", "max", "least", "greatest")
        ) if uses_english_source else any(cue in row.problem for cue in ("最小", "最大")),
        "has_probability_cue": ("probability" in row.problem.lower()) if uses_english_source else ("概率" in row.problem),
        "quantity_count": len(quantities),
        "symbol_count": max(0, len(entities) - 1),
        "task_variant": row.task_variant,
    }
    if row.task_variant == "lean":
        problem_profile.update(
            {
                "has_formal_statement": bool(row.formal_statement_raw.strip() or row.formal_statement.strip()),
                "has_formal_proof": bool(row.formal_proof.strip()),
                "has_bilingual_informal": bool(row.en_informal.strip() and row.zh_informal.strip()),
                "formal_statement_line_count": int(answer_metadata.get("formal_statement_line_count", 0)),
                "formal_proof_line_count": int(answer_metadata.get("formal_proof_line_count", 0)),
            }
        )
    relations = (
        {
            "type": "olympiad_math_source",
            "dataset": row.dataset_name,
            "config": row.config,
            "split": row.split,
            "unique_id": row.unique_id,
            "language": row.language,
        },
        {
            "type": "problem_subject",
            "subject": row.subject,
            "difficulty_tier": row.difficulty_tier,
            "family": family,
            "strategy_hint": strategy_hint,
            "task_variant": row.task_variant,
        },
        {
            "type": "problem_profile",
            **problem_profile,
        },
        {
            "type": "answer_signature",
            "kind": answer_metadata["kind"],
            "canonical": answer_metadata["canonical"],
            "unit": answer_metadata.get("unit"),
        },
    )
    if row.task_variant == "lean":
        relations = relations + (
            {
                "type": "formal_artifact",
                "theorem_name": answer_metadata.get("theorem_name"),
                "imports": list(answer_metadata.get("imports", ())),
                "formal_statement_has_sorry": bool(answer_metadata.get("formal_statement_has_sorry")),
                "formal_proof_has_sorry": bool(answer_metadata.get("formal_proof_has_sorry")),
            },
            {
                "type": "formal_concordance",
                **dict(answer_metadata.get("concordance_profile", {})),
            },
        )
    if focus_tokens:
        relations = relations + ({"type": "question_focus", "tokens": focus_tokens},)

    answer_format = "lean_proof" if row.task_variant == "lean" else "symbolic_expression"
    goal_unit = str(answer_metadata.get("unit") or ("lean_theorem" if row.task_variant == "lean" else "symbolic_expression"))
    program_op = "FormalizeOlympiadMath" if row.task_variant == "lean" else "SolveOlympiadMath"
    task_answer = row.formal_proof if row.task_variant == "lean" else row.answer
    task = AbstractReasoningTask(
        task_id=f"{row.unique_id}_task",
        source_modality="math_olympiad_text",
        source_text=row.problem,
        entities=entities,
        quantities=quantities,
        relations=tuple(relations),
        goal=GoalSpec(
            target_id="answer",
            query=f"olympiad_math_{slug(row.unique_id)}",
            unit=goal_unit,
        ),
        program=Program(
            op=program_op,
            args=(
                row.subject,
                family,
                {
                    "strategy_hint": strategy_hint,
                    "answer_kind": answer_metadata["kind"],
                    "config": row.config,
                    "task_variant": row.task_variant,
                },
            ),
        ),
        trace_template=trace_template_for_family(family),
        answer=task_answer,
        concept_tags=(
            "olympiad_math",
            row.task_variant,
            slug(row.subject),
            slug(row.difficulty_tier),
            slug(family),
        ),
        difficulty=5 if row.task_variant == "lean" else (4 if row.difficulty_tier == "easy" else 5),
        answer_format=answer_format,
        metadata={
            "template": f"olympiad_math_{family}",
            "source_dataset": row.dataset_name,
            "olympiad_math_config": row.config,
            "olympiad_math_unique_id": row.unique_id,
            "language": row.language,
            "subject": row.subject,
            "difficulty_tier": row.difficulty_tier,
            "family_name": family,
            "strategy_hint": strategy_hint,
            "question_focus_tokens": focus_tokens,
            "answer_metadata": answer_metadata,
            "problem_profile": problem_profile,
            "task_variant": row.task_variant,
            "en_informal": row.en_informal,
            "en_nl_proof": row.en_nl_proof,
            "zh_informal": row.zh_informal,
            "zh_nl_proof": row.zh_nl_proof,
            "formal_statement": row.formal_statement,
            "formal_statement_raw": row.formal_statement_raw,
            "formal_proof": row.formal_proof,
        },
    )
    example = OlympiadMathExample(
        example_id=slug(row.unique_id),
        row=row,
        abstract_task=task,
        family_name=family,
        template_name=f"olympiad_math_{family}",
        notes=row_notes(row, answer_metadata),
        metadata={
            "config": row.config,
            "split": row.split,
            "unique_id": row.unique_id,
            "language": row.language,
            "subject": row.subject,
            "difficulty_tier": row.difficulty_tier,
            "answer_format": answer_format,
            "source_dataset": row.dataset_name,
            "task_variant": row.task_variant,
        },
    )
    return example, None


def parse_olympiad_math_rows(
    rows: Sequence[OlympiadMathRow],
) -> Tuple[Tuple[OlympiadMathExample, ...], Dict[str, int]]:
    examples: List[OlympiadMathExample] = []
    failure_counts: Counter[str] = Counter()
    for row in rows:
        example, failure = parse_olympiad_math_row(row)
        if example is not None:
            examples.append(example)
        else:
            failure_counts[failure.reason if failure else "unknown_failure"] += 1
    return tuple(examples), dict(failure_counts)


def input_state_for_example(example: OlympiadMathExample) -> State:
    return {
        "source_modality": example.abstract_task.source_modality,
        "source_text": example.row.problem,
        "answer_format": example.abstract_task.answer_format,
        "subject": example.row.subject,
        "language": example.row.language,
        "difficulty_tier": example.row.difficulty_tier,
        "config": example.row.config,
        "task_variant": example.row.task_variant,
    }


def example_step_states(example: OlympiadMathExample) -> Dict[str, State]:
    task = example.abstract_task
    answer_metadata = task.metadata.get("answer_metadata", {})
    final_state = task.to_state()
    prefix = input_state_for_example(example)
    segment_state = {
        **prefix,
        "goal": final_state["goal"],
        "problem_profile": task.metadata.get("problem_profile", {}),
    }
    if example.row.task_variant == "lean":
        segment_state.update(
            {
                "en_informal": example.row.en_informal,
                "zh_informal": example.row.zh_informal,
                "formal_statement_preview": example.row.answer,
            }
        )
    bind_state = {
        **segment_state,
        "entities": final_state["entities"],
        "quantities": final_state["quantities"],
        "relations": final_state["relations"],
    }
    plan_state = {
        **bind_state,
        "family_name": example.family_name,
        "strategy_hint": task.metadata.get("strategy_hint"),
        "question_focus_tokens": list(task.metadata.get("question_focus_tokens", ())),
    }
    if example.row.task_variant == "lean":
        plan_state.update(
            {
                "theorem_name": answer_metadata.get("theorem_name"),
                "formal_imports": list(answer_metadata.get("imports", ())),
                "concordance_profile": dict(answer_metadata.get("concordance_profile", {})),
            }
        )
    derive_state = {
        **plan_state,
        "program": final_state["program"],
        "answer_kind": answer_metadata.get("kind"),
        "answer_unit": answer_metadata.get("unit"),
    }
    if example.row.task_variant == "lean":
        derive_state.update(
            {
                "formal_statement": example.row.answer,
                "formal_proof": example.row.formal_proof,
                "formal_statement_raw": example.row.formal_statement_raw,
            }
        )
    verify_state = {
        **derive_state,
        "canonical_answer": answer_metadata.get("canonical"),
        "sympy_text": answer_metadata.get("sympy_text"),
        "verification": {
            "sympy_parse_success": example.row.task_variant != "lean",
            "answer_kind": answer_metadata.get("kind"),
        },
    }
    if example.row.task_variant == "lean":
        verify_state["verification"].update(
            {
                "formal_statement_has_sorry": bool(answer_metadata.get("formal_statement_has_sorry")),
                "formal_proof_has_sorry": bool(answer_metadata.get("formal_proof_has_sorry")),
                "theorem_name": answer_metadata.get("theorem_name"),
                "concordance_profile": dict(answer_metadata.get("concordance_profile", {})),
            }
        )
    return {
        "segment": segment_state,
        "bind": bind_state,
        "plan": plan_state,
        "derive": derive_state,
        "verify": verify_state,
        "render": final_state,
    }


def step_action(step_name: str, example: OlympiadMathExample) -> Dict[str, Any]:
    task = example.abstract_task
    answer_metadata = task.metadata.get("answer_metadata", {})
    if step_name == "segment":
        return {
            "subject": example.row.subject,
            "language": example.row.language,
            "difficulty_tier": example.row.difficulty_tier,
            "goal_query": task.goal.query,
            "task_variant": example.row.task_variant,
        }
    if step_name == "bind":
        return {
            "entity_ids": [entity.entity_id for entity in task.entities],
            "quantity_ids": [quantity.quantity_id for quantity in task.quantities],
            "relation_types": [relation["type"] for relation in task.relations],
        }
    if step_name == "plan":
        return {
            "family_name": example.family_name,
            "strategy_hint": task.metadata.get("strategy_hint"),
            "question_focus_tokens": list(task.metadata.get("question_focus_tokens", ())),
            "theorem_name": answer_metadata.get("theorem_name") if example.row.task_variant == "lean" else None,
        }
    if step_name == "derive":
        return {
            "program_op": task.program.op,
            "answer_kind": answer_metadata.get("kind"),
            "problem_profile": task.metadata.get("problem_profile", {}),
            "formal_statement_lines": answer_metadata.get("formal_statement_line_count") if example.row.task_variant == "lean" else None,
            "formal_proof_lines": answer_metadata.get("formal_proof_line_count") if example.row.task_variant == "lean" else None,
        }
    if step_name == "verify":
        return {
            "canonical_answer": answer_metadata.get("canonical"),
            "answer_unit": answer_metadata.get("unit"),
            "formal_proof_has_sorry": answer_metadata.get("formal_proof_has_sorry") if example.row.task_variant == "lean" else None,
        }
    return {
        "ir_keys": sorted(task.to_state()),
        "template_name": example.template_name,
    }


def compile_olympiad_math_trajectory(
    example: OlympiadMathExample,
    *,
    trajectory_index: int,
) -> TrajectoryRecord:
    final_state = example.abstract_task.to_state()
    step_states = example_step_states(example)
    total_possible_reward = sum(
        OLYMPIAD_STEP_WEIGHTS.get(step.name, STEP_WEIGHTS.get(step.name, 0.15))
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
        weight = OLYMPIAD_STEP_WEIGHTS.get(trace_step.name, STEP_WEIGHTS.get(trace_step.name, 0.15))
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
        trajectory_id=f"olympiad_math:{example.row.config}:{example.example_id}:{trajectory_index}",
        split=example.row.split,
        family=example.family_name,
        difficulty=example.abstract_task.difficulty,
        source_modality=example.abstract_task.source_modality,
        concept_tags=example.abstract_task.concept_tags,
        trace_template=tuple(step.name for step in example.abstract_task.trace_template),
        role_bindings={},
        episode_metadata=example.metadata,
        shortcut_checks=(
            (
                "problem subject and config must be preserved across the trajectory",
                "formal statement/proof concordance metadata must remain stable",
                "final rendered theorem name and proof availability must match the official Lean artifact",
            )
            if example.row.task_variant == "lean"
            else (
                "problem subject and config must be preserved across the trajectory",
                "canonical answer must remain stable under SymPy normalization",
                "final rendered answer must match the official benchmark answer string",
            )
        ),
        example=example,
        input_state=initial_input_state,
        output_state=final_state,
        steps=tuple(steps),
        total_reward=cumulative,
        total_possible_reward=total_possible_reward,
    )


def build_olympiad_math_examples(
    *,
    configs: Sequence[str] = DEFAULT_OLYMPIAD_MATH_CONFIGS,
    allow_eval_configs: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[OlympiadMathExample, ...], Dict[str, int]]:
    rows = load_olympiad_math_rows(
        configs=configs,
        allow_eval_configs=allow_eval_configs,
        max_rows=max_rows,
    )
    return parse_olympiad_math_rows(rows)


def compile_olympiad_math_examples(
    examples: Sequence[OlympiadMathExample],
) -> Tuple[TrajectoryRecord, ...]:
    return tuple(
        compile_olympiad_math_trajectory(example, trajectory_index=index)
        for index, example in enumerate(examples)
    )


def build_olympiad_math_trajectories(
    *,
    configs: Sequence[str] = DEFAULT_OLYMPIAD_MATH_CONFIGS,
    allow_eval_configs: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[TrajectoryRecord, ...], Dict[str, int]]:
    examples, failures = build_olympiad_math_examples(
        configs=configs,
        allow_eval_configs=allow_eval_configs,
        max_rows=max_rows,
    )
    return compile_olympiad_math_examples(examples), failures


def write_examples_jsonl(path: str | Path, examples: Iterable[OlympiadMathExample]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_jsonable()))
            handle.write("\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse OlymMATH rows into canonical reasoning IR and trajectories.")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=list(DEFAULT_OLYMPIAD_MATH_CONFIGS),
        help="OlymMATH configs to parse. Defaults to English easy/hard benchmark sets.",
    )
    parser.add_argument(
        "--translation-output",
        type=str,
        default="arc_trajectory_sampler/results/olympiad_math_reasoning_examples.jsonl",
        help="Destination JSONL path for parsed OlymMATH IR examples.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default="arc_trajectory_sampler/results/olympiad_math_reasoning_trajectories.jsonl",
        help="Destination JSONL path for parsed OlymMATH trajectories.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of OlymMATH rows to parse.",
    )
    parser.add_argument(
        "--allow-eval-configs",
        action="store_true",
        help="Allow official OlymMATH benchmark configs. Intended for audit/evaluation only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, failures = build_olympiad_math_examples(
        configs=args.configs,
        allow_eval_configs=args.allow_eval_configs,
        max_rows=args.max_rows,
    )
    trajectories = compile_olympiad_math_examples(examples)
    write_examples_jsonl(args.translation_output, examples)
    write_jsonl(args.trajectory_output, trajectories)
    summary = {
        "parsed_examples": len(examples),
        "failure_counts": failures,
        "trajectory_records": len(trajectories),
        "configs": list(args.configs),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
