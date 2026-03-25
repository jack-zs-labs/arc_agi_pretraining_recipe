from __future__ import annotations

"""Parse GSM8K worked solutions into the canonical reasoning IR."""

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

try:
    from .analyze_gsm8k_template_fit import ALL_GSM8K_SPLITS, ensure_gsm8k_files, load_rows
    from .reasoning_ir import AbstractReasoningTask, EntitySpec, GoalSpec, QuantitySpec
    from .stage1_latent_sampler import Program
    from .stage4_trajectory_dataset import TrajectoryRecord, write_jsonl
    from .word_problem_translation_dataset import (
        WordProblemExample,
        _encode,
        compile_word_problem_trajectory,
        compose_trace,
        partition_inverse_trace,
        rate_scale_trace,
        slug,
    )
except ImportError:  # pragma: no cover - direct script execution
    from analyze_gsm8k_template_fit import ALL_GSM8K_SPLITS, ensure_gsm8k_files, load_rows  # type: ignore
    from reasoning_ir import AbstractReasoningTask, EntitySpec, GoalSpec, QuantitySpec  # type: ignore
    from stage1_latent_sampler import Program  # type: ignore
    from stage4_trajectory_dataset import TrajectoryRecord, write_jsonl  # type: ignore
    from word_problem_translation_dataset import (  # type: ignore
        WordProblemExample,
        _encode,
        compile_word_problem_trajectory,
        compose_trace,
        partition_inverse_trace,
        rate_scale_trace,
        slug,
    )


TAG_RE = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])(?:\d+\.\d+|\.\d+|\d+)(?![A-Za-z_])")
SAFE_EXPR_RE = re.compile(r"^[\d\.\+\-\*/\(\)\s]+$")
QUESTION_RE = re.compile(r"how\s+(many|much)\s+([^?]+)\?", re.I)
ANSWER_TOKEN_RE = re.compile(r"\d+\.\d+|\.\d+|\d+|//|[A-Za-z]+|[%$=()+\-*/:,]")
PURCHASE_RE = re.compile(r"\b(?:buy|costs?|purchase[sd]?|spend on)\s+(?:a|an|the)\s+([a-z][a-z ]+?)(?:\bwhich\b|\bthat\b|[?.!,]|$)", re.I)
NEW_ITEM_RE = re.compile(r"\bnew\s+([a-z][a-z ]+?)(?:\bwhich\b|\bcosts?\b|[?.!,]|$)", re.I)
CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]+\b")

STOP_CAPITALIZED = {
    "A",
    "An",
    "The",
    "If",
    "How",
    "What",
    "When",
    "Where",
    "Why",
    "On",
    "In",
    "At",
    "There",
    "It",
    "This",
    "That",
    "His",
    "Her",
    "Their",
    "They",
    "He",
    "She",
}
VERB_TOKENS = {
    "does",
    "do",
    "did",
    "are",
    "is",
    "were",
    "will",
    "would",
    "can",
    "could",
    "should",
    "has",
    "have",
    "need",
    "needs",
    "remain",
    "remains",
    "left",
    "eat",
    "eaten",
    "brewed",
    "bought",
    "pay",
    "spent",
    "spend",
    "make",
    "makes",
    "made",
    "read",
    "sell",
    "sold",
}
LEADING_DROP = {"more", "many", "much", "total", "altogether", "remaining", "left", "still"}
OP_NAME = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
}
TERM_OP_NAME = {
    "add": "sum",
    "subtract": "offset",
    "multiply": "scale",
    "divide": "partition",
}
IMPLEMENTED_FAMILIES = {"compose_total", "compose_difference"}
BOUNDARY_TOKENS = {":", ","}
OPERATORS = {"+", "-", "*", "/", "//", "(", ")"}
DEFAULT_GSM8K_TRAINING_SPLITS = ("train",)
GSM8K_EVAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class ParserFailure:
    reason: str
    details: str = ""


@dataclass
class QuantityNode:
    quantity_id: str
    value: Fraction
    is_derived: bool
    role: str
    unit: str = ""
    owner_id: str | None = None
    attributes: Dict[str, Any] | None = None

    def to_spec(self) -> QuantitySpec:
        attributes = dict(self.attributes or {})
        if self.value.denominator != 1:
            attributes["exact_value"] = {
                "numerator": self.value.numerator,
                "denominator": self.value.denominator,
            }
        return QuantitySpec(
            quantity_id=self.quantity_id,
            value=number_for_json(self.value),
            unit=self.unit,
            owner_id=self.owner_id,
            role=self.role,
            attributes=attributes,
        )


@dataclass(frozen=True)
class SolvedLinearEquation:
    lhs_coeff: Fraction
    lhs_const: Fraction
    rhs_coeff: Fraction
    rhs_const: Fraction
    coefficient: Fraction
    target: Fraction
    solution: Fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse GSM8K rows into reasoning IR examples and trajectories.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="arc_trajectory_sampler/data/gsm8k",
        help="Directory where GSM8K JSONL files are stored.",
    )
    parser.add_argument(
        "--translation-output",
        type=str,
        default="arc_trajectory_sampler/results/gsm8k_reasoning_examples.jsonl",
        help="Destination JSONL path for parsed GSM8K IR examples.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default="arc_trajectory_sampler/results/gsm8k_reasoning_trajectories.jsonl",
        help="Destination JSONL path for parsed GSM8K trajectories.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of GSM8K rows to parse for a quick smoke run.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_GSM8K_SPLITS,
        default=DEFAULT_GSM8K_TRAINING_SPLITS,
        help="GSM8K splits to parse. Defaults to train only for benchmark hygiene.",
    )
    parser.add_argument(
        "--allow-eval-splits",
        action="store_true",
        help="Allow official GSM8K eval splits such as `test`. Intended for audit/evaluation only.",
    )
    return parser.parse_args()


def validate_gsm8k_splits(
    splits: Sequence[str],
    *,
    allow_eval_splits: bool,
) -> Tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(split) for split in splits))
    blocked = sorted(split for split in normalized if split in GSM8K_EVAL_SPLITS)
    if blocked and not allow_eval_splits:
        blocked_text = ", ".join(blocked)
        raise ValueError(
            f"GSM8K eval split(s) requested without explicit opt-in: {blocked_text}. "
            "Pass allow_eval_splits=True or --allow-eval-splits only for audit/evaluation paths."
        )
    return normalized


def number_for_json(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def value_key(value: Fraction) -> Tuple[int, int]:
    return (value.numerator, value.denominator)


def normalize_expression(expression: str) -> str:
    cleaned = expression.replace("×", "*").replace("$", "").replace(",", "")
    cleaned = re.sub(r"(?<=\d|\))\s*[xX]\s*(?=\d|\.\d|\()", "*", cleaned)
    cleaned = cleaned.replace("//", "/")
    cleaned = cleaned.replace("−", "-").strip()
    return cleaned


def eval_fraction_expression(expression: str) -> Fraction:
    cleaned = normalize_expression(expression)
    if not cleaned or not SAFE_EXPR_RE.fullmatch(cleaned):
        raise ValueError(f"unsafe expression: {expression!r}")
    rewritten = NUMBER_RE.sub(lambda match: f'Fraction("{match.group(0)}")', cleaned)
    return eval(rewritten, {"__builtins__": {}}, {"Fraction": Fraction})


def parse_expression(expression: str) -> Tuple[str, ast.AST, Fraction]:
    cleaned = normalize_expression(expression)
    if not cleaned or not SAFE_EXPR_RE.fullmatch(cleaned):
        raise ValueError(f"unsupported expression: {expression!r}")
    node = ast.parse(cleaned, mode="eval").body
    return cleaned, node, eval_fraction_expression(cleaned)


def strip_uplus(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return strip_uplus(node.operand)
    return node


def top_level_symbol(node: ast.AST) -> str | None:
    node = strip_uplus(node)
    if not isinstance(node, ast.BinOp):
        return None
    if isinstance(node.op, ast.Add):
        return "+"
    if isinstance(node.op, ast.Sub):
        return "-"
    if isinstance(node.op, ast.Mult):
        return "*"
    if isinstance(node.op, ast.Div):
        return "/"
    return None


def flatten_operands(node: ast.AST, symbol: str) -> List[ast.AST]:
    node = strip_uplus(node)
    if symbol == "+" and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return flatten_operands(node.left, symbol) + flatten_operands(node.right, symbol)
    if symbol == "*" and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return flatten_operands(node.left, symbol) + flatten_operands(node.right, symbol)
    if symbol == "-" and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return flatten_operands(node.left, symbol) + [node.right]
    if symbol == "/":
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return [node.left, node.right]
    return [node]


def source_segment(expression: str, node: ast.AST) -> str:
    return ast.get_source_segment(expression, node) or normalize_expression(expression)


def extract_tagged_equations(answer: str) -> List[Tuple[str, ast.AST, Fraction]]:
    equations: List[Tuple[str, ast.AST, Fraction]] = []
    for expression_text, result_text in TAG_RE.findall(answer):
        cleaned, node, expression_value = parse_expression(expression_text)
        result_value = eval_fraction_expression(result_text)
        if expression_value != result_value:
            raise ValueError(
                f"tagged expression mismatch: {expression_text!r} -> {expression_value} but tag result is {result_text!r}"
            )
        equations.append((cleaned, node, result_value))
    return equations


def extract_final_answer(answer: str) -> Fraction | None:
    if "####" not in answer:
        return None
    tail = answer.split("####", 1)[1]
    match = re.search(r"-?(?:\d+\.\d+|\.\d+|\d+)", tail.replace(",", ""))
    if not match:
        return None
    return Fraction(match.group(0))


def is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"(?:\d+\.\d+|\.\d+|\d+)", token))


def is_variable_token(token: str) -> bool:
    return token.isalpha() and len(token) == 1


def is_math_token(token: str) -> bool:
    return is_number_token(token) or is_variable_token(token) or token in OPERATORS or token in {"=", "%", "$"}


def needs_implicit_multiply(previous: str, current: str) -> bool:
    left_atom = is_number_token(previous) or is_variable_token(previous) or previous == ")"
    right_atom = is_number_token(current) or is_variable_token(current) or current == "("
    return left_atom and right_atom


def tokens_to_expression(tokens: Sequence[str]) -> str:
    compact: List[str] = []
    for token in tokens:
        if token == "$":
            continue
        if token == "%":
            compact.extend(["/", "100"])
        else:
            compact.append(token)
    if not compact:
        return ""
    pieces: List[str] = []
    for token in compact:
        if pieces and needs_implicit_multiply(pieces[-1], token):
            pieces.append("*")
        pieces.append(token)
    return normalize_expression("".join(pieces))


def strip_tags(answer: str) -> str:
    return TAG_RE.sub("", answer)


def extract_plain_equations(answer: str) -> List[Tuple[str, str]]:
    equations: List[Tuple[str, str]] = []
    for line in strip_tags(answer).splitlines():
        tokens = ANSWER_TOKEN_RE.findall(line)
        for index, token in enumerate(tokens):
            if token != "=":
                continue
            left = index - 1
            while left >= 0 and tokens[left] != "=" and tokens[left] not in BOUNDARY_TOKENS:
                if tokens[left].isalpha() and len(tokens[left]) > 1:
                    break
                if not is_math_token(tokens[left]):
                    break
                left -= 1
            right = index + 1
            while right < len(tokens) and tokens[right] != "=" and tokens[right] not in BOUNDARY_TOKENS:
                if tokens[right].isalpha() and len(tokens[right]) > 1:
                    break
                if not is_math_token(tokens[right]):
                    break
                right += 1
            lhs = tokens_to_expression(tokens[left + 1 : index])
            rhs = tokens_to_expression(tokens[index + 1 : right])
            if lhs and rhs:
                equations.append((lhs, rhs))
    return equations


def expression_has_variable(expression: str) -> bool:
    return bool(re.search(r"[A-Za-z]", expression))


def parse_linear_expression(expression: str) -> Tuple[str | None, Fraction, Fraction]:
    node = ast.parse(expression, mode="eval").body

    def combine_var(lhs: str | None, rhs: str | None) -> str | None:
        if lhs is None:
            return rhs
        if rhs is None or lhs == rhs:
            return lhs
        raise ValueError(f"multiple variables: {lhs}, {rhs}")

    def visit(current: ast.AST) -> Tuple[str | None, Fraction, Fraction]:
        if isinstance(current, ast.Constant) and isinstance(current.value, int | float):
            return None, Fraction(0), Fraction(str(current.value))
        if isinstance(current, ast.Name):
            if not is_variable_token(current.id):
                raise ValueError(f"unsupported variable token: {current.id}")
            return current.id, Fraction(1), Fraction(0)
        if isinstance(current, ast.UnaryOp):
            var_name, coeff, constant = visit(current.operand)
            if isinstance(current.op, ast.UAdd):
                return var_name, coeff, constant
            if isinstance(current.op, ast.USub):
                return var_name, -coeff, -constant
            raise ValueError(f"unsupported unary op: {ast.dump(current)}")
        if isinstance(current, ast.BinOp):
            left_var, left_coeff, left_const = visit(current.left)
            right_var, right_coeff, right_const = visit(current.right)
            if isinstance(current.op, ast.Add):
                return combine_var(left_var, right_var), left_coeff + right_coeff, left_const + right_const
            if isinstance(current.op, ast.Sub):
                return combine_var(left_var, right_var), left_coeff - right_coeff, left_const - right_const
            if isinstance(current.op, ast.Mult):
                if left_var is None:
                    return right_var, right_coeff * left_const, right_const * left_const
                if right_var is None:
                    return left_var, left_coeff * right_const, left_const * right_const
                raise ValueError(f"nonlinear term: {expression}")
            if isinstance(current.op, ast.Div):
                if right_var is not None or right_coeff != 0:
                    raise ValueError(f"variable in denominator: {expression}")
                return left_var, left_coeff / right_const, left_const / right_const
            if isinstance(current.op, ast.FloorDiv):
                if right_var is not None or right_coeff != 0:
                    raise ValueError(f"variable in denominator: {expression}")
                return left_var, left_coeff / right_const, left_const / right_const
        raise ValueError(f"unsupported linear expression: {expression}")

    return visit(node)


def solve_linear_equation(lhs: str, rhs: str) -> SolvedLinearEquation | None:
    try:
        lhs_var, lhs_coeff, lhs_const = parse_linear_expression(lhs)
        rhs_var, rhs_coeff, rhs_const = parse_linear_expression(rhs)
    except Exception:
        return None
    if lhs_var is None and rhs_var is None:
        return None
    if lhs_var is not None and rhs_var is not None and lhs_var != rhs_var:
        return None
    coefficient = lhs_coeff - rhs_coeff
    if coefficient == 0:
        return None
    target = rhs_const - lhs_const
    return SolvedLinearEquation(
        lhs_coeff=lhs_coeff,
        lhs_const=lhs_const,
        rhs_coeff=rhs_coeff,
        rhs_const=rhs_const,
        coefficient=coefficient,
        target=target,
        solution=target / coefficient,
    )


def fraction_to_expression(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_numeric_equation(expression: str, result: Fraction) -> Tuple[str, ast.AST, Fraction]:
    cleaned, node, expression_value = parse_expression(expression)
    if expression_value != result:
        raise ValueError(f"expression/result mismatch: {expression} -> {expression_value} != {result}")
    return cleaned, node, result


def synthetic_steps_from_linear_solution(solved: SolvedLinearEquation) -> List[Tuple[str, ast.AST, Fraction]]:
    equations: List[Tuple[str, ast.AST, Fraction]] = []
    effective_target = solved.target
    effective_coefficient = solved.coefficient
    if effective_target < 0 and effective_coefficient < 0:
        effective_target = -effective_target
        effective_coefficient = -effective_coefficient

    target_expression = fraction_to_expression(effective_target)
    if solved.lhs_const != 0 and solved.rhs_const != 0:
        lhs_const = fraction_to_expression(solved.lhs_const)
        rhs_const = fraction_to_expression(solved.rhs_const)
        if effective_target == solved.rhs_const - solved.lhs_const:
            target_expression = f"{rhs_const}-({lhs_const})"
        else:
            target_expression = f"{lhs_const}-({rhs_const})"
        equations.append(parse_numeric_equation(target_expression, effective_target))
    elif solved.rhs_const != 0:
        target_expression = fraction_to_expression(solved.rhs_const)
    elif solved.lhs_const != 0 and effective_target != abs(solved.lhs_const):
        lhs_const = fraction_to_expression(solved.lhs_const)
        target_expression = f"0-({lhs_const})"
        equations.append(parse_numeric_equation(target_expression, effective_target))
    if effective_coefficient == 1:
        if equations:
            return equations
        passthrough = f"{fraction_to_expression(effective_target)}+0"
        equations.append(parse_numeric_equation(passthrough, solved.solution))
        return equations
    solve_expression = f"({target_expression})/({fraction_to_expression(effective_coefficient)})"
    equations.append(parse_numeric_equation(solve_expression, solved.solution))
    return equations


def synthesize_fallback_equations(answer: str) -> List[Tuple[str, ast.AST, Fraction]]:
    final_answer = extract_final_answer(answer)
    if final_answer is None:
        return []
    for lhs, rhs in reversed(extract_plain_equations(answer)):
        lhs_has_var = expression_has_variable(lhs)
        rhs_has_var = expression_has_variable(rhs)
        if lhs_has_var ^ rhs_has_var:
            variable_side = lhs if lhs_has_var else rhs
            numeric_side = rhs if lhs_has_var else lhs
            try:
                numeric_value = eval_fraction_expression(numeric_side)
            except Exception:
                numeric_value = None
            if numeric_value == final_answer and has_nontrivial_operator(numeric_side):
                return [parse_numeric_equation(numeric_side, final_answer)]
            if variable_side.strip().isalpha() and numeric_value == final_answer:
                continue
            solved = solve_linear_equation(lhs, rhs)
            if solved and solved.solution == final_answer:
                return synthetic_steps_from_linear_solution(solved)
            continue
        try:
            lhs_value = eval_fraction_expression(lhs)
            rhs_value = eval_fraction_expression(rhs)
        except Exception:
            continue
        if lhs_value == final_answer and has_nontrivial_operator(lhs):
            return [parse_numeric_equation(lhs, final_answer)]
        if rhs_value == final_answer and has_nontrivial_operator(rhs):
            return [parse_numeric_equation(rhs, final_answer)]
        if lhs_value == final_answer:
            passthrough = f"{fraction_to_expression(final_answer)}+0"
            return [parse_numeric_equation(passthrough, final_answer)]
        if rhs_value == final_answer:
            passthrough = f"{fraction_to_expression(final_answer)}+0"
            return [parse_numeric_equation(passthrough, final_answer)]
    return []


def has_nontrivial_operator(expression: str) -> bool:
    stripped = expression.strip()
    return any(op in stripped for op in ("+", "-", "*", "/")) and stripped not in {"+", "-"}


def infer_family(final_symbol: str) -> str | None:
    if final_symbol == "+":
        return "compose_total"
    if final_symbol == "-":
        return "compose_difference"
    if final_symbol == "*":
        return "rate_scale"
    if final_symbol == "/":
        return "partition_inverse"
    return None


def infer_goal_unit(question: str) -> str:
    lowered = question.lower()
    if "$" in question or " money " in f" {lowered} " or " budget" in lowered or "profit" in lowered or "credit" in lowered:
        return "dollars"
    match = QUESTION_RE.search(question)
    if not match:
        return "units"
    phrase = match.group(2).lower()
    tokens = [token for token in re.findall(r"[a-z]+", phrase) if token not in LEADING_DROP]
    collected: List[str] = []
    for token in tokens:
        if token in VERB_TOKENS:
            break
        collected.append(token)
    if not collected:
        return "units"
    unit = " ".join(collected[:3]).strip()
    if unit in {"money", "profit", "credit"}:
        return "dollars"
    return unit or "units"


def infer_purchase_item(question: str) -> str | None:
    for pattern in (NEW_ITEM_RE, PURCHASE_RE):
        match = pattern.search(question)
        if match:
            item = match.group(1).strip().lower()
            item = re.sub(r"\b(?:which|that|for|to)\b.*$", "", item).strip()
            return item or None
    return None


def infer_entities(question: str, goal_unit: str) -> Tuple[EntitySpec, ...]:
    entities: List[EntitySpec] = []
    seen_ids = set()
    for token in CAPITALIZED_RE.findall(question):
        if token in STOP_CAPITALIZED:
            continue
        entity_id = slug(token)
        if entity_id in seen_ids:
            continue
        entities.append(EntitySpec(entity_id=entity_id, label=token, kind="person"))
        seen_ids.add(entity_id)
    if not entities:
        if question.lower().startswith("a coffee shop"):
            entities.append(EntitySpec(entity_id="coffee_shop", label="coffee shop", kind="business"))
            seen_ids.add("coffee_shop")
        else:
            entities.append(EntitySpec(entity_id="scenario", label="scenario", kind="scenario"))
            seen_ids.add("scenario")
    if goal_unit == "dollars":
        purchase_item = infer_purchase_item(question)
        if purchase_item:
            entity_id = slug(purchase_item)
            if entity_id not in seen_ids:
                entities.append(EntitySpec(entity_id=entity_id, label=purchase_item, kind="item"))
    elif goal_unit != "units":
        entity_id = slug(goal_unit)
        if entity_id not in seen_ids:
            entities.append(EntitySpec(entity_id=entity_id, label=goal_unit, kind="item"))
    return tuple(entities)


def infer_goal_query(family: str, row: Dict[str, Any], goal_unit: str) -> str:
    unit_slug = slug(goal_unit) or "units"
    return f"gsm8k_{family}_{row['split']}_{row['index']}_{unit_slug}"


def quantity_role_from_usage(node: QuantityNode, usages: Sequence[Tuple[str, int, bool]], goal_unit: str) -> Tuple[str, str]:
    if node.is_derived:
        return ("derived_term", goal_unit)
    if not usages:
        return ("primitive", "scalar")
    if all(op in {"multiply", "divide"} and position > 0 for op, position, _ in usages):
        role = "scale" if any(op == "multiply" for op, _, _ in usages) else "partition"
        return (role, "scalar")
    if any(is_final for _, _, is_final in usages) or any(op in {"add", "subtract"} for op, _, _ in usages):
        return ("base", goal_unit)
    if any(op == "multiply" for op, position, _ in usages if position == 0):
        return ("rate", goal_unit)
    return ("primitive", goal_unit)


def sorted_term_ops(derivation_rules: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
    seen = []
    for rule in derivation_rules:
        term_name = TERM_OP_NAME.get(rule["op"])
        if term_name is None or term_name in seen:
            continue
        seen.append(term_name)
    return tuple(seen)


def parse_gsm8k_row(row: Dict[str, Any]) -> Tuple[WordProblemExample | None, ParserFailure | None]:
    try:
        tagged_equations = extract_tagged_equations(row["answer"])
    except Exception as exc:
        return None, ParserFailure("invalid_tagged_equations", str(exc))

    equations = list(tagged_equations)
    parser_mode = "answer_trace_supervised"
    if not equations:
        equations = synthesize_fallback_equations(row["answer"])
        if not equations:
            return None, ParserFailure("no_tagged_equations")
        parser_mode = "answer_equation_fallback"

    final_expression, final_node, final_value = equations[-1]
    final_symbol = top_level_symbol(final_node)
    family = infer_family(final_symbol or "")
    if family is None:
        fallback_equations = synthesize_fallback_equations(row["answer"])
        if fallback_equations:
            if tagged_equations and top_level_symbol(tagged_equations[-1][1]) is None:
                equations = list(tagged_equations[:-1]) + fallback_equations
            else:
                equations = fallback_equations
            parser_mode = "answer_equation_fallback"
            final_expression, final_node, final_value = equations[-1]
            final_symbol = top_level_symbol(final_node)
            family = infer_family(final_symbol or "")
        if family is None:
            return None, ParserFailure("unsupported_final_reducer", final_symbol or "none")

    goal_unit = infer_goal_unit(row["question"])
    entities = infer_entities(row["question"], goal_unit)
    primary_owner_id = next((entity.entity_id for entity in entities if entity.kind in {"person", "business", "scenario"}), None)

    quantity_nodes: Dict[str, QuantityNode] = {}
    creation_order: List[str] = []
    primitive_ids_by_value: Dict[Tuple[int, int], str] = {}
    available_ids_by_value: DefaultDict[Tuple[int, int], List[str]] = defaultdict(list)
    usages: DefaultDict[str, List[Tuple[str, int, bool]]] = defaultdict(list)
    derivation_rules: List[Dict[str, Any]] = []

    def get_or_create_literal(value: Fraction) -> str:
        key = value_key(value)
        if available_ids_by_value[key]:
            return available_ids_by_value[key][-1]
        if key in primitive_ids_by_value:
            quantity_id = primitive_ids_by_value[key]
        else:
            quantity_id = f"q_primitive_{len(primitive_ids_by_value)}"
            quantity_nodes[quantity_id] = QuantityNode(
                quantity_id=quantity_id,
                value=value,
                is_derived=False,
                role="primitive",
                owner_id=primary_owner_id,
                attributes={"origin": "answer_trace_literal"},
            )
            primitive_ids_by_value[key] = quantity_id
            creation_order.append(quantity_id)
        available_ids_by_value[key].append(quantity_id)
        return quantity_id

    def resolve_operand(node: ast.AST, expression: str) -> str:
        operand_value = eval_fraction_expression(source_segment(expression, node))
        return get_or_create_literal(operand_value)

    for equation_index, (expression, node, result_value) in enumerate(equations[:-1]):
        symbol = top_level_symbol(node)
        if symbol is None:
            # Some GSM8K solutions restate an already-computed value as <<55=55>>
            # or <<+10=10>>. That does not introduce a new derivation rule.
            get_or_create_literal(result_value)
            continue
        if symbol not in OP_NAME:
            return None, ParserFailure("unsupported_intermediate_op", expression)
        operand_nodes = flatten_operands(node, symbol)
        input_ids = tuple(resolve_operand(child, expression) for child in operand_nodes)
        target_id = f"q_derived_{equation_index}"
        quantity_nodes[target_id] = QuantityNode(
            quantity_id=target_id,
            value=result_value,
            is_derived=True,
            role="derived_term",
            owner_id=primary_owner_id,
            attributes={"origin_expression": expression},
        )
        creation_order.append(target_id)
        available_ids_by_value[value_key(result_value)].append(target_id)
        derivation_rules.append({"target": target_id, "op": OP_NAME[symbol], "inputs": input_ids})
        for input_index, input_id in enumerate(input_ids):
            usages[input_id].append((OP_NAME[symbol], input_index, False))

    final_operand_nodes = flatten_operands(final_node, final_symbol or "")
    final_input_ids = tuple(resolve_operand(child, final_expression) for child in final_operand_nodes)
    program: Program
    trace_template: Tuple[Any, ...]
    concept_tags: Tuple[str, ...]
    metadata: Dict[str, Any] = {
        "template": f"gsm8k_{family}",
        "answer_unit": goal_unit,
        "family_name": family,
        "gsm8k_split": row["split"],
        "gsm8k_index": row["index"],
        "gsm8k_answer": row["answer"],
        "parser_mode": parser_mode,
        "tagged_expressions": [expression for expression, _, _ in equations],
    }
    example_metadata: Dict[str, Any] = {
        "answer_unit": goal_unit,
        "family_name": family,
        "gsm8k_split": row["split"],
        "gsm8k_index": row["index"],
        "parser_mode": parser_mode,
    }

    if family == "compose_total":
        reducer = "sum"
        reduce_input_ids: Tuple[Any, ...] = final_input_ids
        program = Program(
            op="ComposeThenReduce",
            args=(final_input_ids, {"term_ops": sorted_term_ops(derivation_rules), "reducer": reducer}),
        )
        trace_template = compose_trace(reducer)
        concept_tags = ("word_problem", "gsm8k", "composition", "reduction", "addition")
        metadata["reduce_input_ids"] = _encode(reduce_input_ids)
        metadata["reducer"] = reducer
        example_metadata["operation"] = "ComposeThenReduce"
        final_usage_op = "add"
    elif family == "compose_difference":
        reducer = "difference"
        if len(final_input_ids) < 2:
            return None, ParserFailure("invalid_difference_operands", final_expression)
        reduce_input_ids = ((final_input_ids[0],), tuple(final_input_ids[1:]))
        program = Program(
            op="ComposeThenReduce",
            args=(
                (final_input_ids[0],),
                tuple(final_input_ids[1:]),
                {"term_ops": sorted_term_ops(derivation_rules), "reducer": reducer},
            ),
        )
        trace_template = compose_trace(reducer)
        concept_tags = ("word_problem", "gsm8k", "composition", "comparison", "subtraction")
        metadata["reduce_input_ids"] = _encode(reduce_input_ids)
        metadata["reducer"] = reducer
        example_metadata["operation"] = "ComposeThenReduce"
        final_usage_op = "subtract"
    elif family == "rate_scale":
        reducer = "product"
        program = Program(
            op="ApplyRate",
            args=(final_input_ids, {"rate_ops": sorted_term_ops(derivation_rules), "reducer": reducer}),
        )
        trace_template = rate_scale_trace()
        concept_tags = ("word_problem", "gsm8k", "rate", "scaling", "multiplication")
        metadata["final_input_ids"] = _encode(final_input_ids)
        metadata["reducer"] = reducer
        example_metadata["operation"] = "ApplyRate"
        final_usage_op = "multiply"
    else:
        if len(final_input_ids) != 2:
            return None, ParserFailure("invalid_partition_operands", final_expression)
        reducer = "quotient"
        program = Program(
            op="PartitionInverse",
            args=(
                (final_input_ids[0],),
                (final_input_ids[1],),
                {"partition_ops": sorted_term_ops(derivation_rules), "reducer": reducer},
            ),
        )
        trace_template = partition_inverse_trace()
        concept_tags = ("word_problem", "gsm8k", "partition", "inverse_reasoning", "division")
        metadata["final_input_ids"] = _encode(final_input_ids)
        metadata["reducer"] = reducer
        example_metadata["operation"] = "PartitionInverse"
        final_usage_op = "divide"

    for input_index, input_id in enumerate(final_input_ids):
        usages[input_id].append((final_usage_op, input_index, True))

    primitive_quantity_ids = tuple(
        quantity_id for quantity_id in creation_order if not quantity_nodes[quantity_id].is_derived
    )
    derived_quantity_ids = tuple(
        quantity_id for quantity_id in creation_order if quantity_nodes[quantity_id].is_derived
    )

    for quantity_id in creation_order:
        node = quantity_nodes[quantity_id]
        role, unit = quantity_role_from_usage(node, usages[quantity_id], goal_unit)
        node.role = role
        node.unit = unit
        if node.owner_id is None and unit == goal_unit:
            node.owner_id = primary_owner_id

    task = AbstractReasoningTask(
        task_id=f"gsm8k_{row['split']}_{row['index']}_task",
        source_modality="text",
        source_text=row["question"],
        entities=entities,
        quantities=tuple(quantity_nodes[quantity_id].to_spec() for quantity_id in creation_order),
        relations=(
            {"type": "gsm8k_source", "split": row["split"], "index": row["index"]},
            *(
                {"type": "derivation", "target": rule["target"], "op": rule["op"], "inputs": rule["inputs"]}
                for rule in derivation_rules
            ),
        ),
        goal=GoalSpec(
            target_id="answer",
            query=infer_goal_query(family, row, goal_unit),
            unit=goal_unit,
        ),
        program=program,
        trace_template=trace_template,
        answer=number_for_json(final_value),
        concept_tags=concept_tags,
        difficulty=4,
        metadata={
            "primitive_quantity_ids": primitive_quantity_ids,
            "derived_quantity_ids": derived_quantity_ids,
            "derivation_rules": _encode(tuple(derivation_rules)),
            **metadata,
        },
    )
    example = WordProblemExample(
        example_id=f"gsm8k_{row['split']}_{row['index']}",
        source_text=row["question"],
        abstract_task=task,
        family_name=family,
        template_name=f"gsm8k_{family}",
        notes=("parsed from GSM8K worked solution tags",),
        metadata=example_metadata,
    )
    return example, None


def parse_gsm8k_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[Tuple[WordProblemExample, ...], Dict[str, int]]:
    examples: List[WordProblemExample] = []
    failures = Counter()
    for row in rows:
        example, failure = parse_gsm8k_row(row)
        if example is None:
            failures[failure.reason if failure else "unknown_failure"] += 1
            continue
        examples.append(example)
    return tuple(examples), dict(failures)


def build_gsm8k_examples(
    *,
    data_dir: str | Path,
    splits: Sequence[str] = DEFAULT_GSM8K_TRAINING_SPLITS,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[WordProblemExample, ...], Dict[str, int]]:
    paths = ensure_gsm8k_files(Path(data_dir))
    resolved_splits = validate_gsm8k_splits(splits, allow_eval_splits=allow_eval_splits)
    rows = load_rows(paths, splits=resolved_splits, max_rows=max_rows)
    return parse_gsm8k_rows(rows)


def build_gsm8k_trajectories(
    *,
    data_dir: str | Path,
    splits: Sequence[str] = DEFAULT_GSM8K_TRAINING_SPLITS,
    allow_eval_splits: bool = False,
    max_rows: int | None = None,
) -> Tuple[Tuple[TrajectoryRecord, ...], Dict[str, int]]:
    examples, failures = build_gsm8k_examples(
        data_dir=data_dir,
        splits=splits,
        allow_eval_splits=allow_eval_splits,
        max_rows=max_rows,
    )
    trajectories = compile_gsm8k_examples(examples)
    return trajectories, failures


def compile_gsm8k_examples(examples: Sequence[WordProblemExample]) -> Tuple[TrajectoryRecord, ...]:
    records: List[TrajectoryRecord] = []
    for index, example in enumerate(examples):
        split = str(example.metadata.get("gsm8k_split", "train"))
        records.append(compile_word_problem_trajectory(example, split=split, trajectory_index=index))
    return tuple(records)


def write_examples_jsonl(path: str | Path, examples: Iterable[WordProblemExample]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_jsonable()))
            handle.write("\n")
    return output_path


def main() -> None:
    args = parse_args()
    examples, failures = build_gsm8k_examples(
        data_dir=args.data_dir,
        splits=args.splits,
        allow_eval_splits=args.allow_eval_splits,
        max_rows=args.max_rows,
    )
    trajectories = compile_gsm8k_examples(examples)
    write_examples_jsonl(args.translation_output, examples)
    write_jsonl(args.trajectory_output, trajectories)
    summary = {
        "parsed_examples": len(examples),
        "failure_counts": failures,
        "trajectory_records": len(trajectories),
        "splits": list(args.splits),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
