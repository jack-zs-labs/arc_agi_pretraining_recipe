from __future__ import annotations

"""Download GSM8K and estimate fit against the current word-problem templates."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlretrieve


TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
ALL_GSM8K_SPLITS = ("train", "test")

UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_RE = re.compile(r"(?<![A-Za-z])(\d+(?:,\d{3})*(?:\.\d+)?)")
TAG_RE = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")
EXPR_RE = re.compile(r"^\s*\$?\(?\s*([0-9]+(?:\.[0-9]+)?)\s*([+\-*])\s*\$?\(?\s*([0-9]+(?:\.[0-9]+)?)\s*\)?\s*$")
FORBIDDEN_RE = re.compile(
    r"%|percent|average|mean|ratio|fraction|discount|interest|tax|mile per|mph|minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years|quarter|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twice|double|triple|quadruple|thrice|times as many|times as much|as many|as much|dozen|score\b|half\b",
    re.I,
)

ADD_RE = re.compile(r"\b(?:how many|how much)\b.*?\b(?:altogether|total|combined|in total|in all|now)\b", re.I | re.S)
SUB_RE = re.compile(
    r"\b(?:how many|how much)\b.*?\b(?:left|remain|remaining|still have|still has)\b|\bhow much more .* need\b",
    re.I | re.S,
)
COMPARE_RE = re.compile(r"\bhow (?:many|much) more\b.*?\bthan\b|\bdifference between\b|\bhow many fewer\b", re.I | re.S)
MULTIPLY_RE = re.compile(
    r"\b(?:each|every|per|in each|for each)\b.*?\b(?:how many|how much)\b.*?\b(?:altogether|total|in all)?|\bthere (?:are|were)\b.*?\bin each\b.*?\b(?:how many|how much)\b",
    re.I | re.S,
)

FAMILY_TO_OP = {
    "add_change": "+",
    "subtract_change": "-",
    "compare_difference": "-",
    "multiply_groups": "*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate GSM8K fit against the current word-problem templates.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="arc_trajectory_sampler/data/gsm8k",
        help="Directory where GSM8K JSONL files are stored.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/gsm8k_template_fit_summary.json",
        help="Destination summary JSON path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the total number of GSM8K rows to load for a quick smoke run.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_GSM8K_SPLITS,
        default=ALL_GSM8K_SPLITS,
        help="GSM8K splits to load for template-fit analysis.",
    )
    return parser.parse_args()


def ensure_gsm8k_files(data_dir: Path) -> Dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": data_dir / "train.jsonl",
        "test": data_dir / "test.jsonl",
    }
    if not paths["train"].exists():
        urlretrieve(TRAIN_URL, paths["train"])
    if not paths["test"].exists():
        urlretrieve(TEST_URL, paths["test"])
    return paths


def load_rows(
    paths: Dict[str, Path],
    *,
    splits: Sequence[str] = ALL_GSM8K_SPLITS,
    max_rows: int | None = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    row_limit = max_rows if max_rows is not None and max_rows > 0 else None
    for split in splits:
        path = paths[split]
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                payload = json.loads(line)
                payload["split"] = split
                payload["index"] = index
                rows.append(payload)
                if row_limit is not None and len(rows) >= row_limit:
                    return rows
    return rows


def extract_question_numbers(text: str) -> List[float]:
    values: List[float] = [float(item.replace(",", "")) for item in NUMBER_RE.findall(text)]
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in TENS:
            value = TENS[token]
            if index + 1 < len(tokens) and tokens[index + 1] in UNITS and UNITS[tokens[index + 1]] < 10:
                value += UNITS[tokens[index + 1]]
                index += 1
            values.append(float(value))
        elif token in UNITS:
            values.append(float(UNITS[token]))
        index += 1
    return values


def final_binary_expression(answer: str) -> Optional[Tuple[float, str, float, str]]:
    tags = TAG_RE.findall(answer)
    if not tags:
        return None
    expression = tags[-1][0].replace("x", "*").replace("X", "*").replace("×", "*").replace(",", "")
    match = EXPR_RE.match(expression)
    if not match:
        return None
    lhs, op, rhs = match.groups()
    return (float(lhs), op, float(rhs), expression)


def question_family(question: str) -> Optional[str]:
    lowered = question.lower().replace("’", "'")
    if FORBIDDEN_RE.search(lowered):
        return None
    matches = []
    if ADD_RE.search(question):
        matches.append("add_change")
    if SUB_RE.search(question):
        matches.append("subtract_change")
    if COMPARE_RE.search(question):
        matches.append("compare_difference")
    if MULTIPLY_RE.search(question):
        matches.append("multiply_groups")
    if len(matches) == 1:
        return matches[0]
    return None


def strict_match(question: str, answer: str) -> Optional[str]:
    family = question_family(question)
    if family is None:
        return None
    question_numbers = extract_question_numbers(question)
    if len(question_numbers) != 2:
        return None
    final_expression = final_binary_expression(answer)
    if final_expression is None:
        return None
    lhs, op, rhs, _ = final_expression
    if op != FAMILY_TO_OP[family]:
        return None
    if sorted(question_numbers) != sorted([lhs, rhs]):
        return None
    return family


def loose_match(question: str, answer: str) -> Optional[str]:
    family = question_family(question)
    if family is None:
        return None
    question_numbers = extract_question_numbers(question)
    if len(question_numbers) != 2:
        return None
    final_expression = final_binary_expression(answer)
    if final_expression is None:
        return None
    _, op, _, _ = final_expression
    if op != FAMILY_TO_OP[family]:
        return None
    return family


def add_example(store: Dict[str, List[Dict[str, str]]], family: str, row: Dict[str, str], *, expression: Optional[str]) -> None:
    bucket = store[family]
    if len(bucket) >= 5:
        return
    bucket.append(
        {
            "split": row["split"],
            "index": row["index"],
            "question": row["question"],
            "expression": expression,
        }
    )


def summarize_matches(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    strict_counts = Counter()
    loose_counts = Counter()
    strict_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    loose_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        final_expr = final_binary_expression(row["answer"])
        expression_text = final_expr[3] if final_expr is not None else None

        strict_family = strict_match(row["question"], row["answer"])
        if strict_family is not None:
            strict_counts[strict_family] += 1
            add_example(strict_examples, strict_family, row, expression=expression_text)

        loose_family = loose_match(row["question"], row["answer"])
        if loose_family is not None:
            loose_counts[loose_family] += 1
            add_example(loose_examples, loose_family, row, expression=expression_text)

    total_rows = len(rows)
    return {
        "num_examples": total_rows,
        "strict_fit_total": sum(strict_counts.values()),
        "strict_fit_rate": (sum(strict_counts.values()) / total_rows) if total_rows else 0.0,
        "strict_fit_by_family": dict(strict_counts),
        "strict_examples": dict(strict_examples),
        "loose_fit_total": sum(loose_counts.values()),
        "loose_fit_rate": (sum(loose_counts.values()) / total_rows) if total_rows else 0.0,
        "loose_fit_by_family": dict(loose_counts),
        "loose_examples": dict(loose_examples),
        "notes": {
            "strict": "Exact current-template fit: two primitive quantities in the question, no unsupported lexical cues, and the final worked equation is one binary op over those same two quantities.",
            "loose": "Upper bound: same family cue pattern and same final operator, but the worked solution may still involve hidden derivations. Manual inspection is required.",
        },
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    paths = ensure_gsm8k_files(data_dir)
    rows = load_rows(paths, splits=args.splits, max_rows=args.max_rows)
    summary = summarize_matches(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
