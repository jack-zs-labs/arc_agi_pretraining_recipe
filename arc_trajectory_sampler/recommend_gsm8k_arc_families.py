from __future__ import annotations

"""Recommend ARC-style latent families for partial GSM8K coverage."""

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .stage1_latent_sampler import Program, TraceStep
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import Program, TraceStep  # type: ignore


TAG_RE = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")
SEQ_RE = re.compile(r"[+\-*/]")

TOTAL_RE = re.compile(r"altogether|total|in all|combined|together|overall|\bnow\b", re.I)
REMAIN_RE = re.compile(r"left|remain|remaining|change|still have|still has|\bneed\b", re.I)
COMPARE_RE = re.compile(r"how many more|how much more|difference|fewer than|less than", re.I)
RATE_RE = re.compile(r"\beach\b|\bevery\b|\bper\b|\bin each\b|\bfor each\b|\bdaily\b|\bweekly\b|\bmonthly\b|\bhour\b|\bhours\b|\bday\b|\bdays\b|\bweek\b|\bweeks\b", re.I)
PARTITION_RE = re.compile(r"share|shared|equally|average|groups? of|each group|split|divided", re.I)
UNIT_RE = re.compile(
    r"feet|foot|inches|yards|ounces|pounds|gallons|quarts|pints|minutes|hours|days|weeks|months|years|meters|kilometers|centimeters|grams|kilograms|miles|liters",
    re.I,
)


@dataclass(frozen=True)
class ProposedFamilySpec:
    family_name: str
    arc_analog: str
    concept_tags: Tuple[str, ...]
    program: Program
    trace_template: Tuple[TraceStep, ...]
    constraints: Tuple[str, ...]
    rationale: str

    def to_jsonable(self) -> Dict[str, Any]:
        return _encode(self)


FAMILY_SPECS = {
    "compose_total": ProposedFamilySpec(
        family_name="compose_total",
        arc_analog="count_select",
        concept_tags=("word_problem", "composition", "reduction", "addition"),
        program=Program(
            op="ComposeThenReduce",
            args=(
                "derive_terms",
                {"term_ops": ("scale", "offset", "convert"), "reducer": "sum"},
            ),
        ),
        trace_template=(
            TraceStep("segment", "Parse the text into primitive numeric facts."),
            TraceStep("bind", "Bind named quantities and per-term rules."),
            TraceStep("apply", "Derive each hidden term value locally."),
            TraceStep("reduce", "Sum the derived terms into one result."),
            TraceStep("render", "Emit the final answer value."),
        ),
        constraints=(
            "final_query_requests_total_or_combined_quantity",
            "small_number_of_derived_terms",
            "final_reducer_is_sum",
        ),
        rationale="Numeric analog of ARC count/reduce tasks: map local transforms onto terms, then reduce by sum.",
    ),
    "compose_difference": ProposedFamilySpec(
        family_name="compose_difference",
        arc_analog="count_select",
        concept_tags=("word_problem", "composition", "comparison", "subtraction"),
        program=Program(
            op="ComposeThenReduce",
            args=(
                "derive_reference_terms",
                {"term_ops": ("scale", "offset", "convert"), "reducer": "difference"},
            ),
        ),
        trace_template=(
            TraceStep("segment", "Parse the text into primitive numeric facts."),
            TraceStep("bind", "Bind named quantities and comparative cues."),
            TraceStep("apply", "Derive the reference quantities locally."),
            TraceStep("reduce", "Subtract to answer compare or remaining query."),
            TraceStep("render", "Emit the final answer value."),
        ),
        constraints=(
            "final_query_requests_remaining_or_difference",
            "final_reducer_is_subtraction",
            "derived_terms_stay_scalar",
        ),
        rationale="Same overall skeleton as compose-total, but the reducer is subtraction for compare/remaining questions.",
    ),
    "rate_scale": ProposedFamilySpec(
        family_name="rate_scale",
        arc_analog="relational",
        concept_tags=("word_problem", "rate", "scaling", "multiplication"),
        program=Program(
            op="ApplyRate",
            args=(
                "base_quantity",
                {"rate_ops": ("per_unit", "repetition"), "reducer": "product"},
            ),
        ),
        trace_template=(
            TraceStep("segment", "Parse base quantity, rate, and repetition extent."),
            TraceStep("bind", "Bind the base quantity and rate relation."),
            TraceStep("apply", "Apply the rate or repetition transform."),
            TraceStep("render", "Emit the scaled result."),
        ),
        constraints=(
            "question_contains_explicit_rate_or_each_cue",
            "final_reducer_is_multiplication",
            "single_primary_scaled_result",
        ),
        rationale="Numeric analog of ARC relational application: one quantity is transformed by an explicit relation like each/per.",
    ),
    "partition_inverse": ProposedFamilySpec(
        family_name="partition_inverse",
        arc_analog="count_select",
        concept_tags=("word_problem", "partition", "inverse_reasoning", "division"),
        program=Program(
            op="PartitionInverse",
            args=(
                "whole_quantity",
                {"partition_ops": ("share", "group_count", "average"), "reducer": "quotient"},
            ),
        ),
        trace_template=(
            TraceStep("segment", "Parse whole quantity and partition rule."),
            TraceStep("bind", "Bind the group size or number of shares."),
            TraceStep("reduce", "Divide to recover the missing part or group count."),
            TraceStep("render", "Emit the final answer value."),
        ),
        constraints=(
            "question_contains_explicit_share_or_grouping_cue",
            "final_reducer_is_division",
            "partition_rule_is_uniform",
        ),
        rationale="Numeric analog of selecting equal groups or missing partitions from a whole.",
    ),
    "unit_convert": ProposedFamilySpec(
        family_name="unit_convert",
        arc_analog="symbol_map",
        concept_tags=("word_problem", "units", "conversion", "binding"),
        program=Program(
            op="ConvertThenApply",
            args=(
                "quantity_with_units",
                {"conversion_rule": "unit_map", "post_op": "scalar_arithmetic"},
            ),
        ),
        trace_template=(
            TraceStep("segment", "Parse quantities and their source units."),
            TraceStep("bind", "Bind the source and target unit mapping."),
            TraceStep("apply", "Convert to the target unit scale."),
            TraceStep("render", "Emit the converted or post-conversion result."),
        ),
        constraints=(
            "question_contains_explicit_unit_pair",
            "unit_conversion_rule_is_fixed",
            "conversion_precedes_final_scalar_arithmetic",
        ),
        rationale="Closest numeric analog to ARC symbol binding: bind a stable unit map, then apply it before solving.",
    ),
}


def _encode(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [_encode(item) for item in obj]
    if isinstance(obj, list):
        return [_encode(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _encode(value) for key, value in asdict(obj).items()}
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend ARC-style latent families for GSM8K coverage.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="arc_trajectory_sampler/data/gsm8k",
        help="Directory containing GSM8K train/test JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/gsm8k_arc_family_recommendations.json",
        help="Destination JSON path for the recommendation summary.",
    )
    return parser.parse_args()


def final_op(answer: str) -> Optional[str]:
    tags = TAG_RE.findall(answer)
    if not tags:
        return None
    expr = tags[-1][0].replace("x", "*").replace("X", "*").replace("×", "*")
    found = SEQ_RE.findall(expr)
    if not found:
        return None
    unique = set(found)
    return found[-1] if len(unique) == 1 else "mix"


def classify_family(question: str, answer: str) -> Optional[str]:
    op = final_op(answer)
    if op is None or op == "mix":
        return None
    if op == "+" and TOTAL_RE.search(question):
        return "compose_total"
    if op == "-" and (REMAIN_RE.search(question) or COMPARE_RE.search(question)):
        return "compose_difference"
    if op == "*" and RATE_RE.search(question):
        return "rate_scale"
    if op == "/" and PARTITION_RE.search(question):
        return "partition_inverse"
    if op in {"*", "/"} and UNIT_RE.search(question):
        return "unit_convert"
    return None


def load_rows(data_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        path = data_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                payload = json.loads(line)
                payload["split"] = split
                payload["index"] = index
                rows.append(payload)
    return rows


def greedy_cover(family_counts: Dict[str, int], total_rows: int) -> List[Dict[str, Any]]:
    ordered = sorted(family_counts.items(), key=lambda item: item[1], reverse=True)
    cumulative = 0
    result = []
    for rank, (family_name, count) in enumerate(ordered, start=1):
        cumulative += count
        result.append(
            {
                "rank": rank,
                "family_name": family_name,
                "count": count,
                "rate": count / total_rows if total_rows else 0.0,
                "cumulative_count": cumulative,
                "cumulative_rate": cumulative / total_rows if total_rows else 0.0,
            }
        )
    return result


def thresholds(cover: Sequence[Dict[str, Any]], goals: Sequence[float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for goal in goals:
        label = f"{int(goal * 100)}pct"
        picked = next((item for item in cover if item["cumulative_rate"] >= goal), None)
        result[label] = picked
    return result


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    rows = load_rows(data_dir)
    counts: Dict[str, int] = {name: 0 for name in FAMILY_SPECS}
    examples: Dict[str, List[Dict[str, Any]]] = {name: [] for name in FAMILY_SPECS}

    for row in rows:
        family_name = classify_family(row["question"], row["answer"])
        if family_name is None:
            continue
        counts[family_name] += 1
        if len(examples[family_name]) < 5:
            examples[family_name].append(
                {
                    "split": row["split"],
                    "index": row["index"],
                    "question": row["question"],
                }
            )

    nonzero_counts = {name: count for name, count in counts.items() if count > 0}
    total_rows = len(rows)
    cover = greedy_cover(nonzero_counts, total_rows)
    summary = {
        "num_examples": total_rows,
        "family_counts": nonzero_counts,
        "family_rates": {name: count / total_rows for name, count in nonzero_counts.items()},
        "greedy_cover": cover,
        "threshold_recommendations": thresholds(cover, goals=(0.20, 0.25, 0.30)),
        "family_specs": {
            name: {
                **spec.to_jsonable(),
                "coverage_count": counts[name],
                "coverage_rate": counts[name] / total_rows if total_rows else 0.0,
                "examples": examples[name],
            }
            for name, spec in FAMILY_SPECS.items()
        },
        "notes": {
            "method": "Classify each GSM8K problem by final worked operator and question cues, then greedily rank proposed ARC-style numeric families by coverage.",
            "scope": "These are proposed extension families, not exact fits to the current synthetic templates.",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
