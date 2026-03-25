from __future__ import annotations

"""Benchmark epiplex vs BPE on ARC trajectory JSON records."""

import argparse
from collections import Counter, defaultdict
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.dont_write_bytecode = True

EPI_TOKEN_ROOT = Path(__file__).resolve().parents[3] / "research" / "exact_learning" / "epi_token"
if str(EPI_TOKEN_ROOT) not in sys.path:
    sys.path.insert(0, str(EPI_TOKEN_ROOT))

import epiplex_tokenizer_benchmark as bench

try:
    from .stage1_latent_sampler import sample_latent_rule
    from .stage2_episode_sampler import sample_episode
    from .stage4_trajectory_dataset import build_trajectories
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import sample_latent_rule  # type: ignore
    from stage2_episode_sampler import sample_episode  # type: ignore
    from stage4_trajectory_dataset import build_trajectories  # type: ignore


ARC_TRACE_STEPS: Tuple[str, ...] = (
    "segment",
    "select",
    "pick_source",
    "pick_target",
    "group",
    "reduce",
    "read_cue",
    "branch",
    "bind",
    "match",
    "transform",
    "relate",
    "act",
    "apply",
    "render",
    "extract_entities",
    "bind_quantities",
    "choose_operation",
    "compute_answer",
    "emit_ir",
)

KEY_ALIASES: Dict[str, str] = {
    "trajectory_id": "^tid",
    "split": "^sp",
    "family": "^fm",
    "difficulty": "^df",
    "source_modality": "^sm",
    "concept_tags": "^tg",
    "trace_template": "^tt",
    "role_bindings": "^rb",
    "episode_metadata": "^em",
    "shortcut_checks": "^sc",
    "example": "^ex",
    "input_state": "^is",
    "output_state": "^os",
    "workspace_state": "^ws",
    "steps": "^st",
    "total_reward": "^tr",
    "total_possible_reward": "^tp",
    "input_grid": "^ig",
    "output_grid": "^og",
    "workspace_grid": "^wg",
    "example_id": "^eid",
    "input_scene": "^ins",
    "selected_object_ids": "^so",
    "notes": "^nt",
    "metadata": "^md",
    "action": "^ac",
    "reward": "^rw",
    "reward_terms": "^rt",
    "cumulative_reward": "^cr",
    "progress": "^pg",
    "done": "^dn",
    "index": "^ix",
    "name": "^nm",
    "description": "^ds",
    "height": "^h",
    "width": "^w",
    "background_color": "^bg",
    "border_color": "^bc",
    "outline_color": "^oc",
    "marker_position": "^mp",
    "objects": "^ob",
    "object_id": "^oid",
    "shape": "^sh",
    "color": "^cl",
    "top": "^tp0",
    "left": "^lf",
    "mass": "^ms",
    "orientation": "^or",
    "holes": "^ho",
    "is_container": "^ic",
    "tags": "^tags",
    "attributes": "^at",
    "legend": "^lg",
    "branch_schedule": "^bs",
    "cue_trigger": "^ct",
    "cue_kind": "^ck",
    "color_vocab": "^cv",
    "shape_vocab": "^sv",
    "diversity_token": "^dv",
    "branch": "^br",
    "then_action": "^ta",
    "else_action": "^ea",
    "changed_cell_fraction": "^cf",
    "target_grid_shape": "^gs",
    "object_ids": "^oids",
}
INVERSE_KEY_ALIASES: Dict[str, str] = {value: key for key, value in KEY_ALIASES.items()}
BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def is_grid_sequence(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    width = len(value[0])
    if width == 0:
        return False
    for row in value:
        if len(row) != width or not all(isinstance(cell, int) and 0 <= cell < len(BASE36_ALPHABET) for cell in row):
            return False
    return True


def encode_cell(value: int) -> str:
    return BASE36_ALPHABET[value]


def decode_cell(ch: str) -> int:
    return BASE36_ALPHABET.index(ch)


def encode_plain_string(value: str) -> str:
    return f"~{value}" if value.startswith("~") else value


def decode_plain_string(value: str) -> str:
    return value[1:] if value.startswith("~~") else value


def escape_list_item(value: str) -> str:
    return value.replace("~", "~~").replace("|", "~p")


def unescape_list_item(value: str) -> str:
    out: List[str] = []
    idx = 0
    while idx < len(value):
        ch = value[idx]
        if ch != "~":
            out.append(ch)
            idx += 1
            continue
        if idx + 1 >= len(value):
            out.append("~")
            idx += 1
            continue
        nxt = value[idx + 1]
        if nxt == "~":
            out.append("~")
        elif nxt == "p":
            out.append("|")
        else:
            out.append(nxt)
        idx += 2
    return "".join(out)


def encode_grid_string(grid: List[List[int]]) -> str:
    rows = ["".join(encode_cell(cell) for cell in row) for row in grid]
    return "~g:" + "/".join(rows)


def decode_grid_string(value: str) -> List[List[int]]:
    payload = value[len("~g:") :]
    if not payload:
        return []
    return [[decode_cell(ch) for ch in row] for row in payload.split("/")]


def compact_arc_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            KEY_ALIASES.get(key, key): compact_arc_value(item)
            for key, item in value.items()
        }
    if is_grid_sequence(value):
        return encode_grid_string(value)
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            return "~l:" + "|".join(escape_list_item(item) for item in value)
        if value and all(isinstance(item, int) for item in value):
            return "~n:" + ",".join(str(item) for item in value)
        return [compact_arc_value(item) for item in value]
    if isinstance(value, str):
        return encode_plain_string(value)
    return value


def expand_arc_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            INVERSE_KEY_ALIASES.get(key, key): expand_arc_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_arc_value(item) for item in value]
    if isinstance(value, str):
        if value.startswith("~g:"):
            return decode_grid_string(value)
        if value.startswith("~l:"):
            payload = value[len("~l:") :]
            if not payload:
                return []
            return [unescape_list_item(item) for item in payload.split("|")]
        if value.startswith("~n:"):
            payload = value[len("~n:") :]
            if not payload:
                return []
            return [int(item) for item in payload.split(",")]
        return decode_plain_string(value)
    return value


def serialize_record_text(record: Dict[str, Any], text_format: str) -> str:
    if text_format == "raw_json":
        return json.dumps(record)
    if text_format == "arc_text_v1":
        compact = compact_arc_value(record)
        return json.dumps(compact, separators=(",", ":"), sort_keys=True)
    raise ValueError(f"Unsupported text format: {text_format}")


def deserialize_record_text(text: str, text_format: str) -> Dict[str, Any]:
    payload = json.loads(text)
    if text_format == "raw_json":
        return payload
    if text_format == "arc_text_v1":
        expanded = expand_arc_value(payload)
        if not isinstance(expanded, dict):
            raise ValueError("Expanded ARC text payload must be a JSON object.")
        return expanded
    raise ValueError(f"Unsupported text format: {text_format}")


def canonicalize_json_payload(value: Dict[str, Any]) -> Dict[str, Any]:
    # `to_jsonable()` occasionally leaves integer dict keys in Python objects.
    # Any textual JSON encoding coerces those keys to strings, so verification
    # should compare against JSON-canonicalized payloads rather than raw objects.
    canonical = json.loads(json.dumps(value))
    if not isinstance(canonical, dict):
        raise ValueError("Canonical ARC payload must be a JSON object.")
    return canonical


class ArcTrajectoryTokenizer(bench.GreedyBPETokenizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        sp = re.escape(bench.SPACE_SYM)
        self._grid_chars_re = re.compile(rf"^[\[\],\d{sp}]+$")

    def _grid_fragment_stats(self, piece: str) -> tuple[str | None, int, bool]:
        if "," not in piece or self._grid_chars_re.fullmatch(piece) is None:
            return None, 0, False

        cleaned = piece.replace("[", "").replace("]", "").replace(bench.SPACE_SYM, "")
        parts = cleaned.split(",")
        numeric_parts = [part for part in parts if part]
        if len(numeric_parts) < 2 or any(not part.isdigit() for part in numeric_parts):
            return None, 0, False
        all_zero = all(part == "0" for part in numeric_parts)
        if "],"+bench.SPACE_SYM+"[" in piece:
            return "bridge", len(numeric_parts), all_zero
        return "list", len(numeric_parts), all_zero

    def _piece_signal_score(self, piece: str) -> float:
        score = super()._piece_signal_score(piece)
        if not self.use_discriminative_scoring:
            return score

        fragment_kind, item_count, all_zero = self._grid_fragment_stats(piece)
        if fragment_kind is None:
            return score

        growth_bonus = 0.55 * math.log2(item_count + 1.0)
        if fragment_kind == "list":
            score += growth_bonus
            if piece.startswith("[") or piece.endswith("]"):
                score += 0.35 * math.log2(item_count + 1.0)
            if all_zero:
                score += 0.45 * math.log2(item_count + 1.0)
        else:
            score += 0.15 * math.log2(item_count + 1.0)
        return score

    def _boundary_penalty(self, left_piece: str, right_piece: str, merged: str) -> float:
        penalty = super()._boundary_penalty(left_piece, right_piece, merged)
        if not self.use_discriminative_scoring:
            return penalty

        grid_fragment_kind, _, _ = self._grid_fragment_stats(merged)
        if grid_fragment_kind is not None:
            # In ARC grids, comma-space-separated numeric lists are stable units
            # rather than harmful cross-boundary blends.
            internal_spaces = merged[1:].count(bench.SPACE_SYM)
            penalty = max(0.0, penalty - 2.25 * internal_spaces)
            if grid_fragment_kind == "bridge":
                penalty += 0.25
            if merged.count("[") + merged.count("]") >= 2 and grid_fragment_kind == "list":
                penalty = max(0.0, penalty - 0.35)
        return penalty


class ArcRowRunsTokenizer(bench.GreedyBPETokenizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._row_run_header_re = re.compile(r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|$")
        self._row_run_re = re.compile(r"^[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+$")
        self._row_run_sequence_re = re.compile(
            r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+(?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+)*)?#$"
        )
        self._row_run_prefix_re = re.compile(
            r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+(?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+)*)?$"
        )
        self._row_run_fragment_re = re.compile(r"^[0-9a-z@|.;#]+$")

    def _row_run_fragment_stats(self, piece: str) -> tuple[str | None, int]:
        if self._row_run_header_re.fullmatch(piece) is not None:
            return "header", 0
        if self._row_run_re.fullmatch(piece) is not None:
            return "run", 1
        if self._row_run_sequence_re.fullmatch(piece) is not None:
            payload = piece.split("|", 1)[1].rstrip("#")
            run_count = 0 if not payload else payload.count(";") + 1
            return "sequence", run_count
        if self._row_run_prefix_re.fullmatch(piece) is not None:
            payload = piece.split("|", 1)[1]
            run_count = 0 if not payload else payload.count(";") + 1
            return "prefix", run_count
        if self._row_run_fragment_re.fullmatch(piece) is not None and any(ch in piece for ch in ".;@|#"):
            return "fragment", piece.count(";") + piece.count(".")
        return None, 0

    def _piece_signal_score(self, piece: str) -> float:
        score = super()._piece_signal_score(piece)
        if not self.use_discriminative_scoring:
            return score

        fragment_kind, fragment_count = self._row_run_fragment_stats(piece)
        if fragment_kind is None:
            return score
        if fragment_kind == "header":
            score += 2.2
        elif fragment_kind == "run":
            score += 2.0
        elif fragment_kind == "sequence":
            score += 0.55 * math.log2(fragment_count + 2.0) + 0.15 * fragment_count
        elif fragment_kind == "prefix":
            score += 0.35 * math.log2(fragment_count + 2.0)
        else:
            score += 0.08 * fragment_count
        if piece.endswith("#"):
            score += 0.35
        return score

    def _boundary_penalty(self, left_piece: str, right_piece: str, merged: str) -> float:
        penalty = super()._boundary_penalty(left_piece, right_piece, merged)
        if not self.use_discriminative_scoring:
            return penalty

        fragment_kind, fragment_count = self._row_run_fragment_stats(merged)
        if fragment_kind is None:
            return penalty

        discount = 0.0
        if fragment_kind == "header":
            discount = 1.35
        elif fragment_kind == "run":
            discount = 1.6
        elif fragment_kind == "sequence":
            discount = 1.6 + 0.12 * min(fragment_count, 8)
        elif fragment_kind == "prefix":
            discount = 1.1 + 0.08 * min(fragment_count, 8)
        else:
            discount = 0.6
        discount += 0.08 * merged.count(".") + 0.06 * merged.count(";")
        if merged.endswith("#"):
            discount += 0.2
        return max(0.0, penalty - discount)


def baseline_arc_trajectory_signals() -> List[bench.RegexSignal]:
    sp = re.escape(bench.SPACE_SYM)
    trace_steps = "|".join(re.escape(step) for step in ARC_TRACE_STEPS)

    return [
        bench.RegexSignal("json_key", rf"^{sp}?\"[a-z_]+\":(?:{sp})?$", 1.6),
        bench.RegexSignal("trace_step", rf"^{sp}?\"(?:{trace_steps})\"$", 2.0),
        bench.RegexSignal("object_id", rf"^{sp}?\"obj_\d+\"$", 1.8),
        bench.RegexSignal("quoted_identifier", rf"^{sp}?\"[a-z]+(?:_[a-z0-9]+)+\"$", 1.2),
        bench.RegexSignal("json_keyword", rf"^{sp}?(?:true|false|null)$", 1.1),
        bench.RegexSignal("json_number", rf"^{sp}?-?\d+(?:\.\d+)?$", 1.0),
        bench.RegexSignal("grid_row", rf"^\[(?:\d+(?:,{sp}\d+)+)\]$", 2.1),
        bench.RegexSignal("short_enum", rf"^{sp}?\"[a-z]+(?:_[a-z]+)*\"$", 0.8),
    ]


def arc_text_v1_signals() -> List[bench.RegexSignal]:
    trace_steps = "|".join(re.escape(step) for step in ARC_TRACE_STEPS)
    return [
        bench.RegexSignal("compact_key", r'^"\^[a-z0-9]+":$', 1.4),
        bench.RegexSignal("grid_string", r"^~g:[0-9a-z]+(?:/[0-9a-z]+)+$", 2.6),
        bench.RegexSignal("grid_zero_row", r"^(?:0{4,})(?:/0{4,})*$", 2.1),
        bench.RegexSignal("trace_step", rf"^(?:{trace_steps})$", 1.8),
        bench.RegexSignal("object_id", r"^obj_\d+$", 1.5),
        bench.RegexSignal("identifier", r"^[a-z]+(?:_[a-z0-9]+)+$", 1.0),
        bench.RegexSignal("packed_list", r"^~l:[^\"\n]+$", 0.8),
        bench.RegexSignal("packed_ints", r"^~n:-?\d+(?:,-?\d+)+$", 0.8),
    ]


def arc_text_v1_row_runs_signals() -> List[bench.RegexSignal]:
    return arc_text_v1_signals() + [
        bench.RegexSignal("row_runs_header", r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|$", 2.8),
        bench.RegexSignal("row_runs_header_fragment", r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+$", 1.4),
        bench.RegexSignal("row_run", r"^[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+$", 2.5),
        bench.RegexSignal("row_run_prefix", r"^[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+$", 1.1),
        bench.RegexSignal(
            "row_runs_sequence",
            r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+(?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+)*)?#$",
            3.2,
        ),
        bench.RegexSignal("row_runs_end", r"^#$", 0.9),
    ]


def arc_text_v1_row_runs_strict_signals() -> List[bench.RegexSignal]:
    return arc_text_v1_signals() + [
        bench.RegexSignal("row_runs_header", r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|$", 2.8),
        bench.RegexSignal("row_runs_header_fragment", r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+$", 1.2),
        bench.RegexSignal("row_run", r"^[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+$", 2.9),
        bench.RegexSignal(
            "row_runs_sequence",
            r"^[0-9a-z]+x[0-9a-z]+@[0-9a-z]+\|(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+(?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+)*)?#$",
            3.4,
        ),
        bench.RegexSignal("row_runs_end", r"^#$", 1.1),
    ]


def row_aware_arc_trajectory_signals() -> List[bench.RegexSignal]:
    sp = re.escape(bench.SPACE_SYM)
    trace_steps = "|".join(re.escape(step) for step in ARC_TRACE_STEPS)

    return [
        bench.RegexSignal("json_key", rf"^{sp}?\"[a-z_]+\":(?:{sp})?$", 1.6),
        bench.RegexSignal("trace_step", rf"^{sp}?\"(?:{trace_steps})\"$", 2.0),
        bench.RegexSignal("object_id", rf"^{sp}?\"obj_\d+\"$", 1.8),
        bench.RegexSignal("quoted_identifier", rf"^{sp}?\"[a-z]+(?:_[a-z0-9]+)+\"$", 1.2),
        bench.RegexSignal("json_keyword", rf"^{sp}?(?:true|false|null)$", 1.1),
        bench.RegexSignal("json_number", rf"^{sp}?-?\d+(?:\.\d+)?$", 1.0),
        bench.RegexSignal("grid_row", rf"^\[(?:\d+(?:,{sp}\d+)+)\]$", 2.1),
        bench.RegexSignal("grid_list_fragment", rf"^(?:\[\[?|\[)?\d+(?:,{sp}\d+){1,},?$", 2.2),
        bench.RegexSignal("grid_list_closed", rf"^(?:\[\[?|\[)?\d+(?:,{sp}\d+){1,}\]$", 2.4),
        bench.RegexSignal("grid_zero_run", rf"^(?:\[\[?|\[)?0(?:,{sp}0){2,},?$", 2.9),
        bench.RegexSignal("grid_row_prefix", rf"^(?:\[\[?|\[)\d+(?:,{sp}\d+){1,},?$", 1.7),
        bench.RegexSignal("grid_row_bridge", rf"^(?:\d+(?:,{sp}\d+)*)?\],{sp}\[(?:\d+(?:,{sp}\d+)*)?$", 0.6),
        bench.RegexSignal("short_enum", rf"^{sp}?\"[a-z]+(?:_[a-z]+)*\"$", 0.8),
    ]


def tokenizer_base_metrics(tokenizer: bench.GreedyBPETokenizer, texts: Sequence[str]) -> Dict[str, Any]:
    encoded_lengths = [len(tokenizer.encode(text, add_bos=False, add_eos=False)) for text in texts[:500]]
    raw_lengths = [len(text) for text in texts[:500]]
    return {
        "requested_vocab_size": tokenizer.vocab_size,
        "vocab_size": len(tokenizer.vocab),
        "vocab_fill_ratio": len(tokenizer.vocab) / max(1, tokenizer.vocab_size),
        "avg_tokens_per_example": sum(encoded_lengths) / max(1, len(encoded_lengths)),
        "avg_chars_per_example": sum(raw_lengths) / max(1, len(raw_lengths)),
        "avg_chars_per_token": sum(raw_lengths) / max(1, sum(encoded_lengths)),
    }


def arc_intrinsic_metrics(tokenizer: bench.GreedyBPETokenizer, texts: Sequence[str]) -> Dict[str, Any]:
    out = tokenizer_base_metrics(tokenizer, texts)
    sp = re.escape(bench.SPACE_SYM)
    trace_steps = "|".join(re.escape(step) for step in ARC_TRACE_STEPS)

    out["json_keys"] = bench.regex_atomicity(tokenizer, texts, rf"{sp}?\"[a-z_]+\":")
    out["trace_steps"] = bench.regex_atomicity(tokenizer, texts, rf"{sp}?\"(?:{trace_steps})\"")
    out["object_ids"] = bench.regex_atomicity(tokenizer, texts, rf"{sp}?\"obj_\d+\"")
    out["quoted_identifiers"] = bench.regex_atomicity(tokenizer, texts, rf"{sp}?\"[a-z]+(?:_[a-z0-9]+)+\"")
    out["numbers"] = bench.regex_atomicity(tokenizer, texts, rf"{sp}?-?\d+(?:\.\d+)?")
    out["json_keywords"] = bench.regex_atomicity(tokenizer, texts, rf"{sp}?(?:true|false|null)")
    out["grid_rows"] = bench.regex_atomicity(tokenizer, texts, rf"\[(?:\d+(?:,{sp}\d+)+)\]")
    return out


def arc_intrinsic_metrics_v1(tokenizer: bench.GreedyBPETokenizer, texts: Sequence[str]) -> Dict[str, Any]:
    out = tokenizer_base_metrics(tokenizer, texts)
    trace_steps = "|".join(re.escape(step) for step in ARC_TRACE_STEPS)

    out["json_keys"] = bench.regex_atomicity(tokenizer, texts, r'"\^[a-z0-9]+":')
    out["trace_steps"] = bench.regex_atomicity(tokenizer, texts, rf"(?:{trace_steps})")
    out["object_ids"] = bench.regex_atomicity(tokenizer, texts, r"obj_\d+")
    out["quoted_identifiers"] = bench.regex_atomicity(tokenizer, texts, r"[a-z]+(?:_[a-z0-9]+)+")
    out["numbers"] = bench.regex_atomicity(tokenizer, texts, r"-?\d+(?:\.\d+)?")
    out["json_keywords"] = bench.regex_atomicity(tokenizer, texts, r"(?:true|false|null)")
    out["grid_rows"] = bench.regex_atomicity(tokenizer, texts, r"[0-9a-z]{4,}(?:/[0-9a-z]{4,})+")
    return out


def build_arc_tokenizer(
    kind: str,
    *,
    epiplex_profile: str,
    text_format: str,
    target_format: str | None = None,
    vocab_size: int,
    candidate_pool_size: int,
    max_piece_chars: int,
    fit_workers: int,
    verbose: bool,
) -> bench.GreedyBPETokenizer:
    if kind == "bpe":
        return bench.GreedyBPETokenizer(
            vocab_size=vocab_size,
            task="arc_trajectory",
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            use_discriminative_scoring=False,
            num_workers=fit_workers,
            verbose=verbose,
        )

    if text_format == "arc_text_v1":
        if epiplex_profile in {"row_runs_signal", "row_runs_struct", "row_runs_strict"} and target_format != "row_runs_v1":
            raise ValueError(f"{epiplex_profile} requires target_format='row_runs_v1'.")
        tokenizer_kwargs = {
            "vocab_size": vocab_size,
            "task": "arc_trajectory_v1",
            "candidate_pool_size": candidate_pool_size,
            "max_piece_chars": max_piece_chars,
            "use_discriminative_scoring": True,
            "num_workers": fit_workers,
            "verbose": verbose,
        }
        if epiplex_profile == "adaptive":
            tokenizer_kwargs.update(
                alpha_compression=1.2,
                beta_signal=1.0,
                gamma_stability=0.25,
                delta_boundary=0.8,
            )
        tokenizer_cls: type[bench.GreedyBPETokenizer] = bench.GreedyBPETokenizer
        signals = arc_text_v1_signals()
        if target_format == "row_runs_v1" and epiplex_profile == "row_runs_signal":
            signals = arc_text_v1_row_runs_signals()
        elif target_format == "row_runs_v1" and epiplex_profile == "row_runs_strict":
            signals = arc_text_v1_row_runs_strict_signals()
        elif target_format == "row_runs_v1" and epiplex_profile == "row_runs_struct":
            tokenizer_cls = ArcRowRunsTokenizer
            tokenizer_kwargs.update(
                alpha_compression=1.2,
                beta_signal=1.2,
                gamma_stability=0.3,
                delta_boundary=0.6,
            )
            signals = arc_text_v1_row_runs_signals()
        tokenizer = tokenizer_cls(**tokenizer_kwargs)
        tokenizer.signals = signals
        tokenizer._compiled_signals = [(signal, signal.compile()) for signal in tokenizer.signals]
        return tokenizer

    if epiplex_profile == "baseline":
        tokenizer: bench.GreedyBPETokenizer = bench.GreedyBPETokenizer(
            vocab_size=vocab_size,
            task="arc_trajectory",
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            use_discriminative_scoring=True,
            num_workers=fit_workers,
            verbose=verbose,
        )
        tokenizer.signals = baseline_arc_trajectory_signals()
    elif epiplex_profile == "row_aware":
        tokenizer = ArcTrajectoryTokenizer(
            vocab_size=vocab_size,
            task="arc_trajectory",
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            use_discriminative_scoring=True,
            num_workers=fit_workers,
            verbose=verbose,
        )
        tokenizer.signals = row_aware_arc_trajectory_signals()
    elif epiplex_profile == "adaptive":
        tokenizer = ArcTrajectoryTokenizer(
            vocab_size=vocab_size,
            task="arc_trajectory",
            candidate_pool_size=candidate_pool_size,
            max_piece_chars=max_piece_chars,
            use_discriminative_scoring=True,
            alpha_compression=1.35,
            beta_signal=1.1,
            gamma_stability=0.25,
            delta_boundary=0.75,
            num_workers=fit_workers,
            verbose=verbose,
        )
        tokenizer.signals = row_aware_arc_trajectory_signals()
    else:
        raise ValueError(f"Unsupported epiplex profile: {epiplex_profile}")

    tokenizer._compiled_signals = [(signal, signal.compile()) for signal in tokenizer.signals]
    return tokenizer


def sample_trajectory_texts(
    *,
    num_episodes: int,
    seed_start: int,
    num_train: int,
    max_attempts: int,
    include_test: bool,
    text_format: str,
    verify_roundtrip: bool,
) -> Tuple[List[str], Dict[str, int]]:
    texts: List[str] = []
    family_counts: Dict[str, int] = {}

    for offset in range(num_episodes):
        seed = seed_start + offset
        latent = sample_latent_rule(seed=seed)
        episode = sample_episode(
            latent,
            seed=seed,
            num_train=num_train,
            max_attempts=max_attempts,
        )
        records = build_trajectories(episode, include_test=include_test)
        family_counts[latent.family.value] = family_counts.get(latent.family.value, 0) + len(records)
        for record in records:
            payload = record.to_jsonable()
            text = serialize_record_text(payload, text_format)
            if verify_roundtrip:
                recovered = deserialize_record_text(text, text_format)
                if recovered != canonicalize_json_payload(payload):
                    raise AssertionError("ARC text preprocessing must round-trip exactly.")
            texts.append(text)

    return texts, family_counts


def build_arc_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    if "bpe" not in results or "epiplex" not in results:
        return {}

    bpe_intrinsic = results["bpe"]["intrinsic"]
    epiplex_intrinsic = results["epiplex"]["intrinsic"]
    comparison: Dict[str, Any] = {
        "baseline": "bpe",
        "challenger": "epiplex",
        "tokenizer": {},
        "intrinsic": {},
        "notes": [
            "Token-level perplexity is tokenizer-dependent; compare bits-per-char for the fair LM comparison.",
        ],
    }

    for key, higher_is_better in [
        ("vocab_size", True),
        ("vocab_fill_ratio", True),
        ("avg_chars_per_token", True),
        ("avg_tokens_per_example", False),
    ]:
        metric = bench.compare_scalar_metrics(
            bpe_intrinsic.get(key),
            epiplex_intrinsic.get(key),
            higher_is_better=higher_is_better,
        )
        if metric is not None:
            comparison["tokenizer" if key in {"vocab_size", "vocab_fill_ratio"} else "intrinsic"][key] = metric

    for span_name in [
        "json_keys",
        "trace_steps",
        "object_ids",
        "quoted_identifiers",
        "numbers",
        "json_keywords",
        "grid_rows",
    ]:
        for metric_name, higher_is_better in [
            ("single_token_rate", True),
            ("avg_tokens_per_span", False),
        ]:
            metric = bench.compare_scalar_metrics(
                bpe_intrinsic.get(span_name, {}).get(metric_name),
                epiplex_intrinsic.get(span_name, {}).get(metric_name),
                higher_is_better=higher_is_better,
            )
            if metric is not None:
                comparison["intrinsic"][f"{span_name}_{metric_name}"] = metric

    bpe_lm = results["bpe"].get("language_model_characterization", {})
    epiplex_lm = results["epiplex"].get("language_model_characterization", {})
    if bpe_lm and epiplex_lm:
        comparison["language_model"] = {}
        for key in [
            "best_val_loss",
            "best_val_ppl",
            "best_val_nats_per_char",
            "best_val_bits_per_char",
        ]:
            metric = bench.compare_scalar_metrics(
                bpe_lm.get(key),
                epiplex_lm.get(key),
                higher_is_better=False,
            )
            if metric is not None:
                comparison["language_model"][key] = metric

        if "best_val_bits_per_char" in comparison["language_model"]:
            comparison["overall_winner"] = comparison["language_model"]["best_val_bits_per_char"]["winner"]
        elif "best_val_loss" in comparison["language_model"]:
            comparison["overall_winner"] = comparison["language_model"]["best_val_loss"]["winner"]

    return comparison


def evaluate_bigram_nll(
    counts_by_prev: Dict[int, Counter[int]],
    totals_by_prev: Counter[int],
    token_ids: Sequence[int],
    *,
    vocab_size: int,
    smoothing_alpha: float,
) -> float:
    total_nll = 0.0
    total_pairs = 0
    for prev_id, next_id in zip(token_ids, token_ids[1:]):
        pair_count = counts_by_prev[prev_id][next_id]
        context_count = totals_by_prev[prev_id]
        prob = (pair_count + smoothing_alpha) / (context_count + smoothing_alpha * vocab_size)
        total_nll += -math.log(prob)
        total_pairs += 1
    return total_nll / max(1, total_pairs)


def fit_bigram_language_model(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    *,
    vocab_size: int,
    smoothing_alpha: float,
) -> Dict[str, float]:
    counts_by_prev: Dict[int, Counter[int]] = defaultdict(Counter)
    totals_by_prev: Counter[int] = Counter()
    for prev_id, next_id in zip(train_ids, train_ids[1:]):
        counts_by_prev[prev_id][next_id] += 1
        totals_by_prev[prev_id] += 1

    train_loss = evaluate_bigram_nll(
        counts_by_prev,
        totals_by_prev,
        train_ids,
        vocab_size=vocab_size,
        smoothing_alpha=smoothing_alpha,
    )
    val_loss = evaluate_bigram_nll(
        counts_by_prev,
        totals_by_prev,
        val_ids,
        vocab_size=vocab_size,
        smoothing_alpha=smoothing_alpha,
    )
    return {
        "backend": "smoothed_bigram",
        "smoothing_alpha": float(smoothing_alpha),
        "train_loss": float(train_loss),
        "best_val_loss": float(val_loss),
        "best_val_ppl": float(math.exp(min(val_loss, 20.0))),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark epiplex vs BPE on ARC trajectory JSON records.")
    parser.add_argument("--output-dir", type=str, default="results/arc_tokenizer_benchmark")
    parser.add_argument("--tokenizers", nargs="+", default=["bpe", "epiplex"])
    parser.add_argument(
        "--epiplex-profile",
        choices=["baseline", "row_aware", "adaptive", "row_runs_signal", "row_runs_struct", "row_runs_strict"],
        default="baseline",
    )
    parser.add_argument("--text-format", choices=["raw_json", "arc_text_v1"], default="raw_json")
    parser.add_argument("--verify-roundtrip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--num-train", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--include-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--candidate-pool-size", type=int, default=2048)
    parser.add_argument("--max-piece-chars", type=int, default=48)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bigram-smoothing", type=float, default=0.1)
    parser.add_argument("--fit-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bench.set_seed(args.seed)

    train_texts, train_family_counts = sample_trajectory_texts(
        num_episodes=args.train_episodes,
        seed_start=args.seed,
        num_train=args.num_train,
        max_attempts=args.max_attempts,
        include_test=args.include_test,
        text_format=args.text_format,
        verify_roundtrip=args.verify_roundtrip,
    )
    val_texts, val_family_counts = sample_trajectory_texts(
        num_episodes=args.val_episodes,
        seed_start=args.seed + args.train_episodes,
        num_train=args.num_train,
        max_attempts=args.max_attempts,
        include_test=args.include_test,
        text_format=args.text_format,
        verify_roundtrip=args.verify_roundtrip,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "task": "arc_trajectory",
        "seed": args.seed,
        "epiplex_profile": args.epiplex_profile,
        "text_format": args.text_format,
        "verify_roundtrip": args.verify_roundtrip,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "train_records": len(train_texts),
        "val_records": len(val_texts),
        "include_test": args.include_test,
        "train_family_counts": train_family_counts,
        "val_family_counts": val_family_counts,
        "results": {},
    }

    for kind in args.tokenizers:
        tokenizer = build_arc_tokenizer(
            kind,
            epiplex_profile=args.epiplex_profile,
            text_format=args.text_format,
            vocab_size=args.vocab_size,
            candidate_pool_size=args.candidate_pool_size,
            max_piece_chars=args.max_piece_chars,
            fit_workers=args.fit_workers,
            verbose=args.verbose,
        )

        print(f"\n[tokenizer] fitting {kind}")
        tokenizer.fit(train_texts)
        tokenizer_path = out_dir / f"arc_trajectory_{kind}_tokenizer.json"
        tokenizer.save(tokenizer_path)

        metrics: Dict[str, Any] = {
            "intrinsic": (
                arc_intrinsic_metrics_v1(tokenizer, val_texts)
                if args.text_format == "arc_text_v1"
                else arc_intrinsic_metrics(tokenizer, val_texts)
            ),
            "tokenizer_path": str(tokenizer_path),
        }

        if not args.skip_model:
            print(f"[tokenizer] tokenizing corpora for {kind}")
            train_ids = bench.flatten_tokenized_examples(tokenizer, train_texts, add_bos=True, add_eos=True)
            val_ids = bench.flatten_tokenized_examples(tokenizer, val_texts, add_bos=True, add_eos=True)
            if bench.torch is None:
                lm_history = fit_bigram_language_model(
                    train_ids=train_ids,
                    val_ids=val_ids,
                    vocab_size=len(tokenizer.vocab),
                    smoothing_alpha=args.bigram_smoothing,
                )
            else:
                model, lm_history = bench.train_language_model(
                    train_ids=train_ids,
                    val_ids=val_ids,
                    vocab_size=len(tokenizer.vocab),
                    block_size=args.block_size,
                    batch_size=args.batch_size,
                    max_steps=args.max_steps,
                    eval_every=args.eval_every,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    d_model=args.d_model,
                    n_heads=args.n_heads,
                    n_layers=args.n_layers,
                    device=args.device,
                )
            metrics["language_model"] = lm_history
            metrics["language_model_characterization"] = bench.characterize_language_model(
                metrics["intrinsic"],
                lm_history,
            )
            if bench.torch is not None:
                model_path = out_dir / f"arc_trajectory_{kind}_lm.pt"
                bench.torch.save(model.state_dict(), model_path)
                metrics["model_path"] = str(model_path)

        report["results"][kind] = metrics

    report["comparison"] = build_arc_comparison(report["results"])

    report_path = out_dir / "arc_trajectory_benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] report written to {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
