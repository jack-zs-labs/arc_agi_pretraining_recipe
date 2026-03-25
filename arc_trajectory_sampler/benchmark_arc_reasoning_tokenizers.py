from __future__ import annotations

"""Compare BPE vs epiplex on ARC output-grid generation."""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_arc_tokenizers as tok_bench
import epiplex_tokenizer_benchmark as bench

try:
    from .stage1_latent_sampler import sample_latent_rule
    from .stage2_episode_sampler import sample_episode
    from .stage4_trajectory_dataset import build_trajectories, grid_similarity
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import sample_latent_rule  # type: ignore
    from stage2_episode_sampler import sample_episode  # type: ignore
    from stage4_trajectory_dataset import build_trajectories, grid_similarity  # type: ignore


Grid = List[List[int]]

PROMPT_PREFIX = "ARC\nCTX="
PROMPT_SUFFIX = "\nOUT="
TARGET_FORMAT_NATIVE = "native"
TARGET_FORMAT_ROW_RUNS_V1 = "row_runs_v1"
TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1 = "row_runs_payload_v1"
TARGET_FORMAT_ROW_RUNS_ARC_BG_V1 = "row_runs_arc_bg_v1"
ROW_RUNS_END = "#"
DECODE_MODE_GREEDY = "greedy"
DECODE_MODE_ROW_RUNS_CONSTRAINED = "row_runs_constrained"
ARC_COLOR_CHARS = tok_bench.BASE36_ALPHABET[:10]

def build_reasoning_context(record: Dict[str, Any]) -> Dict[str, Any]:
    output_state = record.get("output_state", {})
    trace_targets = output_state.get("trace_targets", {}) if isinstance(output_state, dict) else {}
    return {
        "family": record.get("family"),
        "difficulty": record.get("difficulty"),
        "trace_template": record.get("trace_template"),
        "trace_targets": trace_targets,
        "input_grid": record.get("input_grid"),
    }


def encode_base36_int(value: int) -> str:
    if value < 0:
        raise ValueError("Base36 encoding only supports non-negative integers.")
    if value == 0:
        return tok_bench.BASE36_ALPHABET[0]
    digits: List[str] = []
    base = len(tok_bench.BASE36_ALPHABET)
    while value > 0:
        value, remainder = divmod(value, base)
        digits.append(tok_bench.BASE36_ALPHABET[remainder])
    return "".join(reversed(digits))


def decode_base36_int(text: str) -> int:
    if not text:
        raise ValueError("Base36 payload cannot be empty.")
    base = len(tok_bench.BASE36_ALPHABET)
    value = 0
    for ch in text:
        idx = tok_bench.BASE36_ALPHABET.find(ch)
        if idx < 0:
            raise ValueError(f"Invalid base36 digit: {ch!r}")
        value = value * base + idx
    return value


def dominant_grid_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    if not counts:
        return 0
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def row_runs_from_grid(grid: Grid) -> Tuple[int, List[str]]:
    width = len(grid[0]) if grid else 0
    background = dominant_grid_color(grid)
    runs: List[str] = []
    for row_idx, row in enumerate(grid):
        col_idx = 0
        while col_idx < width:
            value = row[col_idx]
            if value == background:
                col_idx += 1
                continue
            start = col_idx
            col_idx += 1
            while col_idx < width and row[col_idx] == value:
                col_idx += 1
            run_length = col_idx - start
            runs.append(
                ".".join(
                    [
                        encode_base36_int(row_idx),
                        encode_base36_int(start),
                        encode_base36_int(run_length),
                        encode_base36_int(value),
                    ]
                )
            )
    return background, runs


def row_runs_shape_text(height: int, width: int) -> str:
    return f"{encode_base36_int(height)}x{encode_base36_int(width)}"


def row_runs_shape_text_from_grid(grid: Grid) -> str:
    return row_runs_shape_text(len(grid), len(grid[0]) if grid else 0)


def serialize_row_runs_grid(grid: Grid) -> str:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    background, runs = row_runs_from_grid(grid)
    return f"{row_runs_shape_text(height, width)}@{encode_base36_int(background)}|{';'.join(runs)}{ROW_RUNS_END}"


def serialize_row_runs_payload_grid(grid: Grid) -> str:
    background, runs = row_runs_from_grid(grid)
    return f"@{encode_base36_int(background)}|{';'.join(runs)}{ROW_RUNS_END}"


def serialize_row_runs_arc_bg_grid(grid: Grid) -> str:
    background, runs = row_runs_from_grid(grid)
    if background >= len(ARC_COLOR_CHARS):
        raise ValueError("ARC row-runs only supports background colors 0-9.")
    return f"BG={encode_base36_int(background)}\nRUNS={';'.join(runs)}{ROW_RUNS_END}"


def build_grid_from_row_runs(
    height: int,
    width: int,
    background: int,
    body: str,
    *,
    max_color_value: Optional[int] = None,
) -> Optional[Grid]:
    if background >= len(tok_bench.BASE36_ALPHABET) or height <= 0 or width <= 0:
        return None
    if max_color_value is not None and background > max_color_value:
        return None

    grid: Grid = [[background for _ in range(width)] for _ in range(height)]
    if not body:
        return grid

    for run in body.split(";"):
        if not run:
            continue
        parts = run.split(".")
        if len(parts) != 4:
            return None
        row_idx, start, run_length, value = (decode_base36_int(part) for part in parts)
        if row_idx >= height or start >= width or run_length <= 0 or start + run_length > width:
            return None
        if value >= len(tok_bench.BASE36_ALPHABET):
            return None
        if max_color_value is not None and value > max_color_value:
            return None
        for col_idx in range(start, start + run_length):
            grid[row_idx][col_idx] = value
    return grid


def deserialize_row_runs_grid(text: str) -> Optional[Grid]:
    candidate = text.strip()
    if not candidate:
        return None
    if ROW_RUNS_END in candidate:
        candidate = candidate.split(ROW_RUNS_END, 1)[0]
    match = re.fullmatch(r"([0-9a-z]+)x([0-9a-z]+)@([0-9a-z])\|([0-9a-z.;]*)", candidate)
    if match is None:
        return None

    height = decode_base36_int(match.group(1))
    width = decode_base36_int(match.group(2))
    background = decode_base36_int(match.group(3))
    return build_grid_from_row_runs(height, width, background, match.group(4))


def deserialize_row_runs_payload_grid(
    text: str,
    *,
    expected_height: int,
    expected_width: int,
) -> Optional[Grid]:
    candidate = text.strip()
    if not candidate:
        return None
    if ROW_RUNS_END in candidate:
        candidate = candidate.split(ROW_RUNS_END, 1)[0]
    match = re.fullmatch(r"@([0-9a-z])\|([0-9a-z.;]*)", candidate)
    if match is None:
        return None
    background = decode_base36_int(match.group(1))
    return build_grid_from_row_runs(expected_height, expected_width, background, match.group(2))


def deserialize_row_runs_arc_bg_grid(
    text: str,
    *,
    expected_height: int,
    expected_width: int,
) -> Optional[Grid]:
    candidate = text.strip()
    if not candidate:
        return None
    if ROW_RUNS_END in candidate:
        candidate = candidate.split(ROW_RUNS_END, 1)[0]
    match = re.fullmatch(r"BG=([0-9])\nRUNS=([0-9a-z.;]*)", candidate)
    if match is None:
        return None
    background = decode_base36_int(match.group(1))
    return build_grid_from_row_runs(
        expected_height,
        expected_width,
        background,
        match.group(2),
        max_color_value=9,
    )


def serialize_output_grid(grid: Grid, text_format: str, target_format: str) -> str:
    if target_format == TARGET_FORMAT_ROW_RUNS_V1:
        return serialize_row_runs_grid(grid)
    if target_format == TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1:
        return serialize_row_runs_payload_grid(grid)
    if target_format == TARGET_FORMAT_ROW_RUNS_ARC_BG_V1:
        return serialize_row_runs_arc_bg_grid(grid)
    if text_format == "raw_json":
        return json.dumps(grid, separators=(",", ":"))
    if text_format == "arc_text_v1":
        return "/".join("".join(tok_bench.encode_cell(cell) for cell in row) for row in grid)
    raise ValueError(f"Unsupported text format: {text_format}")


def deserialize_output_grid(text: str, text_format: str, target_format: str) -> Optional[Grid]:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        if target_format == TARGET_FORMAT_ROW_RUNS_V1:
            return deserialize_row_runs_grid(candidate)
        if target_format == TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1:
            return None
        if target_format == TARGET_FORMAT_ROW_RUNS_ARC_BG_V1:
            return None
        if text_format == "raw_json":
            payload = json.loads(candidate)
            if tok_bench.is_grid_sequence(payload):
                return payload
            return None
        if text_format == "arc_text_v1":
            if candidate.startswith("~g:"):
                return tok_bench.decode_grid_string(candidate)
            rows = candidate.split("/")
            if not rows or any(not row for row in rows):
                return None
            if any(any(ch not in tok_bench.BASE36_ALPHABET for ch in row) for row in rows):
                return None
            width = len(rows[0])
            if any(len(row) != width for row in rows):
                return None
            return [[tok_bench.decode_cell(ch) for ch in row] for row in rows]
    except Exception:
        return None
    raise ValueError(f"Unsupported text format: {text_format}")


def deserialize_output_grid_for_shape(
    text: str,
    text_format: str,
    target_format: str,
    *,
    expected_height: int,
    expected_width: int,
) -> Optional[Grid]:
    candidate = text.strip()
    if candidate.startswith("OUT="):
        candidate = candidate[len("OUT=") :].strip()

    parsed = deserialize_output_grid(candidate, text_format, target_format)
    if (
        parsed is not None
        and len(parsed) == expected_height
        and (len(parsed[0]) if parsed else 0) == expected_width
    ):
        return parsed

    if target_format == TARGET_FORMAT_ROW_RUNS_V1:
        match = re.search(rf"([0-9a-z]+x[0-9a-z]+@[0-9a-z]\|[0-9a-z.;]*{re.escape(ROW_RUNS_END)})", candidate)
        if match is None:
            return None
        parsed = deserialize_row_runs_grid(match.group(1))
        if (
            parsed is not None
            and len(parsed) == expected_height
            and (len(parsed[0]) if parsed else 0) == expected_width
        ):
            return parsed
        return None
    if target_format == TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1:
        match = re.search(rf"(@[0-9a-z]\|[0-9a-z.;]*{re.escape(ROW_RUNS_END)})", candidate)
        if match is None:
            return None
        parsed = deserialize_row_runs_payload_grid(
            match.group(1),
            expected_height=expected_height,
            expected_width=expected_width,
        )
        if (
            parsed is not None
            and len(parsed) == expected_height
            and (len(parsed[0]) if parsed else 0) == expected_width
        ):
            return parsed
        return None
    if target_format == TARGET_FORMAT_ROW_RUNS_ARC_BG_V1:
        match = re.search(rf"(BG=[0-9]\nRUNS=[0-9a-z.;]*{re.escape(ROW_RUNS_END)})", candidate)
        if match is None:
            return None
        parsed = deserialize_row_runs_arc_bg_grid(
            match.group(1),
            expected_height=expected_height,
            expected_width=expected_width,
        )
        if (
            parsed is not None
            and len(parsed) == expected_height
            and (len(parsed[0]) if parsed else 0) == expected_width
        ):
            return parsed
        return None

    if text_format != "arc_text_v1" or expected_height <= 0 or expected_width <= 0:
        return None

    row_pattern = f"[{re.escape(tok_bench.BASE36_ALPHABET)}]{{{expected_width}}}"
    full_pattern = rf"({row_pattern}(?:/{row_pattern}){{{expected_height - 1}}})"
    match = re.search(full_pattern, candidate)
    if match is None:
        rowlike = re.search(rf"[{re.escape(tok_bench.BASE36_ALPHABET)}/]+", candidate)
        if rowlike is None:
            return None
        rows = [row for row in rowlike.group(0).split("/") if row]
        if not rows:
            return None
        coerced: Grid = []
        for row_idx in range(expected_height):
            raw_row = rows[row_idx] if row_idx < len(rows) else ""
            normalized = raw_row[:expected_width].ljust(expected_width, tok_bench.BASE36_ALPHABET[0])
            coerced.append([tok_bench.decode_cell(ch) for ch in normalized])
        return coerced
    rows = match.group(1).split("/")
    return [[tok_bench.decode_cell(ch) for ch in row] for row in rows]


def reasoning_prompt_text(record: Dict[str, Any], text_format: str, target_format: str) -> str:
    context = build_reasoning_context(record)
    context_text = tok_bench.serialize_record_text(context, text_format)
    prompt = f"{PROMPT_PREFIX}{context_text}"
    if target_format in {TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1, TARGET_FORMAT_ROW_RUNS_ARC_BG_V1}:
        grid = record.get("output_grid")
        if not tok_bench.is_grid_sequence(grid):
            raise ValueError("Payload row-runs prompting requires a 2D integer output grid.")
        prompt += f"\nSHAPE={row_runs_shape_text_from_grid(grid)}"
    return f"{prompt}{PROMPT_SUFFIX}"


def reasoning_target_text(record: Dict[str, Any], text_format: str, target_format: str) -> str:
    grid = record.get("output_grid")
    if not tok_bench.is_grid_sequence(grid):
        raise ValueError("Each reasoning example must include a 2D integer output grid.")
    return serialize_output_grid(grid, text_format, target_format)


def reasoning_training_text(record: Dict[str, Any], text_format: str, target_format: str) -> str:
    return reasoning_prompt_text(record, text_format, target_format) + reasoning_target_text(record, text_format, target_format) + "\n"


def sample_reasoning_examples(
    *,
    num_episodes: int,
    seed_start: int,
    num_train: int,
    max_attempts: int,
    include_test: bool,
    text_format: str,
    verify_roundtrip: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    examples: List[Dict[str, Any]] = []
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
            if verify_roundtrip:
                text = tok_bench.serialize_record_text(payload, text_format)
                recovered = tok_bench.deserialize_record_text(text, text_format)
                if recovered != tok_bench.canonicalize_json_payload(payload):
                    raise AssertionError("Reasoning benchmark serialization must round-trip exactly.")
            examples.append(payload)

    return examples, family_counts


def max_target_token_length(
    tokenizer: bench.GreedyBPETokenizer,
    records: Sequence[Dict[str, Any]],
    text_format: str,
    target_format: str,
) -> int:
    lengths = [
        len(tokenizer.encode(reasoning_target_text(record, text_format, target_format), add_bos=False, add_eos=False))
        for record in records
    ]
    return max(lengths) if lengths else 0


def per_family_fraction(values: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for family, (numerator, denominator) in values.items():
        out[family] = numerator / max(1, denominator)
    return out


def is_base36_fragment(text: str) -> bool:
    return bool(text) and all(ch in tok_bench.BASE36_ALPHABET for ch in text)


def max_base36_digits(value: int) -> int:
    return len(encode_base36_int(max(0, value)))


def row_run_segment_complete(
    segment: str,
    *,
    expected_height: int,
    expected_width: int,
    value_chars: str = tok_bench.BASE36_ALPHABET,
) -> bool:
    parts = segment.split(".")
    if len(parts) != 4 or not all(is_base36_fragment(part) for part in parts):
        return False
    row_idx, start, run_length, value = (decode_base36_int(part) for part in parts)
    return (
        0 <= row_idx < expected_height
        and 0 <= start < expected_width
        and 0 < run_length <= expected_width
        and start + run_length <= expected_width
        and 0 <= value < len(tok_bench.BASE36_ALPHABET)
        and parts[3] in value_chars
    )


def row_run_segment_prefix_ok(
    segment: str,
    *,
    expected_height: int,
    expected_width: int,
    value_chars: str = tok_bench.BASE36_ALPHABET,
) -> bool:
    if segment == "":
        return True
    parts = segment.split(".")
    if len(parts) > 4:
        return False
    field_max_lens = [
        max_base36_digits(expected_height - 1),
        max_base36_digits(expected_width - 1),
        max_base36_digits(expected_width),
        1,
    ]
    completed_parts: List[int] = []
    for field_idx, part in enumerate(parts[:-1]):
        if not is_base36_fragment(part) or len(part) > field_max_lens[field_idx]:
            return False
        value = decode_base36_int(part)
        completed_parts.append(value)
        if field_idx == 0 and value >= expected_height:
            return False
        if field_idx == 1 and value >= expected_width:
            return False
        if field_idx == 2:
            start = completed_parts[1] if len(completed_parts) > 1 else None
            if value <= 0 or value > expected_width:
                return False
            if start is not None and start + value > expected_width:
                return False

    last_idx = len(parts) - 1
    last = parts[-1]
    if last == "":
        return len(parts) <= 4
    if not is_base36_fragment(last) or len(last) > field_max_lens[last_idx]:
        return False

    value = decode_base36_int(last)
    if last_idx == 0:
        return value < expected_height
    if last_idx == 1:
        return value < expected_width
    if last_idx == 2:
        start = completed_parts[1] if len(completed_parts) > 1 else None
        if value > expected_width:
            return False
        if start is not None and value > 0 and start + value > expected_width:
            return False
        return True
    return len(last) == 1 and last in value_chars


def row_runs_body_prefix_ok(
    body: str,
    *,
    expected_height: int,
    expected_width: int,
    value_chars: str = tok_bench.BASE36_ALPHABET,
) -> bool:
    if any(ch not in tok_bench.BASE36_ALPHABET + ".;#" for ch in body):
        return False
    if body.count("#") > 1 or ("#" in body and not body.endswith("#")):
        return False

    completed = body.endswith("#")
    core = body[:-1] if completed else body
    segments = core.split(";")
    for segment in segments[:-1]:
        if segment == "" or not row_run_segment_complete(
            segment,
            expected_height=expected_height,
            expected_width=expected_width,
            value_chars=value_chars,
        ):
            return False
    last = segments[-1] if segments else ""
    if completed:
        return last == "" or row_run_segment_complete(
            last,
            expected_height=expected_height,
            expected_width=expected_width,
            value_chars=value_chars,
        )
    return row_run_segment_prefix_ok(
        last,
        expected_height=expected_height,
        expected_width=expected_width,
        value_chars=value_chars,
    )


def row_runs_prefix_ok(
    text: str,
    *,
    include_shape: bool,
    expected_height: int,
    expected_width: int,
    background_chars: str = tok_bench.BASE36_ALPHABET,
    value_chars: str = tok_bench.BASE36_ALPHABET,
) -> bool:
    if text == "":
        return True
    shape_prefix = row_runs_shape_text(expected_height, expected_width)
    if "|" in text:
        head, body = text.split("|", 1)
        if include_shape:
            if not re.fullmatch(rf"{re.escape(shape_prefix)}@[{re.escape(background_chars)}]", head):
                return False
        else:
            if not re.fullmatch(rf"@[{re.escape(background_chars)}]", head):
                return False
        return row_runs_body_prefix_ok(
            body,
            expected_height=expected_height,
            expected_width=expected_width,
            value_chars=value_chars,
        )

    if include_shape:
        if shape_prefix.startswith(text):
            return True
        if not text.startswith(shape_prefix):
            return False
        suffix = text[len(shape_prefix) :]
        if suffix == "@":
            return True
        return len(suffix) == 2 and suffix.startswith("@") and suffix[1] in background_chars

    if not text.startswith("@"):
        return False
    if text == "@":
        return True
    return len(text) == 2 and text[1] in background_chars


def row_runs_complete(
    text: str,
    *,
    include_shape: bool,
    background_chars: str = tok_bench.BASE36_ALPHABET,
    value_chars: str = tok_bench.BASE36_ALPHABET,
) -> bool:
    background_class = re.escape(background_chars)
    value_class = re.escape(value_chars)
    if include_shape:
        pattern = rf"^[0-9a-z]+x[0-9a-z]+@[{background_class}]\|(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[{value_class}](?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[{value_class}])*)?#$"
    else:
        pattern = rf"^@[{background_class}]\|(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[{value_class}](?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[{value_class}])*)?#$"
    return re.fullmatch(pattern, text) is not None


def row_runs_arc_bg_prefix_ok(
    text: str,
    *,
    expected_height: int,
    expected_width: int,
) -> bool:
    if text == "":
        return True
    if "BG=".startswith(text):
        return True
    if not text.startswith("BG="):
        return False
    suffix = text[len("BG=") :]
    if suffix == "":
        return True
    if suffix[0] not in ARC_COLOR_CHARS:
        return False
    remainder = suffix[1:]
    runs_prefix = "\nRUNS="
    if remainder == "":
        return True
    if runs_prefix.startswith(remainder):
        return True
    if not remainder.startswith(runs_prefix):
        return False
    return row_runs_body_prefix_ok(
        remainder[len(runs_prefix) :],
        expected_height=expected_height,
        expected_width=expected_width,
        value_chars=ARC_COLOR_CHARS,
    )


def row_runs_arc_bg_complete(text: str) -> bool:
    color_class = re.escape(ARC_COLOR_CHARS)
    pattern = rf"^BG=[{color_class}]\nRUNS=(?:[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[{color_class}](?:;[0-9a-z]+\.[0-9a-z]+\.[0-9a-z]+\.[{color_class}])*)?#$"
    return re.fullmatch(pattern, text) is not None


def generate_row_runs_constrained(
    model: bench.TinyCausalLM,
    tokenizer: bench.GreedyBPETokenizer,
    input_ids: List[int],
    *,
    max_new_tokens: int,
    target_format: str,
    expected_height: int,
    expected_width: int,
    device: str,
) -> Tuple[List[int], str]:
    if bench.torch is None:
        raise RuntimeError("PyTorch is required for constrained generation.")

    device_obj = bench.torch.device(device)
    ids = bench.torch.tensor([input_ids], dtype=bench.torch.long, device=device_obj)
    token_texts = [tokenizer.decode_from_pieces([piece]) for piece in tokenizer.pieces]
    generated_text = ""

    for _ in range(max_new_tokens):
        if target_format == TARGET_FORMAT_ROW_RUNS_ARC_BG_V1:
            complete = row_runs_arc_bg_complete(generated_text)
        else:
            complete = row_runs_complete(
                generated_text,
                include_shape=target_format == TARGET_FORMAT_ROW_RUNS_V1,
            )
        if complete:
            break

        idx = ids[:, -model.block_size :]
        logits, _ = model(idx)
        next_logits = logits[:, -1, :].clone()
        allowed_ids: List[int] = []

        for token_id, chunk in enumerate(token_texts):
            if not chunk:
                continue
            candidate = generated_text + chunk
            if target_format == TARGET_FORMAT_ROW_RUNS_ARC_BG_V1:
                ok = row_runs_arc_bg_prefix_ok(
                    candidate,
                    expected_height=expected_height,
                    expected_width=expected_width,
                )
            else:
                ok = row_runs_prefix_ok(
                    candidate,
                    include_shape=target_format == TARGET_FORMAT_ROW_RUNS_V1,
                    expected_height=expected_height,
                    expected_width=expected_width,
                )
            if ok:
                allowed_ids.append(token_id)

        if not allowed_ids:
            break

        mask = bench.torch.full_like(next_logits, float("-inf"))
        mask[:, allowed_ids] = next_logits[:, allowed_ids]
        next_id = bench.torch.argmax(mask, dim=-1, keepdim=True)
        token_id = int(next_id.item())
        ids = bench.torch.cat([ids, next_id], dim=1)
        generated_text += token_texts[token_id]

    return ids[0].tolist(), generated_text


def evaluate_arc_reasoning(
    model: bench.TinyCausalLM,
    tokenizer: bench.GreedyBPETokenizer,
    records: Sequence[Dict[str, Any]],
    *,
    text_format: str,
    target_format: str,
    decode_mode: str,
    max_new_tokens: int,
    device: Optional[str],
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    if bench.torch is None:
        raise RuntimeError("PyTorch is required for ARC reasoning evaluation.")
    if device is None:
        device = "mps" if getattr(bench.torch.backends, "mps", None) and bench.torch.backends.mps.is_available() else (
            "cuda" if bench.torch.cuda.is_available() else "cpu"
        )

    total = 0
    exact = 0
    shape_exact = 0
    parse_failures = 0
    cell_scores: List[float] = []
    family_exact: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    family_parse: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    samples: List[Dict[str, Any]] = []

    for record in records[: max_examples if max_examples is not None else len(records)]:
        family = str(record["family"])
        prompt = reasoning_prompt_text(record, text_format, target_format)
        gold_grid = record["output_grid"]
        gold_text = reasoning_target_text(record, text_format, target_format)

        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if decode_mode == DECODE_MODE_ROW_RUNS_CONSTRAINED and target_format in {
            TARGET_FORMAT_ROW_RUNS_V1,
            TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1,
            TARGET_FORMAT_ROW_RUNS_ARC_BG_V1,
        }:
            out_ids, continuation = generate_row_runs_constrained(
                model,
                tokenizer,
                input_ids,
                max_new_tokens=max_new_tokens,
                target_format=target_format,
                expected_height=len(gold_grid),
                expected_width=len(gold_grid[0]) if gold_grid else 0,
                device=device,
            )
            predicted_text = continuation.strip()
        else:
            out_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.piece_to_id.get(bench.EOS),
                greedy=True,
                device=bench.torch.device(device),
            )
            continuation = tokenizer.decode(out_ids[len(input_ids) :])
            if target_format == TARGET_FORMAT_ROW_RUNS_ARC_BG_V1:
                predicted_text = continuation.strip()
            else:
                predicted_text = continuation.splitlines()[0].strip()
        predicted_grid = deserialize_output_grid_for_shape(
            predicted_text,
            text_format,
            target_format,
            expected_height=len(gold_grid),
            expected_width=len(gold_grid[0]) if gold_grid else 0,
        )

        total += 1
        family_exact[family][1] += 1
        family_parse[family][1] += 1

        if predicted_grid is None:
            parse_failures += 1
            family_parse[family][0] += 0
            cell_scores.append(0.0)
            if len(samples) < 8:
                samples.append(
                    {
                        "family": family,
                        "trajectory_id": record["trajectory_id"],
                        "gold": gold_text,
                        "pred": predicted_text,
                        "parsed": False,
                        "cell_accuracy": 0.0,
                    }
                )
            continue

        family_parse[family][0] += 1
        score = float(grid_similarity(tuple(tuple(row) for row in predicted_grid), tuple(tuple(row) for row in gold_grid)))
        cell_scores.append(score)
        same_shape = (
            len(predicted_grid) == len(gold_grid)
            and (len(predicted_grid[0]) if predicted_grid else 0) == (len(gold_grid[0]) if gold_grid else 0)
        )
        exact_match = predicted_grid == gold_grid
        exact += int(exact_match)
        shape_exact += int(same_shape)
        family_exact[family][0] += int(exact_match)

        if len(samples) < 8:
            samples.append(
                {
                    "family": family,
                    "trajectory_id": record["trajectory_id"],
                    "gold": gold_text,
                    "pred": predicted_text,
                    "parsed": True,
                    "cell_accuracy": score,
                    "exact": exact_match,
                }
            )

    return {
        "n": total,
        "exact_grid_rate": exact / max(1, total),
        "shape_exact_rate": shape_exact / max(1, total),
        "parse_success_rate": (total - parse_failures) / max(1, total),
        "mean_cell_accuracy": statistics.mean(cell_scores) if cell_scores else 0.0,
        "median_cell_accuracy": statistics.median(cell_scores) if cell_scores else 0.0,
        "per_family_exact_grid_rate": per_family_fraction({key: (vals[0], vals[1]) for key, vals in family_exact.items()}),
        "per_family_parse_success_rate": per_family_fraction({key: (vals[0], vals[1]) for key, vals in family_parse.items()}),
        "samples": samples,
    }


def build_reasoning_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    if "bpe" not in results or "epiplex" not in results:
        return {}

    comparison: Dict[str, Any] = {
        "baseline": "bpe",
        "challenger": "epiplex",
        "generation": {},
        "language_model": {},
        "notes": [
            "Reasoning metrics are exact output-grid generation metrics on leakage-free prompts.",
        ],
    }

    for key, higher_is_better in [
        ("exact_grid_rate", True),
        ("shape_exact_rate", True),
        ("parse_success_rate", True),
        ("mean_cell_accuracy", True),
        ("median_cell_accuracy", True),
    ]:
        metric = bench.compare_scalar_metrics(
            results["bpe"]["generation"].get(key),
            results["epiplex"]["generation"].get(key),
            higher_is_better=higher_is_better,
        )
        if metric is not None:
            comparison["generation"][key] = metric

    for key in ["best_val_loss", "best_val_ppl"]:
        metric = bench.compare_scalar_metrics(
            results["bpe"]["language_model"].get(key),
            results["epiplex"]["language_model"].get(key),
            higher_is_better=False,
        )
        if metric is not None:
            comparison["language_model"][key] = metric

    exact_metric = comparison["generation"].get("exact_grid_rate")
    if exact_metric is not None and exact_metric.get("winner") not in {None, "tie"}:
        comparison["overall_winner"] = exact_metric["winner"]
    elif "mean_cell_accuracy" in comparison["generation"]:
        comparison["overall_winner"] = comparison["generation"]["mean_cell_accuracy"]["winner"]
    elif "shape_exact_rate" in comparison["generation"]:
        comparison["overall_winner"] = comparison["generation"]["shape_exact_rate"]["winner"]
    return comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BPE vs epiplex on ARC output-grid generation.")
    parser.add_argument("--output-dir", type=str, default="results/arc_reasoning_tokenizer_benchmark")
    parser.add_argument("--tokenizers", nargs="+", default=["bpe", "epiplex"])
    parser.add_argument(
        "--epiplex-profile",
        choices=["baseline", "row_aware", "adaptive", "row_runs_signal", "row_runs_struct", "row_runs_strict"],
        default="adaptive",
    )
    parser.add_argument("--text-format", choices=["raw_json", "arc_text_v1"], default="arc_text_v1")
    parser.add_argument(
        "--target-format",
        choices=[
            TARGET_FORMAT_NATIVE,
            TARGET_FORMAT_ROW_RUNS_V1,
            TARGET_FORMAT_ROW_RUNS_PAYLOAD_V1,
            TARGET_FORMAT_ROW_RUNS_ARC_BG_V1,
        ],
        default=TARGET_FORMAT_NATIVE,
    )
    parser.add_argument(
        "--decode-mode",
        choices=[DECODE_MODE_GREEDY, DECODE_MODE_ROW_RUNS_CONSTRAINED],
        default=DECODE_MODE_GREEDY,
    )
    parser.add_argument("--verify-roundtrip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--val-episodes", type=int, default=16)
    parser.add_argument("--num-train", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--include-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--max-piece-chars", type=int, default=48)
    parser.add_argument("--fit-workers", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    if bench.torch is None:
        raise RuntimeError("PyTorch is required for ARC reasoning benchmark. Install torch and rerun.")

    args = build_arg_parser().parse_args()
    bench.set_seed(args.seed)

    train_records, train_family_counts = sample_reasoning_examples(
        num_episodes=args.train_episodes,
        seed_start=args.seed,
        num_train=args.num_train,
        max_attempts=args.max_attempts,
        include_test=args.include_test,
        text_format=args.text_format,
        verify_roundtrip=args.verify_roundtrip,
    )
    val_records, val_family_counts = sample_reasoning_examples(
        num_episodes=args.val_episodes,
        seed_start=args.seed + args.train_episodes,
        num_train=args.num_train,
        max_attempts=args.max_attempts,
        include_test=args.include_test,
        text_format=args.text_format,
        verify_roundtrip=args.verify_roundtrip,
    )

    train_texts = [reasoning_training_text(record, args.text_format, args.target_format) for record in train_records]
    val_texts = [reasoning_training_text(record, args.text_format, args.target_format) for record in val_records]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "task": "arc_reasoning_output_grid",
        "seed": args.seed,
        "epiplex_profile": args.epiplex_profile,
        "text_format": args.text_format,
        "target_format": args.target_format,
        "decode_mode": args.decode_mode,
        "verify_roundtrip": args.verify_roundtrip,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "include_test": args.include_test,
        "train_family_counts": train_family_counts,
        "val_family_counts": val_family_counts,
        "results": {},
    }

    for kind in args.tokenizers:
        tokenizer = tok_bench.build_arc_tokenizer(
            kind,
            epiplex_profile=args.epiplex_profile,
            text_format=args.text_format,
            target_format=args.target_format,
            vocab_size=args.vocab_size,
            candidate_pool_size=args.candidate_pool_size,
            max_piece_chars=args.max_piece_chars,
            fit_workers=args.fit_workers,
            verbose=args.verbose,
        )

        print(f"\n[tokenizer] fitting {kind}")
        tokenizer.fit(train_texts)
        tokenizer_path = out_dir / f"arc_reasoning_{kind}_tokenizer.json"
        tokenizer.save(tokenizer_path)

        train_ids = bench.flatten_tokenized_examples(tokenizer, train_texts, add_bos=True, add_eos=True)
        val_ids = bench.flatten_tokenized_examples(tokenizer, val_texts, add_bos=True, add_eos=True)
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

        model_path = out_dir / f"arc_reasoning_{kind}_lm.pt"
        bench.torch.save(model.state_dict(), model_path)

        generation = evaluate_arc_reasoning(
            model,
            tokenizer,
            val_records,
            text_format=args.text_format,
            target_format=args.target_format,
            decode_mode=args.decode_mode,
            max_new_tokens=max(
                args.max_new_tokens,
                max_target_token_length(tokenizer, val_records, args.text_format, args.target_format) + 16,
            ),
            device=args.device,
            max_examples=args.max_eval_examples,
        )

        report["results"][kind] = {
            "tokenizer_path": str(tokenizer_path),
            "model_path": str(model_path),
            "tokenizer_intrinsic": (
                tok_bench.arc_intrinsic_metrics_v1(tokenizer, val_texts)
                if args.text_format == "arc_text_v1"
                else tok_bench.arc_intrinsic_metrics(tokenizer, val_texts)
            ),
            "language_model": lm_history,
            "generation": generation,
        }

    report["comparison"] = build_reasoning_comparison(report["results"])
    report_path = out_dir / "arc_reasoning_tokenizer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] report written to {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
