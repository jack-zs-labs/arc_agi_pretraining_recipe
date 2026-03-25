from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This decode eval requires torch. Install requirements-models.txt or use .venv_atari.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import (
    AttentionBackendConfig,
    AttentionBackendPresetName,
    DecoderKVCache,
    DecoderLanguageModel,
    DecoderModelConfig,
    InferenceBudget,
    reasoning_budget_policy_for_benchmark,
)
from model_stack_training_ablation import (
    BYTE_VOCAB_SIZE,
    auto_device,
    chunk_token_stream,
    encode_text,
    make_batches,
    parameter_count,
    sample_arc_texts,
    synchronize_device,
)


TARGET_ACTION_MARKER = "target_action="
DECODE_MODE_CHOICES = ("greedy", "policy_default", "context_trie", "schema_json", "context_then_schema")
EFFORT_CHOICES = ("fast", "balanced", "deep")
CandidateKey = tuple[str, str, str]
TargetCandidate = tuple[str, tuple[int, ...]]
SchemaBank = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train small decoder presets, then evaluate ARC structured-step quality with cached decode."
    )
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--latent-kv-dim", type=int, default=24)
    parser.add_argument("--scale-invariant-tau", type=float, default=10.0)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=("mla_default", "mla_sia_prefill_l1"),
        default=["mla_default", "mla_sia_prefill_l1"],
    )
    parser.add_argument(
        "--decode-modes",
        nargs="+",
        choices=DECODE_MODE_CHOICES,
        default=["greedy"],
        help="Decode strategies to evaluate after training. `context_trie` constrains cached decode to train-split target_action candidates.",
    )
    parser.add_argument("--max-eval-examples", type=int, default=64)
    parser.add_argument("--reasoning-effort", choices=EFFORT_CHOICES, default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arc-include-verifier-targets",
        action="store_true",
        help="Append verifier supervision to structured ARC text records.",
    )
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--csv-output", type=str, default="", help="Optional CSV output path.")
    return parser.parse_args()


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_prompt_metadata(prompt_text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in prompt_text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key] = value
    return metadata


def candidate_bucket_keys(metadata: dict[str, str]) -> list[CandidateKey]:
    family = metadata.get("family", "")
    step_index = metadata.get("step_index", "")
    previous_action = metadata.get("previous_action", "")
    return [
        (family, step_index, previous_action),
        (family, step_index, ""),
        (family, "", ""),
        ("", step_index, ""),
        ("", "", ""),
    ]


def build_prompt_target_pairs(texts: list[str]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for text in texts:
        marker_index = text.rfind(TARGET_ACTION_MARKER)
        if marker_index < 0:
            continue
        prompt_text = text[:marker_index]
        target_text = text[marker_index:]
        metadata = parse_prompt_metadata(prompt_text)
        prompt_tokens = encode_text(prompt_text)[:-1]
        target_tokens = encode_text(target_text)[:-1]
        if not prompt_tokens or not target_tokens:
            continue
        examples.append(
            {
                "family": metadata.get("family", ""),
                "step_index": metadata.get("step_index", ""),
                "previous_action": metadata.get("previous_action", ""),
                "prompt_text": prompt_text,
                "target_text": target_text,
                "prompt_tokens": prompt_tokens,
                "target_tokens": target_tokens,
            }
        )
    return examples


def build_target_candidate_index(texts: list[str]) -> dict[CandidateKey, list[TargetCandidate]]:
    buckets: dict[CandidateKey, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for example in build_prompt_target_pairs(texts):
        candidate = (example["target_text"], tuple(int(token) for token in example["target_tokens"]))
        metadata = {
            "family": str(example["family"]),
            "step_index": str(example["step_index"]),
            "previous_action": str(example["previous_action"]),
        }
        for key in candidate_bucket_keys(metadata):
            buckets[key][candidate[0]] = candidate[1]
    return {
        key: sorted(bucket.items(), key=lambda item: item[0])
        for key, bucket in buckets.items()
    }


def build_schema_bank(texts: list[str]) -> SchemaBank:
    render_changed_fractions: set[float] = set()
    for example in build_prompt_target_pairs(texts):
        payload = json.loads(example["target_text"][len(TARGET_ACTION_MARKER) :])
        if payload["name"] == "render":
            render_changed_fractions.add(float(payload["action"]["changed_cell_fraction"]))
    return {
        "render_changed_fractions": tuple(sorted(render_changed_fractions)),
    }


def serialize_target_action_text(name: str, action: dict[str, object]) -> str:
    return (
        TARGET_ACTION_MARKER
        + json.dumps(
            {
                "action": action,
                "name": name,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def prompt_workspace_tokens(prompt_text: str) -> list[str]:
    for line in prompt_text.splitlines():
        if line.startswith("workspace_tokens="):
            payload = line[len("workspace_tokens=") :]
            return payload.split()
    return []


def extract_prompt_objects(prompt_text: str) -> list[dict[str, object]]:
    objects: dict[int, dict[str, object]] = {}
    for token in prompt_workspace_tokens(prompt_text):
        if not token.startswith("obj["):
            continue
        prefix, _, value = token.partition("=")
        if not _:
            continue
        index_end = prefix.find("]")
        if index_end < 4:
            continue
        try:
            object_index = int(prefix[4:index_end])
        except ValueError:
            continue
        suffix = prefix[index_end + 1 :]
        descriptor = objects.setdefault(
            object_index,
            {
                "index": object_index,
                "object_id": f"obj_{object_index}",
                "tags": set(),
                "selected": False,
                "focus": False,
            },
        )
        if suffix == ".tag":
            descriptor["tags"].add(value)
        elif suffix == ".selected" and value == "1":
            descriptor["selected"] = True
        elif suffix == ".focus" and value == "1":
            descriptor["focus"] = True
    return [objects[index] for index in sorted(objects)]


def extract_previous_action_payload(prompt_text: str) -> dict[str, object] | None:
    previous_action = parse_prompt_metadata(prompt_text).get("previous_action", "")
    if not previous_action.startswith("{"):
        return None
    try:
        payload = json.loads(previous_action)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def schema_candidates_for_example(example: dict[str, object], schema_bank: SchemaBank) -> list[TargetCandidate]:
    step_index = str(example["step_index"])
    objects = extract_prompt_objects(str(example["prompt_text"]))
    object_ids = [str(obj["object_id"]) for obj in objects]
    source_ids = [str(obj["object_id"]) for obj in objects if "source" in obj["tags"]]
    target_ids = [str(obj["object_id"]) for obj in objects if "target" in obj["tags"]]
    selected_ids = [str(obj["object_id"]) for obj in objects if bool(obj["selected"])]
    output_scene_objects = [
        "connector" if "connector" in obj["tags"] else str(obj["object_id"])
        for obj in objects
    ]
    previous_action_payload = extract_previous_action_payload(str(example["prompt_text"]))
    candidates: dict[str, tuple[int, ...]] = {}

    def add_candidate(name: str, action: dict[str, object]) -> None:
        text = serialize_target_action_text(name, action)
        candidates[text] = tuple(encode_text(text)[:-1])

    if step_index == "0" and object_ids:
        add_candidate("segment", {"object_ids": object_ids})

    if step_index in {"1", "2"}:
        for object_id in source_ids or selected_ids or object_ids:
            add_candidate("pick_source", {"selected_object_ids": [object_id]})
        for object_id in target_ids or selected_ids or object_ids:
            add_candidate("pick_target", {"selected_object_ids": [object_id]})
        for object_id in selected_ids or object_ids:
            add_candidate("select", {"selected_object_ids": [object_id]})

    if step_index == "4" and previous_action_payload is not None:
        previous_action = previous_action_payload.get("action", {})
        if isinstance(previous_action, dict):
            target_shape = previous_action.get("target_grid_shape") or previous_action.get("grid_shape")
            if isinstance(target_shape, list) and len(target_shape) == 2 and output_scene_objects:
                normalized_shape = [int(target_shape[0]), int(target_shape[1])]
                for changed_fraction in schema_bank.get("render_changed_fractions", ()):
                    add_candidate(
                        "render",
                        {
                            "changed_cell_fraction": changed_fraction,
                            "grid_shape": normalized_shape,
                            "output_scene_objects": output_scene_objects,
                            "target_grid_shape": normalized_shape,
                            "workspace_grid_shape": normalized_shape,
                        },
                    )

    return sorted(candidates.items(), key=lambda item: item[0])


class TokenTrie:
    def __init__(self) -> None:
        self.children: dict[int, TokenTrie] = {}
        self.is_terminal = False

    def insert(self, tokens: tuple[int, ...]) -> None:
        node = self
        for token in tokens:
            node = node.children.setdefault(token, TokenTrie())
        node.is_terminal = True


def build_token_trie(candidates: list[TargetCandidate], *, target_length: int) -> TokenTrie | None:
    matching = [tokens for _text, tokens in candidates if len(tokens) == target_length]
    if not matching:
        return None
    root = TokenTrie()
    for tokens in matching:
        root.insert(tokens)
    return root


def exact_context_key(example: dict[str, object]) -> CandidateKey:
    return (
        str(example["family"]),
        str(example["step_index"]),
        str(example["previous_action"]),
    )


def lookup_target_candidates(
    example: dict[str, object],
    candidate_index: dict[CandidateKey, list[TargetCandidate]],
) -> tuple[list[TargetCandidate], CandidateKey | None]:
    metadata = {
        "family": str(example["family"]),
        "step_index": str(example["step_index"]),
        "previous_action": str(example["previous_action"]),
    }
    for key in candidate_bucket_keys(metadata):
        candidates = candidate_index.get(key)
        if candidates:
            return candidates, key
    return [], None


def build_model_config(
    args: argparse.Namespace,
    *,
    preset_name: AttentionBackendPresetName,
    max_position_embeddings: int,
) -> DecoderModelConfig:
    attention = AttentionBackendConfig.from_preset(
        preset_name,
        latent_kv_dim=args.latent_kv_dim,
        scale_invariant_tau=args.scale_invariant_tau,
    )
    return DecoderModelConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        max_position_embeddings=max(max_position_embeddings, 512),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        attention=attention,
    )


def evaluate_teacher_forced_loss(
    model: DecoderLanguageModel,
    windows: list[list[int]],
    *,
    batch_size: int,
    device: torch.device,
    reasoning_effort: str,
) -> float:
    if not windows:
        return float("nan")
    losses: list[float] = []
    policy = reasoning_budget_policy_for_benchmark(
        "arc",
        effort=reasoning_effort,
        attention_window=model.config.attention.sliding_window,
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
            inputs = batch_tensor[:, :-1]
            labels = batch_tensor[:, 1:]
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            inputs, labels = trim_batch_to_budget(inputs, labels, budget.max_prompt_tokens)
            logits = model.forward(inputs, budget=budget).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def decode_tokens_to_text(tokens: list[int]) -> str:
    return bytes(max(0, min(token, 255)) for token in tokens).decode("utf-8", errors="replace")


def trim_batch_to_budget(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    budget_prompt_limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if budget_prompt_limit is None or inputs.size(1) <= budget_prompt_limit:
        return inputs, labels
    return inputs[:, -budget_prompt_limit:], labels[:, -budget_prompt_limit:]


def select_next_token(
    next_logits: torch.Tensor,
    *,
    decode_mode: str,
    trie_node: TokenTrie | None,
) -> tuple[int, TokenTrie | None]:
    if decode_mode in {"context_trie", "schema_json", "context_then_schema"} and trie_node is not None and trie_node.children:
        allowed = torch.tensor(sorted(trie_node.children.keys()), dtype=torch.long, device=next_logits.device)
        allowed_logits = next_logits.index_select(-1, allowed)
        next_index = int(torch.argmax(allowed_logits, dim=-1).item())
        token = int(allowed[next_index].item())
        return token, trie_node.children[token]
    token = int(torch.argmax(next_logits, dim=-1).item())
    next_node = trie_node.children.get(token) if trie_node is not None else None
    return token, next_node


def resolved_decode_mode(
    decode_mode_request: str,
    *,
    structured_decode_mode_hint: str | None,
) -> str:
    if decode_mode_request != "policy_default":
        return decode_mode_request
    return structured_decode_mode_hint or "greedy"


def matching_candidates_for_example(
    example: dict[str, object],
    *,
    decode_mode: str,
    target_candidate_index: dict[CandidateKey, list[TargetCandidate]],
    schema_bank: SchemaBank,
) -> tuple[list[TargetCandidate], CandidateKey | None, str]:
    target_length = len(example["target_tokens"])
    schema_candidates = [
        candidate
        for candidate in schema_candidates_for_example(example, schema_bank)
        if len(candidate[1]) == target_length
    ]
    if decode_mode == "greedy":
        return [], None, "greedy"
    if decode_mode == "context_then_schema":
        exact_key = exact_context_key(example)
        exact_candidates = [
            candidate
            for candidate in target_candidate_index.get(exact_key, [])
            if len(candidate[1]) == target_length
        ]
        if exact_candidates:
            return exact_candidates, exact_key, "exact_context"
        return schema_candidates, None, "schema_fallback"

    candidates, matched_key = lookup_target_candidates(example, target_candidate_index)
    matching_candidates = [candidate for candidate in candidates if len(candidate[1]) == target_length]
    if decode_mode == "schema_json":
        candidates_by_text = {text: tokens for text, tokens in matching_candidates}
        for text, tokens in schema_candidates:
            candidates_by_text[text] = tokens
        matching_candidates = sorted(candidates_by_text.items(), key=lambda item: item[0])
        return matching_candidates, matched_key, "context_plus_schema"
    return matching_candidates, matched_key, "context"


def run_cached_decode_eval(
    model: DecoderLanguageModel,
    examples: list[dict[str, object]],
    *,
    device: torch.device,
    decode_mode: str,
    decode_mode_request: str,
    reasoning_effort: str,
    target_candidate_index: dict[CandidateKey, list[TargetCandidate]],
    schema_bank: SchemaBank,
) -> dict[str, object]:
    exact_matches = 0
    token_matches = 0
    token_total = 0
    prefill_samples_ms: list[float] = []
    decode_samples_ms: list[float] = []
    candidate_counts: list[int] = []
    exact_candidate_coverage = 0
    constrained_examples = 0
    exact_context_used = 0
    schema_fallback_used = 0
    failures: list[dict[str, object]] = []
    policy = reasoning_budget_policy_for_benchmark(
        "arc",
        effort=reasoning_effort,
        attention_window=model.config.attention.sliding_window,
    )
    model.eval()
    with torch.inference_mode():
        for example in examples:
            prompt_tokens = example["prompt_tokens"]
            target_tokens = example["target_tokens"]
            matching_candidates, matched_key, candidate_source = matching_candidates_for_example(
                example,
                decode_mode=decode_mode,
                target_candidate_index=target_candidate_index,
                schema_bank=schema_bank,
            )
            trie_root = (
                build_token_trie(matching_candidates, target_length=len(target_tokens))
                if decode_mode in {"context_trie", "schema_json", "context_then_schema"}
                else None
            )
            trie_node = trie_root
            candidate_counts.append(len(matching_candidates))
            exact_candidate_coverage += int(any(text == example["target_text"] for text, _tokens in matching_candidates))
            constrained_examples += int(trie_root is not None)
            exact_context_used += int(candidate_source == "exact_context")
            schema_fallback_used += int(candidate_source == "schema_fallback")
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=len(prompt_tokens),
                target_tokens=len(target_tokens),
                use_kv_cache=True,
                max_new_tokens=len(target_tokens),
            )
            cache = DecoderKVCache.create(model.config.num_hidden_layers)

            synchronize_device(device)
            prefill_start = time.perf_counter()
            outputs = model.forward(prompt_tensor, budget=budget, cache=cache)
            synchronize_device(device)
            prefill_samples_ms.append((time.perf_counter() - prefill_start) * 1000.0)

            next_logits = outputs.logits[:, -1, :]
            cache = outputs.cache
            if cache is None:
                raise RuntimeError("Expected cache to be populated for cached decode evaluation.")

            predicted_tokens: list[int] = []
            synchronize_device(device)
            decode_start = time.perf_counter()
            for _ in range(len(target_tokens)):
                token_id, trie_node = select_next_token(
                    next_logits[0],
                    decode_mode=decode_mode,
                    trie_node=trie_node,
                )
                predicted_tokens.append(token_id)
                next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                outputs = model.forward(next_token, budget=budget, cache=cache)
                next_logits = outputs.logits[:, -1, :]
                cache = outputs.cache
            synchronize_device(device)
            decode_samples_ms.append((time.perf_counter() - decode_start) * 1000.0)

            matches = sum(int(pred == target) for pred, target in zip(predicted_tokens, target_tokens))
            token_matches += matches
            token_total += len(target_tokens)
            exact = matches == len(target_tokens)
            exact_matches += int(exact)
            if not exact and len(failures) < 3:
                failures.append(
                    {
                        "decode_mode": decode_mode,
                        "candidate_source": candidate_source,
                        "candidate_bucket_key": list(matched_key) if matched_key is not None else None,
                        "candidate_count": len(matching_candidates),
                        "prompt_preview": example["prompt_text"][-240:],
                        "target_text": example["target_text"],
                        "predicted_text": decode_tokens_to_text(predicted_tokens),
                    }
                )
    return {
        "decode_mode_request": decode_mode_request,
        "decode_mode": decode_mode,
        "reasoning_effort": reasoning_effort,
        "examples": len(examples),
        "exact_match_rate": float(exact_matches / max(len(examples), 1)),
        "token_accuracy": float(token_matches / max(token_total, 1)),
        "candidate_count_median": statistics.median(candidate_counts) if candidate_counts else 0.0,
        "candidate_exact_coverage_rate": float(exact_candidate_coverage / max(len(examples), 1)),
        "constraint_applied_rate": float(constrained_examples / max(len(examples), 1)),
        "exact_context_applied_rate": float(exact_context_used / max(len(examples), 1)),
        "schema_fallback_rate": float(schema_fallback_used / max(len(examples), 1)),
        "prefill_latency_median_ms": statistics.median(prefill_samples_ms) if prefill_samples_ms else float("nan"),
        "decode_latency_median_ms": statistics.median(decode_samples_ms) if decode_samples_ms else float("nan"),
        "failures": failures,
    }


def train_and_evaluate_preset(
    *,
    preset_name: AttentionBackendPresetName,
    args: argparse.Namespace,
    train_windows: list[list[int]],
    val_windows: list[list[int]],
    eval_examples: list[dict[str, object]],
    decode_modes: list[str],
    target_candidate_index: dict[CandidateKey, list[TargetCandidate]],
    schema_bank: SchemaBank,
    max_position_embeddings: int,
    device: torch.device,
) -> list[dict[str, object]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = DecoderLanguageModel(
        build_model_config(
            args,
            preset_name=preset_name,
            max_position_embeddings=max_position_embeddings,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batches = make_batches(train_windows, batch_size=args.batch_size, rng=random.Random(args.seed))
    if not batches:
        raise ValueError("Training windows are empty.")

    initial_val_loss = evaluate_teacher_forced_loss(
        model,
        val_windows,
        batch_size=args.batch_size,
        device=device,
        reasoning_effort=args.reasoning_effort,
    )
    step_losses: list[float] = []
    tokens_processed = 0
    model.train()
    synchronize_device(device)
    start_time = time.perf_counter()
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
        inputs = batch_tensor[:, :-1]
        labels = batch_tensor[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        budget = reasoning_budget_policy_for_benchmark(
            "arc",
            effort=args.reasoning_effort,
            attention_window=model.config.attention.sliding_window,
        ).build_inference_budget(
            model.config,
            prompt_tokens=inputs.size(1),
            use_kv_cache=False,
            max_new_tokens=0,
        )
        inputs, labels = trim_batch_to_budget(inputs, labels, budget.max_prompt_tokens)
        logits = model.forward(inputs, budget=budget).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.item()))
        tokens_processed += int(labels.numel())
    synchronize_device(device)
    elapsed = time.perf_counter() - start_time
    final_val_loss = evaluate_teacher_forced_loss(
        model,
        val_windows,
        batch_size=args.batch_size,
        device=device,
        reasoning_effort=args.reasoning_effort,
    )
    policy = reasoning_budget_policy_for_benchmark(
        "arc",
        effort=args.reasoning_effort,
        attention_window=model.config.attention.sliding_window,
    )
    common = {
        "preset_name": preset_name,
        "attention_preset": preset_name,
        "backend": model.config.attention.backend,
        "parameter_count": parameter_count(model),
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "train_loss_history": step_losses,
        "train_tokens_per_second": float(tokens_processed / max(elapsed, 1e-9)),
        "train_elapsed_seconds": elapsed,
        "reasoning_effort": args.reasoning_effort,
        "train_budget_policy": policy.resolved_budget_dict(
            model.config,
            prompt_tokens=args.seq_len,
            use_kv_cache=False,
            max_new_tokens=0,
        ),
        "decode_budget_hint": policy.resolved_budget_dict(
            model.config,
            prompt_tokens=args.seq_len,
            use_kv_cache=True,
        ),
    }
    return [
        {
            **common,
            **run_cached_decode_eval(
                model,
                eval_examples,
                device=device,
                decode_mode=resolved_decode_mode(
                    decode_mode,
                    structured_decode_mode_hint=policy.structured_decode_mode_hint,
                ),
                decode_mode_request=decode_mode,
                reasoning_effort=args.reasoning_effort,
                target_candidate_index=target_candidate_index,
                schema_bank=schema_bank,
            ),
        }
        for decode_mode in decode_modes
    ]


def main() -> None:
    args = parse_args()
    device = auto_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_texts = sample_arc_texts(
        num_episodes=args.train_episodes,
        seed_start=args.seed,
        serialization="structured_workspace",
        include_verifier_targets=args.arc_include_verifier_targets,
    )
    val_texts = sample_arc_texts(
        num_episodes=args.val_episodes,
        seed_start=args.seed + 10_000,
        serialization="structured_workspace",
        include_verifier_targets=args.arc_include_verifier_targets,
    )
    train_windows = chunk_token_stream(train_texts, seq_len=args.seq_len)
    val_windows = chunk_token_stream(val_texts, seq_len=args.seq_len)
    eval_examples = build_prompt_target_pairs(val_texts)[: args.max_eval_examples]
    target_candidate_index = build_target_candidate_index(train_texts)
    schema_bank = build_schema_bank(train_texts)
    if not eval_examples:
        raise ValueError("No decode evaluation examples were produced from the validation set.")
    longest_record = max(len(encode_text(text)) - 1 for text in train_texts + val_texts)
    results: list[dict[str, object]] = []
    for preset_name in args.presets:
        results.extend(
            train_and_evaluate_preset(
                preset_name=preset_name,
                args=args,
                train_windows=train_windows,
                val_windows=val_windows,
                eval_examples=eval_examples,
                decode_modes=args.decode_modes,
                target_candidate_index=target_candidate_index,
                schema_bank=schema_bank,
                max_position_embeddings=longest_record + 32,
                device=device,
            )
        )

    payload = {
        "device": device.type,
        "seed": args.seed,
        "presets": args.presets,
        "decode_modes": args.decode_modes,
        "reasoning_effort": args.reasoning_effort,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "seq_len": args.seq_len,
        "max_eval_examples": len(eval_examples),
        "arc_include_verifier_targets": args.arc_include_verifier_targets,
        "results": results,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv_output:
        write_csv(
            args.csv_output,
            [
                {
                    "device": payload["device"],
                    "seed": payload["seed"],
                    "attention_preset": item["attention_preset"],
                    "reasoning_effort": item["reasoning_effort"],
                    "decode_mode_request": item["decode_mode_request"],
                    "decode_mode": item["decode_mode"],
                    "backend": item["backend"],
                    "parameter_count": item["parameter_count"],
                    "initial_val_loss": item["initial_val_loss"],
                    "final_val_loss": item["final_val_loss"],
                    "train_tokens_per_second": item["train_tokens_per_second"],
                    "exact_match_rate": item["exact_match_rate"],
                    "token_accuracy": item["token_accuracy"],
                    "candidate_count_median": item["candidate_count_median"],
                    "candidate_exact_coverage_rate": item["candidate_exact_coverage_rate"],
                    "constraint_applied_rate": item["constraint_applied_rate"],
                    "exact_context_applied_rate": item["exact_context_applied_rate"],
                    "schema_fallback_rate": item["schema_fallback_rate"],
                    "prefill_latency_median_ms": item["prefill_latency_median_ms"],
                    "decode_latency_median_ms": item["decode_latency_median_ms"],
                    "examples": item["examples"],
                }
                for item in results
            ],
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
