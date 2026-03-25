from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Iterable, Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("This holdout evaluator requires torch. Install requirements-models.txt or use .venv_atari.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from arc_trajectory_sampler.mixed_reasoning_dataset import build_oscar_scope_reasoning_examples, texts_from_examples
from arc_trajectory_sampler.oscar_scope_reasoning import OSCAR_WORKFLOW_REASONING_FAMILIES
from models import DecoderLanguageModel, reasoning_budget_policy_for_benchmark
from models.reasoning_tokenizer import add_tokenizer_cli_arguments, build_reasoning_tokenizer
from train_integrated_reasoning_stack import (
    DecisionActionExample,
    build_decision_candidate_masks,
    build_decision_head_vocabularies,
    build_model_config,
    build_oscar_auxiliary_vocabularies,
    convert_decision_action_examples,
    count_parameters,
    cross_entropy_loss,
    decision_argument_head_key,
    encode_example_sequence,
    evaluate_benchmark_loss,
    evaluate_decision_action_accuracy,
    evaluate_oscar_auxiliary,
    finalize_decision_action_examples,
    oscar_auxiliary_labels_for_example,
    parse_reasoning_fields,
    sample_decision_action_batch,
    sample_oscar_auxiliary_batch,
    set_text_tokenizer,
    chunk_token_stream,
    trim_batch_to_budget,
)


TRACE_STEP_TRANSFER = "transfer_primary_intervention"
INTERVENTION_TRACE_STEPS = (
    "select_focus_kpi",
    "apply_primary_intervention",
    "apply_followup_intervention",
)
WORKFLOW_METRIC_KEYS = (
    "oscar_workflow_kpi_accuracy",
    "oscar_workflow_improvement_accuracy",
    "oscar_workflow_motif_accuracy",
    "oscar_workflow_reward_bucket_accuracy",
    "oscar_workflow_reward_score_mse",
    "oscar_workflow_action_step_accuracy",
    "oscar_workflow_action_kpi_family_accuracy",
    "oscar_workflow_action_intervention_family_accuracy",
    "oscar_workflow_action_exact_match",
    "oscar_workflow_transfer_intervention_accuracy",
    "oscar_workflow_rollout_step_accuracy",
    "oscar_workflow_rollout_kpi_accuracy",
    "oscar_workflow_rollout_intervention_accuracy",
    "oscar_workflow_rollout_trajectory_exact_match",
)


@dataclass(frozen=True)
class OscarWorkflowRolloutState:
    previous_action: str = "<start>"
    selected_kpi_family: str | None = None
    selected_primary_intervention_family: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Oscar workflow reasoning on all but one business environment, "
            "then evaluate held-out-domain workflow abstraction and intervention transfer."
        )
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=OSCAR_WORKFLOW_REASONING_FAMILIES,
        default=list(OSCAR_WORKFLOW_REASONING_FAMILIES),
    )
    parser.add_argument(
        "--holdout-environments",
        nargs="+",
        default=["all"],
        help="Exact workflow-environment labels to hold out, or `all` for every discovered environment.",
    )
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--max-documents", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decision-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--decision-action-loss-weight", type=float, default=1.0)
    parser.add_argument("--decision-action-train-scale", type=float, default=1.0)
    parser.add_argument("--workflow-action-aux-train-scale", type=float, default=2.0)
    parser.add_argument("--workflow-rollout-train-scale", type=float, default=2.0)
    parser.add_argument("--workflow-rollout-batch-size", type=int, default=2)
    parser.add_argument("--workflow-scheduled-sampling-max-prob", type=float, default=0.5)
    parser.add_argument("--oscar-workflow-action-step-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-action-kpi-loss-weight", type=float, default=0.2)
    parser.add_argument("--oscar-workflow-action-intervention-loss-weight", type=float, default=0.3)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--attention-preset", choices=("mla_default", "mla_sia_prefill_l1"), default="mla_default")
    parser.add_argument("--latent-kv-dim", type=int, default=24)
    parser.add_argument("--architecture", choices=("dense", "moe"), default="dense")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--router-jitter-noise", type=float, default=0.01)
    parser.add_argument("--moe-auxiliary-loss-weight", type=float, default=1e-2)
    parser.add_argument("--train-reasoning-effort", choices=("fast", "balanced", "deep"), default="balanced")
    parser.add_argument("--eval-reasoning-effort", choices=("fast", "balanced", "deep"), default="deep")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/oscar_workflow_holdout_eval",
    )
    add_tokenizer_cli_arguments(
        parser,
        default_kind="byte",
        default_vocab_size=2048,
        default_task="generic",
        default_fit_verbose=False,
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def example_environment_mentions(example) -> tuple[str, ...]:
    auxiliary_targets = example.auxiliary_targets or {}
    mentions = {
        str(auxiliary_targets.get(key, "")).strip()
        for key in ("workflow_environment", "source_workflow_environment", "target_workflow_environment")
    }
    return tuple(sorted(value for value in mentions if value))


def split_examples_by_holdout(
    examples,
    *,
    holdout_environment: str,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    train_examples = []
    eval_examples = []
    for example in examples:
        mentions = example_environment_mentions(example)
        if holdout_environment in mentions:
            eval_examples.append(example)
        else:
            train_examples.append(example)
    return tuple(train_examples), tuple(eval_examples)


def family_counts(examples) -> dict[str, int]:
    counts = Counter(str((example.auxiliary_targets or {}).get("family", "unknown")) for example in examples)
    return dict(sorted(counts.items()))


def action_step_counts(examples: Iterable[DecisionActionExample]) -> dict[str, int]:
    counts = Counter(example.trace_step for example in examples)
    return dict(sorted(counts.items()))


def build_decision_train_artifacts(
    train_examples,
) -> dict[str, object]:
    raw_train_action_examples = {
        "oscar_scope_reasoning": convert_decision_action_examples(train_examples)[0],
    }
    decision_head_vocabularies = build_decision_head_vocabularies(raw_train_action_examples)
    decision_full_action_vocabularies = decision_head_vocabularies["full_action_vocabularies"]
    decision_name_vocabularies = decision_head_vocabularies["name_vocabularies"]
    decision_argument_vocabularies = decision_head_vocabularies["argument_vocabularies"]
    decision_full_action_components = decision_head_vocabularies["full_action_components"]
    decision_candidate_masks = build_decision_candidate_masks(
        raw_train_action_examples,
        full_action_vocabularies=decision_full_action_vocabularies,
        name_vocabularies=decision_name_vocabularies,
        argument_vocabularies=decision_argument_vocabularies,
        full_action_components=decision_full_action_components,
    )
    train_action_examples = finalize_decision_action_examples(
        raw_train_action_examples,
        full_action_vocabularies=decision_full_action_vocabularies,
        name_vocabularies=decision_name_vocabularies,
        argument_vocabularies=decision_argument_vocabularies,
        full_action_components=decision_full_action_components,
        candidate_masks=decision_candidate_masks,
    )
    return {
        "train_action_examples": train_action_examples,
        "raw_train_action_examples": raw_train_action_examples,
        "decision_benchmark_adapter_names": tuple(
            name for name, examples in train_action_examples.items() if examples
        ),
        "decision_full_action_vocabularies": decision_full_action_vocabularies,
        "decision_name_vocabularies": decision_name_vocabularies,
        "decision_argument_vocabularies": decision_argument_vocabularies,
        "decision_full_action_components": decision_full_action_components,
        "decision_candidate_masks": decision_candidate_masks,
    }


def filter_covered_decision_examples(
    raw_examples: Sequence[DecisionActionExample],
    *,
    full_action_vocabularies: dict[str, tuple[str, ...]],
    name_vocabularies: dict[str, tuple[str, ...]],
    argument_vocabularies: dict[str, tuple[str, ...]],
) -> tuple[tuple[DecisionActionExample, ...], dict[str, object]]:
    name_to_index = {
        head_name: {name: index for index, name in enumerate(vocabulary)}
        for head_name, vocabulary in name_vocabularies.items()
    }
    argument_to_index = {
        head_name: {argument: index for index, argument in enumerate(vocabulary)}
        for head_name, vocabulary in argument_vocabularies.items()
    }
    covered: list[DecisionActionExample] = []
    uncovered_by_reason = Counter()
    for example in raw_examples:
        full_vocab = full_action_vocabularies.get(example.output_head)
        if not full_vocab:
            uncovered_by_reason["missing_output_head"] += 1
            continue
        if example.target_action not in full_vocab:
            uncovered_by_reason["unseen_action"] += 1
            continue
        name_lookup = name_to_index.get(example.output_head)
        if name_lookup is None or example.target_action_name not in name_lookup:
            uncovered_by_reason["unseen_action_name"] += 1
            continue
        name_id = name_lookup[example.target_action_name]
        argument_head = decision_argument_head_key(example.output_head, name_id)
        argument_lookup = argument_to_index.get(argument_head)
        if argument_lookup is None or example.target_argument_key not in argument_lookup:
            uncovered_by_reason["unseen_argument"] += 1
            continue
        covered.append(example)
    summary = {
        "total": len(raw_examples),
        "covered": len(covered),
        "coverage": (len(covered) / len(raw_examples)) if raw_examples else 0.0,
        "uncovered_by_reason": dict(sorted(uncovered_by_reason.items())),
    }
    return tuple(covered), summary


def finalize_eval_action_examples(
    raw_examples: Sequence[DecisionActionExample],
    *,
    full_action_vocabularies: dict[str, tuple[str, ...]],
    name_vocabularies: dict[str, tuple[str, ...]],
    argument_vocabularies: dict[str, tuple[str, ...]],
    full_action_components: dict[str, tuple[tuple[int, int], ...]],
    candidate_masks: dict[str, dict[str, tuple[bool, ...]]],
) -> tuple[DecisionActionExample, ...]:
    return finalize_decision_action_examples(
        {"oscar_scope_reasoning": tuple(raw_examples)},
        full_action_vocabularies=full_action_vocabularies,
        name_vocabularies=name_vocabularies,
        argument_vocabularies=argument_vocabularies,
        full_action_components=full_action_components,
        candidate_masks=candidate_masks,
    )["oscar_scope_reasoning"]


def subset_action_examples(
    examples: Sequence[DecisionActionExample],
    *,
    allowed_trace_steps: Sequence[str],
) -> tuple[DecisionActionExample, ...]:
    allowed = set(allowed_trace_steps)
    return tuple(example for example in examples if example.trace_step in allowed)


def remove_reasoning_fields(text: str, *, keys: Sequence[str]) -> str:
    prefixes = tuple(f"{key}=" for key in keys)
    lines = [line for line in text.splitlines() if not line.startswith(prefixes)]
    return "\n".join(lines) + "\n"


def sanitize_workflow_example(example):
    return replace(example, text=remove_reasoning_fields(example.text, keys=("auxiliary_targets",)))


def replace_reasoning_field(text: str, *, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = f"{key}="
    updated = False
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = prefix + value
            updated = True
            break
    if not updated:
        lines.append(prefix + value)
    return "\n".join(lines) + "\n"


def action_name_from_serialized(action_text: str) -> str:
    if action_text == "<start>":
        return "<start>"
    try:
        payload = json.loads(action_text)
    except json.JSONDecodeError:
        return action_text
    if isinstance(payload, dict):
        return str(payload.get("name", action_text))
    return action_text


def rollout_condition_example(
    example,
    *,
    state: OscarWorkflowRolloutState,
) -> object:
    conditioned = replace_reasoning_field(example.text, key="previous_action", value=state.previous_action)
    fields = parse_reasoning_fields(conditioned)
    candidate_bucket = fields.get("candidate_bucket", "")
    if "|prev:" in candidate_bucket:
        bucket_prefix = candidate_bucket.rsplit("|prev:", 1)[0]
        conditioned = replace_reasoning_field(
            conditioned,
            key="candidate_bucket",
            value=f"{bucket_prefix}|prev:{action_name_from_serialized(state.previous_action)}",
        )
    state_tokens = fields.get("state_tokens", "").split()
    filtered_tokens = [
        token
        for token in state_tokens
        if not (
            token.startswith("selected.kpi_family=")
            or token.startswith("selected.primary_intervention_family=")
        )
    ]
    if example.trace_step in {"apply_primary_intervention", "apply_followup_intervention"} and state.selected_kpi_family:
        filtered_tokens.append(f"selected.kpi_family={state.selected_kpi_family}")
    if example.trace_step == "apply_followup_intervention" and state.selected_primary_intervention_family:
        filtered_tokens.append(
            f"selected.primary_intervention_family={state.selected_primary_intervention_family}"
        )
    conditioned = replace_reasoning_field(
        conditioned,
        key="state_tokens",
        value=" ".join(filtered_tokens).strip(),
    )
    conditioned = remove_reasoning_fields(conditioned, keys=("auxiliary_targets",))
    return replace(example, text=conditioned)


def build_workflow_rollout_trajectories(examples) -> tuple[tuple[object, ...], ...]:
    grouped: dict[str, list[object]] = {}
    for example in examples:
        auxiliary_targets = example.auxiliary_targets or {}
        if str(auxiliary_targets.get("family", "")) != "oscar_workflow_intervention_trace":
            continue
        if parse_reasoning_fields(example.text).get("record_type") != "decision_action":
            continue
        grouped.setdefault(example.trajectory_id, []).append(sanitize_workflow_example(example))
    trajectories: list[tuple[object, ...]] = []
    for trajectory_examples in grouped.values():
        ordered = tuple(sorted(trajectory_examples, key=lambda item: item.step_index))
        expected = tuple(range(len(ordered)))
        actual = tuple(example.step_index for example in ordered)
        if actual != expected:
            continue
        trajectories.append(ordered)
    trajectories.sort(key=lambda trajectory: trajectory[0].trajectory_id if trajectory else "")
    return tuple(trajectories)


def scheduled_sampling_probability(
    *,
    train_step_index: int,
    total_steps: int,
    max_probability: float,
) -> float:
    if max_probability <= 0.0:
        return 0.0
    if total_steps <= 1:
        return max_probability
    progress = max(0.0, min(float(train_step_index) / float(total_steps - 1), 1.0))
    return progress * max_probability


def build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        attention_preset=args.attention_preset,
        latent_kv_dim=args.latent_kv_dim,
        num_experts=args.num_experts,
        experts_per_token=args.experts_per_token,
        router_jitter_noise=args.router_jitter_noise,
        moe_auxiliary_loss_weight=args.moe_auxiliary_loss_weight,
        core_max_rows=0,
        disable_core_auxiliary_heads=True,
        core_query_positive_loss_weight=0.0,
        core_source_count_loss_weight=0.0,
        core_trace_length_loss_weight=0.0,
        core_dependency_kind_loss_weight=0.0,
        core_infoflow_data_edge_loss_weight=0.0,
        core_source_membership_loss_weight=0.0,
        core_direct_edge_loss_weight=0.0,
        disable_oscar_auxiliary_heads=False,
        oscar_family_loss_weight=1.0,
        oscar_section_depth_loss_weight=0.25,
        oscar_doc_group_loss_weight=0.25,
        oscar_doc_title_loss_weight=0.25,
        oscar_section_path_loss_weight=0.25,
        oscar_concept_loss_weight=0.5,
        oscar_section_parent_loss_weight=0.5,
        oscar_related_doc_loss_weight=0.5,
        oscar_workflow_kpi_loss_weight=1.0,
        oscar_workflow_improvement_loss_weight=1.0,
        oscar_workflow_motif_loss_weight=1.0,
        oscar_workflow_reward_bucket_loss_weight=1.0,
        oscar_workflow_reward_score_loss_weight=1.0,
        oscar_workflow_action_step_loss_weight=args.oscar_workflow_action_step_loss_weight,
        oscar_workflow_action_kpi_loss_weight=args.oscar_workflow_action_kpi_loss_weight,
        oscar_workflow_action_intervention_loss_weight=args.oscar_workflow_action_intervention_loss_weight,
        disable_oscar_graph_auxiliary_heads=True,
        oscar_graph_family_loss_weight=0.0,
        oscar_graph_domain_loss_weight=0.0,
        oscar_graph_relation_loss_weight=0.0,
        oscar_graph_neighbor_loss_weight=0.0,
        oscar_graph_path_via_loss_weight=0.0,
        oscar_graph_path_target_loss_weight=0.0,
        oscar_graph_grounding_loss_weight=0.0,
        oscar_graph_rollout_motif_loss_weight=0.0,
        oscar_graph_rollout_step_loss_weight=0.0,
        disable_decision_action_heads=False,
        decision_action_projection_hidden_size=0,
        decision_action_loss_weight=args.decision_action_loss_weight,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
    )


def fit_tokenizer(args: argparse.Namespace, train_examples) -> object:
    return build_reasoning_tokenizer(
        texts_from_examples(train_examples),
        kind=args.tokenizer,
        vocab_size=args.tokenizer_vocab_size,
        task=args.tokenizer_task,
        min_freq=args.tokenizer_min_freq,
        candidate_pool_size=args.tokenizer_candidate_pool_size,
        max_piece_chars=args.tokenizer_max_piece_chars,
        fit_workers=args.tokenizer_fit_workers,
        fit_verbose=args.tokenizer_fit_verbose,
        load_path=args.tokenizer_load,
        save_path=args.tokenizer_save,
    )


def build_single_oscar_auxiliary_labels(
    example,
    *,
    vocabularies: dict[str, object],
    device: torch.device,
    state: OscarWorkflowRolloutState | None = None,
) -> dict[str, torch.Tensor]:
    structured = oscar_auxiliary_labels_for_example(example, vocabularies=vocabularies)
    if state is not None:
        canonical_kpi_to_id = dict(vocabularies.get("workflow_canonical_kpi_to_id", {}))
        canonical_intervention_to_id = dict(vocabularies.get("workflow_canonical_intervention_to_id", {}))
        if state.selected_kpi_family and state.selected_kpi_family in canonical_kpi_to_id:
            structured["workflow_canonical_kpi_context_id"] = canonical_kpi_to_id[state.selected_kpi_family]
            structured["workflow_canonical_kpi_context_active_mask"] = 1
        if (
            state.selected_primary_intervention_family
            and state.selected_primary_intervention_family in canonical_intervention_to_id
        ):
            structured["workflow_source_canonical_intervention_id"] = canonical_intervention_to_id[
                state.selected_primary_intervention_family
            ]
            structured["workflow_source_canonical_intervention_active_mask"] = 1
    labels = {
        name: torch.tensor([int(structured[name])], dtype=torch.long, device=device)
        for name in (
            "family_id",
            "section_depth_bucket",
            "doc_group_id",
            "doc_title_id",
            "section_path_id",
            "section_parent_id",
            "workflow_canonical_kpi_context_id",
            "workflow_source_canonical_intervention_id",
            "workflow_action_step_id",
            "workflow_action_kpi_family_id",
            "workflow_action_intervention_family_id",
            "workflow_kpi_id",
            "workflow_improvement_id",
            "workflow_motif_id",
            "workflow_reward_bucket_id",
        )
    }
    for name in (
        "concept_multihot",
        "related_doc_multihot",
        "workflow_active_bottleneck_multihot",
        "workflow_action_kpi_candidate_mask",
        "workflow_action_intervention_candidate_mask",
    ):
        labels[name] = torch.tensor([structured[name]], dtype=torch.float32, device=device)
    labels["workflow_reward_score"] = torch.tensor(
        [float(structured["workflow_reward_score"])],
        dtype=torch.float32,
        device=device,
    )
    for name in (
        "workflow_canonical_kpi_context_active_mask",
        "workflow_source_canonical_intervention_active_mask",
        "workflow_action_step_active_mask",
        "workflow_action_kpi_active_mask",
        "workflow_action_intervention_active_mask",
        "workflow_transfer_action_active_mask",
        "workflow_intervention_trace_action_active_mask",
        "workflow_kpi_active_mask",
        "workflow_improvement_active_mask",
        "workflow_motif_active_mask",
        "workflow_reward_active_mask",
    ):
        labels[name] = torch.tensor([float(structured[name])], dtype=torch.float32, device=device)
    return labels


def masked_argmax(logits: torch.Tensor, candidate_mask: torch.Tensor | None = None) -> int:
    row = logits[0]
    if candidate_mask is not None:
        mask = candidate_mask[0].to(dtype=torch.bool)
        if bool(torch.any(mask)):
            row = row.masked_fill(~mask, torch.finfo(row.dtype).min)
    return int(torch.argmax(row).item())


def decode_oscar_action_prediction_from_outputs(
    model,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: dict[str, torch.Tensor],
    vocabularies: dict[str, object],
    trace_step: str,
) -> dict[str, object]:
    if model.oscar_auxiliary_heads is None:
        return {}
    heads = model.oscar_auxiliary_heads
    with torch.no_grad():
        pooled = heads.pooled_hidden_state(hidden_states.detach(), attention_mask=attention_mask)
        features = heads.proj(pooled)
        step_features = heads.workflow_structured_features(
            features,
            labels=labels,
            include_action_step=False,
            include_kpi_context=False,
            include_source_intervention=False,
        )
        step_logits = heads.workflow_action_step_head(step_features)
        predicted_step_id = masked_argmax(step_logits)
        kpi_features = heads.workflow_structured_features(
            features,
            labels=labels,
            include_action_step=True,
            include_kpi_context=False,
            include_source_intervention=False,
        )
        kpi_logits = heads.workflow_action_kpi_head(kpi_features)
        predicted_kpi_id = masked_argmax(kpi_logits, labels["workflow_action_kpi_candidate_mask"])
        intervention_features = heads.workflow_structured_features(
            features,
            labels=labels,
            include_action_step=True,
            include_kpi_context=True,
            include_source_intervention=True,
        )
        intervention_logits = heads.workflow_action_intervention_head(intervention_features)
        predicted_intervention_id = masked_argmax(
            intervention_logits,
            labels["workflow_action_intervention_candidate_mask"],
        )
    canonical_kpis = tuple(vocabularies.get("workflow_canonical_kpi_ids", ()))
    canonical_interventions = tuple(vocabularies.get("workflow_canonical_intervention_ids", ()))
    action_step_ids = tuple(vocabularies.get("workflow_action_step_ids", ()))
    predicted_step_label = action_step_ids[predicted_step_id] if predicted_step_id < len(action_step_ids) else "none"
    predicted_kpi_family = canonical_kpis[predicted_kpi_id] if predicted_kpi_id < len(canonical_kpis) else ""
    predicted_intervention_family = (
        canonical_interventions[predicted_intervention_id]
        if predicted_intervention_id < len(canonical_interventions)
        else ""
    )
    gold_step_id = int(labels["workflow_action_step_id"][0].item())
    gold_kpi_id = int(labels["workflow_action_kpi_family_id"][0].item())
    gold_intervention_id = int(labels["workflow_action_intervention_family_id"][0].item())
    step_correct = predicted_step_id == gold_step_id
    if predicted_step_label == "select_kpi_family":
        action_name = "select_kpi_family"
        predicted_payload = {"kpi_family_id": predicted_kpi_family}
    elif predicted_step_label == "select_intervention_family":
        action_name = "select_intervention_family"
        predicted_payload = {"intervention_family_id": predicted_intervention_family}
    elif trace_step == "select_focus_kpi":
        action_name = "select_kpi_family"
        predicted_payload = {"kpi_family_id": predicted_kpi_family}
    else:
        action_name = "select_intervention_family"
        predicted_payload = {"intervention_family_id": predicted_intervention_family}
    if trace_step == "select_focus_kpi":
        target_correct = predicted_kpi_id == gold_kpi_id
    else:
        target_correct = predicted_intervention_id == gold_intervention_id
    predicted_action = json.dumps(
        {"action": predicted_payload, "name": action_name},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "predicted_action": predicted_action,
        "predicted_step_label": predicted_step_label,
        "predicted_kpi_family": predicted_kpi_family,
        "predicted_intervention_family": predicted_intervention_family,
        "step_correct": step_correct,
        "target_correct": target_correct,
        "exact_match": step_correct and target_correct,
    }


def predict_oscar_rollout_step(
    model,
    *,
    example,
    vocabularies: dict[str, object],
    device: torch.device,
    effort: str,
    state: OscarWorkflowRolloutState,
    seq_len: int,
) -> dict[str, object]:
    if model.oscar_auxiliary_heads is None:
        return {}
    conditioned_example = rollout_condition_example(example, state=state)
    encoded_tokens, encoded_mask = encode_example_sequence(conditioned_example, seq_len=seq_len)
    token_batch = torch.tensor([encoded_tokens], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded_mask], dtype=torch.long, device=device)
    inputs = token_batch[:, :-1]
    mask = attention_mask[:, :-1]
    policy = reasoning_budget_policy_for_benchmark(
        "oscar_scope_reasoning",
        effort=effort,
        attention_window=model.config.attention.sliding_window,
    )
    budget = policy.build_inference_budget(
        model.config,
        prompt_tokens=inputs.size(1),
        use_kv_cache=False,
        max_new_tokens=0,
    )
    if budget.max_prompt_tokens is not None and inputs.size(1) > budget.max_prompt_tokens:
        inputs = inputs[:, -budget.max_prompt_tokens :]
        mask = mask[:, -budget.max_prompt_tokens :]
    labels = build_single_oscar_auxiliary_labels(
        conditioned_example,
        vocabularies=vocabularies,
        device=device,
        state=state,
    )
    with torch.no_grad():
        outputs = model(inputs, attention_mask=mask, budget=budget)
    prediction = decode_oscar_action_prediction_from_outputs(
        model,
        hidden_states=outputs.last_hidden_state,
        attention_mask=mask,
        labels=labels,
        vocabularies=vocabularies,
        trace_step=example.trace_step,
    )
    prediction["example"] = conditioned_example
    return prediction


def workflow_rollout_state_after_action(
    *,
    state: OscarWorkflowRolloutState,
    trace_step: str,
    action_text: str,
    example=None,
) -> OscarWorkflowRolloutState:
    selected_kpi_family = state.selected_kpi_family
    selected_primary_intervention_family = state.selected_primary_intervention_family
    payload: dict[str, object] = {}
    if action_text and action_text != "<start>":
        try:
            parsed = json.loads(action_text)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            inner = parsed.get("action", {})
            if isinstance(inner, dict):
                payload = inner
    auxiliary_targets = (example.auxiliary_targets or {}) if example is not None else {}
    if trace_step == "select_focus_kpi":
        selected_kpi_family = str(
            payload.get(
                "kpi_family_id",
                auxiliary_targets.get("workflow_canonical_kpi_id", selected_kpi_family or ""),
            )
        ).strip() or selected_kpi_family
    elif trace_step == "apply_primary_intervention":
        selected_primary_intervention_family = str(
            payload.get(
                "intervention_family_id",
                auxiliary_targets.get(
                    "workflow_canonical_primary_improvement_id",
                    selected_primary_intervention_family or "",
                ),
            )
        ).strip() or selected_primary_intervention_family
    return OscarWorkflowRolloutState(
        previous_action=action_text or "<start>",
        selected_kpi_family=selected_kpi_family,
        selected_primary_intervention_family=selected_primary_intervention_family,
    )


def rollout_action_prediction_from_outputs(
    model,
    *,
    outputs,
    attention_mask: torch.Tensor,
    labels: dict[str, torch.Tensor],
    vocabularies: dict[str, object],
    trace_step: str,
) -> dict[str, object]:
    return decode_oscar_action_prediction_from_outputs(
        model,
        hidden_states=outputs.last_hidden_state,
        attention_mask=attention_mask,
        labels=labels,
        vocabularies=vocabularies,
        trace_step=trace_step,
    )


def sample_workflow_rollout_training_loss(
    model,
    *,
    trajectories: Sequence[tuple[object, ...]],
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
    effort: str,
    vocabularies: dict[str, object],
    train_step_index: int,
    total_steps: int,
    max_scheduled_sampling_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not trajectories or model.oscar_auxiliary_heads is None or batch_size <= 0:
        zero = next(model.parameters()).new_zeros(())
        return zero, zero
    if batch_size >= len(trajectories):
        selected = [rng.choice(trajectories) for _ in range(batch_size)]
    else:
        selected = rng.sample(list(trajectories), batch_size)
    sampling_probability = scheduled_sampling_probability(
        train_step_index=train_step_index,
        total_steps=total_steps,
        max_probability=max_scheduled_sampling_probability,
    )
    rollout_main_losses: list[torch.Tensor] = []
    rollout_aux_losses: list[torch.Tensor] = []
    for trajectory in selected:
        state = OscarWorkflowRolloutState()
        step_main_losses: list[torch.Tensor] = []
        step_aux_losses: list[torch.Tensor] = []
        for example in trajectory:
            conditioned_example = rollout_condition_example(example, state=state)
            encoded_tokens, encoded_mask = encode_example_sequence(conditioned_example, seq_len=seq_len)
            token_batch = torch.tensor([encoded_tokens], dtype=torch.long, device=device)
            attention_batch = torch.tensor([encoded_mask], dtype=torch.long, device=device)
            inputs = token_batch[:, :-1]
            targets = token_batch[:, 1:]
            attention_mask = attention_batch[:, :-1]
            policy = reasoning_budget_policy_for_benchmark(
                "oscar_scope_reasoning",
                effort=effort,
                attention_window=model.config.attention.sliding_window,
            )
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            inputs, targets = trim_batch_to_budget(inputs, targets, budget.max_prompt_tokens)
            attention_mask = attention_mask[:, -inputs.size(1) :]
            labels = build_single_oscar_auxiliary_labels(
                conditioned_example,
                vocabularies=vocabularies,
                device=device,
                state=state,
            )
            outputs = model(
                inputs,
                attention_mask=attention_mask,
                budget=budget,
                task_name="oscar_scope_reasoning",
                task_auxiliary_labels=labels,
            )
            step_main_losses.append(cross_entropy_loss(outputs.logits, targets))
            step_aux_losses.append(
                (outputs.task_auxiliary_loss if outputs.task_auxiliary_loss is not None else inputs.new_zeros((), dtype=torch.float32))
                + (outputs.auxiliary_loss if outputs.auxiliary_loss is not None else inputs.new_zeros((), dtype=torch.float32))
            )
            prediction = rollout_action_prediction_from_outputs(
                model,
                outputs=outputs,
                attention_mask=attention_mask,
                labels=labels,
                vocabularies=vocabularies,
                trace_step=example.trace_step,
            )
            gold_fields = parse_reasoning_fields(example.text)
            gold_action = str(gold_fields.get("target_action", "<start>"))
            use_predicted = rng.random() < sampling_probability
            if use_predicted:
                state = workflow_rollout_state_after_action(
                    state=state,
                    trace_step=example.trace_step,
                    action_text=str(prediction.get("predicted_action", "<start>")),
                    example=example,
                )
            else:
                state = workflow_rollout_state_after_action(
                    state=state,
                    trace_step=example.trace_step,
                    action_text=gold_action,
                    example=example,
                )
        if step_main_losses:
            rollout_main_losses.append(torch.stack(step_main_losses).mean())
            rollout_aux_losses.append(torch.stack(step_aux_losses).mean())
    if not rollout_main_losses:
        zero = next(model.parameters()).new_zeros(())
        return zero, zero
    return torch.stack(rollout_main_losses).mean(), torch.stack(rollout_aux_losses).mean()


def evaluate_oscar_workflow_rollouts(
    model,
    examples,
    *,
    device: torch.device,
    effort: str,
    vocabularies: dict[str, object],
    seq_len: int,
) -> dict[str, float]:
    workflow_examples = [
        example
        for example in examples
        if str((example.auxiliary_targets or {}).get("family", "")) == "oscar_workflow_intervention_trace"
        and parse_reasoning_fields(example.text).get("record_type") == "decision_action"
    ]
    if not workflow_examples or model.oscar_auxiliary_heads is None:
        return {}
    trajectories: dict[str, list[object]] = {}
    for example in workflow_examples:
        trajectories.setdefault(example.trajectory_id, []).append(example)
    total_steps = 0
    step_correct = 0
    kpi_steps = 0
    kpi_correct = 0
    intervention_steps = 0
    intervention_correct = 0
    exact_trajectories = 0
    total_trajectories = 0
    for trajectory_id, trajectory_examples in sorted(trajectories.items()):
        _ = trajectory_id
        ordered = sorted(trajectory_examples, key=lambda item: item.step_index)
        state = OscarWorkflowRolloutState()
        trajectory_exact = True
        for example in ordered:
            prediction = predict_oscar_rollout_step(
                model,
                example=example,
                vocabularies=vocabularies,
                device=device,
                effort=effort,
                state=state,
                seq_len=seq_len,
            )
            if not prediction:
                continue
            total_steps += 1
            step_correct += int(bool(prediction["step_correct"]))
            if example.trace_step == "select_focus_kpi":
                kpi_steps += 1
                kpi_correct += int(bool(prediction["target_correct"]))
                state = OscarWorkflowRolloutState(
                    previous_action=str(prediction["predicted_action"]),
                    selected_kpi_family=str(prediction["predicted_kpi_family"]) or None,
                    selected_primary_intervention_family=None,
                )
            else:
                intervention_steps += 1
                intervention_correct += int(bool(prediction["target_correct"]))
                state = OscarWorkflowRolloutState(
                    previous_action=str(prediction["predicted_action"]),
                    selected_kpi_family=state.selected_kpi_family,
                    selected_primary_intervention_family=(
                        str(prediction["predicted_intervention_family"]) or None
                        if example.trace_step == "apply_primary_intervention"
                        else state.selected_primary_intervention_family
                    ),
                )
            trajectory_exact = trajectory_exact and bool(prediction["exact_match"])
        total_trajectories += 1
        exact_trajectories += int(trajectory_exact and bool(ordered))
    return {
        "oscar_workflow_rollout_step_accuracy": (step_correct / total_steps) if total_steps else 0.0,
        "oscar_workflow_rollout_kpi_accuracy": (kpi_correct / kpi_steps) if kpi_steps else 0.0,
        "oscar_workflow_rollout_intervention_accuracy": (
            intervention_correct / intervention_steps
        ) if intervention_steps else 0.0,
        "oscar_workflow_rollout_trajectory_exact_match": (
            exact_trajectories / total_trajectories
        ) if total_trajectories else 0.0,
        "oscar_workflow_rollout_trajectory_count": float(total_trajectories),
    }


def train_for_holdout(
    *,
    args: argparse.Namespace,
    device: torch.device,
    train_examples,
    train_action_reasoning_examples,
    train_rollout_trajectories: tuple[tuple[object, ...], ...],
    train_action_examples: tuple[DecisionActionExample, ...],
    tokenizer,
    model_config,
    oscar_auxiliary_vocabularies: dict[str, object],
    decision_full_action_vocabularies: dict[str, tuple[str, ...]],
    decision_name_vocabularies: dict[str, tuple[str, ...]],
    decision_argument_vocabularies: dict[str, tuple[str, ...]],
    seed: int,
) -> dict[str, object]:
    set_text_tokenizer(tokenizer)
    model = DecoderLanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(seed)
    train_policy = reasoning_budget_policy_for_benchmark(
        "oscar_scope_reasoning",
        effort=args.train_reasoning_effort,
        attention_window=model.config.attention.sliding_window,
    )
    train_loss_history: list[float] = []
    train_task_aux_history: list[float] = []
    train_decision_history: list[float] = []
    train_action_focus_history: list[float] = []
    train_rollout_main_history: list[float] = []
    train_rollout_aux_history: list[float] = []
    train_router_entropy_history: list[float] = []
    for step_index in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs, targets, attention_mask, task_auxiliary_labels = sample_oscar_auxiliary_batch(
            train_examples,
            batch_size=args.batch_size,
            rng=rng,
            device=device,
            seq_len=args.seq_len,
            vocabularies=oscar_auxiliary_vocabularies,
        )
        train_budget = train_policy.build_inference_budget(
            model.config,
            prompt_tokens=inputs.size(1),
            use_kv_cache=False,
            max_new_tokens=0,
        )
        inputs, targets = trim_batch_to_budget(inputs, targets, train_budget.max_prompt_tokens)
        attention_mask = attention_mask[:, -inputs.size(1) :]
        outputs = model(
            inputs,
            attention_mask=attention_mask,
            budget=train_budget,
            task_name="oscar_scope_reasoning",
            task_auxiliary_labels=task_auxiliary_labels,
        )
        main_loss = cross_entropy_loss(outputs.logits, targets)
        router_aux_loss = outputs.auxiliary_loss if outputs.auxiliary_loss is not None else main_loss.new_zeros(())
        task_aux_loss = outputs.task_auxiliary_loss if outputs.task_auxiliary_loss is not None else main_loss.new_zeros(())
        decision_aux_loss = main_loss.new_zeros(())
        if train_action_examples and model.decision_action_heads is not None:
            decision_inputs, decision_attention_mask, decision_labels = sample_decision_action_batch(
                train_action_examples,
                batch_size=args.decision_batch_size,
                rng=rng,
                device=device,
                seq_len=args.seq_len,
                full_action_vocabularies=decision_full_action_vocabularies,
                name_vocabularies=decision_name_vocabularies,
                argument_vocabularies=decision_argument_vocabularies,
            )
            decision_budget = train_policy.build_inference_budget(
                model.config,
                prompt_tokens=decision_inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            if decision_budget.max_prompt_tokens is not None and decision_inputs.size(1) > decision_budget.max_prompt_tokens:
                decision_inputs = decision_inputs[:, -decision_budget.max_prompt_tokens :]
                decision_attention_mask = decision_attention_mask[:, -decision_budget.max_prompt_tokens :]
            decision_outputs = model(
                decision_inputs,
                attention_mask=decision_attention_mask,
                budget=decision_budget,
                task_name="decision_action",
                task_auxiliary_labels=decision_labels,
            )
            decision_aux_loss = (
                decision_outputs.task_auxiliary_loss
                if decision_outputs.task_auxiliary_loss is not None
                else main_loss.new_zeros(())
            )
        action_focus_aux_loss = main_loss.new_zeros(())
        if train_action_reasoning_examples:
            action_inputs, _action_targets, action_attention_mask, action_task_labels = sample_oscar_auxiliary_batch(
                train_action_reasoning_examples,
                batch_size=args.decision_batch_size,
                rng=rng,
                device=device,
                seq_len=args.seq_len,
                vocabularies=oscar_auxiliary_vocabularies,
            )
            action_budget = train_policy.build_inference_budget(
                model.config,
                prompt_tokens=action_inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            action_inputs, _action_targets = trim_batch_to_budget(
                action_inputs,
                _action_targets,
                action_budget.max_prompt_tokens,
            )
            action_attention_mask = action_attention_mask[:, -action_inputs.size(1) :]
            action_outputs = model(
                action_inputs,
                attention_mask=action_attention_mask,
                budget=action_budget,
                task_name="oscar_scope_reasoning",
                task_auxiliary_labels=action_task_labels,
            )
            action_focus_aux_loss = (
                action_outputs.task_auxiliary_loss
                if action_outputs.task_auxiliary_loss is not None
                else main_loss.new_zeros(())
            )
        rollout_main_loss = main_loss.new_zeros(())
        rollout_aux_loss = main_loss.new_zeros(())
        if train_rollout_trajectories and args.workflow_rollout_train_scale > 0.0:
            rollout_main_loss, rollout_aux_loss = sample_workflow_rollout_training_loss(
                model,
                trajectories=train_rollout_trajectories,
                batch_size=args.workflow_rollout_batch_size,
                rng=rng,
                device=device,
                seq_len=args.seq_len,
                effort=args.train_reasoning_effort,
                vocabularies=oscar_auxiliary_vocabularies,
                train_step_index=step_index,
                total_steps=args.steps,
                max_scheduled_sampling_probability=args.workflow_scheduled_sampling_max_prob,
            )
        total_loss = (
            main_loss
            + router_aux_loss
            + task_aux_loss
            + (decision_aux_loss * args.decision_action_train_scale)
            + (action_focus_aux_loss * args.workflow_action_aux_train_scale)
            + ((rollout_main_loss + rollout_aux_loss) * args.workflow_rollout_train_scale)
        )
        total_loss.backward()
        if args.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        synchronize_device(device)
        train_loss_history.append(float(main_loss.detach().item()))
        train_task_aux_history.append(float(task_aux_loss.detach().item()))
        train_decision_history.append(float(decision_aux_loss.detach().item()))
        train_action_focus_history.append(float(action_focus_aux_loss.detach().item()))
        train_rollout_main_history.append(float(rollout_main_loss.detach().item()))
        train_rollout_aux_history.append(float(rollout_aux_loss.detach().item()))
        if outputs.router_entropy is not None:
            train_router_entropy_history.append(float(outputs.router_entropy.detach().item()))
    return {
        "model": model,
        "train_main_loss_mean": statistics.fmean(train_loss_history) if train_loss_history else float("nan"),
        "train_task_aux_loss_mean": statistics.fmean(train_task_aux_history) if train_task_aux_history else float("nan"),
        "train_decision_aux_loss_mean": (
            statistics.fmean(train_decision_history) if train_decision_history else float("nan")
        ),
        "train_action_focus_aux_loss_mean": (
            statistics.fmean(train_action_focus_history) if train_action_focus_history else float("nan")
        ),
        "train_rollout_main_loss_mean": (
            statistics.fmean(train_rollout_main_history) if train_rollout_main_history else float("nan")
        ),
        "train_rollout_aux_loss_mean": (
            statistics.fmean(train_rollout_aux_history) if train_rollout_aux_history else float("nan")
        ),
        "train_router_entropy_mean": (
            statistics.fmean(train_router_entropy_history) if train_router_entropy_history else float("nan")
        ),
    }


def select_holdout_environments(
    examples,
    requested: Sequence[str],
) -> tuple[str, ...]:
    discovered = tuple(sorted({environment for example in examples for environment in example_environment_mentions(example)}))
    if not requested or "all" in requested:
        return discovered
    requested_set = tuple(dict.fromkeys(requested))
    missing = [environment for environment in requested_set if environment not in discovered]
    if missing:
        raise SystemExit(
            "Unknown holdout environment(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(discovered)
        )
    return requested_set


def summarize_holdout(
    *,
    holdout_environment: str,
    train_examples,
    eval_examples,
    raw_eval_action_examples: tuple[DecisionActionExample, ...],
    covered_eval_action_examples: tuple[DecisionActionExample, ...],
    raw_transfer_action_examples: tuple[DecisionActionExample, ...],
    covered_transfer_action_examples: tuple[DecisionActionExample, ...],
    raw_intervention_action_examples: tuple[DecisionActionExample, ...],
    covered_intervention_action_examples: tuple[DecisionActionExample, ...],
    train_metrics: dict[str, object],
    val_loss: float,
    oscar_aux_eval_metrics: dict[str, float],
    rollout_eval_metrics: dict[str, float],
    decision_action_eval_metrics: dict[str, float],
    transfer_decision_eval_metrics: dict[str, float],
    intervention_decision_eval_metrics: dict[str, float],
    tokenizer,
    model,
    output_dir: Path,
) -> dict[str, object]:
    row = {
        "holdout_environment": holdout_environment,
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "parameter_count": count_parameters(model),
        "tokenizer_kind": tokenizer.kind,
        "train_main_loss_mean": train_metrics["train_main_loss_mean"],
        "train_task_aux_loss_mean": train_metrics["train_task_aux_loss_mean"],
        "train_decision_aux_loss_mean": train_metrics["train_decision_aux_loss_mean"],
        "train_action_focus_aux_loss_mean": train_metrics["train_action_focus_aux_loss_mean"],
        "train_rollout_main_loss_mean": train_metrics["train_rollout_main_loss_mean"],
        "train_rollout_aux_loss_mean": train_metrics["train_rollout_aux_loss_mean"],
        "heldout_lm_loss": val_loss,
        "decision_action_coverage": (
            len(covered_eval_action_examples) / len(raw_eval_action_examples)
            if raw_eval_action_examples else 0.0
        ),
        "transfer_action_coverage": (
            len(covered_transfer_action_examples) / len(raw_transfer_action_examples)
            if raw_transfer_action_examples else 0.0
        ),
        "intervention_action_coverage": (
            len(covered_intervention_action_examples) / len(raw_intervention_action_examples)
            if raw_intervention_action_examples else 0.0
        ),
        "heldout_decision_action_accuracy": decision_action_eval_metrics.get(
            "oscar_scope_reasoning_decision_action_accuracy",
            float("nan"),
        ),
        "heldout_transfer_action_accuracy": transfer_decision_eval_metrics.get(
            "oscar_scope_reasoning_decision_action_accuracy",
            float("nan"),
        ),
        "heldout_intervention_action_accuracy": intervention_decision_eval_metrics.get(
            "oscar_scope_reasoning_decision_action_accuracy",
            float("nan"),
        ),
    }
    for metric_name in WORKFLOW_METRIC_KEYS:
        row[metric_name] = rollout_eval_metrics.get(
            metric_name,
            oscar_aux_eval_metrics.get(metric_name, float("nan")),
        )
    return {
        "row": row,
        "holdout_environment": holdout_environment,
        "holdout_slug": slugify(holdout_environment),
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "train_family_counts": family_counts(train_examples),
        "eval_family_counts": family_counts(eval_examples),
        "train_environments": sorted({environment for example in train_examples for environment in example_environment_mentions(example)}),
        "eval_environment_mentions": sorted({environment for example in eval_examples for environment in example_environment_mentions(example)}),
        "raw_eval_action_count": len(raw_eval_action_examples),
        "covered_eval_action_count": len(covered_eval_action_examples),
        "raw_transfer_action_count": len(raw_transfer_action_examples),
        "covered_transfer_action_count": len(covered_transfer_action_examples),
        "raw_intervention_action_count": len(raw_intervention_action_examples),
        "covered_intervention_action_count": len(covered_intervention_action_examples),
        "eval_action_step_counts": action_step_counts(covered_eval_action_examples),
        "train_metrics": {key: value for key, value in train_metrics.items() if key != "model"},
        "heldout_lm_loss": val_loss,
        "oscar_aux_eval_metrics": oscar_aux_eval_metrics,
        "rollout_eval_metrics": rollout_eval_metrics,
        "decision_action_eval_metrics": decision_action_eval_metrics,
        "transfer_decision_eval_metrics": transfer_decision_eval_metrics,
        "intervention_decision_eval_metrics": intervention_decision_eval_metrics,
        "artifacts_dir": str(output_dir.resolve()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_or_nan(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if float(value) == float(value)]
    if not filtered:
        return float("nan")
    return statistics.fmean(filtered)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = build_oscar_scope_reasoning_examples(
        auto_discover=True,
        max_documents=args.max_documents,
        max_examples=args.max_examples,
        families=args.families,
    )
    if not all_examples:
        raise SystemExit("No Oscar workflow reasoning examples were discovered.")
    holdout_environments = select_holdout_environments(all_examples, args.holdout_environments)
    rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []

    for holdout_index, holdout_environment in enumerate(holdout_environments):
        train_examples, eval_examples = split_examples_by_holdout(
            all_examples,
            holdout_environment=holdout_environment,
        )
        if not train_examples or not eval_examples:
            runs.append(
                {
                    "holdout_environment": holdout_environment,
                    "skipped": True,
                    "reason": "empty_train_or_eval_split",
                    "train_example_count": len(train_examples),
                    "eval_example_count": len(eval_examples),
                }
            )
            continue

        tokenizer = fit_tokenizer(args, train_examples)
        set_text_tokenizer(tokenizer)
        train_windows = chunk_token_stream(texts_from_examples(train_examples), seq_len=args.seq_len)
        eval_windows = chunk_token_stream(texts_from_examples(eval_examples), seq_len=args.seq_len)
        if not train_windows or not eval_windows:
            runs.append(
                {
                    "holdout_environment": holdout_environment,
                    "skipped": True,
                    "reason": "empty_token_windows",
                    "train_window_count": len(train_windows),
                    "eval_window_count": len(eval_windows),
                }
            )
            continue

        decision_train_artifacts = build_decision_train_artifacts(train_examples)
        train_action_reasoning_examples = tuple(
            example
            for example in train_examples
            if parse_reasoning_fields(example.text).get("record_type") == "decision_action"
        )
        train_rollout_trajectories = build_workflow_rollout_trajectories(train_examples)
        raw_eval_action_examples = convert_decision_action_examples(eval_examples)[0]
        covered_eval_raw_examples, eval_action_coverage = filter_covered_decision_examples(
            raw_eval_action_examples,
            full_action_vocabularies=decision_train_artifacts["decision_full_action_vocabularies"],
            name_vocabularies=decision_train_artifacts["decision_name_vocabularies"],
            argument_vocabularies=decision_train_artifacts["decision_argument_vocabularies"],
        )
        covered_eval_action_examples = finalize_eval_action_examples(
            covered_eval_raw_examples,
            full_action_vocabularies=decision_train_artifacts["decision_full_action_vocabularies"],
            name_vocabularies=decision_train_artifacts["decision_name_vocabularies"],
            argument_vocabularies=decision_train_artifacts["decision_argument_vocabularies"],
            full_action_components=decision_train_artifacts["decision_full_action_components"],
            candidate_masks=decision_train_artifacts["decision_candidate_masks"],
        )
        raw_transfer_action_examples = subset_action_examples(
            raw_eval_action_examples,
            allowed_trace_steps=(TRACE_STEP_TRANSFER,),
        )
        covered_transfer_action_examples = subset_action_examples(
            covered_eval_action_examples,
            allowed_trace_steps=(TRACE_STEP_TRANSFER,),
        )
        raw_intervention_action_examples = subset_action_examples(
            raw_eval_action_examples,
            allowed_trace_steps=INTERVENTION_TRACE_STEPS,
        )
        covered_intervention_action_examples = subset_action_examples(
            covered_eval_action_examples,
            allowed_trace_steps=INTERVENTION_TRACE_STEPS,
        )

        oscar_auxiliary_vocabularies = build_oscar_auxiliary_vocabularies(train_examples)
        model_args = build_model_args(args)
        model_config = build_model_config(
            model_args,
            architecture=args.architecture,
            vocab_size=tokenizer.vocab_size,
            decision_benchmark_adapter_names=decision_train_artifacts["decision_benchmark_adapter_names"],
            decision_output_sizes={
                name: len(vocabulary)
                for name, vocabulary in decision_train_artifacts["decision_full_action_vocabularies"].items()
            },
            decision_name_output_sizes={
                name: len(vocabulary)
                for name, vocabulary in decision_train_artifacts["decision_name_vocabularies"].items()
            },
            decision_argument_output_sizes={
                name: len(vocabulary)
                for name, vocabulary in decision_train_artifacts["decision_argument_vocabularies"].items()
            },
            oscar_auxiliary_vocabularies=oscar_auxiliary_vocabularies,
            oscar_graph_auxiliary_vocabularies={},
        )
        train_metrics = train_for_holdout(
            args=args,
            device=device,
            train_examples=train_examples,
            train_action_reasoning_examples=train_action_reasoning_examples,
            train_rollout_trajectories=train_rollout_trajectories,
            train_action_examples=decision_train_artifacts["train_action_examples"].get("oscar_scope_reasoning", ()),
            tokenizer=tokenizer,
            model_config=model_config,
            oscar_auxiliary_vocabularies=oscar_auxiliary_vocabularies,
            decision_full_action_vocabularies=decision_train_artifacts["decision_full_action_vocabularies"],
            decision_name_vocabularies=decision_train_artifacts["decision_name_vocabularies"],
            decision_argument_vocabularies=decision_train_artifacts["decision_argument_vocabularies"],
            seed=args.seed + (holdout_index * 997),
        )
        model = train_metrics["model"]
        val_loss = evaluate_benchmark_loss(
            model,
            eval_windows,
            batch_size=args.batch_size,
            device=device,
            benchmark_name="oscar_scope_reasoning",
            effort=args.eval_reasoning_effort,
        )
        oscar_aux_eval_metrics = evaluate_oscar_auxiliary(
            model,
            eval_examples,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            effort=args.eval_reasoning_effort,
            vocabularies=oscar_auxiliary_vocabularies,
        )
        rollout_eval_metrics = evaluate_oscar_workflow_rollouts(
            model,
            eval_examples,
            device=device,
            effort=args.eval_reasoning_effort,
            vocabularies=oscar_auxiliary_vocabularies,
            seq_len=args.seq_len,
        )
        decision_action_eval_metrics = evaluate_decision_action_accuracy(
            model,
            {"oscar_scope_reasoning": covered_eval_action_examples},
            batch_size=args.decision_batch_size,
            seq_len=args.seq_len,
            device=device,
            full_action_vocabularies=decision_train_artifacts["decision_full_action_vocabularies"],
            benchmark_efforts={"oscar_scope_reasoning": args.eval_reasoning_effort},
        ) if covered_eval_action_examples else {}
        transfer_decision_eval_metrics = evaluate_decision_action_accuracy(
            model,
            {"oscar_scope_reasoning": covered_transfer_action_examples},
            batch_size=args.decision_batch_size,
            seq_len=args.seq_len,
            device=device,
            full_action_vocabularies=decision_train_artifacts["decision_full_action_vocabularies"],
            benchmark_efforts={"oscar_scope_reasoning": args.eval_reasoning_effort},
        ) if covered_transfer_action_examples else {}
        intervention_decision_eval_metrics = evaluate_decision_action_accuracy(
            model,
            {"oscar_scope_reasoning": covered_intervention_action_examples},
            batch_size=args.decision_batch_size,
            seq_len=args.seq_len,
            device=device,
            full_action_vocabularies=decision_train_artifacts["decision_full_action_vocabularies"],
            benchmark_efforts={"oscar_scope_reasoning": args.eval_reasoning_effort},
        ) if covered_intervention_action_examples else {}
        run_summary = summarize_holdout(
            holdout_environment=holdout_environment,
            train_examples=train_examples,
            eval_examples=eval_examples,
            raw_eval_action_examples=raw_eval_action_examples,
            covered_eval_action_examples=covered_eval_action_examples,
            raw_transfer_action_examples=raw_transfer_action_examples,
            covered_transfer_action_examples=covered_transfer_action_examples,
            raw_intervention_action_examples=raw_intervention_action_examples,
            covered_intervention_action_examples=covered_intervention_action_examples,
            train_metrics=train_metrics,
            val_loss=val_loss,
            oscar_aux_eval_metrics=oscar_aux_eval_metrics,
            rollout_eval_metrics=rollout_eval_metrics,
            decision_action_eval_metrics=decision_action_eval_metrics,
            transfer_decision_eval_metrics=transfer_decision_eval_metrics,
            intervention_decision_eval_metrics=intervention_decision_eval_metrics,
            tokenizer=tokenizer,
            model=model,
            output_dir=output_dir,
        )
        run_summary["decision_action_coverage"] = eval_action_coverage
        run_summary["transfer_action_coverage"] = {
            "total": len(raw_transfer_action_examples),
            "covered": len(covered_transfer_action_examples),
            "coverage": (
                len(covered_transfer_action_examples) / len(raw_transfer_action_examples)
                if raw_transfer_action_examples else 0.0
            ),
        }
        run_summary["intervention_action_coverage"] = {
            "total": len(raw_intervention_action_examples),
            "covered": len(covered_intervention_action_examples),
            "coverage": (
                len(covered_intervention_action_examples) / len(raw_intervention_action_examples)
                if raw_intervention_action_examples else 0.0
            ),
        }
        rows.append(run_summary["row"])
        runs.append(run_summary)
        del train_metrics["model"]
        del model
        synchronize_device(device)

    aggregate_rows = [row for row in rows if isinstance(row, dict)]
    aggregate = {
        "holdout_count": len(aggregate_rows),
        "mean_heldout_lm_loss": mean_or_nan([float(row["heldout_lm_loss"]) for row in aggregate_rows]),
        "mean_oscar_workflow_motif_accuracy": mean_or_nan(
            [float(row["oscar_workflow_motif_accuracy"]) for row in aggregate_rows]
        ),
        "mean_oscar_workflow_action_exact_match": mean_or_nan(
            [float(row["oscar_workflow_action_exact_match"]) for row in aggregate_rows]
        ),
        "mean_oscar_workflow_rollout_trajectory_exact_match": mean_or_nan(
            [float(row["oscar_workflow_rollout_trajectory_exact_match"]) for row in aggregate_rows]
        ),
        "mean_oscar_workflow_transfer_intervention_accuracy": mean_or_nan(
            [float(row["oscar_workflow_transfer_intervention_accuracy"]) for row in aggregate_rows]
        ),
        "mean_transfer_action_accuracy": mean_or_nan(
            [float(row["heldout_transfer_action_accuracy"]) for row in aggregate_rows]
        ),
        "mean_transfer_action_coverage": mean_or_nan(
            [float(row["transfer_action_coverage"]) for row in aggregate_rows]
        ),
    }
    summary = {
        "requested": {
            "families": list(args.families),
            "holdout_environments": list(holdout_environments),
            "max_examples": args.max_examples,
            "max_documents": args.max_documents,
            "architecture": args.architecture,
            "attention_preset": args.attention_preset,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "decision_batch_size": args.decision_batch_size,
            "seq_len": args.seq_len,
            "train_reasoning_effort": args.train_reasoning_effort,
            "eval_reasoning_effort": args.eval_reasoning_effort,
            "decision_action_loss_weight": args.decision_action_loss_weight,
            "decision_action_train_scale": args.decision_action_train_scale,
            "workflow_action_aux_train_scale": args.workflow_action_aux_train_scale,
            "workflow_rollout_train_scale": args.workflow_rollout_train_scale,
            "workflow_rollout_batch_size": args.workflow_rollout_batch_size,
            "workflow_scheduled_sampling_max_prob": args.workflow_scheduled_sampling_max_prob,
            "oscar_workflow_action_step_loss_weight": args.oscar_workflow_action_step_loss_weight,
            "oscar_workflow_action_kpi_loss_weight": args.oscar_workflow_action_kpi_loss_weight,
            "oscar_workflow_action_intervention_loss_weight": args.oscar_workflow_action_intervention_loss_weight,
            "device": device.type,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "rows": rows,
        "runs": runs,
        "notes": [
            "Training excludes every reasoning trace that mentions the held-out business environment, including cross-case analogy and transfer examples.",
            "Decision-action vocabularies are learned from the non-heldout environments only, so coverage metrics report how much of the held-out action surface remains representable without label leakage.",
            "Oscar auxiliary vocabularies are also built from the non-heldout split, so workflow KPI/improvement/motif/reward metrics are the most reliable holdout indicators.",
        ],
    }
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
