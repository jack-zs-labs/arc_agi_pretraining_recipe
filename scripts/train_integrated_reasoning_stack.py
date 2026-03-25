from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Sequence

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    dist = None
    F = None
    DistributedDataParallel = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler.mixed_reasoning_dataset import (
    ReasoningTextExample,
    build_arc_reasoning_examples,
    build_core_reasoning_examples,
    build_dclm_text_examples,
    build_gsm8k_reasoning_examples,
    build_mmlu_reasoning_examples,
    build_mmlu_pro_reasoning_examples,
    build_mmlu_redux_reasoning_examples,
    build_oscar_graph_reasoning_examples,
    build_oscar_scope_examples,
    build_oscar_scope_reasoning_examples,
    build_olympiad_math_reasoning_examples,
    split_examples,
    texts_from_examples,
)
from arc_trajectory_sampler.olympiad_math_parser import DEFAULT_OLYMPIAD_MATH_CONFIGS, OLYMPIAD_MATH_SUPPORTED_CONFIGS
from arc_trajectory_sampler.oscar_graph_reasoning import OSCAR_GRAPH_REASONING_FAMILIES, build_oscar_canonical_graph
from arc_trajectory_sampler.oscar_scope_corpus import DEFAULT_OSCAR_SCOPE_VIEWS, OSCAR_SCOPE_VIEW_CHOICES
from arc_trajectory_sampler.oscar_scope_reasoning import (
    OSCAR_SCOPE_CONCEPT_TAGS,
    OSCAR_SCOPE_REASONING_FAMILIES,
    OSCAR_WORKFLOW_ACTION_STEP_IDS,
    OSCAR_WORKFLOW_BOTTLENECK_IDS,
    OSCAR_WORKFLOW_CANONICAL_INTERVENTION_IDS,
    OSCAR_WORKFLOW_CANONICAL_KPI_IDS,
    OSCAR_WORKFLOW_IMPROVEMENT_IDS,
    OSCAR_WORKFLOW_KPI_IDS,
    OSCAR_WORKFLOW_MOTIF_IDS,
    OSCAR_WORKFLOW_REWARD_BUCKETS,
    _workflow_canonical_intervention_id,
    _workflow_canonical_kpi_id,
)
from models.reasoning_tokenizer import (
    ReasoningTokenizer,
    add_tokenizer_cli_arguments,
    build_reasoning_tokenizer,
)
if torch is not None:
    from models import (
        AttentionBackendConfig,
        CoReAuxiliaryConfig,
        DecoderLanguageModel,
        DecoderModelConfig,
        DecisionActionConfig,
        MoEConfig,
        OscarAuxiliaryConfig,
        OscarGraphAuxiliaryConfig,
        ReasoningEffort,
        reasoning_budget_policy_for_benchmark,
        resolve_effort,
    )
else:  # pragma: no cover - only used for data-only mode without torch installed
    AttentionBackendConfig = None
    CoReAuxiliaryConfig = None
    DecoderLanguageModel = None
    DecoderModelConfig = None
    DecisionActionConfig = None
    MoEConfig = None
    OscarAuxiliaryConfig = None
    OscarGraphAuxiliaryConfig = None
    ReasoningEffort = str
    reasoning_budget_policy_for_benchmark = None
    resolve_effort = None

ARCHITECTURE_CHOICES = ("dense", "moe")
EFFORT_CHOICES: tuple[ReasoningEffort, ...] = ("fast", "balanced", "deep")
CORE_DEPENDENCY_KIND_TO_ID = {
    "control": 0,
    "data": 1,
    "infoflow": 2,
}
CORE_CATEGORY_TO_ID = {
    "trace": 0,
    "list_source": 1,
}
OSCAR_FAMILY_TO_ID = {
    "oscar_section_anchor": 0,
    "oscar_outline_next_heading": 1,
    "oscar_concept_tags": 2,
    "oscar_workflow_environment": 3,
    "oscar_workflow_kpi_tags": 4,
    "oscar_workflow_bottleneck_tags": 5,
    "oscar_workflow_improvement_tags": 6,
    "oscar_workflow_kpi_improvement": 7,
    "oscar_workflow_intervention_trace": 8,
    "oscar_workflow_case_analogy": 9,
    "oscar_workflow_transfer": 10,
}
OSCAR_GRAPH_FAMILY_TO_ID = {
    "oscar_graph_relation": 0,
    "oscar_graph_neighbors": 1,
    "oscar_graph_path_completion": 2,
    "oscar_graph_grounding": 3,
    "oscar_graph_executor_rollout": 4,
    "oscar_graph_executor_trace": 5,
}
CORE_GRAPH_BACKEND_TO_ID = {
    "heuristic": 0,
    "python_ast": 1,
}
CORE_MAX_SOURCE_CANDIDATES = 16
DECISION_OUTPUT_HEAD_ALIASES = {
    "mmlu_pro": "mmlu",
    "mmlu_redux": "mmlu",
}
TEXT_TOKENIZER: ReasoningTokenizer | None = None
DEFAULT_EXPORT_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = ""

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class DecisionActionExample:
    benchmark: str
    output_head: str
    text: str
    trajectory_id: str
    trace_step: str
    target_action: str
    target_action_name: str = ""
    target_action_name_id: int = -1
    target_argument_head_key: str = ""
    target_argument_key: str = ""
    target_argument_id: int = -1
    target_action_id: int = -1
    full_candidate_mask: tuple[bool, ...] = ()
    full_candidate_name_ids: tuple[int, ...] = ()
    full_candidate_argument_ids: tuple[int, ...] = ()
    name_candidate_mask: tuple[bool, ...] = ()
    target_argument_candidate_mask: tuple[bool, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an integrated mixed-benchmark reasoning stack.")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURE_CHOICES, default=list(ARCHITECTURE_CHOICES))
    parser.add_argument("--attention-preset", choices=("mla_default", "mla_sia_prefill_l1"), default="mla_default")
    parser.add_argument("--arc-train-episodes", type=int, default=16)
    parser.add_argument("--arc-val-episodes", type=int, default=4)
    parser.add_argument("--gsm8k-max-rows", type=int, default=48)
    parser.add_argument("--mmlu-max-rows", type=int, default=48)
    parser.add_argument("--mmlu-pro-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-max-rows", type=int, default=0)
    parser.add_argument("--mmlu-redux-label-mode", choices=("original", "corrected_single"), default="corrected_single")
    parser.add_argument("--olympiad-math-max-rows", type=int, default=0)
    parser.add_argument(
        "--olympiad-math-configs",
        nargs="+",
        choices=OLYMPIAD_MATH_SUPPORTED_CONFIGS,
        default=DEFAULT_OLYMPIAD_MATH_CONFIGS,
    )
    parser.add_argument("--core-max-rows", type=int, default=0)
    parser.add_argument("--core-graph-backend", choices=("auto", "heuristic", "python_ast"), default="auto")
    parser.add_argument("--include-dclm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dclm-dataset-id", type=str, default="mlfoundations/dclm-baseline-1.0-parquet")
    parser.add_argument("--dclm-split", type=str, default="train")
    parser.add_argument("--dclm-text-field", type=str, default="text")
    parser.add_argument("--dclm-max-documents", type=int, default=0)
    parser.add_argument("--dclm-shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dclm-shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--dclm-min-text-chars", type=int, default=0)
    parser.add_argument("--dclm-min-language-score", type=float, default=0.0)
    parser.add_argument("--dclm-min-fasttext-score", type=float, default=0.0)
    parser.add_argument("--include-oscar-scope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-roots", nargs="+", default=[])
    parser.add_argument("--oscar-scope-paths", nargs="+", default=[])
    parser.add_argument("--oscar-scope-max-documents", type=int, default=16)
    parser.add_argument("--oscar-scope-max-chunks", type=int, default=192)
    parser.add_argument(
        "--oscar-scope-views",
        nargs="+",
        choices=OSCAR_SCOPE_VIEW_CHOICES,
        default=list(DEFAULT_OSCAR_SCOPE_VIEWS),
    )
    parser.add_argument("--include-oscar-scope-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-scope-reasoning-max-examples", type=int, default=256)
    parser.add_argument(
        "--oscar-scope-reasoning-families",
        nargs="+",
        choices=OSCAR_SCOPE_REASONING_FAMILIES,
        default=list(OSCAR_SCOPE_REASONING_FAMILIES),
    )
    parser.add_argument("--include-oscar-graph-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oscar-graph-reasoning-max-examples", type=int, default=128)
    parser.add_argument(
        "--oscar-graph-reasoning-families",
        nargs="+",
        choices=OSCAR_GRAPH_REASONING_FAMILIES,
        default=list(OSCAR_GRAPH_REASONING_FAMILIES),
    )
    parser.add_argument("--disable-core-auxiliary-heads", action="store_true")
    parser.add_argument("--disable-oscar-auxiliary-heads", action="store_true")
    parser.add_argument("--disable-oscar-graph-auxiliary-heads", action="store_true")
    parser.add_argument("--core-query-positive-loss-weight", type=float, default=0.1)
    parser.add_argument("--core-source-count-loss-weight", type=float, default=0.05)
    parser.add_argument("--core-trace-length-loss-weight", type=float, default=0.05)
    parser.add_argument("--core-dependency-kind-loss-weight", type=float, default=0.05)
    parser.add_argument("--core-infoflow-data-edge-loss-weight", type=float, default=0.05)
    parser.add_argument("--core-source-membership-loss-weight", type=float, default=0.1)
    parser.add_argument("--core-direct-edge-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-family-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-section-depth-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-doc-group-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-doc-title-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-section-path-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-concept-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-section-parent-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-related-doc-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-kpi-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-improvement-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-motif-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-reward-bucket-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-reward-score-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-workflow-action-step-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-workflow-action-kpi-loss-weight", type=float, default=0.2)
    parser.add_argument("--oscar-workflow-action-intervention-loss-weight", type=float, default=0.3)
    parser.add_argument("--oscar-graph-family-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-graph-domain-loss-weight", type=float, default=0.05)
    parser.add_argument("--oscar-graph-relation-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-graph-neighbor-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-graph-path-via-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-graph-path-target-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-graph-grounding-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-graph-rollout-motif-loss-weight", type=float, default=0.1)
    parser.add_argument("--oscar-graph-rollout-step-loss-weight", type=float, default=0.1)
    parser.add_argument("--disable-decision-action-heads", action="store_true")
    parser.add_argument("--decision-action-loss-weight", type=float, default=0.2)
    parser.add_argument("--decision-action-projection-hidden-size", type=int, default=0)
    parser.add_argument("--decision-batch-size", type=int, default=4)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--train-reasoning-effort", choices=EFFORT_CHOICES, default="balanced")
    parser.add_argument("--eval-reasoning-effort", choices=EFFORT_CHOICES, default="deep")
    parser.add_argument("--arc-train-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--gsm8k-train-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--mmlu-train-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--arc-eval-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--gsm8k-eval-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--mmlu-eval-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--oscar-scope-train-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--oscar-scope-eval-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--oscar-scope-reasoning-train-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--oscar-scope-reasoning-eval-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--oscar-graph-reasoning-train-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--oscar-graph-reasoning-eval-effort", choices=EFFORT_CHOICES, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--latent-kv-dim", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--router-jitter-noise", type=float, default=0.0)
    parser.add_argument("--moe-auxiliary-loss-weight", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--force-full-train-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gsm8k-data-dir", type=str, default="arc_trajectory_sampler/data/gsm8k")
    parser.add_argument("--mmlu-data-dir", type=str, default="arc_trajectory_sampler/data/mmlu")
    parser.add_argument("--core-data-dir", type=str, default="arc_trajectory_sampler/data/core")
    parser.add_argument("--include-verifier-targets", action="store_true")
    parser.add_argument("--data-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-dir", type=str, default="")
    parser.add_argument("--corpus-manifest", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--csv-output", type=str, default="")
    add_tokenizer_cli_arguments(parser, default_kind="epiplex", default_vocab_size=4096, default_task="generic")
    return parser.parse_args()


def auto_device() -> torch.device:
    if torch is None:  # pragma: no cover - guarded by require_training_runtime
        raise SystemExit(
            "Torch is required for training. Install requirements-models.txt or use .venv_atari."
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(device_name: str) -> torch.device:
    if torch is None:  # pragma: no cover - guarded by require_training_runtime
        raise SystemExit(
            "Torch is required for training. Install requirements-models.txt or use .venv_atari."
        )
    if device_name == "auto":
        return auto_device()
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise SystemExit("MPS requested but not available.")
        return torch.device("mps")
    return torch.device("cpu")


def initialize_distributed(device_name: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)
    if torch is None or dist is None:
        raise SystemExit("Distributed launch requested but torch.distributed is unavailable.")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if device_name in {"auto", "cuda"}:
        if not torch.cuda.is_available():
            raise SystemExit("Distributed CUDA launch requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def resolve_runtime_device(args: argparse.Namespace, distributed_context: DistributedContext) -> torch.device:
    if not distributed_context.enabled:
        return resolve_device(args.device)
    if args.device in {"auto", "cuda"}:
        return torch.device("cuda", distributed_context.local_rank)
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "mps":
        raise SystemExit("Distributed training does not support MPS.")
    return resolve_device(args.device)


def synchronize_device(device: torch.device) -> None:
    if torch is None:  # pragma: no cover - guarded by require_training_runtime
        raise SystemExit(
            "Torch is required for training. Install requirements-models.txt or use .venv_atari."
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def distributed_barrier(distributed_context: DistributedContext) -> None:
    if distributed_context.enabled and dist is not None:
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if DistributedDataParallel is not None and isinstance(model, DistributedDataParallel):
        return model.module
    return model


def precision_mode(args: argparse.Namespace, device: torch.device) -> str:
    if args.precision == "auto":
        return "bf16" if device.type == "cuda" else "fp32"
    if args.precision == "bf16" and device.type != "cuda":
        raise SystemExit("BF16 precision is only supported for CUDA devices in this trainer.")
    return args.precision


def autocast_context(device: torch.device, precision: str):
    if torch is None or precision != "bf16":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def set_step_seed(
    *,
    seed: int,
    distributed_context: DistributedContext,
    architecture_index: int,
    step_index: int,
    micro_step_index: int = 0,
) -> int:
    step_seed = (
        int(seed)
        + (distributed_context.rank * 1_000_003)
        + (architecture_index * 100_003)
        + (step_index * 1_003)
        + micro_step_index
    )
    random.seed(step_seed)
    if torch is not None:
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)
    return step_seed


def sample_rng_for_step(
    *,
    seed: int,
    distributed_context: DistributedContext,
    architecture_index: int,
    step_index: int,
    micro_step_index: int = 0,
) -> random.Random:
    resolved_seed = set_step_seed(
        seed=seed,
        distributed_context=distributed_context,
        architecture_index=architecture_index,
        step_index=step_index,
        micro_step_index=micro_step_index,
    )
    return random.Random(resolved_seed)


def set_text_tokenizer(tokenizer: ReasoningTokenizer) -> None:
    global TEXT_TOKENIZER
    TEXT_TOKENIZER = tokenizer


def require_text_tokenizer() -> ReasoningTokenizer:
    if TEXT_TOKENIZER is None:
        raise RuntimeError("Text tokenizer has not been initialized yet.")
    return TEXT_TOKENIZER


def encode_text(text: str) -> list[int]:
    return require_text_tokenizer().encode(text, add_eos=True)


def bucket_core_source_count(count: int) -> int:
    return max(0, min(int(count), 3))


def bucket_core_trace_length(trace_nodes: Sequence[object]) -> int:
    return max(0, min(len(trace_nodes), 4))


def core_node_key(node: dict[str, object] | None) -> tuple[int, str | None, str | None] | None:
    if not isinstance(node, dict):
        return None
    line = node.get("line")
    if line is None:
        return None
    name = node.get("name")
    use_kind = node.get("use_kind")
    return (
        int(line),
        None if name in (None, "") else str(name),
        None if use_kind in (None, "") else str(use_kind),
    )


def core_node_sort_key(node: dict[str, object]) -> tuple[int, str, str]:
    key = core_node_key(node)
    if key is None:
        return (0, "", "")
    line, name, use_kind = key
    return (line, name or "", use_kind or "")


def clamp_core_feature(value: float, *, scale: float = 128.0) -> float:
    normalized = value / scale
    return max(-8.0, min(normalized, 8.0))


def core_source_candidate_features_for_example(
    example: ReasoningTextExample,
    *,
    max_candidates: int = CORE_MAX_SOURCE_CANDIDATES,
) -> dict[str, object]:
    auxiliary_targets = example.auxiliary_targets or {}
    dependency_kind = str(auxiliary_targets.get("dependency_kind", "control"))
    category = str(auxiliary_targets.get("category", "trace"))
    graph_backend = str(auxiliary_targets.get("graph_backend", "heuristic"))
    query_source = auxiliary_targets.get("query_source")
    query_source = query_source if isinstance(query_source, dict) else None
    query_target = auxiliary_targets.get("query_target")
    query_target = query_target if isinstance(query_target, dict) else {"line": 0}
    query_target_line = int(query_target.get("line", 0))
    query_source_line = int(query_source.get("line", 0)) if query_source is not None else 0
    query_target_name = None if query_target.get("name") in (None, "") else str(query_target.get("name"))
    query_source_name = None if query_source is None or query_source.get("name") in (None, "") else str(query_source.get("name"))
    positive_source_keys = {
        key
        for key in (core_node_key(node) for node in auxiliary_targets.get("source_set", ()))
        if key is not None
    }
    query_target_key = core_node_key(query_target)
    direct_edge_source_keys: set[tuple[int, str | None, str | None]] = set()
    for edge in auxiliary_targets.get("direct_dependency_edges", ()):
        if not isinstance(edge, dict):
            continue
        target_key = core_node_key(edge.get("to"))
        source_key = core_node_key(edge.get("from"))
        if target_key is not None and source_key is not None and target_key == query_target_key:
            direct_edge_source_keys.add(source_key)
    deduped_candidates: dict[tuple[int, str | None, str | None], dict[str, object]] = {}
    for node in auxiliary_targets.get("candidate_sources", ()):
        if not isinstance(node, dict):
            continue
        key = core_node_key(node)
        if key is not None:
            deduped_candidates[key] = node
    if not deduped_candidates and query_source is not None:
        key = core_node_key(query_source)
        if key is not None:
            deduped_candidates[key] = query_source
    ordered_candidates = sorted(
        deduped_candidates.values(),
        key=lambda node: (
            core_node_key(node) not in positive_source_keys,
            core_node_key(node) not in direct_edge_source_keys,
            core_node_sort_key(node),
        ),
    )[:max_candidates]
    features: list[list[float]] = []
    source_membership_labels: list[int] = []
    direct_edge_labels: list[int] = []
    for candidate in ordered_candidates:
        candidate_key = core_node_key(candidate)
        candidate_line = int(candidate.get("line", 0))
        candidate_name = None if candidate.get("name") in (None, "") else str(candidate.get("name"))
        candidate_use_kind = None if candidate.get("use_kind") in (None, "") else str(candidate.get("use_kind"))
        feature_vector = [
            clamp_core_feature(float(candidate_line)),
            clamp_core_feature(float(query_target_line)),
            clamp_core_feature(float(query_source_line)) if query_source is not None else 0.0,
            clamp_core_feature(float(candidate_line - query_target_line)),
            clamp_core_feature(float(abs(candidate_line - query_target_line))),
            clamp_core_feature(float(candidate_line - query_source_line)) if query_source is not None else 0.0,
            1.0 if candidate_name is None else 0.0,
            1.0 if candidate_name is not None else 0.0,
            1.0 if candidate_use_kind is not None else 0.0,
            1.0 if candidate_name is not None and candidate_name == query_target_name else 0.0,
            1.0 if candidate_name is not None and candidate_name == query_source_name else 0.0,
            1.0 if candidate_line == query_target_line else 0.0,
            1.0 if query_source is not None and candidate_line == query_source_line else 0.0,
            1.0 if candidate_line < query_target_line else 0.0,
            1.0 if candidate_line > query_target_line else 0.0,
            1.0 if dependency_kind == "control" else 0.0,
            1.0 if dependency_kind == "data" else 0.0,
            1.0 if dependency_kind == "infoflow" else 0.0,
            1.0 if category == "trace" else 0.0,
            1.0 if category == "list_source" else 0.0,
            1.0 if graph_backend == "python_ast" else 0.0,
            1.0 if graph_backend == "heuristic" else 0.0,
        ]
        features.append(feature_vector)
        source_membership_labels.append(int(candidate_key in positive_source_keys))
        direct_edge_labels.append(int(candidate_key in direct_edge_source_keys))
    return {
        "features": features,
        "source_membership_labels": source_membership_labels,
        "direct_edge_to_target_labels": direct_edge_labels,
    }


def core_auxiliary_labels_for_example(example: ReasoningTextExample) -> dict[str, object]:
    auxiliary_targets = example.auxiliary_targets or {}
    dependency_kind = str(auxiliary_targets.get("dependency_kind", "control"))
    candidate_payload = core_source_candidate_features_for_example(example)
    return {
        "query_positive": int(bool(auxiliary_targets.get("query_positive", False))),
        "source_count_bucket": bucket_core_source_count(len(auxiliary_targets.get("source_set", ()))),
        "trace_length_bucket": bucket_core_trace_length(auxiliary_targets.get("trace_nodes", ())),
        "dependency_kind_id": CORE_DEPENDENCY_KIND_TO_ID.get(dependency_kind, 0),
        "infoflow_has_data_edge": int(bool(auxiliary_targets.get("infoflow_has_data_edge", False))),
        "source_candidate_features": candidate_payload["features"],
        "source_membership_labels": candidate_payload["source_membership_labels"],
        "direct_edge_to_target_labels": candidate_payload["direct_edge_to_target_labels"],
    }


def bucket_oscar_section_depth(depth: int) -> int:
    return max(0, min(max(depth, 1), 5) - 1)


def build_oscar_auxiliary_vocabularies(
    examples: Sequence[ReasoningTextExample],
) -> dict[str, object]:
    doc_groups = tuple(
        sorted(
            {
                str((example.auxiliary_targets or {}).get("doc_group", "unknown"))
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
            }
        )
    )
    doc_titles = tuple(
        sorted(
            {
                str((example.auxiliary_targets or {}).get("doc_title", "unknown"))
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
            }
            | {
                str(title)
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
                for title in (example.auxiliary_targets or {}).get("related_doc_titles", ())
                if str(title)
            }
        )
    )
    section_paths = tuple(
        sorted(
            {
                str(
                    (example.auxiliary_targets or {}).get(
                        "section_path_label",
                        (example.auxiliary_targets or {}).get("section_path", "unknown"),
                    )
                )
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
            }
            | {
                str((example.auxiliary_targets or {}).get("doc_title", "unknown"))
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
            }
            | {
                str((example.auxiliary_targets or {}).get("section_parent_label", "unknown"))
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
            }
            | {
                str(path_label)
                for example in examples
                if example.benchmark == "oscar_scope_reasoning"
                for path_label in (example.auxiliary_targets or {}).get("section_ancestor_labels", ())
                if str(path_label)
            }
        )
    )
    return {
        "doc_groups": doc_groups,
        "doc_titles": doc_titles,
        "section_paths": section_paths,
        "workflow_bottleneck_ids": OSCAR_WORKFLOW_BOTTLENECK_IDS,
        "workflow_kpi_ids": OSCAR_WORKFLOW_KPI_IDS,
        "workflow_improvement_ids": OSCAR_WORKFLOW_IMPROVEMENT_IDS,
        "workflow_motif_ids": OSCAR_WORKFLOW_MOTIF_IDS,
        "workflow_reward_buckets": OSCAR_WORKFLOW_REWARD_BUCKETS,
        "workflow_canonical_kpi_ids": OSCAR_WORKFLOW_CANONICAL_KPI_IDS,
        "workflow_canonical_intervention_ids": OSCAR_WORKFLOW_CANONICAL_INTERVENTION_IDS,
        "workflow_action_step_ids": OSCAR_WORKFLOW_ACTION_STEP_IDS,
        "doc_group_to_id": {value: index for index, value in enumerate(doc_groups)},
        "doc_title_to_id": {value: index for index, value in enumerate(doc_titles)},
        "section_path_to_id": {value: index for index, value in enumerate(section_paths)},
        "concept_tags": OSCAR_SCOPE_CONCEPT_TAGS,
        "concept_tag_to_id": {value: index for index, value in enumerate(OSCAR_SCOPE_CONCEPT_TAGS)},
        "workflow_bottleneck_to_id": {
            value: index for index, value in enumerate(OSCAR_WORKFLOW_BOTTLENECK_IDS)
        },
        "workflow_kpi_to_id": {value: index for index, value in enumerate(OSCAR_WORKFLOW_KPI_IDS)},
        "workflow_improvement_to_id": {
            value: index for index, value in enumerate(OSCAR_WORKFLOW_IMPROVEMENT_IDS)
        },
        "workflow_motif_to_id": {value: index for index, value in enumerate(OSCAR_WORKFLOW_MOTIF_IDS)},
        "workflow_reward_bucket_to_id": {
            value: index for index, value in enumerate(OSCAR_WORKFLOW_REWARD_BUCKETS)
        },
        "workflow_canonical_kpi_to_id": {
            value: index for index, value in enumerate(OSCAR_WORKFLOW_CANONICAL_KPI_IDS)
        },
        "workflow_canonical_intervention_to_id": {
            value: index for index, value in enumerate(OSCAR_WORKFLOW_CANONICAL_INTERVENTION_IDS)
        },
        "workflow_action_step_to_id": {
            value: index for index, value in enumerate(OSCAR_WORKFLOW_ACTION_STEP_IDS)
        },
    }


def oscar_auxiliary_labels_for_example(
    example: ReasoningTextExample,
    *,
    vocabularies: dict[str, object],
) -> dict[str, object]:
    auxiliary_targets = example.auxiliary_targets or {}
    family = str(auxiliary_targets.get("family", "oscar_section_anchor"))
    doc_group = str(auxiliary_targets.get("doc_group", "unknown"))
    doc_title = str(auxiliary_targets.get("doc_title", "unknown"))
    section_path = str(auxiliary_targets.get("section_path_label", auxiliary_targets.get("section_path", doc_title)))
    section_parent = str(auxiliary_targets.get("section_parent_label", doc_title))
    concept_tags = {
        str(tag)
        for tag in auxiliary_targets.get("concept_tags", ())
        if str(tag) in OSCAR_SCOPE_CONCEPT_TAGS
    }
    related_doc_titles = {
        str(title)
        for title in auxiliary_targets.get("related_doc_titles", ())
        if str(title)
    }
    concept_multihot = [0.0] * len(OSCAR_SCOPE_CONCEPT_TAGS)
    concept_tag_to_id = dict(vocabularies.get("concept_tag_to_id", {}))
    for tag in concept_tags:
        concept_multihot[concept_tag_to_id[tag]] = 1.0
    doc_group_to_id = dict(vocabularies.get("doc_group_to_id", {}))
    doc_title_to_id = dict(vocabularies.get("doc_title_to_id", {}))
    section_path_to_id = dict(vocabularies.get("section_path_to_id", {}))
    workflow_bottleneck_ids = tuple(vocabularies.get("workflow_bottleneck_ids", ()))
    workflow_bottleneck_to_id = dict(vocabularies.get("workflow_bottleneck_to_id", {}))
    workflow_kpi_to_id = dict(vocabularies.get("workflow_kpi_to_id", {}))
    workflow_improvement_to_id = dict(vocabularies.get("workflow_improvement_to_id", {}))
    workflow_motif_to_id = dict(vocabularies.get("workflow_motif_to_id", {}))
    workflow_reward_bucket_to_id = dict(vocabularies.get("workflow_reward_bucket_to_id", {}))
    workflow_canonical_kpi_ids = tuple(vocabularies.get("workflow_canonical_kpi_ids", ()))
    workflow_canonical_kpi_to_id = dict(vocabularies.get("workflow_canonical_kpi_to_id", {}))
    workflow_canonical_intervention_ids = tuple(vocabularies.get("workflow_canonical_intervention_ids", ()))
    workflow_canonical_intervention_to_id = dict(vocabularies.get("workflow_canonical_intervention_to_id", {}))
    workflow_action_step_to_id = dict(vocabularies.get("workflow_action_step_to_id", {}))
    related_doc_multihot = [0.0] * len(tuple(vocabularies.get("doc_titles", ())))
    for title in related_doc_titles:
        title_id = doc_title_to_id.get(title)
        if title_id is not None:
            related_doc_multihot[title_id] = 1.0
    section_depth = int(auxiliary_targets.get("section_depth", 1))
    focus_kpi_id = str(auxiliary_targets.get("focus_kpi_id", ""))
    focus_improvement_id = str(auxiliary_targets.get("focus_improvement_id", ""))
    workflow_motif_id = str(auxiliary_targets.get("workflow_motif_id", ""))
    reward_bucket_label = str(auxiliary_targets.get("workflow_reward_bucket_label", ""))
    reward_score = float(auxiliary_targets.get("workflow_reward_score", 0.0) or 0.0)
    active_bottleneck_ids = {
        str(bottleneck_id)
        for bottleneck_id in auxiliary_targets.get("active_bottleneck_ids", ())
        if str(bottleneck_id) in workflow_bottleneck_to_id
    }
    active_bottleneck_multihot = [0.0] * len(workflow_bottleneck_ids)
    for bottleneck_id in active_bottleneck_ids:
        active_bottleneck_multihot[workflow_bottleneck_to_id[bottleneck_id]] = 1.0
    canonical_kpi_context_id = str(auxiliary_targets.get("workflow_canonical_kpi_id", ""))
    source_canonical_intervention_id = str(auxiliary_targets.get("source_workflow_canonical_improvement_id", ""))
    target_action_name = str(auxiliary_targets.get("target_action_name", "none"))
    if target_action_name not in workflow_action_step_to_id:
        target_action_name = "none"
    action_step_active_mask = 1 if target_action_name != "none" else 0
    action_step_id = workflow_action_step_to_id.get(target_action_name, 0)
    action_kpi_active_mask = 1 if target_action_name == "select_kpi_family" else 0
    action_intervention_active_mask = 1 if target_action_name == "select_intervention_family" else 0
    action_kpi_family_id = canonical_kpi_context_id if canonical_kpi_context_id in workflow_canonical_kpi_to_id else ""
    if str(example.trace_step) == "apply_primary_intervention":
        action_intervention_family_id = str(auxiliary_targets.get("workflow_canonical_primary_improvement_id", ""))
    elif str(example.trace_step) == "apply_followup_intervention":
        action_intervention_family_id = str(auxiliary_targets.get("workflow_canonical_followup_improvement_id", ""))
    else:
        action_intervention_family_id = str(auxiliary_targets.get("workflow_canonical_improvement_id", ""))
    kpi_candidate_ids = {
        str(auxiliary_targets.get("workflow_canonical_kpi_id", ""))
    }
    for candidate_id in auxiliary_targets.get("candidate_kpi_ids", ()):
        candidate_id = str(candidate_id)
        canonical_candidate = _workflow_canonical_kpi_id(candidate_id)
        if canonical_candidate in workflow_canonical_kpi_to_id:
            kpi_candidate_ids.add(canonical_candidate)
    intervention_candidate_ids = set()
    for candidate_id in auxiliary_targets.get("candidate_canonical_improvement_ids", ()):
        candidate_id = str(candidate_id)
        if candidate_id in workflow_canonical_intervention_to_id:
            intervention_candidate_ids.add(candidate_id)
    if not intervention_candidate_ids:
        for candidate_id in auxiliary_targets.get("candidate_improvement_ids", ()):
            candidate_id = str(candidate_id)
            canonical_candidate = _workflow_canonical_intervention_id(candidate_id)
            if canonical_candidate in workflow_canonical_intervention_to_id:
                intervention_candidate_ids.add(canonical_candidate)
    primary_canonical_intervention_id = str(auxiliary_targets.get("workflow_canonical_primary_improvement_id", ""))
    if str(example.trace_step) == "apply_followup_intervention" and primary_canonical_intervention_id:
        intervention_candidate_ids.discard(primary_canonical_intervention_id)
    if action_intervention_family_id in workflow_canonical_intervention_to_id:
        intervention_candidate_ids.add(action_intervention_family_id)
    workflow_action_kpi_candidate_mask = [0.0] * len(workflow_canonical_kpi_ids)
    for candidate_id in kpi_candidate_ids:
        candidate_index = workflow_canonical_kpi_to_id.get(candidate_id)
        if candidate_index is not None:
            workflow_action_kpi_candidate_mask[candidate_index] = 1.0
    if not any(workflow_action_kpi_candidate_mask) and action_kpi_family_id in workflow_canonical_kpi_to_id:
        workflow_action_kpi_candidate_mask[workflow_canonical_kpi_to_id[action_kpi_family_id]] = 1.0
    workflow_action_intervention_candidate_mask = [0.0] * len(workflow_canonical_intervention_ids)
    for candidate_id in intervention_candidate_ids:
        candidate_index = workflow_canonical_intervention_to_id.get(candidate_id)
        if candidate_index is not None:
            workflow_action_intervention_candidate_mask[candidate_index] = 1.0
    if (
        not any(workflow_action_intervention_candidate_mask)
        and action_intervention_family_id in workflow_canonical_intervention_to_id
    ):
        workflow_action_intervention_candidate_mask[
            workflow_canonical_intervention_to_id[action_intervention_family_id]
        ] = 1.0
    has_workflow_kpi = 1 if focus_kpi_id in workflow_kpi_to_id else 0
    has_workflow_improvement = 1 if focus_improvement_id in workflow_improvement_to_id else 0
    has_workflow_motif = 1 if workflow_motif_id in workflow_motif_to_id else 0
    has_workflow_reward = 1 if reward_bucket_label in workflow_reward_bucket_to_id else 0
    has_canonical_kpi_context = 1 if canonical_kpi_context_id in workflow_canonical_kpi_to_id else 0
    has_source_canonical_intervention = 1 if source_canonical_intervention_id in workflow_canonical_intervention_to_id else 0
    is_transfer_action = 1 if family == "oscar_workflow_transfer" and action_intervention_active_mask else 0
    is_intervention_trace_action = 1 if family == "oscar_workflow_intervention_trace" and action_step_active_mask else 0
    return {
        "family_id": OSCAR_FAMILY_TO_ID.get(family, 0),
        "section_depth_bucket": bucket_oscar_section_depth(section_depth),
        "doc_group_id": doc_group_to_id.get(doc_group, 0),
        "doc_title_id": doc_title_to_id.get(doc_title, 0),
        "section_path_id": section_path_to_id.get(section_path, 0),
        "section_parent_id": section_path_to_id.get(section_parent, 0),
        "concept_multihot": concept_multihot,
        "related_doc_multihot": related_doc_multihot,
        "workflow_kpi_id": workflow_kpi_to_id.get(focus_kpi_id, 0),
        "workflow_improvement_id": workflow_improvement_to_id.get(focus_improvement_id, 0),
        "workflow_motif_id": workflow_motif_to_id.get(workflow_motif_id, 0),
        "workflow_reward_bucket_id": workflow_reward_bucket_to_id.get(reward_bucket_label, 0),
        "workflow_reward_score": max(0.0, min(1.0, reward_score)),
        "workflow_active_bottleneck_multihot": active_bottleneck_multihot,
        "workflow_canonical_kpi_context_id": workflow_canonical_kpi_to_id.get(canonical_kpi_context_id, 0),
        "workflow_canonical_kpi_context_active_mask": has_canonical_kpi_context,
        "workflow_source_canonical_intervention_id": workflow_canonical_intervention_to_id.get(
            source_canonical_intervention_id,
            0,
        ),
        "workflow_source_canonical_intervention_active_mask": has_source_canonical_intervention,
        "workflow_action_step_id": action_step_id,
        "workflow_action_step_active_mask": action_step_active_mask,
        "workflow_action_kpi_family_id": workflow_canonical_kpi_to_id.get(action_kpi_family_id, 0),
        "workflow_action_kpi_active_mask": action_kpi_active_mask,
        "workflow_action_kpi_candidate_mask": workflow_action_kpi_candidate_mask,
        "workflow_action_intervention_family_id": workflow_canonical_intervention_to_id.get(
            action_intervention_family_id,
            0,
        ),
        "workflow_action_intervention_active_mask": action_intervention_active_mask,
        "workflow_action_intervention_candidate_mask": workflow_action_intervention_candidate_mask,
        "workflow_transfer_action_active_mask": is_transfer_action,
        "workflow_intervention_trace_action_active_mask": is_intervention_trace_action,
        "workflow_kpi_active_mask": has_workflow_kpi,
        "workflow_improvement_active_mask": has_workflow_improvement,
        "workflow_motif_active_mask": has_workflow_motif,
        "workflow_reward_active_mask": has_workflow_reward,
    }


def build_oscar_graph_auxiliary_vocabularies(
    examples: Sequence[ReasoningTextExample],
) -> dict[str, object]:
    if not any(example.benchmark == "oscar_graph_reasoning" for example in examples):
        return {}
    nodes, edges = build_oscar_canonical_graph()
    node_ids = tuple(node.node_id for node in nodes)
    node_categories = tuple(sorted({node.category for node in nodes}))
    graph_domains = tuple(sorted({node.domain for node in nodes} | {edge.domain for edge in edges}))
    relations = tuple(sorted({edge.relation for edge in edges}))
    rollout_motifs = tuple(
        sorted(
            {
                str((example.auxiliary_targets or {}).get("rollout_motif_label", ""))
                for example in examples
                if example.benchmark == "oscar_graph_reasoning" and str((example.auxiliary_targets or {}).get("rollout_motif_label", ""))
            }
        )
    )
    rollout_max_steps = max(
        (
            len(tuple((example.auxiliary_targets or {}).get("rollout_step_node_ids", ())))
            for example in examples
            if example.benchmark == "oscar_graph_reasoning"
        ),
        default=0,
    )
    outgoing_targets_by_source: dict[str, set[str]] = defaultdict(set)
    incoming_sources_by_target: dict[str, set[str]] = defaultdict(set)
    outgoing_relations_by_source: dict[str, set[str]] = defaultdict(set)
    incoming_relations_by_target: dict[str, set[str]] = defaultdict(set)
    outgoing_targets_by_source_relation: dict[tuple[str, str], set[str]] = defaultdict(set)
    incoming_sources_by_target_relation: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in edges:
        outgoing_targets_by_source[edge.source_id].add(edge.target_id)
        incoming_sources_by_target[edge.target_id].add(edge.source_id)
        outgoing_relations_by_source[edge.source_id].add(edge.relation)
        incoming_relations_by_target[edge.target_id].add(edge.relation)
        outgoing_targets_by_source_relation[(edge.source_id, edge.relation)].add(edge.target_id)
        incoming_sources_by_target_relation[(edge.target_id, edge.relation)].add(edge.source_id)
    return {
        "node_ids": node_ids,
        "node_id_to_id": {value: index for index, value in enumerate(node_ids)},
        "node_id_to_category": {node.node_id: node.category for node in nodes},
        "node_id_to_domain": {node.node_id: node.domain for node in nodes},
        "node_categories": node_categories,
        "node_category_to_id": {value: index for index, value in enumerate(node_categories)},
        "graph_domains": graph_domains,
        "graph_domain_to_id": {value: index for index, value in enumerate(graph_domains)},
        "relations": relations,
        "relation_to_id": {value: index for index, value in enumerate(relations)},
        "rollout_motifs": rollout_motifs,
        "rollout_motif_to_id": {value: index for index, value in enumerate(rollout_motifs)},
        "rollout_max_steps": rollout_max_steps,
        "outgoing_targets_by_source": {key: tuple(sorted(value)) for key, value in outgoing_targets_by_source.items()},
        "incoming_sources_by_target": {key: tuple(sorted(value)) for key, value in incoming_sources_by_target.items()},
        "outgoing_relations_by_source": {key: tuple(sorted(value)) for key, value in outgoing_relations_by_source.items()},
        "incoming_relations_by_target": {key: tuple(sorted(value)) for key, value in incoming_relations_by_target.items()},
        "outgoing_targets_by_source_relation": {
            key: tuple(sorted(value))
            for key, value in outgoing_targets_by_source_relation.items()
        },
        "incoming_sources_by_target_relation": {
            key: tuple(sorted(value))
            for key, value in incoming_sources_by_target_relation.items()
        },
        "max_out_degree": max((len(value) for value in outgoing_targets_by_source.values()), default=1),
        "max_in_degree": max((len(value) for value in incoming_sources_by_target.values()), default=1),
    }


def _one_hot_from_id(index: int | None, size: int) -> list[float]:
    vector = [0.0] * max(size, 0)
    if index is not None and 0 <= index < size:
        vector[index] = 1.0
    return vector


def _safe_string_set(values: object) -> set[str]:
    if not isinstance(values, (list, tuple)):
        return set()
    return {str(value) for value in values if str(value)}


def oscar_graph_node_candidate_features(
    node_id: str,
    *,
    auxiliary_targets: dict[str, object],
    vocabularies: dict[str, object],
) -> list[float]:
    node_id_to_id = dict(vocabularies.get("node_id_to_id", {}))
    node_id_to_category = dict(vocabularies.get("node_id_to_category", {}))
    node_id_to_domain = dict(vocabularies.get("node_id_to_domain", {}))
    node_category_to_id = dict(vocabularies.get("node_category_to_id", {}))
    graph_domain_to_id = dict(vocabularies.get("graph_domain_to_id", {}))
    node_ids = tuple(vocabularies.get("node_ids", ()))
    node_categories = tuple(vocabularies.get("node_categories", ()))
    graph_domains = tuple(vocabularies.get("graph_domains", ()))
    outgoing_targets_by_source = dict(vocabularies.get("outgoing_targets_by_source", {}))
    incoming_sources_by_target = dict(vocabularies.get("incoming_sources_by_target", {}))
    outgoing_relations_by_source = dict(vocabularies.get("outgoing_relations_by_source", {}))
    outgoing_targets_by_source_relation = dict(vocabularies.get("outgoing_targets_by_source_relation", {}))
    max_out_degree = max(int(vocabularies.get("max_out_degree", 1)), 1)
    max_in_degree = max(int(vocabularies.get("max_in_degree", 1)), 1)

    node_category = str(node_id_to_category.get(node_id, "unknown"))
    node_domain = str(node_id_to_domain.get(node_id, "unknown"))
    source_node_id = str(auxiliary_targets.get("source_node_id", ""))
    graph_domain = str(auxiliary_targets.get("graph_domain", ""))
    target_category = str(auxiliary_targets.get("target_category", ""))
    first_relation = str(auxiliary_targets.get("first_relation", ""))
    second_relation = str(auxiliary_targets.get("second_relation", ""))
    local_graph = auxiliary_targets.get("local_graph", {})
    local_node_ids = {
        str(node.get("id", ""))
        for node in local_graph.get("nodes", [])
        if isinstance(node, dict) and str(node.get("id", ""))
    } if isinstance(local_graph, dict) else set()

    outgoing_targets = set(outgoing_targets_by_source.get(node_id, ()))
    incoming_sources = set(incoming_sources_by_target.get(node_id, ()))
    features = []
    features.extend(_one_hot_from_id(node_id_to_id.get(node_id), len(node_ids)))
    features.extend(_one_hot_from_id(node_category_to_id.get(node_category), len(node_categories)))
    features.extend(_one_hot_from_id(graph_domain_to_id.get(node_domain), len(graph_domains)))
    features.extend(
        [
            1.0 if node_id == source_node_id else 0.0,
            1.0 if node_domain == graph_domain else 0.0,
            1.0 if node_category == target_category else 0.0,
            1.0 if node_id in local_node_ids else 0.0,
            len(outgoing_targets) / float(max_out_degree),
            len(incoming_sources) / float(max_in_degree),
            1.0 if source_node_id and node_id in set(outgoing_targets_by_source.get(source_node_id, ())) else 0.0,
            1.0 if source_node_id and source_node_id in incoming_sources else 0.0,
            1.0
            if source_node_id and first_relation and node_id in set(outgoing_targets_by_source_relation.get((source_node_id, first_relation), ()))
            else 0.0,
            1.0
            if second_relation and second_relation in set(outgoing_relations_by_source.get(node_id, ()))
            else 0.0,
            1.0 if bool(outgoing_targets) else 0.0,
        ]
    )
    return features


def oscar_graph_relation_candidate_features(
    relation: str,
    *,
    auxiliary_targets: dict[str, object],
    vocabularies: dict[str, object],
) -> list[float]:
    relation_to_id = dict(vocabularies.get("relation_to_id", {}))
    relations = tuple(vocabularies.get("relations", ()))
    outgoing_relations_by_source = dict(vocabularies.get("outgoing_relations_by_source", {}))
    incoming_relations_by_target = dict(vocabularies.get("incoming_relations_by_target", {}))
    outgoing_targets_by_source_relation = dict(vocabularies.get("outgoing_targets_by_source_relation", {}))
    source_node_id = str(auxiliary_targets.get("source_node_id", ""))
    target_node_id = str(auxiliary_targets.get("target_node_id", ""))
    local_graph = auxiliary_targets.get("local_graph", {})
    local_relations = {
        str(edge.get("relation", ""))
        for edge in local_graph.get("edges", [])
        if isinstance(edge, dict) and str(edge.get("relation", ""))
    } if isinstance(local_graph, dict) else set()
    features = []
    features.extend(_one_hot_from_id(relation_to_id.get(relation), len(relations)))
    features.extend(
        [
            1.0 if relation in set(outgoing_relations_by_source.get(source_node_id, ())) else 0.0,
            1.0 if relation in set(incoming_relations_by_target.get(target_node_id, ())) else 0.0,
            1.0 if relation in local_relations else 0.0,
        ]
    )
    return features


def oscar_graph_rollout_step_candidate_features(
    candidate_node_id: str,
    *,
    current_node_id: str,
    step_index: int,
    auxiliary_targets: dict[str, object],
    vocabularies: dict[str, object],
) -> list[float]:
    node_id_to_id = dict(vocabularies.get("node_id_to_id", {}))
    node_id_to_category = dict(vocabularies.get("node_id_to_category", {}))
    node_id_to_domain = dict(vocabularies.get("node_id_to_domain", {}))
    node_category_to_id = dict(vocabularies.get("node_category_to_id", {}))
    graph_domain_to_id = dict(vocabularies.get("graph_domain_to_id", {}))
    node_ids = tuple(vocabularies.get("node_ids", ()))
    node_categories = tuple(vocabularies.get("node_categories", ()))
    graph_domains = tuple(vocabularies.get("graph_domains", ()))
    rollout_max_steps = max(int(vocabularies.get("rollout_max_steps", 1)), 1)
    outgoing_targets_by_source = dict(vocabularies.get("outgoing_targets_by_source", {}))
    outgoing_targets_by_source_relation = dict(vocabularies.get("outgoing_targets_by_source_relation", {}))
    local_graph = auxiliary_targets.get("local_graph", {})
    local_node_ids = {
        str(node.get("id", ""))
        for node in local_graph.get("nodes", [])
        if isinstance(node, dict) and str(node.get("id", ""))
    } if isinstance(local_graph, dict) else set()
    rollout_step_relations = [str(value) for value in auxiliary_targets.get("rollout_step_relations", ()) if str(value)]
    rollout_motif_label = str(auxiliary_targets.get("rollout_motif_label", ""))
    rollout_motif_to_id = dict(vocabularies.get("rollout_motif_to_id", {}))
    current_category = str(node_id_to_category.get(current_node_id, "unknown"))
    candidate_category = str(node_id_to_category.get(candidate_node_id, "unknown"))
    candidate_domain = str(node_id_to_domain.get(candidate_node_id, "unknown"))
    graph_domain = str(auxiliary_targets.get("graph_domain", ""))
    expected_relation = rollout_step_relations[step_index] if step_index < len(rollout_step_relations) else ""
    step_one_hot = [0.0] * rollout_max_steps
    if 0 <= step_index < rollout_max_steps:
        step_one_hot[step_index] = 1.0
    features = []
    features.extend(_one_hot_from_id(node_id_to_id.get(candidate_node_id), len(node_ids)))
    features.extend(_one_hot_from_id(node_category_to_id.get(candidate_category), len(node_categories)))
    features.extend(_one_hot_from_id(graph_domain_to_id.get(candidate_domain), len(graph_domains)))
    features.extend(step_one_hot)
    features.extend(_one_hot_from_id(rollout_motif_to_id.get(rollout_motif_label), len(tuple(vocabularies.get("rollout_motifs", ())))))
    features.extend(
        [
            1.0 if candidate_node_id in set(outgoing_targets_by_source.get(current_node_id, ())) else 0.0,
            1.0
            if expected_relation and candidate_node_id in set(outgoing_targets_by_source_relation.get((current_node_id, expected_relation), ()))
            else 0.0,
            1.0 if candidate_node_id in local_node_ids else 0.0,
            1.0 if current_category == candidate_category else 0.0,
            1.0 if candidate_domain == graph_domain else 0.0,
            1.0 if candidate_node_id == current_node_id else 0.0,
        ]
    )
    return features


def oscar_graph_auxiliary_labels_for_example(
    example: ReasoningTextExample,
    *,
    vocabularies: dict[str, object],
) -> dict[str, object]:
    auxiliary_targets = example.auxiliary_targets or {}
    family = str(auxiliary_targets.get("family", "oscar_graph_relation"))
    graph_domain = str(auxiliary_targets.get("graph_domain", "primitive"))
    graph_domain_to_id = dict(vocabularies.get("graph_domain_to_id", {}))
    node_ids = tuple(vocabularies.get("node_ids", ()))
    rollout_motif_label = str(auxiliary_targets.get("rollout_motif_label", ""))
    rollout_motif_to_id = dict(vocabularies.get("rollout_motif_to_id", {}))

    relation_candidates = [str(value) for value in auxiliary_targets.get("relation_candidates", ()) if str(value)]
    relation_candidate_features = [
        oscar_graph_relation_candidate_features(relation, auxiliary_targets=auxiliary_targets, vocabularies=vocabularies)
        for relation in relation_candidates
    ]
    relation_candidate_labels = [1.0 if relation == str(auxiliary_targets.get("relation", "")) else 0.0 for relation in relation_candidates]

    neighbor_candidate_ids = [str(value) for value in auxiliary_targets.get("candidate_target_ids", ()) if str(value)]
    neighbor_target_ids = _safe_string_set(auxiliary_targets.get("target_node_ids"))
    neighbor_candidate_features = [
        oscar_graph_node_candidate_features(node_id, auxiliary_targets=auxiliary_targets, vocabularies=vocabularies)
        for node_id in neighbor_candidate_ids
    ]
    neighbor_target_labels = [1.0 if node_id in neighbor_target_ids else 0.0 for node_id in neighbor_candidate_ids]

    path_candidate_ids = [str(value) for value in auxiliary_targets.get("candidate_node_ids", ()) if str(value)]
    path_candidate_features = [
        oscar_graph_node_candidate_features(node_id, auxiliary_targets=auxiliary_targets, vocabularies=vocabularies)
        for node_id in path_candidate_ids
    ]
    via_node_id = str(auxiliary_targets.get("via_node_id", ""))
    path_target_node_id = str(auxiliary_targets.get("target_node_id", ""))
    path_via_labels = [1.0 if node_id == via_node_id else 0.0 for node_id in path_candidate_ids]
    path_target_labels = [1.0 if node_id == path_target_node_id else 0.0 for node_id in path_candidate_ids]

    grounding_candidate_ids = [str(value) for value in auxiliary_targets.get("candidate_node_ids", ()) if str(value)]
    if not grounding_candidate_ids and str(auxiliary_targets.get("target_node_id", "")) in node_ids:
        grounding_candidate_ids = [str(auxiliary_targets["target_node_id"])]
    grounding_candidate_features = [
        oscar_graph_node_candidate_features(node_id, auxiliary_targets=auxiliary_targets, vocabularies=vocabularies)
        for node_id in grounding_candidate_ids
    ]
    grounding_target_node_id = str(auxiliary_targets.get("target_node_id", ""))
    grounding_labels = [1.0 if node_id == grounding_target_node_id else 0.0 for node_id in grounding_candidate_ids]

    rollout_candidate_ids = [str(value) for value in auxiliary_targets.get("rollout_candidate_node_ids", ()) if str(value)]
    rollout_step_node_ids = [str(value) for value in auxiliary_targets.get("rollout_step_node_ids", ()) if str(value)]
    rollout_source_node_id = str(auxiliary_targets.get("source_node_id", ""))
    rollout_step_candidate_features: list[list[list[float]]] = []
    rollout_step_labels: list[list[float]] = []
    current_node_id = rollout_source_node_id
    for step_index, step_node_id in enumerate(rollout_step_node_ids):
        step_features = [
            oscar_graph_rollout_step_candidate_features(
                candidate_node_id,
                current_node_id=current_node_id,
                step_index=step_index,
                auxiliary_targets=auxiliary_targets,
                vocabularies=vocabularies,
            )
            for candidate_node_id in rollout_candidate_ids
        ]
        step_labels = [1.0 if candidate_node_id == step_node_id else 0.0 for candidate_node_id in rollout_candidate_ids]
        rollout_step_candidate_features.append(step_features)
        rollout_step_labels.append(step_labels)
        current_node_id = step_node_id

    return {
        "family_id": OSCAR_GRAPH_FAMILY_TO_ID.get(family, 0),
        "graph_domain_id": graph_domain_to_id.get(graph_domain, 0),
        "rollout_motif_id": rollout_motif_to_id.get(rollout_motif_label, 0),
        "relation_candidate_features": relation_candidate_features,
        "relation_candidate_labels": relation_candidate_labels,
        "neighbor_candidate_features": neighbor_candidate_features,
        "neighbor_target_labels": neighbor_target_labels,
        "path_via_candidate_features": path_candidate_features,
        "path_via_labels": path_via_labels,
        "path_target_candidate_features": path_candidate_features,
        "path_target_labels": path_target_labels,
        "grounding_candidate_features": grounding_candidate_features,
        "grounding_labels": grounding_labels,
        "rollout_step_candidate_features": rollout_step_candidate_features,
        "rollout_step_labels": rollout_step_labels,
    }


def pack_candidate_feature_batch(
    structured_labels: Sequence[dict[str, object]],
    *,
    feature_key: str,
    label_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_candidates = max(
        max((len(structured_label.get(feature_key, [])) for structured_label in structured_labels), default=0),
        1,
    )
    feature_dim = max(
        (
            len(feature_vector)
            for structured_label in structured_labels
            for feature_vector in structured_label.get(feature_key, [])
        ),
        default=1,
    )
    candidate_features = torch.zeros(
        (len(structured_labels), max_candidates, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    candidate_mask = torch.zeros(
        (len(structured_labels), max_candidates),
        dtype=torch.float32,
        device=device,
    )
    candidate_labels = torch.zeros(
        (len(structured_labels), max_candidates),
        dtype=torch.float32,
        device=device,
    )
    for batch_index, structured_label in enumerate(structured_labels):
        features = list(structured_label.get(feature_key, []))
        labels = list(structured_label.get(label_key, []))
        for candidate_index, feature_vector in enumerate(features[:max_candidates]):
            candidate_features[batch_index, candidate_index, : len(feature_vector)] = torch.tensor(
                feature_vector,
                dtype=torch.float32,
                device=device,
            )
            candidate_mask[batch_index, candidate_index] = 1.0
            if candidate_index < len(labels):
                candidate_labels[batch_index, candidate_index] = float(labels[candidate_index])
    return candidate_features, candidate_mask, candidate_labels


def pack_stepwise_candidate_feature_batch(
    structured_labels: Sequence[dict[str, object]],
    *,
    feature_key: str,
    label_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps = max(
        max((len(structured_label.get(feature_key, [])) for structured_label in structured_labels), default=0),
        1,
    )
    max_candidates = max(
        (
            len(step_features)
            for structured_label in structured_labels
            for step_features in structured_label.get(feature_key, [])
        ),
        default=1,
    )
    feature_dim = max(
        (
            len(feature_vector)
            for structured_label in structured_labels
            for step_features in structured_label.get(feature_key, [])
            for feature_vector in step_features
        ),
        default=1,
    )
    candidate_features = torch.zeros(
        (len(structured_labels), max_steps, max_candidates, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    candidate_mask = torch.zeros(
        (len(structured_labels), max_steps, max_candidates),
        dtype=torch.float32,
        device=device,
    )
    candidate_labels = torch.zeros(
        (len(structured_labels), max_steps, max_candidates),
        dtype=torch.float32,
        device=device,
    )
    step_active_mask = torch.zeros(
        (len(structured_labels), max_steps),
        dtype=torch.float32,
        device=device,
    )
    for batch_index, structured_label in enumerate(structured_labels):
        steps = list(structured_label.get(feature_key, []))
        step_labels = list(structured_label.get(label_key, []))
        for step_index, step_features in enumerate(steps[:max_steps]):
            if not step_features:
                continue
            step_active_mask[batch_index, step_index] = 1.0
            current_labels = step_labels[step_index] if step_index < len(step_labels) else []
            for candidate_index, feature_vector in enumerate(step_features[:max_candidates]):
                candidate_features[batch_index, step_index, candidate_index, : len(feature_vector)] = torch.tensor(
                    feature_vector,
                    dtype=torch.float32,
                    device=device,
                )
                candidate_mask[batch_index, step_index, candidate_index] = 1.0
                if candidate_index < len(current_labels):
                    candidate_labels[batch_index, step_index, candidate_index] = float(current_labels[candidate_index])
    return candidate_features, candidate_mask, candidate_labels, step_active_mask


def decision_output_head_for_benchmark(benchmark_name: str) -> str:
    return DECISION_OUTPUT_HEAD_ALIASES.get(benchmark_name, benchmark_name)


def decision_argument_head_key(output_head: str, name_id: int) -> str:
    return f"{output_head}|{name_id}"


def parse_reasoning_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def parse_target_action_structure(target_action: str) -> tuple[str, str, str]:
    try:
        payload = json.loads(target_action)
    except json.JSONDecodeError:
        return target_action, target_action, target_action
    if not isinstance(payload, dict):
        return target_action, target_action, target_action
    action_name = str(payload.get("name", ""))
    action_payload = payload.get("action", {})
    canonical_payload = json.dumps(action_payload, sort_keys=True, separators=(",", ":"))
    canonical_action = json.dumps(
        {
            "action": action_payload,
            "name": action_name,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical_action, action_name, canonical_payload


def action_example_from_reasoning_text(example: ReasoningTextExample) -> DecisionActionExample | None:
    fields = parse_reasoning_fields(example.text)
    target_action = fields.get("target_action")
    if target_action is None or fields.get("record_type") != "decision_action":
        return None
    canonical_action, target_action_name, target_argument_key = parse_target_action_structure(target_action)
    input_lines: list[str] = []
    for raw_line in example.text.splitlines():
        line = raw_line.strip()
        if (
            line.startswith("target_action=")
            or line.startswith("target_stop=")
            or line.startswith("target_answer=")
        ):
            continue
        if line:
            input_lines.append(line)
    return DecisionActionExample(
        benchmark=example.benchmark,
        output_head=decision_output_head_for_benchmark(example.benchmark),
        text="\n".join(input_lines) + "\n",
        trajectory_id=example.trajectory_id,
        trace_step=example.trace_step,
        target_action=canonical_action,
        target_action_name=target_action_name,
        target_argument_key=target_argument_key,
    )


def convert_decision_action_examples(
    examples: Sequence[ReasoningTextExample],
) -> tuple[tuple[DecisionActionExample, ...], int]:
    converted: list[DecisionActionExample] = []
    skipped = 0
    for example in examples:
        converted_example = action_example_from_reasoning_text(example)
        if converted_example is None:
            skipped += 1
            continue
        converted.append(converted_example)
    return tuple(converted), skipped


def build_decision_head_vocabularies(
    examples_by_benchmark: dict[str, tuple[DecisionActionExample, ...]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    full_actions_by_head: dict[str, set[str]] = defaultdict(set)
    names_by_head: dict[str, set[str]] = defaultdict(set)
    for examples in examples_by_benchmark.values():
        for example in examples:
            full_actions_by_head[example.output_head].add(example.target_action)
            names_by_head[example.output_head].add(example.target_action_name)
    full_action_vocabularies = {
        head_name: tuple(sorted(actions))
        for head_name, actions in sorted(full_actions_by_head.items())
    }
    name_vocabularies = {
        head_name: tuple(sorted(names))
        for head_name, names in sorted(names_by_head.items())
    }
    name_to_index = {
        head_name: {name: index for index, name in enumerate(vocabulary)}
        for head_name, vocabulary in name_vocabularies.items()
    }
    arguments_by_head: dict[str, set[str]] = defaultdict(set)
    for examples in examples_by_benchmark.values():
        for example in examples:
            name_id = name_to_index[example.output_head][example.target_action_name]
            arguments_by_head[decision_argument_head_key(example.output_head, name_id)].add(example.target_argument_key)
    argument_vocabularies = {
        head_key: tuple(sorted(arguments))
        for head_key, arguments in sorted(arguments_by_head.items())
    }
    argument_to_index = {
        head_key: {argument: index for index, argument in enumerate(vocabulary)}
        for head_key, vocabulary in argument_vocabularies.items()
    }
    full_action_components: dict[str, tuple[tuple[int, int], ...]] = {}
    for head_name, vocabulary in full_action_vocabularies.items():
        components: list[tuple[int, int]] = []
        for action in vocabulary:
            _canonical_action, action_name, argument_key = parse_target_action_structure(action)
            name_id = name_to_index[head_name][action_name]
            argument_head = decision_argument_head_key(head_name, name_id)
            argument_id = argument_to_index[argument_head][argument_key]
            components.append((name_id, argument_id))
        full_action_components[head_name] = tuple(components)
    return {
        "full_action_vocabularies": full_action_vocabularies,
        "name_vocabularies": name_vocabularies,
        "argument_vocabularies": argument_vocabularies,
        "full_action_components": full_action_components,
    }


def build_decision_candidate_masks(
    examples_by_benchmark: dict[str, tuple[DecisionActionExample, ...]],
    *,
    full_action_vocabularies: dict[str, tuple[str, ...]],
    name_vocabularies: dict[str, tuple[str, ...]],
    argument_vocabularies: dict[str, tuple[str, ...]],
    full_action_components: dict[str, tuple[tuple[int, int], ...]],
) -> dict[str, dict[str, tuple[bool, ...]]]:
    action_to_index = {
        head_name: {action: index for index, action in enumerate(vocabulary)}
        for head_name, vocabulary in full_action_vocabularies.items()
    }
    bucket_to_indices: dict[str, set[int]] = defaultdict(set)
    for examples in examples_by_benchmark.values():
        for example in examples:
            fields = parse_reasoning_fields(example.text)
            bucket = fields.get("candidate_bucket", f"benchmark:{example.benchmark}|trace:{example.trace_step}")
            bucket_key = f"{example.output_head}|{bucket}"
            bucket_to_indices[bucket_key].add(action_to_index[example.output_head][example.target_action])
    full_masks: dict[str, tuple[bool, ...]] = {}
    name_masks: dict[str, tuple[bool, ...]] = {}
    argument_masks: dict[str, tuple[bool, ...]] = {}
    for bucket_key, indices in bucket_to_indices.items():
        head_name, _ = bucket_key.split("|", 1)
        full_mask = [False] * len(full_action_vocabularies[head_name])
        name_mask = [False] * len(name_vocabularies[head_name])
        arguments_by_head: dict[str, set[int]] = defaultdict(set)
        for index in indices:
            full_mask[index] = True
            name_id, argument_id = full_action_components[head_name][index]
            name_mask[name_id] = True
            arguments_by_head[decision_argument_head_key(head_name, name_id)].add(argument_id)
        full_masks[bucket_key] = tuple(full_mask)
        name_masks[bucket_key] = tuple(name_mask)
        for argument_head, argument_indices in arguments_by_head.items():
            argument_mask = [False] * len(argument_vocabularies[argument_head])
            for argument_index in argument_indices:
                argument_mask[argument_index] = True
            argument_masks[f"{bucket_key}|arg:{argument_head}"] = tuple(argument_mask)
    return {
        "full_masks": full_masks,
        "name_masks": name_masks,
        "argument_masks": argument_masks,
    }


def finalize_decision_action_examples(
    examples_by_benchmark: dict[str, tuple[DecisionActionExample, ...]],
    *,
    full_action_vocabularies: dict[str, tuple[str, ...]],
    name_vocabularies: dict[str, tuple[str, ...]],
    argument_vocabularies: dict[str, tuple[str, ...]],
    full_action_components: dict[str, tuple[tuple[int, int], ...]],
    candidate_masks: dict[str, dict[str, tuple[bool, ...]]],
) -> dict[str, tuple[DecisionActionExample, ...]]:
    full_action_to_index = {
        head_name: {action: index for index, action in enumerate(vocabulary)}
        for head_name, vocabulary in full_action_vocabularies.items()
    }
    name_to_index = {
        head_name: {name: index for index, name in enumerate(vocabulary)}
        for head_name, vocabulary in name_vocabularies.items()
    }
    argument_to_index = {
        head_name: {argument: index for index, argument in enumerate(vocabulary)}
        for head_name, vocabulary in argument_vocabularies.items()
    }
    finalized: dict[str, tuple[DecisionActionExample, ...]] = {}
    for benchmark_name, examples in examples_by_benchmark.items():
        updated: list[DecisionActionExample] = []
        for example in examples:
            fields = parse_reasoning_fields(example.text)
            bucket = fields.get("candidate_bucket", f"benchmark:{benchmark_name}|trace:{example.trace_step}")
            bucket_key = f"{example.output_head}|{bucket}"
            action_id = full_action_to_index[example.output_head][example.target_action]
            action_name_id = name_to_index[example.output_head][example.target_action_name]
            argument_head = decision_argument_head_key(example.output_head, action_name_id)
            argument_id = argument_to_index[argument_head][example.target_argument_key]
            full_candidate_mask = candidate_masks["full_masks"].get(
                bucket_key,
                tuple(True for _ in full_action_vocabularies[example.output_head]),
            )
            name_candidate_mask = candidate_masks["name_masks"].get(
                bucket_key,
                tuple(True for _ in name_vocabularies[example.output_head]),
            )
            argument_candidate_mask = candidate_masks["argument_masks"].get(
                f"{bucket_key}|arg:{argument_head}",
                tuple(True for _ in argument_vocabularies[argument_head]),
            )
            components = full_action_components[example.output_head]
            updated.append(
                DecisionActionExample(
                    benchmark=example.benchmark,
                    output_head=example.output_head,
                    text=example.text,
                    trajectory_id=example.trajectory_id,
                    trace_step=example.trace_step,
                    target_action=example.target_action,
                    target_action_name=example.target_action_name,
                    target_action_name_id=action_name_id,
                    target_argument_head_key=argument_head,
                    target_argument_key=example.target_argument_key,
                    target_argument_id=argument_id,
                    target_action_id=action_id,
                    full_candidate_mask=full_candidate_mask,
                    full_candidate_name_ids=tuple(component[0] for component in components),
                    full_candidate_argument_ids=tuple(component[1] for component in components),
                    name_candidate_mask=name_candidate_mask,
                    target_argument_candidate_mask=argument_candidate_mask,
                )
            )
        finalized[benchmark_name] = tuple(updated)
    return finalized


def encode_example_sequence(
    example: ReasoningTextExample,
    *,
    seq_len: int,
) -> tuple[list[int], list[int]]:
    tokenizer = require_text_tokenizer()
    tokens = encode_text(example.text)
    max_tokens = seq_len + 1
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    attention_mask = [1] * len(tokens)
    if len(tokens) < max_tokens:
        pad = [tokenizer.window_pad_token_id] * (max_tokens - len(tokens))
        mask_pad = [0] * (max_tokens - len(tokens))
        tokens = tokens + pad
        attention_mask = attention_mask + mask_pad
    return tokens, attention_mask


def sample_core_batch(
    examples: Sequence[ReasoningTextExample],
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if not examples:
        raise ValueError("Cannot sample from an empty CoRe example set.")
    indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
    batch_examples = [examples[index] for index in indices]
    encoded = [encode_example_sequence(example, seq_len=seq_len) for example in batch_examples]
    token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
    structured_labels = [core_auxiliary_labels_for_example(example) for example in batch_examples]
    labels_by_name = {
        name: torch.tensor(
            [int(structured_label[name]) for structured_label in structured_labels],
            dtype=torch.long,
            device=device,
        )
        for name in (
            "query_positive",
            "source_count_bucket",
            "trace_length_bucket",
            "dependency_kind_id",
            "infoflow_has_data_edge",
        )
    }
    max_candidates = max(
        max((len(structured_label["source_candidate_features"]) for structured_label in structured_labels), default=0),
        1,
    )
    feature_dim = max(
        (
            len(feature_vector)
            for structured_label in structured_labels
            for feature_vector in structured_label["source_candidate_features"]
        ),
        default=1,
    )
    source_candidate_features = torch.zeros(
        (len(batch_examples), max_candidates, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    source_candidate_mask = torch.zeros(
        (len(batch_examples), max_candidates),
        dtype=torch.float32,
        device=device,
    )
    source_membership_labels = torch.zeros(
        (len(batch_examples), max_candidates),
        dtype=torch.float32,
        device=device,
    )
    direct_edge_labels = torch.zeros(
        (len(batch_examples), max_candidates),
        dtype=torch.float32,
        device=device,
    )
    for batch_index, structured_label in enumerate(structured_labels):
        features = structured_label["source_candidate_features"]
        membership = structured_label["source_membership_labels"]
        direct_edges = structured_label["direct_edge_to_target_labels"]
        for candidate_index, feature_vector in enumerate(features[:max_candidates]):
            source_candidate_features[batch_index, candidate_index, : len(feature_vector)] = torch.tensor(
                feature_vector,
                dtype=torch.float32,
                device=device,
            )
            source_candidate_mask[batch_index, candidate_index] = 1.0
            source_membership_labels[batch_index, candidate_index] = float(membership[candidate_index])
            direct_edge_labels[batch_index, candidate_index] = float(direct_edges[candidate_index])
    labels_by_name["source_candidate_features"] = source_candidate_features
    labels_by_name["source_candidate_mask"] = source_candidate_mask
    labels_by_name["source_membership_labels"] = source_membership_labels
    labels_by_name["direct_edge_to_target_labels"] = direct_edge_labels
    return token_batch[:, :-1], token_batch[:, 1:], attention_mask[:, :-1], labels_by_name


def sample_oscar_auxiliary_batch(
    examples: Sequence[ReasoningTextExample],
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
    vocabularies: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if not examples:
        raise ValueError("Cannot sample from an empty Oscar reasoning example set.")
    indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
    batch_examples = [examples[index] for index in indices]
    encoded = [encode_example_sequence(example, seq_len=seq_len) for example in batch_examples]
    token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
    structured_labels = [
        oscar_auxiliary_labels_for_example(example, vocabularies=vocabularies)
        for example in batch_examples
    ]
    labels_by_name = {
        name: torch.tensor(
            [int(structured_label[name]) for structured_label in structured_labels],
            dtype=torch.long,
            device=device,
        )
        for name in (
            "family_id",
            "section_depth_bucket",
            "doc_group_id",
            "doc_title_id",
            "section_path_id",
            "section_parent_id",
        )
    }
    labels_by_name["concept_multihot"] = torch.tensor(
        [structured_label["concept_multihot"] for structured_label in structured_labels],
        dtype=torch.float32,
        device=device,
    )
    labels_by_name["related_doc_multihot"] = torch.tensor(
        [structured_label["related_doc_multihot"] for structured_label in structured_labels],
        dtype=torch.float32,
        device=device,
    )
    labels_by_name["workflow_reward_score"] = torch.tensor(
        [float(structured_label["workflow_reward_score"]) for structured_label in structured_labels],
        dtype=torch.float32,
        device=device,
    )
    labels_by_name["workflow_active_bottleneck_multihot"] = torch.tensor(
        [structured_label["workflow_active_bottleneck_multihot"] for structured_label in structured_labels],
        dtype=torch.float32,
        device=device,
    )
    labels_by_name["workflow_action_kpi_candidate_mask"] = torch.tensor(
        [structured_label["workflow_action_kpi_candidate_mask"] for structured_label in structured_labels],
        dtype=torch.float32,
        device=device,
    )
    labels_by_name["workflow_action_intervention_candidate_mask"] = torch.tensor(
        [structured_label["workflow_action_intervention_candidate_mask"] for structured_label in structured_labels],
        dtype=torch.float32,
        device=device,
    )
    for name in (
        "workflow_canonical_kpi_context_id",
        "workflow_canonical_kpi_context_active_mask",
        "workflow_source_canonical_intervention_id",
        "workflow_source_canonical_intervention_active_mask",
        "workflow_action_step_id",
        "workflow_action_step_active_mask",
        "workflow_action_kpi_family_id",
        "workflow_action_kpi_active_mask",
        "workflow_action_intervention_family_id",
        "workflow_action_intervention_active_mask",
        "workflow_transfer_action_active_mask",
        "workflow_intervention_trace_action_active_mask",
        "workflow_kpi_id",
        "workflow_improvement_id",
        "workflow_motif_id",
        "workflow_reward_bucket_id",
        "workflow_kpi_active_mask",
        "workflow_improvement_active_mask",
        "workflow_motif_active_mask",
        "workflow_reward_active_mask",
    ):
        dtype = torch.long if name.endswith("_id") else torch.float32
        labels_by_name[name] = torch.tensor(
            [structured_label[name] for structured_label in structured_labels],
            dtype=dtype,
            device=device,
        )
    return token_batch[:, :-1], token_batch[:, 1:], attention_mask[:, :-1], labels_by_name


def sample_oscar_graph_auxiliary_batch(
    examples: Sequence[ReasoningTextExample],
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
    vocabularies: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if not examples:
        raise ValueError("Cannot sample from an empty Oscar graph reasoning example set.")
    indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
    batch_examples = [examples[index] for index in indices]
    encoded = [encode_example_sequence(example, seq_len=seq_len) for example in batch_examples]
    token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
    structured_labels = [
        oscar_graph_auxiliary_labels_for_example(example, vocabularies=vocabularies)
        for example in batch_examples
    ]
    labels_by_name = {
        name: torch.tensor(
            [int(structured_label[name]) for structured_label in structured_labels],
            dtype=torch.long,
            device=device,
        )
        for name in (
            "family_id",
            "graph_domain_id",
            "rollout_motif_id",
        )
    }
    for feature_key, label_key, tensor_prefix in (
        ("relation_candidate_features", "relation_candidate_labels", "relation_candidate"),
        ("neighbor_candidate_features", "neighbor_target_labels", "neighbor_candidate"),
        ("path_via_candidate_features", "path_via_labels", "path_via_candidate"),
        ("path_target_candidate_features", "path_target_labels", "path_target_candidate"),
        ("grounding_candidate_features", "grounding_labels", "grounding_candidate"),
    ):
        candidate_features, candidate_mask, candidate_labels = pack_candidate_feature_batch(
            structured_labels,
            feature_key=feature_key,
            label_key=label_key,
            device=device,
        )
        labels_by_name[f"{tensor_prefix}_features"] = candidate_features
        labels_by_name[f"{tensor_prefix}_mask"] = candidate_mask
        labels_by_name[label_key] = candidate_labels
    rollout_step_candidate_features, rollout_step_candidate_mask, rollout_step_labels, rollout_step_active_mask = (
        pack_stepwise_candidate_feature_batch(
            structured_labels,
            feature_key="rollout_step_candidate_features",
            label_key="rollout_step_labels",
            device=device,
        )
    )
    labels_by_name["rollout_step_candidate_features"] = rollout_step_candidate_features
    labels_by_name["rollout_step_candidate_mask"] = rollout_step_candidate_mask
    labels_by_name["rollout_step_labels"] = rollout_step_labels
    labels_by_name["rollout_step_active_mask"] = rollout_step_active_mask
    return token_batch[:, :-1], token_batch[:, 1:], attention_mask[:, :-1], labels_by_name


def sample_decision_action_batch(
    examples: Sequence[DecisionActionExample],
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
    full_action_vocabularies: dict[str, tuple[str, ...]],
    name_vocabularies: dict[str, tuple[str, ...]],
    argument_vocabularies: dict[str, tuple[str, ...]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if not examples:
        raise ValueError("Cannot sample from an empty decision-action example set.")
    indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
    batch_examples = [examples[index] for index in indices]
    encoded = [encode_example_sequence(ReasoningTextExample(
        benchmark=example.benchmark,
        text=example.text,
        trajectory_id=example.trajectory_id,
        step_index=0,
        trace_step=example.trace_step,
        auxiliary_targets=None,
    ), seq_len=seq_len) for example in batch_examples]
    token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
    max_full_head_size = max((len(vocabulary) for vocabulary in full_action_vocabularies.values()), default=1)
    max_name_head_size = max((len(vocabulary) for vocabulary in name_vocabularies.values()), default=1)
    max_argument_head_size = max((len(vocabulary) for vocabulary in argument_vocabularies.values()), default=1)
    full_candidate_masks = torch.zeros((len(batch_examples), max_full_head_size), dtype=torch.bool, device=device)
    full_candidate_name_ids = torch.zeros((len(batch_examples), max_full_head_size), dtype=torch.long, device=device)
    full_candidate_argument_ids = torch.zeros((len(batch_examples), max_full_head_size), dtype=torch.long, device=device)
    name_candidate_masks = torch.zeros((len(batch_examples), max_name_head_size), dtype=torch.bool, device=device)
    target_argument_candidate_masks = torch.zeros(
        (len(batch_examples), max_argument_head_size),
        dtype=torch.bool,
        device=device,
    )
    for batch_index, example in enumerate(batch_examples):
        full_mask = torch.tensor(example.full_candidate_mask, dtype=torch.bool, device=device)
        full_candidate_masks[batch_index, : full_mask.numel()] = full_mask
        full_name_ids = torch.tensor(example.full_candidate_name_ids, dtype=torch.long, device=device)
        full_candidate_name_ids[batch_index, : full_name_ids.numel()] = full_name_ids
        full_argument_ids = torch.tensor(example.full_candidate_argument_ids, dtype=torch.long, device=device)
        full_candidate_argument_ids[batch_index, : full_argument_ids.numel()] = full_argument_ids
        name_mask = torch.tensor(example.name_candidate_mask, dtype=torch.bool, device=device)
        name_candidate_masks[batch_index, : name_mask.numel()] = name_mask
        argument_mask = torch.tensor(example.target_argument_candidate_mask, dtype=torch.bool, device=device)
        target_argument_candidate_masks[batch_index, : argument_mask.numel()] = argument_mask
    labels: dict[str, object] = {
        "benchmark_names": [example.benchmark for example in batch_examples],
        "output_heads": [example.output_head for example in batch_examples],
        "target_action_ids": torch.tensor(
            [example.target_action_id for example in batch_examples],
            dtype=torch.long,
            device=device,
        ),
        "target_name_ids": torch.tensor(
            [example.target_action_name_id for example in batch_examples],
            dtype=torch.long,
            device=device,
        ),
        "target_argument_output_heads": [example.target_argument_head_key for example in batch_examples],
        "target_argument_ids": torch.tensor(
            [example.target_argument_id for example in batch_examples],
            dtype=torch.long,
            device=device,
        ),
        "full_candidate_masks": full_candidate_masks,
        "full_candidate_name_ids": full_candidate_name_ids,
        "full_candidate_argument_ids": full_candidate_argument_ids,
        "name_candidate_masks": name_candidate_masks,
        "target_argument_candidate_masks": target_argument_candidate_masks,
    }
    return token_batch[:, :-1], attention_mask[:, :-1], labels


def chunk_token_stream(texts: Iterable[str], *, seq_len: int) -> list[list[int]]:
    tokenizer = require_text_tokenizer()
    windows: list[list[int]] = []
    stride = max(seq_len // 2, 1)
    for text in texts:
        tokens = encode_text(text)
        if len(tokens) < 2:
            continue
        if len(tokens) <= seq_len + 1:
            padded = tokens + [tokenizer.window_pad_token_id] * max(0, seq_len + 1 - len(tokens))
            windows.append(padded[: seq_len + 1])
            continue
        for start in range(0, len(tokens) - 1, stride):
            window = tokens[start : start + seq_len + 1]
            if len(window) < seq_len + 1:
                window = window + [tokenizer.window_pad_token_id] * (seq_len + 1 - len(window))
            windows.append(window[: seq_len + 1])
            if start + seq_len + 1 >= len(tokens):
                break
    return windows


def sample_batch(
    windows: Sequence[Sequence[int]] | torch.Tensor,
    *,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    if torch is not None and isinstance(windows, torch.Tensor):
        if windows.size(0) == 0:
            raise ValueError("Cannot sample from an empty window set.")
        indices = torch.tensor(
            [rng.randrange(windows.size(0)) for _ in range(batch_size)],
            dtype=torch.long,
        )
        return windows.index_select(0, indices).to(device=device, dtype=torch.long)
    if not windows:
        raise ValueError("Cannot sample from an empty window set.")
    indices = [rng.randrange(len(windows)) for _ in range(batch_size)]
    batch = [windows[index] for index in indices]
    return torch.tensor(batch, dtype=torch.long, device=device)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def trim_batch_to_budget(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    budget_prompt_limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if budget_prompt_limit is None or inputs.size(1) <= budget_prompt_limit:
        return inputs, targets
    return inputs[:, -budget_prompt_limit:], targets[:, -budget_prompt_limit:]


def evaluate_benchmark_loss(
    model: DecoderLanguageModel,
    windows: Sequence[Sequence[int]] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    benchmark_name: str,
    effort: ReasoningEffort,
) -> float:
    if torch is not None and isinstance(windows, torch.Tensor):
        if windows.size(0) == 0:
            return float("nan")
        num_windows = int(windows.size(0))
    else:
        if not windows:
            return float("nan")
        num_windows = len(windows)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    policy = reasoning_budget_policy_for_benchmark(
        benchmark_name,
        effort=effort,
        attention_window=model.config.attention.sliding_window,
    )
    with torch.no_grad():
        for start in range(0, num_windows, batch_size):
            if torch is not None and isinstance(windows, torch.Tensor):
                batch = windows[start : start + batch_size].to(device=device, dtype=torch.long)
            else:
                batch = torch.tensor(windows[start : start + batch_size], dtype=torch.long, device=device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            inputs, targets = trim_batch_to_budget(inputs, targets, budget.max_prompt_tokens)
            outputs = model(inputs, budget=budget)
            loss = cross_entropy_loss(outputs.logits, targets)
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(total_batches, 1)


def evaluate_core_auxiliary(
    model: DecoderLanguageModel,
    examples: Sequence[ReasoningTextExample],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    effort: ReasoningEffort,
) -> dict[str, float]:
    if not examples:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    policy = reasoning_budget_policy_for_benchmark(
        "core",
        effort=effort,
        attention_window=model.config.attention.sliding_window,
    )
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            encoded = [encode_example_sequence(example, seq_len=seq_len) for example in batch_examples]
            token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
            structured_labels = [core_auxiliary_labels_for_example(example) for example in batch_examples]
            labels = {
                name: torch.tensor(
                    [int(structured_label[name]) for structured_label in structured_labels],
                    dtype=torch.long,
                    device=device,
                )
                for name in (
                    "query_positive",
                    "source_count_bucket",
                    "trace_length_bucket",
                    "dependency_kind_id",
                    "infoflow_has_data_edge",
                )
            }
            max_candidates = max(
                max((len(structured_label["source_candidate_features"]) for structured_label in structured_labels), default=0),
                1,
            )
            feature_dim = max(
                (
                    len(feature_vector)
                    for structured_label in structured_labels
                    for feature_vector in structured_label["source_candidate_features"]
                ),
                default=1,
            )
            source_candidate_features = torch.zeros(
                (len(batch_examples), max_candidates, feature_dim),
                dtype=torch.float32,
                device=device,
            )
            source_candidate_mask = torch.zeros(
                (len(batch_examples), max_candidates),
                dtype=torch.float32,
                device=device,
            )
            source_membership_labels = torch.zeros(
                (len(batch_examples), max_candidates),
                dtype=torch.float32,
                device=device,
            )
            direct_edge_labels = torch.zeros(
                (len(batch_examples), max_candidates),
                dtype=torch.float32,
                device=device,
            )
            for batch_index, structured_label in enumerate(structured_labels):
                features = structured_label["source_candidate_features"]
                membership = structured_label["source_membership_labels"]
                direct_edges = structured_label["direct_edge_to_target_labels"]
                for candidate_index, feature_vector in enumerate(features[:max_candidates]):
                    source_candidate_features[batch_index, candidate_index, : len(feature_vector)] = torch.tensor(
                        feature_vector,
                        dtype=torch.float32,
                        device=device,
                    )
                    source_candidate_mask[batch_index, candidate_index] = 1.0
                    source_membership_labels[batch_index, candidate_index] = float(membership[candidate_index])
                    direct_edge_labels[batch_index, candidate_index] = float(direct_edges[candidate_index])
            labels["source_candidate_features"] = source_candidate_features
            labels["source_candidate_mask"] = source_candidate_mask
            labels["source_membership_labels"] = source_membership_labels
            labels["direct_edge_to_target_labels"] = direct_edge_labels
            inputs = token_batch[:, :-1]
            targets = token_batch[:, 1:]
            mask = attention_mask[:, :-1]
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            inputs, targets = trim_batch_to_budget(inputs, targets, budget.max_prompt_tokens)
            mask = mask[:, -inputs.size(1) :]
            outputs = model(
                inputs,
                attention_mask=mask,
                budget=budget,
                task_name="core",
                task_auxiliary_labels=labels,
            )
            if outputs.task_auxiliary_metrics is not None:
                for name, value in outputs.task_auxiliary_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
            batches += 1
    if batches == 0:
        return {}
    return {
        name: value / batches
        for name, value in totals.items()
    }


def evaluate_oscar_auxiliary(
    model: DecoderLanguageModel,
    examples: Sequence[ReasoningTextExample],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    effort: ReasoningEffort,
    vocabularies: dict[str, object],
) -> dict[str, float]:
    if not examples:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    policy = reasoning_budget_policy_for_benchmark(
        "oscar_scope_reasoning",
        effort=effort,
        attention_window=model.config.attention.sliding_window,
    )
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            encoded = [encode_example_sequence(example, seq_len=seq_len) for example in batch_examples]
            token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
            structured_labels = [
                oscar_auxiliary_labels_for_example(example, vocabularies=vocabularies)
                for example in batch_examples
            ]
            labels = {
                name: torch.tensor(
                    [int(structured_label[name]) for structured_label in structured_labels],
                    dtype=torch.long,
                    device=device,
                )
                for name in (
                    "family_id",
                    "section_depth_bucket",
                    "doc_group_id",
                    "doc_title_id",
                    "section_path_id",
                    "section_parent_id",
                )
            }
            labels["concept_multihot"] = torch.tensor(
                [structured_label["concept_multihot"] for structured_label in structured_labels],
                dtype=torch.float32,
                device=device,
            )
            labels["related_doc_multihot"] = torch.tensor(
                [structured_label["related_doc_multihot"] for structured_label in structured_labels],
                dtype=torch.float32,
                device=device,
            )
            labels["workflow_reward_score"] = torch.tensor(
                [float(structured_label["workflow_reward_score"]) for structured_label in structured_labels],
                dtype=torch.float32,
                device=device,
            )
            labels["workflow_active_bottleneck_multihot"] = torch.tensor(
                [structured_label["workflow_active_bottleneck_multihot"] for structured_label in structured_labels],
                dtype=torch.float32,
                device=device,
            )
            labels["workflow_action_kpi_candidate_mask"] = torch.tensor(
                [structured_label["workflow_action_kpi_candidate_mask"] for structured_label in structured_labels],
                dtype=torch.float32,
                device=device,
            )
            labels["workflow_action_intervention_candidate_mask"] = torch.tensor(
                [structured_label["workflow_action_intervention_candidate_mask"] for structured_label in structured_labels],
                dtype=torch.float32,
                device=device,
            )
            for name in (
                "workflow_canonical_kpi_context_id",
                "workflow_canonical_kpi_context_active_mask",
                "workflow_source_canonical_intervention_id",
                "workflow_source_canonical_intervention_active_mask",
                "workflow_action_step_id",
                "workflow_action_step_active_mask",
                "workflow_action_kpi_family_id",
                "workflow_action_kpi_active_mask",
                "workflow_action_intervention_family_id",
                "workflow_action_intervention_active_mask",
                "workflow_transfer_action_active_mask",
                "workflow_intervention_trace_action_active_mask",
                "workflow_kpi_id",
                "workflow_improvement_id",
                "workflow_motif_id",
                "workflow_reward_bucket_id",
                "workflow_kpi_active_mask",
                "workflow_improvement_active_mask",
                "workflow_motif_active_mask",
                "workflow_reward_active_mask",
            ):
                dtype = torch.long if name.endswith("_id") else torch.float32
                labels[name] = torch.tensor(
                    [structured_label[name] for structured_label in structured_labels],
                    dtype=dtype,
                    device=device,
                )
            inputs = token_batch[:, :-1]
            targets = token_batch[:, 1:]
            mask = attention_mask[:, :-1]
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            inputs, targets = trim_batch_to_budget(inputs, targets, budget.max_prompt_tokens)
            mask = mask[:, -inputs.size(1) :]
            outputs = model(
                inputs,
                attention_mask=mask,
                budget=budget,
                task_name="oscar_scope_reasoning",
                task_auxiliary_labels=labels,
            )
            if outputs.task_auxiliary_metrics is not None:
                for name, value in outputs.task_auxiliary_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
            batches += 1
    if batches == 0:
        return {}
    return {
        name: value / batches
        for name, value in totals.items()
    }


def evaluate_oscar_graph_auxiliary(
    model: DecoderLanguageModel,
    examples: Sequence[ReasoningTextExample],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    effort: ReasoningEffort,
    vocabularies: dict[str, object],
) -> dict[str, float]:
    if not examples:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    policy = reasoning_budget_policy_for_benchmark(
        "oscar_graph_reasoning",
        effort=effort,
        attention_window=model.config.attention.sliding_window,
    )
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            encoded = [encode_example_sequence(example, seq_len=seq_len) for example in batch_examples]
            token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
            attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
            structured_labels = [
                oscar_graph_auxiliary_labels_for_example(example, vocabularies=vocabularies)
                for example in batch_examples
            ]
            labels = {
                name: torch.tensor(
                    [int(structured_label[name]) for structured_label in structured_labels],
                    dtype=torch.long,
                    device=device,
                )
                for name in (
                    "family_id",
                    "graph_domain_id",
                    "rollout_motif_id",
                )
            }
            for feature_key, label_key, tensor_prefix in (
                ("relation_candidate_features", "relation_candidate_labels", "relation_candidate"),
                ("neighbor_candidate_features", "neighbor_target_labels", "neighbor_candidate"),
                ("path_via_candidate_features", "path_via_labels", "path_via_candidate"),
                ("path_target_candidate_features", "path_target_labels", "path_target_candidate"),
                ("grounding_candidate_features", "grounding_labels", "grounding_candidate"),
            ):
                candidate_features, candidate_mask, candidate_labels = pack_candidate_feature_batch(
                    structured_labels,
                    feature_key=feature_key,
                    label_key=label_key,
                    device=device,
                )
                labels[f"{tensor_prefix}_features"] = candidate_features
                labels[f"{tensor_prefix}_mask"] = candidate_mask
                labels[label_key] = candidate_labels
            rollout_step_candidate_features, rollout_step_candidate_mask, rollout_step_labels, rollout_step_active_mask = (
                pack_stepwise_candidate_feature_batch(
                    structured_labels,
                    feature_key="rollout_step_candidate_features",
                    label_key="rollout_step_labels",
                    device=device,
                )
            )
            labels["rollout_step_candidate_features"] = rollout_step_candidate_features
            labels["rollout_step_candidate_mask"] = rollout_step_candidate_mask
            labels["rollout_step_labels"] = rollout_step_labels
            labels["rollout_step_active_mask"] = rollout_step_active_mask
            inputs = token_batch[:, :-1]
            targets = token_batch[:, 1:]
            mask = attention_mask[:, :-1]
            budget = policy.build_inference_budget(
                model.config,
                prompt_tokens=inputs.size(1),
                use_kv_cache=False,
                max_new_tokens=0,
            )
            inputs, targets = trim_batch_to_budget(inputs, targets, budget.max_prompt_tokens)
            mask = mask[:, -inputs.size(1) :]
            outputs = model(
                inputs,
                attention_mask=mask,
                budget=budget,
                task_name="oscar_graph_reasoning",
                task_auxiliary_labels=labels,
            )
            if outputs.task_auxiliary_metrics is not None:
                for name, value in outputs.task_auxiliary_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
            batches += 1
    if batches == 0:
        return {}
    return {
        name: value / batches
        for name, value in totals.items()
    }


def evaluate_decision_action_accuracy(
    model: DecoderLanguageModel,
    examples_by_benchmark: dict[str, tuple[DecisionActionExample, ...]],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    full_action_vocabularies: dict[str, tuple[str, ...]],
    benchmark_efforts: dict[str, ReasoningEffort],
) -> dict[str, float]:
    if model.decision_action_heads is None:
        return {}
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    step_correct_counts: dict[tuple[str, str], int] = defaultdict(int)
    step_total_counts: dict[tuple[str, str], int] = defaultdict(int)
    policy_cache: dict[str, object] = {}
    with torch.no_grad():
        for benchmark_name, examples in examples_by_benchmark.items():
            if not examples:
                continue
            correct = 0
            total = 0
            for start in range(0, len(examples), batch_size):
                batch_examples = examples[start : start + batch_size]
                encoded = [encode_example_sequence(ReasoningTextExample(
                    benchmark=example.benchmark,
                    text=example.text,
                    trajectory_id=example.trajectory_id,
                    step_index=0,
                    trace_step=example.trace_step,
                    auxiliary_targets=None,
                ), seq_len=seq_len) for example in batch_examples]
                token_batch = torch.tensor([tokens for tokens, _mask in encoded], dtype=torch.long, device=device)
                attention_mask = torch.tensor([mask for _tokens, mask in encoded], dtype=torch.long, device=device)
                inputs = token_batch[:, :-1]
                attention = attention_mask[:, :-1]
                if benchmark_name not in policy_cache:
                    policy_cache[benchmark_name] = reasoning_budget_policy_for_benchmark(
                        benchmark_name,
                        effort=benchmark_efforts.get(benchmark_name, "deep"),
                        attention_window=model.config.attention.sliding_window,
                    )
                budget = policy_cache[benchmark_name].build_inference_budget(
                    model.config,
                    prompt_tokens=inputs.size(1),
                    use_kv_cache=False,
                    max_new_tokens=0,
                )
                if budget.max_prompt_tokens is not None and inputs.size(1) > budget.max_prompt_tokens:
                    inputs = inputs[:, -budget.max_prompt_tokens :]
                    attention = attention[:, -budget.max_prompt_tokens :]
                outputs = model(inputs, attention_mask=attention, budget=budget)
                max_head_size = max((len(vocabulary) for vocabulary in full_action_vocabularies.values()), default=1)
                full_candidate_masks = torch.zeros((len(batch_examples), max_head_size), dtype=torch.bool, device=device)
                full_candidate_name_ids = torch.zeros((len(batch_examples), max_head_size), dtype=torch.long, device=device)
                full_candidate_argument_ids = torch.zeros(
                    (len(batch_examples), max_head_size),
                    dtype=torch.long,
                    device=device,
                )
                for batch_index, example in enumerate(batch_examples):
                    full_mask = torch.tensor(example.full_candidate_mask, dtype=torch.bool, device=device)
                    full_candidate_masks[batch_index, : full_mask.numel()] = full_mask
                    name_ids = torch.tensor(example.full_candidate_name_ids, dtype=torch.long, device=device)
                    full_candidate_name_ids[batch_index, : name_ids.numel()] = name_ids
                    argument_ids = torch.tensor(example.full_candidate_argument_ids, dtype=torch.long, device=device)
                    full_candidate_argument_ids[batch_index, : argument_ids.numel()] = argument_ids
                predicted_ids = model.decision_action_heads.predict(
                    outputs.last_hidden_state,
                    attention_mask=attention,
                    benchmark_names=[example.benchmark for example in batch_examples],
                    output_heads=[example.output_head for example in batch_examples],
                    full_candidate_masks=full_candidate_masks,
                    full_candidate_name_ids=full_candidate_name_ids,
                    full_candidate_argument_ids=full_candidate_argument_ids,
                )
                targets = torch.tensor(
                    [example.target_action_id for example in batch_examples],
                    dtype=torch.long,
                    device=device,
                )
                batch_correct = (predicted_ids == targets)
                correct += int(batch_correct.sum().item())
                for example, is_correct in zip(batch_examples, batch_correct.tolist(), strict=True):
                    step_key = (benchmark_name, example.trace_step)
                    step_correct_counts[step_key] += int(bool(is_correct))
                    step_total_counts[step_key] += 1
                total += len(batch_examples)
            totals[f"{benchmark_name}_decision_action_accuracy"] = correct / max(total, 1)
            counts[benchmark_name] = total
    if not totals:
        return {}
    overall_correct = 0.0
    overall_total = 0
    for benchmark_name, accuracy in totals.items():
        short_name = benchmark_name.removesuffix("_decision_action_accuracy")
        count = counts.get(short_name, 0)
        overall_correct += accuracy * count
        overall_total += count
    for (benchmark_name, trace_step), total in sorted(step_total_counts.items()):
        totals[f"{benchmark_name}_decision_action_accuracy_{trace_step}"] = (
            step_correct_counts[(benchmark_name, trace_step)] / max(total, 1)
        )
    totals["decision_action_accuracy_mean"] = overall_correct / max(overall_total, 1)
    return totals


def build_model_config(
    args: argparse.Namespace,
    *,
    architecture: str,
    vocab_size: int,
    decision_benchmark_adapter_names: Sequence[str],
    decision_output_sizes: dict[str, int],
    decision_name_output_sizes: dict[str, int],
    decision_argument_output_sizes: dict[str, int],
    oscar_auxiliary_vocabularies: dict[str, object],
    oscar_graph_auxiliary_vocabularies: dict[str, object],
) -> DecoderModelConfig:
    attention = AttentionBackendConfig.from_preset(
        args.attention_preset,
        latent_kv_dim=args.latent_kv_dim,
    )
    moe = MoEConfig()
    if architecture == "moe":
        moe = MoEConfig.reference(
            num_experts=args.num_experts,
            experts_per_token=args.experts_per_token,
            router_jitter_noise=args.router_jitter_noise,
            auxiliary_loss_weight=args.moe_auxiliary_loss_weight,
        )
    core_auxiliary = CoReAuxiliaryConfig()
    if args.core_max_rows > 0 and not args.disable_core_auxiliary_heads:
        core_auxiliary = CoReAuxiliaryConfig.reference(
            query_positive_loss_weight=args.core_query_positive_loss_weight,
            source_count_loss_weight=args.core_source_count_loss_weight,
            trace_length_loss_weight=args.core_trace_length_loss_weight,
            dependency_kind_loss_weight=args.core_dependency_kind_loss_weight,
            infoflow_data_edge_loss_weight=args.core_infoflow_data_edge_loss_weight,
            source_membership_loss_weight=args.core_source_membership_loss_weight,
            direct_edge_loss_weight=args.core_direct_edge_loss_weight,
        )
    oscar_auxiliary = OscarAuxiliaryConfig()
    if (
        oscar_auxiliary_vocabularies.get("doc_groups")
        and oscar_auxiliary_vocabularies.get("doc_titles")
        and oscar_auxiliary_vocabularies.get("section_paths")
        and not args.disable_oscar_auxiliary_heads
    ):
        oscar_auxiliary = OscarAuxiliaryConfig.reference(
            family_output_size=len(OSCAR_FAMILY_TO_ID),
            doc_group_output_size=len(tuple(oscar_auxiliary_vocabularies["doc_groups"])),
            doc_title_output_size=len(tuple(oscar_auxiliary_vocabularies["doc_titles"])),
            section_path_output_size=len(tuple(oscar_auxiliary_vocabularies["section_paths"])),
            concept_output_size=len(OSCAR_SCOPE_CONCEPT_TAGS),
            section_parent_output_size=len(tuple(oscar_auxiliary_vocabularies["section_paths"])),
            workflow_bottleneck_output_size=len(tuple(oscar_auxiliary_vocabularies["workflow_bottleneck_ids"])),
            workflow_kpi_output_size=len(tuple(oscar_auxiliary_vocabularies["workflow_kpi_ids"])),
            workflow_improvement_output_size=len(tuple(oscar_auxiliary_vocabularies["workflow_improvement_ids"])),
            workflow_motif_output_size=len(tuple(oscar_auxiliary_vocabularies["workflow_motif_ids"])),
            workflow_reward_bucket_output_size=len(tuple(oscar_auxiliary_vocabularies["workflow_reward_buckets"])),
            workflow_canonical_kpi_output_size=len(
                tuple(oscar_auxiliary_vocabularies["workflow_canonical_kpi_ids"])
            ),
            workflow_canonical_intervention_output_size=len(
                tuple(oscar_auxiliary_vocabularies["workflow_canonical_intervention_ids"])
            ),
            workflow_action_step_output_size=len(tuple(oscar_auxiliary_vocabularies["workflow_action_step_ids"])),
            family_loss_weight=args.oscar_family_loss_weight,
            section_depth_loss_weight=args.oscar_section_depth_loss_weight,
            doc_group_loss_weight=args.oscar_doc_group_loss_weight,
            doc_title_loss_weight=args.oscar_doc_title_loss_weight,
            section_path_loss_weight=args.oscar_section_path_loss_weight,
            concept_loss_weight=args.oscar_concept_loss_weight,
            section_parent_loss_weight=args.oscar_section_parent_loss_weight,
            related_doc_loss_weight=args.oscar_related_doc_loss_weight,
            workflow_kpi_loss_weight=args.oscar_workflow_kpi_loss_weight,
            workflow_improvement_loss_weight=args.oscar_workflow_improvement_loss_weight,
            workflow_motif_loss_weight=args.oscar_workflow_motif_loss_weight,
            workflow_reward_bucket_loss_weight=args.oscar_workflow_reward_bucket_loss_weight,
            workflow_reward_score_loss_weight=args.oscar_workflow_reward_score_loss_weight,
            workflow_action_step_loss_weight=args.oscar_workflow_action_step_loss_weight,
            workflow_action_kpi_loss_weight=args.oscar_workflow_action_kpi_loss_weight,
            workflow_action_intervention_loss_weight=args.oscar_workflow_action_intervention_loss_weight,
        )
    oscar_graph_auxiliary = OscarGraphAuxiliaryConfig()
    if (
        oscar_graph_auxiliary_vocabularies.get("graph_domains")
        and not args.disable_oscar_graph_auxiliary_heads
    ):
        oscar_graph_auxiliary = OscarGraphAuxiliaryConfig.reference(
            family_output_size=len(OSCAR_GRAPH_FAMILY_TO_ID),
            domain_output_size=len(tuple(oscar_graph_auxiliary_vocabularies["graph_domains"])),
            motif_output_size=len(tuple(oscar_graph_auxiliary_vocabularies.get("rollout_motifs", ()))),
            rollout_max_steps=int(oscar_graph_auxiliary_vocabularies.get("rollout_max_steps", 0)),
            family_loss_weight=args.oscar_graph_family_loss_weight,
            domain_loss_weight=args.oscar_graph_domain_loss_weight,
            relation_loss_weight=args.oscar_graph_relation_loss_weight,
            neighbor_loss_weight=args.oscar_graph_neighbor_loss_weight,
            path_via_loss_weight=args.oscar_graph_path_via_loss_weight,
            path_target_loss_weight=args.oscar_graph_path_target_loss_weight,
            grounding_loss_weight=args.oscar_graph_grounding_loss_weight,
            rollout_motif_loss_weight=args.oscar_graph_rollout_motif_loss_weight,
            rollout_step_loss_weight=args.oscar_graph_rollout_step_loss_weight,
        )
    decision_action = DecisionActionConfig()
    if decision_output_sizes and not args.disable_decision_action_heads:
        projection_hidden_size = None
        if args.decision_action_projection_hidden_size > 0:
            projection_hidden_size = args.decision_action_projection_hidden_size
        decision_action = DecisionActionConfig.reference(
            projection_hidden_size=projection_hidden_size,
            action_loss_weight=args.decision_action_loss_weight,
            benchmark_adapter_names=tuple(decision_benchmark_adapter_names),
            benchmark_output_sizes=tuple(sorted((name, int(size)) for name, size in decision_output_sizes.items())),
            benchmark_name_output_sizes=tuple(
                sorted((name, int(size)) for name, size in decision_name_output_sizes.items())
            ),
            benchmark_argument_output_sizes=tuple(
                sorted((name, int(size)) for name, size in decision_argument_output_sizes.items())
            ),
        )
    return DecoderModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max(args.seq_len + 1, 4096),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        attention=attention,
        moe=moe,
        core_auxiliary=core_auxiliary,
        oscar_auxiliary=oscar_auxiliary,
        oscar_graph_auxiliary=oscar_graph_auxiliary,
        decision_action=decision_action,
    )


def count_parameters(model: torch.nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        try:
            total += parameter.numel()
        except ValueError:
            continue
    return total


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def serialize_reasoning_example(example: ReasoningTextExample) -> dict[str, Any]:
    return asdict(example)


def deserialize_reasoning_example(payload: dict[str, Any]) -> ReasoningTextExample:
    return ReasoningTextExample(
        benchmark=str(payload["benchmark"]),
        text=str(payload["text"]),
        trajectory_id=str(payload["trajectory_id"]),
        step_index=int(payload["step_index"]),
        trace_step=str(payload["trace_step"]),
        auxiliary_targets=payload.get("auxiliary_targets"),
    )


def write_reasoning_examples(path: Path, examples: Sequence[ReasoningTextExample]) -> None:
    write_jsonl(path, [serialize_reasoning_example(example) for example in examples])


def read_reasoning_examples(path: Path) -> tuple[ReasoningTextExample, ...]:
    return tuple(deserialize_reasoning_example(payload) for payload in read_jsonl(path))


def windows_to_tensor(windows: Sequence[Sequence[int]]) -> torch.Tensor:
    if torch is None:
        raise SystemExit("Torch is required to export packed token windows.")
    if not windows:
        return torch.empty((0, 0), dtype=torch.int32)
    return torch.tensor(windows, dtype=torch.int32)


def save_window_tensor(path: Path, windows: Sequence[Sequence[int]]) -> None:
    if torch is None:
        raise SystemExit("Torch is required to export packed token windows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(windows_to_tensor(windows), path)


def load_window_tensor(path: Path) -> torch.Tensor:
    if torch is None:
        raise SystemExit("Torch is required to load packed token windows.")
    if not path.exists():
        return torch.empty((0, 0), dtype=torch.int32)
    return torch.load(path, map_location="cpu")


def normalize_window_container(windows: Sequence[Sequence[int]] | torch.Tensor) -> Sequence[Sequence[int]] | torch.Tensor:
    if torch is not None and isinstance(windows, torch.Tensor):
        return windows.to(dtype=torch.int32, device="cpu")
    return [list(window) for window in windows]


def window_count(windows: Sequence[Sequence[int]] | torch.Tensor) -> int:
    if torch is not None and isinstance(windows, torch.Tensor):
        return int(windows.size(0))
    return len(windows)


def resolved_tokenizer_load_path(tokenizer_summary: dict[str, object], *, manifest_dir: Path | None = None) -> str:
    candidate = str(
        tokenizer_summary.get("load_path")
        or tokenizer_summary.get("save_path")
        or ""
    ).strip()
    if not candidate:
        return ""
    path = Path(candidate)
    if not path.is_absolute() and manifest_dir is not None:
        path = manifest_dir / path
    return str(path.resolve())


def load_tokenizer_from_summary(
    tokenizer_summary: dict[str, object],
    *,
    manifest_dir: Path | None = None,
) -> ReasoningTokenizer:
    kind = str(tokenizer_summary["kind"])
    load_path = resolved_tokenizer_load_path(tokenizer_summary, manifest_dir=manifest_dir)
    return build_reasoning_tokenizer(
        (),
        kind=kind,
        vocab_size=int(tokenizer_summary.get("requested_vocab_size", tokenizer_summary["vocab_size"])),
        task=str(tokenizer_summary.get("task", "generic")),
        min_freq=2,
        candidate_pool_size=2048,
        max_piece_chars=24,
        fit_workers=1,
        fit_verbose=False,
        load_path=load_path,
        save_path="",
    )


def build_decision_artifacts(
    train_examples: dict[str, tuple[ReasoningTextExample, ...]],
    val_examples: dict[str, tuple[ReasoningTextExample, ...]],
) -> dict[str, object]:
    raw_train_action_examples = {
        benchmark_name: convert_decision_action_examples(examples)[0]
        for benchmark_name, examples in train_examples.items()
        if benchmark_name != "core"
    }
    raw_val_action_examples = {
        benchmark_name: convert_decision_action_examples(examples)[0]
        for benchmark_name, examples in val_examples.items()
        if benchmark_name != "core"
    }
    combined_examples = {
        benchmark_name: tuple(raw_train_action_examples.get(benchmark_name, ()))
        + tuple(raw_val_action_examples.get(benchmark_name, ()))
        for benchmark_name in sorted(set(raw_train_action_examples) | set(raw_val_action_examples))
    }
    decision_head_vocabularies = build_decision_head_vocabularies(combined_examples)
    decision_full_action_vocabularies = decision_head_vocabularies["full_action_vocabularies"]
    decision_name_vocabularies = decision_head_vocabularies["name_vocabularies"]
    decision_argument_vocabularies = decision_head_vocabularies["argument_vocabularies"]
    decision_full_action_components = decision_head_vocabularies["full_action_components"]
    decision_candidate_masks = build_decision_candidate_masks(
        combined_examples,
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
    val_action_examples = finalize_decision_action_examples(
        raw_val_action_examples,
        full_action_vocabularies=decision_full_action_vocabularies,
        name_vocabularies=decision_name_vocabularies,
        argument_vocabularies=decision_argument_vocabularies,
        full_action_components=decision_full_action_components,
        candidate_masks=decision_candidate_masks,
    )
    decision_benchmark_adapter_names = tuple(
        sorted(
            {
                benchmark_name
                for examples_by_benchmark in (train_action_examples, val_action_examples)
                for benchmark_name, examples in examples_by_benchmark.items()
                if examples
            }
        )
    )
    return {
        "train_action_examples": train_action_examples,
        "val_action_examples": val_action_examples,
        "decision_benchmark_adapter_names": decision_benchmark_adapter_names,
        "decision_full_action_vocabularies": decision_full_action_vocabularies,
        "decision_name_vocabularies": decision_name_vocabularies,
        "decision_argument_vocabularies": decision_argument_vocabularies,
    }


def windows_by_benchmark(
    examples_by_benchmark: dict[str, tuple[ReasoningTextExample, ...]],
    *,
    seq_len: int,
) -> dict[str, list[list[int]]]:
    return {
        name: chunk_token_stream(texts_from_examples(examples), seq_len=seq_len)
        for name, examples in examples_by_benchmark.items()
    }


def finalize_dataset_bundle(
    args: argparse.Namespace,
    *,
    train_examples: dict[str, tuple[ReasoningTextExample, ...]],
    val_examples: dict[str, tuple[ReasoningTextExample, ...]],
    tokenizer: ReasoningTokenizer,
    train_windows: dict[str, Sequence[Sequence[int]]] | None = None,
    val_windows: dict[str, Sequence[Sequence[int]]] | None = None,
) -> dict[str, object]:
    set_text_tokenizer(tokenizer)
    resolved_train_windows = (
        {name: normalize_window_container(windows) for name, windows in train_windows.items()}
        if train_windows is not None
        else windows_by_benchmark(train_examples, seq_len=args.seq_len)
    )
    resolved_val_windows = (
        {name: normalize_window_container(windows) for name, windows in val_windows.items()}
        if val_windows is not None
        else windows_by_benchmark(val_examples, seq_len=args.seq_len)
    )
    decision_artifacts = build_decision_artifacts(train_examples, val_examples)
    benchmarks = [name for name, windows in resolved_train_windows.items() if window_count(windows) > 0]
    return {
        "train_examples": train_examples,
        "val_examples": val_examples,
        "train_action_examples": decision_artifacts["train_action_examples"],
        "val_action_examples": decision_artifacts["val_action_examples"],
        "decision_benchmark_adapter_names": decision_artifacts["decision_benchmark_adapter_names"],
        "decision_full_action_vocabularies": decision_artifacts["decision_full_action_vocabularies"],
        "decision_name_vocabularies": decision_artifacts["decision_name_vocabularies"],
        "decision_argument_vocabularies": decision_artifacts["decision_argument_vocabularies"],
        "train_windows": resolved_train_windows,
        "val_windows": resolved_val_windows,
        "benchmarks": benchmarks,
        "tokenizer": tokenizer,
        "tokenizer_summary": tokenizer.summary(),
    }


def export_dataset_bundle(
    *,
    args: argparse.Namespace,
    bundle: dict[str, object],
    export_dir: Path,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = export_dir / "examples"
    windows_dir = export_dir / "windows"
    train_examples = bundle["train_examples"]
    val_examples = bundle["val_examples"]
    train_windows = bundle["train_windows"]
    val_windows = bundle["val_windows"]
    train_example_paths: dict[str, str] = {}
    val_example_paths: dict[str, str] = {}
    train_window_paths: dict[str, str] = {}
    val_window_paths: dict[str, str] = {}
    for benchmark_name, examples in train_examples.items():
        path = examples_dir / f"train_{benchmark_name}.jsonl"
        write_reasoning_examples(path, examples)
        train_example_paths[benchmark_name] = str(path.resolve())
    for benchmark_name, examples in val_examples.items():
        path = examples_dir / f"val_{benchmark_name}.jsonl"
        write_reasoning_examples(path, examples)
        val_example_paths[benchmark_name] = str(path.resolve())
    for benchmark_name, windows in train_windows.items():
        path = windows_dir / f"train_{benchmark_name}.pt"
        save_window_tensor(path, windows)
        train_window_paths[benchmark_name] = str(path.resolve())
    for benchmark_name, windows in val_windows.items():
        path = windows_dir / f"val_{benchmark_name}.pt"
        save_window_tensor(path, windows)
        val_window_paths[benchmark_name] = str(path.resolve())
    tokenizer_summary = dict(bundle["tokenizer_summary"])
    manifest = {
        "format_version": 1,
        "created_at_unix_s": time.time(),
        "seq_len": args.seq_len,
        "seed": args.seed,
        "tokenizer": tokenizer_summary,
        "tokenizer_path": resolved_tokenizer_load_path(tokenizer_summary, manifest_dir=export_dir),
        "active_train_benchmarks": list(bundle["benchmarks"]),
        "train_examples": train_example_paths,
        "val_examples": val_example_paths,
        "train_windows": train_window_paths,
        "val_windows": val_window_paths,
        "train_example_counts": {name: len(examples) for name, examples in train_examples.items()},
        "val_example_counts": {name: len(examples) for name, examples in val_examples.items()},
        "train_window_counts": {name: len(windows) for name, windows in train_windows.items()},
        "val_window_counts": {name: len(windows) for name, windows in val_windows.items()},
        "export_args": {
            "include_verifier_targets": args.include_verifier_targets,
            "include_dclm": args.include_dclm,
            "dclm_dataset_id": args.dclm_dataset_id,
            "dclm_split": args.dclm_split,
            "dclm_text_field": args.dclm_text_field,
            "dclm_max_documents": args.dclm_max_documents,
            "dclm_shuffle": args.dclm_shuffle,
            "dclm_shuffle_buffer_size": args.dclm_shuffle_buffer_size,
            "dclm_min_text_chars": args.dclm_min_text_chars,
            "dclm_min_language_score": args.dclm_min_language_score,
            "dclm_min_fasttext_score": args.dclm_min_fasttext_score,
            "include_oscar_scope": args.include_oscar_scope,
            "include_oscar_scope_reasoning": args.include_oscar_scope_reasoning,
            "include_oscar_graph_reasoning": args.include_oscar_graph_reasoning,
            "validation_fraction": args.validation_fraction,
            "core_graph_backend": args.core_graph_backend,
            "architectures": list(args.architectures),
            "oscar_scope_views": list(args.oscar_scope_views),
            "oscar_scope_reasoning_families": list(args.oscar_scope_reasoning_families),
            "oscar_graph_reasoning_families": list(args.oscar_graph_reasoning_families),
        },
    }
    manifest_path = export_dir / DEFAULT_EXPORT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_exported_dataset_bundle(args: argparse.Namespace, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_summary = dict(manifest["tokenizer"])
    tokenizer_path = str(manifest.get("tokenizer_path", "")).strip()
    if tokenizer_path:
        tokenizer_summary["load_path"] = tokenizer_path
    tokenizer = load_tokenizer_from_summary(tokenizer_summary, manifest_dir=manifest_path.parent)
    train_examples = {
        benchmark_name: read_reasoning_examples(Path(path))
        for benchmark_name, path in manifest.get("train_examples", {}).items()
    }
    val_examples = {
        benchmark_name: read_reasoning_examples(Path(path))
        for benchmark_name, path in manifest.get("val_examples", {}).items()
    }
    train_windows = {
        benchmark_name: load_window_tensor(Path(path))
        for benchmark_name, path in manifest.get("train_windows", {}).items()
    }
    val_windows = {
        benchmark_name: load_window_tensor(Path(path))
        for benchmark_name, path in manifest.get("val_windows", {}).items()
    }
    return finalize_dataset_bundle(
        args,
        train_examples=train_examples,
        val_examples=val_examples,
        tokenizer=tokenizer,
        train_windows=train_windows,
        val_windows=val_windows,
    )


def effort_for_stage(
    args: argparse.Namespace,
    *,
    benchmark_name: str,
    stage: str,
) -> ReasoningEffort:
    default_effort = getattr(args, f"{stage}_reasoning_effort")
    override_effort = getattr(args, f"{benchmark_name}_{stage}_effort", None)
    return resolve_effort(default_effort, override_effort)


def require_training_runtime() -> None:
    if torch is None or F is None or DecoderLanguageModel is None:
        raise SystemExit(
            "This integrated training script requires torch and the local model stack. "
            "Install requirements-models.txt or use .venv_atari."
        )


def build_dataset_bundle(args: argparse.Namespace) -> dict[str, object]:
    arc_train = build_arc_reasoning_examples(
        num_episodes=args.arc_train_episodes,
        seed_start=args.seed * 1000,
        include_verifier_targets=args.include_verifier_targets,
    )
    arc_val = build_arc_reasoning_examples(
        num_episodes=args.arc_val_episodes,
        seed_start=(args.seed * 1000) + 100_000,
        include_verifier_targets=args.include_verifier_targets,
    )

    gsm8k_train = ()
    gsm8k_val = ()
    if args.gsm8k_max_rows > 0:
        gsm8k_all = build_gsm8k_reasoning_examples(
            data_dir=args.gsm8k_data_dir,
            max_rows=args.gsm8k_max_rows,
            include_verifier_targets=False,
        )
        gsm8k_train, gsm8k_val = split_examples(gsm8k_all, validation_fraction=args.validation_fraction)

    mmlu_train = ()
    mmlu_val = ()
    if args.mmlu_max_rows > 0:
        mmlu_all = build_mmlu_reasoning_examples(
            data_dir=args.mmlu_data_dir,
            max_rows=args.mmlu_max_rows,
            include_verifier_targets=False,
        )
        mmlu_train, mmlu_val = split_examples(mmlu_all, validation_fraction=args.validation_fraction)

    mmlu_pro_eval = ()
    if args.mmlu_pro_max_rows > 0:
        mmlu_pro_eval = build_mmlu_pro_reasoning_examples(
            max_rows=args.mmlu_pro_max_rows,
            include_verifier_targets=False,
        )

    mmlu_redux_eval = ()
    if args.mmlu_redux_max_rows > 0:
        mmlu_redux_eval = build_mmlu_redux_reasoning_examples(
            max_rows=args.mmlu_redux_max_rows,
            include_verifier_targets=False,
            label_mode=args.mmlu_redux_label_mode,
        )

    olympiad_math_train = ()
    olympiad_math_val = ()
    if args.olympiad_math_max_rows > 0:
        olympiad_math_all = build_olympiad_math_reasoning_examples(
            configs=args.olympiad_math_configs,
            max_rows=args.olympiad_math_max_rows,
            include_verifier_targets=False,
        )
        olympiad_math_train, olympiad_math_val = split_examples(
            olympiad_math_all,
            validation_fraction=args.validation_fraction,
        )

    core_train = ()
    core_val = ()
    if args.core_max_rows > 0:
        core_all = build_core_reasoning_examples(
            data_dir=args.core_data_dir,
            max_rows=args.core_max_rows,
            graph_backend=args.core_graph_backend,
        )
        core_train, core_val = split_examples(core_all, validation_fraction=args.validation_fraction)

    dclm_train = ()
    dclm_val = ()
    if args.include_dclm and args.dclm_max_documents > 0:
        dclm_all = build_dclm_text_examples(
            dataset_id=args.dclm_dataset_id,
            split=args.dclm_split,
            text_field=args.dclm_text_field,
            max_documents=args.dclm_max_documents,
            shuffle=args.dclm_shuffle,
            shuffle_buffer_size=args.dclm_shuffle_buffer_size,
            seed=args.seed,
            min_text_chars=args.dclm_min_text_chars,
            min_language_score=args.dclm_min_language_score,
            min_fasttext_score=args.dclm_min_fasttext_score,
        )
        dclm_train, dclm_val = split_examples(
            dclm_all,
            validation_fraction=args.validation_fraction,
        )

    oscar_scope_train = ()
    oscar_scope_val = ()
    if args.include_oscar_scope:
        oscar_scope_native_all = build_oscar_scope_examples(
            roots=args.oscar_scope_roots,
            paths=args.oscar_scope_paths,
            auto_discover=args.oscar_scope_auto_discover,
            max_documents=None if args.oscar_scope_max_documents <= 0 else args.oscar_scope_max_documents,
            max_chunks=None if args.oscar_scope_max_chunks <= 0 else args.oscar_scope_max_chunks,
            views=args.oscar_scope_views,
        )
        native_train, native_val = split_examples(
            oscar_scope_native_all,
            validation_fraction=args.validation_fraction,
        )
        oscar_scope_train = native_train
        oscar_scope_val = native_val
    oscar_scope_reasoning_train = ()
    oscar_scope_reasoning_val = ()
    if args.include_oscar_scope_reasoning and args.oscar_scope_reasoning_max_examples > 0:
        oscar_scope_reasoning_all = build_oscar_scope_reasoning_examples(
            roots=args.oscar_scope_roots,
            paths=args.oscar_scope_paths,
            auto_discover=args.oscar_scope_auto_discover,
            max_documents=None if args.oscar_scope_max_documents <= 0 else args.oscar_scope_max_documents,
            max_examples=args.oscar_scope_reasoning_max_examples,
            views=args.oscar_scope_views,
            families=args.oscar_scope_reasoning_families,
        )
        oscar_scope_reasoning_train, oscar_scope_reasoning_val = split_examples(
            oscar_scope_reasoning_all,
            validation_fraction=args.validation_fraction,
        )
    oscar_graph_reasoning_train = ()
    oscar_graph_reasoning_val = ()
    if args.include_oscar_graph_reasoning and args.oscar_graph_reasoning_max_examples > 0:
        oscar_graph_reasoning_all = build_oscar_graph_reasoning_examples(
            max_examples=args.oscar_graph_reasoning_max_examples,
            families=args.oscar_graph_reasoning_families,
        )
        oscar_graph_reasoning_train, oscar_graph_reasoning_val = split_examples(
            oscar_graph_reasoning_all,
            validation_fraction=args.validation_fraction,
        )

    train_examples = {
        "arc": arc_train,
        "core": core_train,
        "dclm": dclm_train,
        "gsm8k": gsm8k_train,
        "mmlu": mmlu_train,
        "oscar_graph_reasoning": oscar_graph_reasoning_train,
        "oscar_scope": oscar_scope_train,
        "oscar_scope_reasoning": oscar_scope_reasoning_train,
        "olympiad_math": olympiad_math_train,
    }
    val_examples = {
        "arc": arc_val,
        "core": core_val,
        "dclm": dclm_val,
        "gsm8k": gsm8k_val,
        "mmlu": mmlu_val,
        "mmlu_pro": mmlu_pro_eval,
        "mmlu_redux": mmlu_redux_eval,
        "oscar_graph_reasoning": oscar_graph_reasoning_val,
        "oscar_scope": oscar_scope_val,
        "oscar_scope_reasoning": oscar_scope_reasoning_val,
        "olympiad_math": olympiad_math_val,
    }
    train_texts_by_benchmark = {
        name: list(texts_from_examples(examples))
        for name, examples in train_examples.items()
    }
    tokenizer_fit_texts = [
        text
        for benchmark_texts in train_texts_by_benchmark.values()
        for text in benchmark_texts
    ]
    tokenizer = build_reasoning_tokenizer(
        tokenizer_fit_texts,
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
    return finalize_dataset_bundle(
        args,
        train_examples=train_examples,
        val_examples=val_examples,
        tokenizer=tokenizer,
    )


def build_data_only_payload(args: argparse.Namespace, bundle: dict[str, object]) -> dict[str, object]:
    train_examples = bundle["train_examples"]
    val_examples = bundle["val_examples"]
    train_action_examples = bundle["train_action_examples"]
    val_action_examples = bundle["val_action_examples"]
    decision_full_action_vocabularies = bundle["decision_full_action_vocabularies"]
    decision_name_vocabularies = bundle["decision_name_vocabularies"]
    decision_argument_vocabularies = bundle["decision_argument_vocabularies"]
    train_windows = bundle["train_windows"]
    val_windows = bundle["val_windows"]
    return {
        "mode": "data_only",
        "seed": args.seed,
        "include_verifier_targets": args.include_verifier_targets,
        "include_oscar_scope": args.include_oscar_scope,
        "include_oscar_scope_reasoning": args.include_oscar_scope_reasoning,
        "include_oscar_graph_reasoning": args.include_oscar_graph_reasoning,
        "seq_len": args.seq_len,
        "validation_fraction": args.validation_fraction,
        "core_graph_backend": args.core_graph_backend,
        "oscar_scope_views": list(args.oscar_scope_views),
        "oscar_scope_reasoning_families": list(args.oscar_scope_reasoning_families),
        "oscar_graph_reasoning_families": list(args.oscar_graph_reasoning_families),
        "tokenizer": bundle["tokenizer_summary"],
        "active_train_benchmarks": list(bundle["benchmarks"]),
        "train_example_counts": {name: len(examples) for name, examples in train_examples.items()},
        "val_example_counts": {name: len(examples) for name, examples in val_examples.items()},
        "train_action_example_counts": {name: len(examples) for name, examples in train_action_examples.items()},
        "val_action_example_counts": {name: len(examples) for name, examples in val_action_examples.items()},
        "decision_head_sizes": {
            name: len(vocabulary) for name, vocabulary in decision_full_action_vocabularies.items()
        },
        "decision_benchmark_adapter_names": list(bundle["decision_benchmark_adapter_names"]),
        "decision_name_head_sizes": {
            name: len(vocabulary) for name, vocabulary in decision_name_vocabularies.items()
        },
        "decision_argument_head_sizes": {
            name: len(vocabulary) for name, vocabulary in decision_argument_vocabularies.items()
        },
        "train_trace_step_counts": {
            name: dict(sorted(Counter(example.trace_step for example in examples).items()))
            for name, examples in train_examples.items()
        },
        "val_trace_step_counts": {
            name: dict(sorted(Counter(example.trace_step for example in examples).items()))
            for name, examples in val_examples.items()
        },
        "train_task_kind_counts": {
            name: dict(
                sorted(
                    Counter(str((example.auxiliary_targets or {}).get("task_kind", "none")) for example in examples).items()
                )
            )
            for name, examples in train_examples.items()
        },
        "val_task_kind_counts": {
            name: dict(
                sorted(
                    Counter(str((example.auxiliary_targets or {}).get("task_kind", "none")) for example in examples).items()
                )
            )
            for name, examples in val_examples.items()
        },
        "train_window_counts": {name: len(windows) for name, windows in train_windows.items()},
        "val_window_counts": {name: len(windows) for name, windows in val_windows.items()},
    }


def write_payload(args: argparse.Namespace, payload: dict[str, object]) -> None:
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def result_rows_for_csv(results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            key: (json.dumps(value) if isinstance(value, (list, dict)) else value)
            for key, value in result.items()
        }
        for result in results
    ]


def artifact_root(args: argparse.Namespace) -> Path | None:
    for candidate in (args.output, args.csv_output):
        if candidate:
            return Path(candidate).parent
    return None


def progress_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    root = artifact_root(args)
    if root is None:
        return {
            "progress_json": None,
            "partial_json": None,
            "partial_csv": None,
        }
    return {
        "progress_json": root / "progress.json",
        "partial_json": root / "partial_summary.json",
        "partial_csv": root / "partial_summary.csv",
    }


def checkpoint_root(args: argparse.Namespace) -> Path | None:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    root = artifact_root(args)
    if root is None:
        return None
    return root / "checkpoints"


def build_training_payload(
    *,
    args: argparse.Namespace,
    device_type: str,
    tokenizer_summary: dict[str, object],
    results: Sequence[dict[str, object]],
    distributed_context: DistributedContext,
    precision: str,
    corpus_manifest: str = "",
) -> dict[str, object]:
    return {
        "seed": args.seed,
        "device": device_type,
        "precision": precision,
        "distributed": {
            "enabled": distributed_context.enabled,
            "world_size": distributed_context.world_size,
            "rank": distributed_context.rank,
            "backend": distributed_context.backend,
        },
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "force_full_train_layers": args.force_full_train_layers,
        "corpus_manifest": corpus_manifest,
        "core_graph_backend": args.core_graph_backend,
        "tokenizer": tokenizer_summary,
        "results": list(results),
    }


def write_partial_results(
    *,
    args: argparse.Namespace,
    device_type: str,
    tokenizer_summary: dict[str, object],
    results: Sequence[dict[str, object]],
    paths: dict[str, Path | None],
    distributed_context: DistributedContext,
    precision: str,
    corpus_manifest: str = "",
) -> None:
    payload = build_training_payload(
        args=args,
        device_type=device_type,
        tokenizer_summary=tokenizer_summary,
        results=results,
        distributed_context=distributed_context,
        precision=precision,
        corpus_manifest=corpus_manifest,
    )
    write_json(paths["partial_json"], payload)
    if paths["partial_csv"] is not None:
        write_csv(str(paths["partial_csv"]), result_rows_for_csv(results))


def write_progress_snapshot(
    *,
    args: argparse.Namespace,
    device_type: str,
    paths: dict[str, Path | None],
    status: str,
    total_architectures: int,
    completed_architectures: int,
    current_architecture: str | None = None,
    step_index: int | None = None,
    total_steps: int | None = None,
    current_benchmark: str | None = None,
    elapsed_s: float | None = None,
    train_main_loss_mean: float | None = None,
    train_decision_aux_loss_mean: float | None = None,
    partial_results: Sequence[dict[str, object]] = (),
) -> None:
    payload: dict[str, object] = {
        "status": status,
        "seed": args.seed,
        "device": device_type,
        "core_graph_backend": args.core_graph_backend,
        "total_architectures": total_architectures,
        "completed_architectures": completed_architectures,
        "current_architecture": current_architecture,
        "current_step": step_index,
        "total_steps": total_steps,
        "current_benchmark": current_benchmark,
        "elapsed_s": elapsed_s,
        "train_main_loss_mean": train_main_loss_mean,
        "train_decision_aux_loss_mean": train_decision_aux_loss_mean,
        "completed_result_summaries": [
            {
                "architecture": result.get("architecture"),
                "val_loss_mean": result.get("val_loss_mean"),
                "decision_action_accuracy_mean": (
                    result.get("decision_action_eval_metrics", {}) if isinstance(result.get("decision_action_eval_metrics"), dict) else {}
                ).get("decision_action_accuracy_mean"),
            }
            for result in partial_results
        ],
        "updated_at_unix_s": time.time(),
    }
    write_json(paths["progress_json"], payload)


def write_data_only_csv(args: argparse.Namespace, payload: dict[str, object]) -> None:
    if not args.csv_output:
        return
    row: dict[str, object] = {
        "mode": payload["mode"],
        "seed": payload["seed"],
        "include_verifier_targets": payload["include_verifier_targets"],
        "seq_len": payload["seq_len"],
        "validation_fraction": payload["validation_fraction"],
        "core_graph_backend": payload["core_graph_backend"],
        "tokenizer_kind": payload["tokenizer"]["kind"],
        "tokenizer_vocab_size": payload["tokenizer"]["vocab_size"],
        "active_train_benchmarks_json": json.dumps(payload["active_train_benchmarks"]),
    }
    for prefix, values in (
        ("train_examples", payload["train_example_counts"]),
        ("val_examples", payload["val_example_counts"]),
        ("train_action_examples", payload["train_action_example_counts"]),
        ("val_action_examples", payload["val_action_example_counts"]),
        ("decision_head_sizes", payload["decision_head_sizes"]),
        ("train_windows", payload["train_window_counts"]),
        ("val_windows", payload["val_window_counts"]),
    ):
        for name, value in values.items():
            row[f"{prefix}_{name}"] = value
    write_csv(args.csv_output, [row])


def build_optimizer_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    resolved_total_steps = max(int(total_steps), 1)
    resolved_warmup_steps = max(0, min(int(warmup_steps), resolved_total_steps - 1))
    resolved_min_scale = float(max(0.0, min(min_learning_rate_scale, 1.0)))

    def lr_lambda(step: int) -> float:
        if resolved_warmup_steps > 0 and step < resolved_warmup_steps:
            return float(step + 1) / float(resolved_warmup_steps)
        if resolved_total_steps <= resolved_warmup_steps:
            return 1.0
        decay_progress = (step - resolved_warmup_steps) / float(max(resolved_total_steps - resolved_warmup_steps, 1))
        decay_progress = max(0.0, min(decay_progress, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return resolved_min_scale + ((1.0 - resolved_min_scale) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_budget_for_stage(
    *,
    args: argparse.Namespace,
    model_config: DecoderModelConfig,
    benchmark_name: str,
    stage: str,
    prompt_tokens: int,
    use_kv_cache: bool,
    max_new_tokens: int | None,
) -> tuple[object, object]:
    policy = reasoning_budget_policy_for_benchmark(
        benchmark_name,
        effort=effort_for_stage(args, benchmark_name=benchmark_name, stage=stage),
        attention_window=model_config.attention.sliding_window,
    )
    budget = policy.build_inference_budget(
        model_config,
        prompt_tokens=prompt_tokens,
        use_kv_cache=use_kv_cache,
        max_new_tokens=max_new_tokens,
    )
    if stage == "train" and args.force_full_train_layers:
        budget = replace(budget, active_layers=model_config.num_hidden_layers)
    return policy, budget


def resolved_budget_policy_payload(
    *,
    args: argparse.Namespace,
    model_config: DecoderModelConfig,
    benchmark_name: str,
    stage: str,
    prompt_tokens: int,
    use_kv_cache: bool,
    max_new_tokens: int | None = None,
) -> dict[str, object]:
    policy, budget = build_budget_for_stage(
        args=args,
        model_config=model_config,
        benchmark_name=benchmark_name,
        stage=stage,
        prompt_tokens=prompt_tokens,
        use_kv_cache=use_kv_cache,
        max_new_tokens=max_new_tokens,
    )
    resolved = asdict(budget.resolve(model_config))
    resolved["benchmark"] = benchmark_name
    resolved["effort"] = effort_for_stage(args, benchmark_name=benchmark_name, stage=stage)
    resolved["structured_decode_mode_hint"] = getattr(policy, "structured_decode_mode_hint", None)
    return resolved


def latest_checkpoint_pointer(architecture_dir: Path) -> Path:
    return architecture_dir / "latest.txt"


def resolve_resume_checkpoint_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_dir():
        pointer = latest_checkpoint_pointer(candidate)
        if pointer.exists():
            resolved = pointer.read_text(encoding="utf-8").strip()
            if not resolved:
                raise SystemExit(f"Checkpoint pointer is empty: {pointer}")
            return Path(resolved)
        checkpoints = sorted(candidate.glob("step_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        raise SystemExit(f"No checkpoints found under {candidate}")
    return candidate


def save_training_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_root_dir: Path | None,
    architecture: str,
    architecture_index: int,
    step_index: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config: DecoderModelConfig,
    tokenizer_summary: dict[str, object],
    results: Sequence[dict[str, object]],
    train_state: dict[str, object],
) -> Path | None:
    if checkpoint_root_dir is None:
        return None
    architecture_dir = checkpoint_root_dir / architecture
    architecture_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = architecture_dir / f"step_{step_index:08d}.pt"
    torch.save(
        {
            "format_version": 1,
            "seed": args.seed,
            "architecture": architecture,
            "architecture_index": architecture_index,
            "step_index": step_index,
            "model_config": asdict(model_config),
            "tokenizer_summary": tokenizer_summary,
            "results": list(results),
            "train_state": train_state,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args_snapshot": {
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "decision_batch_size": args.decision_batch_size,
                "steps": args.steps,
                "grad_accumulation_steps": args.grad_accumulation_steps,
                "precision": args.precision,
                "force_full_train_layers": args.force_full_train_layers,
            },
        },
        checkpoint_path,
    )
    latest_checkpoint_pointer(architecture_dir).write_text(str(checkpoint_path.resolve()), encoding="utf-8")
    return checkpoint_path


def load_training_checkpoint(path: Path, *, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    distributed_context = initialize_distributed(args.device)
    paths = progress_paths(args)
    checkpoint_dir = checkpoint_root(args)
    manifest_path = Path(args.corpus_manifest).resolve() if args.corpus_manifest else None
    export_manifest_path: Path | None = None
    corpus_manifest_str = str(manifest_path) if manifest_path is not None else ""

    if distributed_context.enabled and not args.corpus_manifest:
        raise SystemExit(
            "Distributed training requires a pre-exported corpus manifest. "
            "Run scripts/export_integrated_reasoning_corpus.py first, then pass --corpus-manifest."
        )

    if args.export_dir and args.tokenizer in {"epiplex", "rust_bpe"} and not args.tokenizer_load and not args.tokenizer_save:
        args.tokenizer_save = str((Path(args.export_dir).resolve() / "reasoning_tokenizer.json"))

    if distributed_context.is_primary:
        write_progress_snapshot(
            args=args,
            device_type=args.device,
            paths=paths,
            status="building_data",
            total_architectures=len(args.architectures),
            completed_architectures=0,
        )

    bundle = (
        load_exported_dataset_bundle(args, manifest_path)
        if manifest_path is not None
        else build_dataset_bundle(args)
    )

    if args.export_dir and distributed_context.is_primary:
        export_manifest_path = export_dataset_bundle(
            args=args,
            bundle=bundle,
            export_dir=Path(args.export_dir).resolve(),
        )
        if manifest_path is None:
            corpus_manifest_str = str(export_manifest_path)

    distributed_barrier(distributed_context)

    if args.data_only:
        payload = build_data_only_payload(args, bundle)
        if export_manifest_path is not None:
            payload["export_manifest"] = str(export_manifest_path)
        if distributed_context.is_primary:
            write_payload(args, payload)
            write_data_only_csv(args, payload)
            print(json.dumps(payload, indent=2))
        distributed_barrier(distributed_context)
        if distributed_context.enabled and dist is not None and dist.is_initialized():
            dist.destroy_process_group()
        return

    require_training_runtime()
    device = resolve_runtime_device(args, distributed_context)
    precision = precision_mode(args, device)
    grad_accumulation_steps = max(int(args.grad_accumulation_steps), 1)
    checkpoint_every = max(int(args.checkpoint_every), 1)
    progress_enabled = bool(args.progress and tqdm is not None and distributed_context.is_primary)

    train_examples = bundle["train_examples"]
    val_examples = bundle["val_examples"]
    train_action_examples = bundle["train_action_examples"]
    val_action_examples = bundle["val_action_examples"]
    decision_full_action_vocabularies = bundle["decision_full_action_vocabularies"]
    decision_name_vocabularies = bundle["decision_name_vocabularies"]
    decision_argument_vocabularies = bundle["decision_argument_vocabularies"]
    decision_benchmark_adapter_names = bundle["decision_benchmark_adapter_names"]
    train_windows = bundle["train_windows"]
    val_windows = bundle["val_windows"]
    benchmarks = bundle["benchmarks"]
    tokenizer_summary = bundle["tokenizer_summary"]
    oscar_auxiliary_vocabularies = build_oscar_auxiliary_vocabularies(
        tuple(train_examples.get("oscar_scope_reasoning", ())) + tuple(val_examples.get("oscar_scope_reasoning", ()))
    )
    oscar_graph_auxiliary_vocabularies = build_oscar_graph_auxiliary_vocabularies(
        tuple(train_examples.get("oscar_graph_reasoning", ())) + tuple(val_examples.get("oscar_graph_reasoning", ()))
    )
    if not benchmarks:
        raise SystemExit("No train windows were generated for the integrated reasoning stack.")

    resume_payload: dict[str, object] | None = None
    resume_architecture_index = -1
    resume_architecture_name = ""
    if args.resume_from:
        resume_checkpoint = resolve_resume_checkpoint_path(args.resume_from)
        resume_payload = load_training_checkpoint(resume_checkpoint, device=torch.device("cpu"))
        resume_architecture_index = int(resume_payload.get("architecture_index", -1))
        resume_architecture_name = str(resume_payload.get("architecture", ""))

    results: list[dict[str, object]] = list(resume_payload.get("results", [])) if resume_payload is not None else []
    architecture_iterator: Iterable[str] = args.architectures
    if progress_enabled:
        architecture_iterator = tqdm(args.architectures, desc="architectures", leave=True)
    if distributed_context.is_primary:
        write_progress_snapshot(
            args=args,
            device_type=device.type,
            paths=paths,
            status="starting",
            total_architectures=len(args.architectures),
            completed_architectures=len(results),
            partial_results=results,
        )

    try:
        for architecture_index, architecture in enumerate(architecture_iterator):
            if resume_payload is not None and architecture_index < resume_architecture_index:
                continue

            model_config = build_model_config(
                args,
                architecture=architecture,
                vocab_size=int(tokenizer_summary["vocab_size"]),
                decision_benchmark_adapter_names=decision_benchmark_adapter_names,
                decision_output_sizes={
                    name: len(vocabulary)
                    for name, vocabulary in decision_full_action_vocabularies.items()
                },
                decision_name_output_sizes={
                    name: len(vocabulary)
                    for name, vocabulary in decision_name_vocabularies.items()
                },
                decision_argument_output_sizes={
                    name: len(vocabulary)
                    for name, vocabulary in decision_argument_vocabularies.items()
                },
                oscar_auxiliary_vocabularies=oscar_auxiliary_vocabularies,
                oscar_graph_auxiliary_vocabularies=oscar_graph_auxiliary_vocabularies,
            )
            base_model = DecoderLanguageModel(model_config).to(device)
            model: torch.nn.Module = base_model
            if distributed_context.enabled:
                model = DistributedDataParallel(
                    base_model,
                    device_ids=[distributed_context.local_rank] if device.type == "cuda" else None,
                    broadcast_buffers=False,
                )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = build_optimizer_scheduler(
                optimizer,
                total_steps=args.steps,
                warmup_steps=args.lr_warmup_steps,
                min_learning_rate_scale=args.min_learning_rate_scale,
            )
            benchmark_cycle = [benchmarks[index % len(benchmarks)] for index in range(args.steps)]
            train_main_loss_total = 0.0
            train_aux_loss_total = 0.0
            train_task_aux_loss_total = 0.0
            train_decision_aux_loss_total = 0.0
            train_router_entropy_total = 0.0
            train_task_aux_metric_totals: dict[str, float] = {}
            train_task_aux_steps = 0
            train_decision_metric_totals: dict[str, float] = {}
            train_decision_steps = 0
            train_tokens = 0
            elapsed_offset = 0.0
            start_step = 0

            if resume_payload is not None and architecture_index == resume_architecture_index:
                if architecture != resume_architecture_name:
                    raise SystemExit(
                        f"Resume checkpoint architecture mismatch: expected {resume_architecture_name}, got {architecture}."
                    )
                unwrap_model(model).load_state_dict(resume_payload["model_state"])
                optimizer.load_state_dict(resume_payload["optimizer_state"])
                scheduler.load_state_dict(resume_payload["scheduler_state"])
                train_state = dict(resume_payload.get("train_state", {}))
                start_step = int(resume_payload.get("step_index", 0))
                elapsed_offset = float(train_state.get("elapsed_s_total", 0.0))
                train_main_loss_total = float(train_state.get("train_main_loss_total", 0.0))
                train_aux_loss_total = float(train_state.get("train_aux_loss_total", 0.0))
                train_task_aux_loss_total = float(train_state.get("train_task_aux_loss_total", 0.0))
                train_decision_aux_loss_total = float(train_state.get("train_decision_aux_loss_total", 0.0))
                train_router_entropy_total = float(train_state.get("train_router_entropy_total", 0.0))
                train_task_aux_metric_totals = {
                    name: float(value)
                    for name, value in dict(
                        train_state.get(
                            "train_task_aux_metric_totals",
                            train_state.get("train_core_aux_metric_totals", {}),
                        )
                    ).items()
                }
                train_task_aux_steps = int(
                    train_state.get("train_task_aux_steps", train_state.get("train_core_aux_steps", 0))
                )
                train_decision_metric_totals = {
                    name: float(value)
                    for name, value in dict(train_state.get("train_decision_metric_totals", {})).items()
                }
                train_decision_steps = int(train_state.get("train_decision_steps", 0))
                train_tokens = int(train_state.get("train_tokens", 0))

            if distributed_context.is_primary:
                write_progress_snapshot(
                    args=args,
                    device_type=device.type,
                    paths=paths,
                    status="training",
                    total_architectures=len(args.architectures),
                    completed_architectures=len(results),
                    current_architecture=architecture,
                    step_index=start_step,
                    total_steps=args.steps,
                    partial_results=results,
                )

            step_progress = None
            step_iterator: Iterable[int] = range(start_step + 1, args.steps + 1)
            if progress_enabled:
                step_progress = tqdm(step_iterator, desc=f"{architecture} train", leave=False)
                step_iterator = step_progress
            start_time = time.perf_counter()
            model.train()

            for step_index in step_iterator:
                benchmark_name = benchmark_cycle[step_index - 1]
                optimizer.zero_grad(set_to_none=True)
                for micro_step_index in range(grad_accumulation_steps):
                    rng = sample_rng_for_step(
                        seed=args.seed,
                        distributed_context=distributed_context,
                        architecture_index=architecture_index,
                        step_index=step_index,
                        micro_step_index=micro_step_index,
                    )
                    attention_mask = None
                    task_name = None
                    task_auxiliary_labels = None
                    if benchmark_name == "core" and model_config.core_auxiliary.enabled and train_examples["core"]:
                        inputs, targets, attention_mask, task_auxiliary_labels = sample_core_batch(
                            train_examples["core"],
                            batch_size=args.batch_size,
                            rng=rng,
                            device=device,
                            seq_len=args.seq_len,
                        )
                        task_name = "core"
                    elif (
                        benchmark_name == "oscar_scope_reasoning"
                        and model_config.oscar_auxiliary.enabled
                        and train_examples["oscar_scope_reasoning"]
                    ):
                        inputs, targets, attention_mask, task_auxiliary_labels = sample_oscar_auxiliary_batch(
                            train_examples["oscar_scope_reasoning"],
                            batch_size=args.batch_size,
                            rng=rng,
                            device=device,
                            seq_len=args.seq_len,
                            vocabularies=oscar_auxiliary_vocabularies,
                        )
                        task_name = "oscar_scope_reasoning"
                    elif (
                        benchmark_name == "oscar_graph_reasoning"
                        and model_config.oscar_graph_auxiliary.enabled
                        and train_examples["oscar_graph_reasoning"]
                    ):
                        inputs, targets, attention_mask, task_auxiliary_labels = sample_oscar_graph_auxiliary_batch(
                            train_examples["oscar_graph_reasoning"],
                            batch_size=args.batch_size,
                            rng=rng,
                            device=device,
                            seq_len=args.seq_len,
                            vocabularies=oscar_graph_auxiliary_vocabularies,
                        )
                        task_name = "oscar_graph_reasoning"
                    else:
                        batch = sample_batch(
                            train_windows[benchmark_name],
                            batch_size=args.batch_size,
                            rng=rng,
                            device=device,
                        )
                        inputs = batch[:, :-1]
                        targets = batch[:, 1:]

                    _train_policy, train_budget = build_budget_for_stage(
                        args=args,
                        model_config=model_config,
                        benchmark_name=benchmark_name,
                        stage="train",
                        prompt_tokens=inputs.size(1),
                        use_kv_cache=False,
                        max_new_tokens=0,
                    )
                    inputs, targets = trim_batch_to_budget(inputs, targets, train_budget.max_prompt_tokens)
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, -inputs.size(1) :]

                    sync_context = (
                        model.no_sync()
                        if distributed_context.enabled and micro_step_index + 1 < grad_accumulation_steps
                        else nullcontext()
                    )
                    with sync_context:
                        with autocast_context(device, precision):
                            outputs = model(
                                inputs,
                                attention_mask=attention_mask,
                                budget=train_budget,
                                task_name=task_name,
                                task_auxiliary_labels=task_auxiliary_labels,
                            )
                            main_loss = cross_entropy_loss(outputs.logits, targets)
                            aux_loss = (
                                outputs.auxiliary_loss
                                if outputs.auxiliary_loss is not None
                                else main_loss.new_zeros(())
                            )
                            task_aux_loss = (
                                outputs.task_auxiliary_loss
                                if outputs.task_auxiliary_loss is not None
                                else main_loss.new_zeros(())
                            )
                            decision_aux_loss = main_loss.new_zeros(())
                            decision_outputs = None
                            if (
                                unwrap_model(model).decision_action_heads is not None
                                and benchmark_name != "core"
                                and train_action_examples.get(benchmark_name)
                            ):
                                decision_inputs, decision_attention_mask, decision_labels = sample_decision_action_batch(
                                    train_action_examples[benchmark_name],
                                    batch_size=args.decision_batch_size,
                                    rng=rng,
                                    device=device,
                                    seq_len=args.seq_len,
                                    full_action_vocabularies=decision_full_action_vocabularies,
                                    name_vocabularies=decision_name_vocabularies,
                                    argument_vocabularies=decision_argument_vocabularies,
                                )
                                _decision_policy, decision_budget = build_budget_for_stage(
                                    args=args,
                                    model_config=model_config,
                                    benchmark_name=benchmark_name,
                                    stage="train",
                                    prompt_tokens=decision_inputs.size(1),
                                    use_kv_cache=False,
                                    max_new_tokens=0,
                                )
                                if (
                                    decision_budget.max_prompt_tokens is not None
                                    and decision_inputs.size(1) > decision_budget.max_prompt_tokens
                                ):
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
                            total_loss = (
                                main_loss + aux_loss + task_aux_loss + decision_aux_loss
                            ) / grad_accumulation_steps
                        total_loss.backward()

                    train_main_loss_total += float(main_loss.detach().item())
                    train_aux_loss_total += float(aux_loss.detach().item())
                    train_task_aux_loss_total += float(task_aux_loss.detach().item())
                    train_decision_aux_loss_total += float(decision_aux_loss.detach().item())
                    if outputs.router_entropy is not None:
                        train_router_entropy_total += float(outputs.router_entropy.detach().item())
                    if outputs.task_auxiliary_metrics is not None:
                        for name, value in outputs.task_auxiliary_metrics.items():
                            train_task_aux_metric_totals[name] = train_task_aux_metric_totals.get(name, 0.0) + float(value)
                        train_task_aux_steps += 1
                    if decision_outputs is not None and decision_outputs.task_auxiliary_metrics is not None:
                        for name, value in decision_outputs.task_auxiliary_metrics.items():
                            train_decision_metric_totals[name] = train_decision_metric_totals.get(name, 0.0) + float(value)
                        train_decision_steps += 1
                    train_tokens += int(inputs.numel())

                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                if step_progress is not None:
                    step_progress.set_postfix(
                        benchmark=benchmark_name,
                        lm=f"{train_main_loss_total / max(step_index * grad_accumulation_steps, 1):.3f}",
                        dec=f"{train_decision_aux_loss_total / max(step_index * grad_accumulation_steps, 1):.3f}",
                    )

                if distributed_context.is_primary and (step_index % checkpoint_every == 0 or step_index == args.steps):
                    elapsed_so_far = elapsed_offset + max(time.perf_counter() - start_time, 0.0)
                    write_progress_snapshot(
                        args=args,
                        device_type=device.type,
                        paths=paths,
                        status="training",
                        total_architectures=len(args.architectures),
                        completed_architectures=len(results),
                        current_architecture=architecture,
                        step_index=step_index,
                        total_steps=args.steps,
                        current_benchmark=benchmark_name,
                        elapsed_s=elapsed_so_far,
                        train_main_loss_mean=train_main_loss_total / max(step_index * grad_accumulation_steps, 1),
                        train_decision_aux_loss_mean=train_decision_aux_loss_total / max(step_index * grad_accumulation_steps, 1),
                        partial_results=results,
                    )
                    save_training_checkpoint(
                        args=args,
                        checkpoint_root_dir=checkpoint_dir,
                        architecture=architecture,
                        architecture_index=architecture_index,
                        step_index=step_index,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        model_config=model_config,
                        tokenizer_summary=tokenizer_summary,
                        results=results,
                        train_state={
                            "elapsed_s_total": elapsed_so_far,
                            "train_main_loss_total": train_main_loss_total,
                            "train_aux_loss_total": train_aux_loss_total,
                            "train_task_aux_loss_total": train_task_aux_loss_total,
                            "train_decision_aux_loss_total": train_decision_aux_loss_total,
                            "train_router_entropy_total": train_router_entropy_total,
                            "train_task_aux_metric_totals": train_task_aux_metric_totals,
                            "train_task_aux_steps": train_task_aux_steps,
                            "train_decision_metric_totals": train_decision_metric_totals,
                            "train_decision_steps": train_decision_steps,
                            "train_tokens": train_tokens,
                        },
                    )

            synchronize_device(device)
            elapsed = elapsed_offset + max(time.perf_counter() - start_time, 1e-9)
            distributed_barrier(distributed_context)

            if distributed_context.is_primary:
                eval_model = unwrap_model(model)
                write_progress_snapshot(
                    args=args,
                    device_type=device.type,
                    paths=paths,
                    status="evaluating",
                    total_architectures=len(args.architectures),
                    completed_architectures=len(results),
                    current_architecture=architecture,
                    step_index=args.steps,
                    total_steps=args.steps,
                    elapsed_s=elapsed,
                    train_main_loss_mean=train_main_loss_total / max(args.steps * grad_accumulation_steps, 1),
                    train_decision_aux_loss_mean=train_decision_aux_loss_total / max(args.steps * grad_accumulation_steps, 1),
                    partial_results=results,
                )
                val_losses = {
                    benchmark_name: evaluate_benchmark_loss(
                        eval_model,
                        windows,
                        batch_size=args.batch_size,
                        device=device,
                        benchmark_name=benchmark_name,
                        effort=effort_for_stage(args, benchmark_name=benchmark_name, stage="eval"),
                    )
                    for benchmark_name, windows in val_windows.items()
                }
                mean_val_loss = sum(loss for loss in val_losses.values() if loss == loss) / max(
                    sum(int(loss == loss) for loss in val_losses.values()),
                    1,
                )
                train_budget_policies = {
                    benchmark_name: resolved_budget_policy_payload(
                        args=args,
                        model_config=model_config,
                        benchmark_name=benchmark_name,
                        stage="train",
                        prompt_tokens=args.seq_len,
                        use_kv_cache=False,
                        max_new_tokens=0,
                    )
                    for benchmark_name in benchmarks
                }
                eval_budget_policies = {
                    benchmark_name: resolved_budget_policy_payload(
                        args=args,
                        model_config=model_config,
                        benchmark_name=benchmark_name,
                        stage="eval",
                        prompt_tokens=args.seq_len,
                        use_kv_cache=False,
                        max_new_tokens=0,
                    )
                    for benchmark_name in val_windows
                }
                generation_budget_hints = {
                    benchmark_name: resolved_budget_policy_payload(
                        args=args,
                        model_config=model_config,
                        benchmark_name=benchmark_name,
                        stage="eval",
                        prompt_tokens=args.seq_len,
                        use_kv_cache=True,
                        max_new_tokens=None,
                    )
                    for benchmark_name in val_windows
                }
                core_aux_eval_metrics = evaluate_core_auxiliary(
                    eval_model,
                    val_examples["core"],
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                    effort=effort_for_stage(args, benchmark_name="core", stage="eval"),
                ) if val_examples["core"] and model_config.core_auxiliary.enabled else {}
                oscar_aux_eval_metrics = evaluate_oscar_auxiliary(
                    eval_model,
                    val_examples["oscar_scope_reasoning"],
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                    effort=effort_for_stage(args, benchmark_name="oscar_scope_reasoning", stage="eval"),
                    vocabularies=oscar_auxiliary_vocabularies,
                ) if val_examples["oscar_scope_reasoning"] and model_config.oscar_auxiliary.enabled else {}
                oscar_graph_aux_eval_metrics = evaluate_oscar_graph_auxiliary(
                    eval_model,
                    val_examples["oscar_graph_reasoning"],
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                    effort=effort_for_stage(args, benchmark_name="oscar_graph_reasoning", stage="eval"),
                    vocabularies=oscar_graph_auxiliary_vocabularies,
                ) if val_examples["oscar_graph_reasoning"] and model_config.oscar_graph_auxiliary.enabled else {}
                decision_action_eval_metrics = evaluate_decision_action_accuracy(
                    eval_model,
                    val_action_examples,
                    batch_size=args.decision_batch_size,
                    seq_len=args.seq_len,
                    device=device,
                    full_action_vocabularies=decision_full_action_vocabularies,
                    benchmark_efforts={
                        benchmark_name: effort_for_stage(args, benchmark_name=benchmark_name, stage="eval")
                        for benchmark_name in val_action_examples
                    },
                ) if any(val_action_examples.values()) and eval_model.decision_action_heads is not None else {}
                expert_fraction = None
                if architecture == "moe":
                    probe_rng = random.Random(args.seed + architecture_index + 17)
                    probe_batch = sample_batch(train_windows[benchmarks[0]], batch_size=1, rng=probe_rng, device=device)
                    with torch.no_grad():
                        _probe_policy, probe_budget = build_budget_for_stage(
                            args=args,
                            model_config=model_config,
                            benchmark_name=benchmarks[0],
                            stage="eval",
                            prompt_tokens=probe_batch[:, :-1].size(1),
                            use_kv_cache=False,
                            max_new_tokens=0,
                        )
                        probe_outputs = eval_model(
                            probe_batch[:, :-1],
                            budget=probe_budget,
                        )
                    if probe_outputs.expert_fraction is not None:
                        expert_fraction = [
                            float(value)
                            for value in probe_outputs.expert_fraction.detach().cpu().tolist()
                        ]

                results.append(
                    {
                        "architecture": architecture,
                        "attention_preset": args.attention_preset,
                        "core_graph_backend": args.core_graph_backend,
                        "device": device.type,
                        "precision": precision,
                        "distributed_world_size": distributed_context.world_size,
                        "parameter_count": count_parameters(eval_model),
                        "train_steps": args.steps,
                        "grad_accumulation_steps": grad_accumulation_steps,
                        "force_full_train_layers": args.force_full_train_layers,
                        "corpus_manifest": corpus_manifest_str,
                        "resume_from": args.resume_from,
                        "train_tokens_per_second": train_tokens / elapsed,
                        "train_main_loss_mean": train_main_loss_total / max(args.steps * grad_accumulation_steps, 1),
                        "train_aux_loss_mean": train_aux_loss_total / max(args.steps * grad_accumulation_steps, 1),
                        "train_task_aux_loss_mean": train_task_aux_loss_total / max(args.steps * grad_accumulation_steps, 1),
                        "train_decision_aux_loss_mean": train_decision_aux_loss_total / max(args.steps * grad_accumulation_steps, 1),
                        "train_router_entropy_mean": train_router_entropy_total / max(args.steps * grad_accumulation_steps, 1),
                        "train_task_aux_metrics": {
                            name: value / max(train_task_aux_steps, 1)
                            for name, value in train_task_aux_metric_totals.items()
                        },
                        "train_decision_aux_metrics": {
                            name: value / max(train_decision_steps, 1)
                            for name, value in train_decision_metric_totals.items()
                        },
                        "val_loss_mean": mean_val_loss,
                        "benchmark_val_losses": val_losses,
                        "arc_val_loss": val_losses.get("arc"),
                        "core_val_loss": val_losses.get("core"),
                        "dclm_val_loss": val_losses.get("dclm"),
                        "gsm8k_val_loss": val_losses.get("gsm8k"),
                        "mmlu_val_loss": val_losses.get("mmlu"),
                        "mmlu_pro_val_loss": val_losses.get("mmlu_pro"),
                        "mmlu_redux_val_loss": val_losses.get("mmlu_redux"),
                        "oscar_graph_reasoning_val_loss": val_losses.get("oscar_graph_reasoning"),
                        "oscar_scope_val_loss": val_losses.get("oscar_scope"),
                        "oscar_scope_reasoning_val_loss": val_losses.get("oscar_scope_reasoning"),
                        "olympiad_math_val_loss": val_losses.get("olympiad_math"),
                        "arc_train_examples": len(train_examples["arc"]),
                        "arc_val_examples": len(val_examples["arc"]),
                        "core_train_examples": len(train_examples["core"]),
                        "core_val_examples": len(val_examples["core"]),
                        "dclm_train_examples": len(train_examples["dclm"]),
                        "dclm_val_examples": len(val_examples["dclm"]),
                        "gsm8k_train_examples": len(train_examples["gsm8k"]),
                        "gsm8k_val_examples": len(val_examples["gsm8k"]),
                        "mmlu_train_examples": len(train_examples["mmlu"]),
                        "mmlu_val_examples": len(val_examples["mmlu"]),
                        "oscar_graph_reasoning_train_examples": len(train_examples["oscar_graph_reasoning"]),
                        "oscar_graph_reasoning_val_examples": len(val_examples["oscar_graph_reasoning"]),
                        "oscar_scope_train_examples": len(train_examples["oscar_scope"]),
                        "oscar_scope_val_examples": len(val_examples["oscar_scope"]),
                        "oscar_scope_reasoning_train_examples": len(train_examples["oscar_scope_reasoning"]),
                        "oscar_scope_reasoning_val_examples": len(val_examples["oscar_scope_reasoning"]),
                        "olympiad_math_train_examples": len(train_examples["olympiad_math"]),
                        "mmlu_pro_eval_examples": len(val_examples["mmlu_pro"]),
                        "mmlu_redux_eval_examples": len(val_examples["mmlu_redux"]),
                        "olympiad_math_eval_examples": len(val_examples["olympiad_math"]),
                        "train_action_examples": {name: len(examples) for name, examples in train_action_examples.items()},
                        "val_action_examples": {name: len(examples) for name, examples in val_action_examples.items()},
                        "decision_head_sizes": {
                            name: len(vocabulary)
                            for name, vocabulary in decision_full_action_vocabularies.items()
                        },
                        "decision_name_head_sizes": {
                            name: len(vocabulary)
                            for name, vocabulary in decision_name_vocabularies.items()
                        },
                        "decision_argument_head_sizes": {
                            name: len(vocabulary)
                            for name, vocabulary in decision_argument_vocabularies.items()
                        },
                        "train_budget_policies": train_budget_policies,
                        "eval_budget_policies": eval_budget_policies,
                        "generation_budget_hints": generation_budget_hints,
                        "core_aux_eval_metrics": core_aux_eval_metrics,
                        "oscar_aux_eval_metrics": oscar_aux_eval_metrics,
                        "oscar_graph_aux_eval_metrics": oscar_graph_aux_eval_metrics,
                        "oscar_auxiliary_vocab_sizes": {
                            "doc_groups": len(tuple(oscar_auxiliary_vocabularies.get("doc_groups", ()))),
                            "doc_titles": len(tuple(oscar_auxiliary_vocabularies.get("doc_titles", ()))),
                            "section_paths": len(tuple(oscar_auxiliary_vocabularies.get("section_paths", ()))),
                            "concept_tags": len(OSCAR_SCOPE_CONCEPT_TAGS),
                        },
                        "oscar_graph_auxiliary_vocab_sizes": {
                            "graph_domains": len(tuple(oscar_graph_auxiliary_vocabularies.get("graph_domains", ()))),
                            "node_ids": len(tuple(oscar_graph_auxiliary_vocabularies.get("node_ids", ()))),
                            "node_categories": len(tuple(oscar_graph_auxiliary_vocabularies.get("node_categories", ()))),
                            "relations": len(tuple(oscar_graph_auxiliary_vocabularies.get("relations", ()))),
                            "rollout_motifs": len(tuple(oscar_graph_auxiliary_vocabularies.get("rollout_motifs", ()))),
                            "rollout_max_steps": int(oscar_graph_auxiliary_vocabularies.get("rollout_max_steps", 0)),
                        },
                        "decision_action_eval_metrics": decision_action_eval_metrics,
                        "expert_fraction": expert_fraction,
                    }
                )
                write_partial_results(
                    args=args,
                    device_type=device.type,
                    tokenizer_summary=tokenizer_summary,
                    results=results,
                    paths=paths,
                    distributed_context=distributed_context,
                    precision=precision,
                    corpus_manifest=corpus_manifest_str,
                )
                write_progress_snapshot(
                    args=args,
                    device_type=device.type,
                    paths=paths,
                    status="architecture_complete",
                    total_architectures=len(args.architectures),
                    completed_architectures=len(results),
                    current_architecture=architecture,
                    step_index=args.steps,
                    total_steps=args.steps,
                    elapsed_s=elapsed,
                    train_main_loss_mean=train_main_loss_total / max(args.steps * grad_accumulation_steps, 1),
                    train_decision_aux_loss_mean=train_decision_aux_loss_total / max(args.steps * grad_accumulation_steps, 1),
                    partial_results=results,
                )
                if progress_enabled and hasattr(architecture_iterator, "set_postfix"):
                    architecture_iterator.set_postfix(
                        done=f"{len(results)}/{len(args.architectures)}",
                        val=f"{results[-1]['val_loss_mean']:.3f}",
                    )
            distributed_barrier(distributed_context)

        if distributed_context.is_primary:
            payload = build_training_payload(
                args=args,
                device_type=device.type,
                tokenizer_summary=tokenizer_summary,
                results=results,
                distributed_context=distributed_context,
                precision=precision,
                corpus_manifest=corpus_manifest_str,
            )
            write_payload(args, payload)
            if args.csv_output:
                write_csv(args.csv_output, result_rows_for_csv(results))
            write_progress_snapshot(
                args=args,
                device_type=device.type,
                paths=paths,
                status="completed",
                total_architectures=len(args.architectures),
                completed_architectures=len(results),
                partial_results=results,
            )
            print(json.dumps(payload, indent=2))
    finally:
        distributed_barrier(distributed_context)
        if distributed_context.enabled and dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
