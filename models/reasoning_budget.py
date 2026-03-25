from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from .config import DecoderModelConfig, InferenceBudget, ResolvedInferenceBudget


BenchmarkName = Literal[
    "arc",
    "gsm8k",
    "mmlu",
    "mmlu_pro",
    "mmlu_redux",
    "core",
    "olympiad_math",
    "oscar_graph_reasoning",
    "oscar_scope",
    "oscar_scope_reasoning",
    "dclm",
]
ReasoningEffort = Literal["fast", "balanced", "deep"]


@dataclass(frozen=True)
class BenchmarkReasoningBudgetPolicy:
    benchmark: BenchmarkName
    effort: ReasoningEffort
    active_layer_fraction: float
    prompt_token_limit: int | None
    default_generation_tokens: int
    cache_token_buffer: int = 0
    attention_window: int | None = None
    use_kv_cache: bool = True
    structured_decode_mode_hint: str | None = None
    temperature: float = 0.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.active_layer_fraction <= 1.0:
            raise ValueError("active_layer_fraction must be in the interval (0, 1].")
        if self.prompt_token_limit is not None and self.prompt_token_limit <= 0:
            raise ValueError("prompt_token_limit must be positive when provided.")
        if self.default_generation_tokens < 0:
            raise ValueError("default_generation_tokens must be non-negative.")
        if self.cache_token_buffer < 0:
            raise ValueError("cache_token_buffer must be non-negative.")
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention_window must be positive when provided.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")

    def active_layers_for(self, config: DecoderModelConfig) -> int:
        return max(1, math.ceil(config.num_hidden_layers * self.active_layer_fraction))

    def build_inference_budget(
        self,
        config: DecoderModelConfig,
        *,
        prompt_tokens: int | None = None,
        target_tokens: int | None = None,
        use_kv_cache: bool | None = None,
        max_new_tokens: int | None = None,
        stop_token_ids: tuple[int, ...] = (),
    ) -> InferenceBudget:
        resolved_use_kv_cache = self.use_kv_cache if use_kv_cache is None else use_kv_cache
        resolved_prompt_tokens = self.prompt_token_limit
        if prompt_tokens is not None and resolved_prompt_tokens is not None:
            resolved_prompt_tokens = min(resolved_prompt_tokens, prompt_tokens)
        elif prompt_tokens is not None and resolved_prompt_tokens is None:
            resolved_prompt_tokens = prompt_tokens

        resolved_max_new_tokens = self.default_generation_tokens if max_new_tokens is None else max_new_tokens
        if target_tokens is not None and max_new_tokens is None:
            resolved_max_new_tokens = target_tokens

        max_cache_tokens = None
        if resolved_use_kv_cache:
            prompt_cache_tokens = resolved_prompt_tokens if resolved_prompt_tokens is not None else prompt_tokens
            if prompt_cache_tokens is not None:
                max_cache_tokens = prompt_cache_tokens + resolved_max_new_tokens + self.cache_token_buffer

        return InferenceBudget(
            max_prompt_tokens=resolved_prompt_tokens,
            max_new_tokens=resolved_max_new_tokens,
            active_layers=self.active_layers_for(config),
            attention_window=self.attention_window,
            max_cache_tokens=max_cache_tokens,
            use_kv_cache=resolved_use_kv_cache,
            stop_token_ids=stop_token_ids,
            temperature=self.temperature,
            top_k=self.top_k,
        )

    def resolved_budget_dict(
        self,
        config: DecoderModelConfig,
        *,
        prompt_tokens: int | None = None,
        target_tokens: int | None = None,
        use_kv_cache: bool | None = None,
        max_new_tokens: int | None = None,
        stop_token_ids: tuple[int, ...] = (),
    ) -> dict[str, object]:
        budget = self.build_inference_budget(
            config,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            use_kv_cache=use_kv_cache,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        ).resolve(config)
        payload = asdict(budget)
        payload["benchmark"] = self.benchmark
        payload["effort"] = self.effort
        payload["structured_decode_mode_hint"] = self.structured_decode_mode_hint
        return payload


_POLICY_TABLE: dict[BenchmarkName, dict[ReasoningEffort, dict[str, object]]] = {
    "arc": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 384,
            "default_generation_tokens": 96,
            "cache_token_buffer": 16,
            "structured_decode_mode_hint": "context_trie",
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 768,
            "default_generation_tokens": 128,
            "cache_token_buffer": 24,
            "structured_decode_mode_hint": "context_trie",
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 1536,
            "default_generation_tokens": 192,
            "cache_token_buffer": 32,
            "structured_decode_mode_hint": "context_trie",
        },
    },
    "gsm8k": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 256,
            "default_generation_tokens": 48,
            "cache_token_buffer": 8,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 512,
            "default_generation_tokens": 64,
            "cache_token_buffer": 16,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 1024,
            "default_generation_tokens": 96,
            "cache_token_buffer": 24,
        },
    },
    "mmlu": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 192,
            "default_generation_tokens": 24,
            "cache_token_buffer": 4,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 320,
            "default_generation_tokens": 32,
            "cache_token_buffer": 8,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 512,
            "default_generation_tokens": 48,
            "cache_token_buffer": 12,
        },
    },
    "mmlu_pro": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 320,
            "default_generation_tokens": 40,
            "cache_token_buffer": 8,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 512,
            "default_generation_tokens": 56,
            "cache_token_buffer": 12,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 768,
            "default_generation_tokens": 80,
            "cache_token_buffer": 16,
        },
    },
    "mmlu_redux": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 224,
            "default_generation_tokens": 28,
            "cache_token_buffer": 4,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 384,
            "default_generation_tokens": 40,
            "cache_token_buffer": 8,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 640,
            "default_generation_tokens": 56,
            "cache_token_buffer": 12,
        },
    },
    "core": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 768,
            "default_generation_tokens": 48,
            "cache_token_buffer": 16,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 1536,
            "default_generation_tokens": 80,
            "cache_token_buffer": 24,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 3072,
            "default_generation_tokens": 128,
            "cache_token_buffer": 32,
        },
    },
    "olympiad_math": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 640,
            "default_generation_tokens": 96,
            "cache_token_buffer": 16,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 1280,
            "default_generation_tokens": 160,
            "cache_token_buffer": 24,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 2048,
            "default_generation_tokens": 256,
            "cache_token_buffer": 32,
        },
    },
    "oscar_scope": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 1024,
            "default_generation_tokens": 64,
            "cache_token_buffer": 16,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 2048,
            "default_generation_tokens": 96,
            "cache_token_buffer": 24,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 3072,
            "default_generation_tokens": 160,
            "cache_token_buffer": 32,
        },
    },
    "oscar_scope_reasoning": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 1024,
            "default_generation_tokens": 96,
            "cache_token_buffer": 16,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 2048,
            "default_generation_tokens": 128,
            "cache_token_buffer": 24,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 3072,
            "default_generation_tokens": 192,
            "cache_token_buffer": 32,
        },
    },
    "oscar_graph_reasoning": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 1024,
            "default_generation_tokens": 96,
            "cache_token_buffer": 16,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 2048,
            "default_generation_tokens": 160,
            "cache_token_buffer": 24,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 3072,
            "default_generation_tokens": 224,
            "cache_token_buffer": 32,
        },
    },
    "dclm": {
        "fast": {
            "active_layer_fraction": 0.5,
            "prompt_token_limit": 1024,
            "default_generation_tokens": 64,
            "cache_token_buffer": 16,
        },
        "balanced": {
            "active_layer_fraction": 0.75,
            "prompt_token_limit": 2048,
            "default_generation_tokens": 96,
            "cache_token_buffer": 24,
        },
        "deep": {
            "active_layer_fraction": 1.0,
            "prompt_token_limit": 3072,
            "default_generation_tokens": 160,
            "cache_token_buffer": 32,
        },
    },
}


def reasoning_budget_policy_for_benchmark(
    benchmark: BenchmarkName,
    *,
    effort: ReasoningEffort = "balanced",
    attention_window: int | None = None,
) -> BenchmarkReasoningBudgetPolicy:
    try:
        spec = dict(_POLICY_TABLE[benchmark][effort])
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark/effort pair: {benchmark}/{effort}") from exc
    if attention_window is not None:
        spec["attention_window"] = attention_window
    return BenchmarkReasoningBudgetPolicy(
        benchmark=benchmark,
        effort=effort,
        **spec,
    )


def resolve_effort(
    default_effort: ReasoningEffort,
    override_effort: ReasoningEffort | None,
) -> ReasoningEffort:
    return override_effort if override_effort is not None else default_effort
