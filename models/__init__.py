from __future__ import annotations

from .config import (
    AttentionBackendConfig,
    AttentionBackendName,
    AttentionBackendPresetName,
    CoReAuxiliaryConfig,
    DecoderModelConfig,
    DecisionActionConfig,
    InferenceBudget,
    MoEConfig,
    OscarAuxiliaryConfig,
    OscarGraphAuxiliaryConfig,
    ResolvedInferenceBudget,
)
from .reasoning_budget import (
    BenchmarkName,
    BenchmarkReasoningBudgetPolicy,
    ReasoningEffort,
    reasoning_budget_policy_for_benchmark,
    resolve_effort,
)


_TORCH_EXPORTS = {
    "BenchmarkLatentPolicy",
    "DecoderKVCache",
    "DecoderLanguageModel",
    "DecoderModelOutput",
    "DecisionActionAuxiliaryHeads",
}


def __getattr__(name: str):
    if name in _TORCH_EXPORTS:
        from .benchmark_latent_policy import BenchmarkLatentPolicy
        from .cache import DecoderKVCache
        from .decoder import DecoderLanguageModel, DecoderModelOutput
        from .decision_auxiliary import DecisionActionAuxiliaryHeads

        exports = {
            "BenchmarkLatentPolicy": BenchmarkLatentPolicy,
            "DecoderKVCache": DecoderKVCache,
            "DecoderLanguageModel": DecoderLanguageModel,
            "DecoderModelOutput": DecoderModelOutput,
            "DecisionActionAuxiliaryHeads": DecisionActionAuxiliaryHeads,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttentionBackendConfig",
    "AttentionBackendName",
    "AttentionBackendPresetName",
    "BenchmarkLatentPolicy",
    "BenchmarkName",
    "BenchmarkReasoningBudgetPolicy",
    "CoReAuxiliaryConfig",
    "DecoderKVCache",
    "DecoderLanguageModel",
    "DecoderModelConfig",
    "DecoderModelOutput",
    "DecisionActionAuxiliaryHeads",
    "DecisionActionConfig",
    "InferenceBudget",
    "MoEConfig",
    "OscarAuxiliaryConfig",
    "OscarGraphAuxiliaryConfig",
    "ReasoningEffort",
    "ResolvedInferenceBudget",
    "reasoning_budget_policy_for_benchmark",
    "resolve_effort",
]
