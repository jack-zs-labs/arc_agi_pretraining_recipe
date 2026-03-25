from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple, cast


AttentionBackendName = Literal["eager", "sdpa", "hybrid", "mla", "sia", "sia_hybrid", "mla_sia"]
AttentionBackendPresetName = Literal["mla_default", "mla_sia_prefill_l1"]
ScaleInvariantDecodeMode = Literal["all_tokens", "prefill_only"]


@dataclass(frozen=True)
class CoReAuxiliaryConfig:
    enabled: bool = False
    projection_hidden_size: int | None = None
    query_positive_loss_weight: float = 0.1
    source_count_loss_weight: float = 0.05
    trace_length_loss_weight: float = 0.05
    dependency_kind_loss_weight: float = 0.05
    infoflow_data_edge_loss_weight: float = 0.05
    source_membership_loss_weight: float = 0.1
    direct_edge_loss_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.projection_hidden_size is not None and self.projection_hidden_size <= 0:
            raise ValueError("projection_hidden_size must be positive when provided.")
        for field_name in (
            "query_positive_loss_weight",
            "source_count_loss_weight",
            "trace_length_loss_weight",
            "dependency_kind_loss_weight",
            "infoflow_data_edge_loss_weight",
            "source_membership_loss_weight",
            "direct_edge_loss_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")

    @classmethod
    def reference(
        cls,
        *,
        projection_hidden_size: int | None = None,
        query_positive_loss_weight: float = 0.1,
        source_count_loss_weight: float = 0.05,
        trace_length_loss_weight: float = 0.05,
        dependency_kind_loss_weight: float = 0.05,
        infoflow_data_edge_loss_weight: float = 0.05,
        source_membership_loss_weight: float = 0.1,
        direct_edge_loss_weight: float = 0.05,
    ) -> "CoReAuxiliaryConfig":
        return cls(
            enabled=True,
            projection_hidden_size=projection_hidden_size,
            query_positive_loss_weight=query_positive_loss_weight,
            source_count_loss_weight=source_count_loss_weight,
            trace_length_loss_weight=trace_length_loss_weight,
            dependency_kind_loss_weight=dependency_kind_loss_weight,
            infoflow_data_edge_loss_weight=infoflow_data_edge_loss_weight,
            source_membership_loss_weight=source_membership_loss_weight,
            direct_edge_loss_weight=direct_edge_loss_weight,
        )

    def resolved_projection_hidden_size(self, model_hidden_size: int) -> int:
        return self.projection_hidden_size or model_hidden_size


@dataclass(frozen=True)
class OscarAuxiliaryConfig:
    enabled: bool = False
    projection_hidden_size: int | None = None
    family_output_size: int = 3
    section_depth_output_size: int = 5
    doc_group_output_size: int = 0
    doc_title_output_size: int = 0
    section_path_output_size: int = 0
    concept_output_size: int = 0
    section_parent_output_size: int = 0
    workflow_bottleneck_output_size: int = 0
    workflow_kpi_output_size: int = 0
    workflow_improvement_output_size: int = 0
    workflow_motif_output_size: int = 0
    workflow_reward_bucket_output_size: int = 0
    workflow_canonical_kpi_output_size: int = 0
    workflow_canonical_intervention_output_size: int = 0
    workflow_action_step_output_size: int = 0
    family_loss_weight: float = 0.05
    section_depth_loss_weight: float = 0.05
    doc_group_loss_weight: float = 0.05
    doc_title_loss_weight: float = 0.1
    section_path_loss_weight: float = 0.1
    concept_loss_weight: float = 0.1
    section_parent_loss_weight: float = 0.1
    related_doc_loss_weight: float = 0.1
    workflow_kpi_loss_weight: float = 0.1
    workflow_improvement_loss_weight: float = 0.1
    workflow_motif_loss_weight: float = 0.1
    workflow_reward_bucket_loss_weight: float = 0.1
    workflow_reward_score_loss_weight: float = 0.05
    workflow_action_step_loss_weight: float = 0.1
    workflow_action_kpi_loss_weight: float = 0.15
    workflow_action_intervention_loss_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.projection_hidden_size is not None and self.projection_hidden_size <= 0:
            raise ValueError("projection_hidden_size must be positive when provided.")
        for field_name in (
            "family_output_size",
            "section_depth_output_size",
            "doc_group_output_size",
            "doc_title_output_size",
            "section_path_output_size",
            "concept_output_size",
            "section_parent_output_size",
            "workflow_bottleneck_output_size",
            "workflow_kpi_output_size",
            "workflow_improvement_output_size",
            "workflow_motif_output_size",
            "workflow_reward_bucket_output_size",
            "workflow_canonical_kpi_output_size",
            "workflow_canonical_intervention_output_size",
            "workflow_action_step_output_size",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in (
            "family_loss_weight",
            "section_depth_loss_weight",
            "doc_group_loss_weight",
            "doc_title_loss_weight",
            "section_path_loss_weight",
            "concept_loss_weight",
            "section_parent_loss_weight",
            "related_doc_loss_weight",
            "workflow_kpi_loss_weight",
            "workflow_improvement_loss_weight",
            "workflow_motif_loss_weight",
            "workflow_reward_bucket_loss_weight",
            "workflow_reward_score_loss_weight",
            "workflow_action_step_loss_weight",
            "workflow_action_kpi_loss_weight",
            "workflow_action_intervention_loss_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.enabled:
            for field_name in (
                "doc_group_output_size",
                "doc_title_output_size",
                "section_path_output_size",
                "concept_output_size",
                "section_parent_output_size",
                "workflow_bottleneck_output_size",
                "workflow_kpi_output_size",
                "workflow_improvement_output_size",
                "workflow_motif_output_size",
                "workflow_reward_bucket_output_size",
                "workflow_canonical_kpi_output_size",
                "workflow_canonical_intervention_output_size",
                "workflow_action_step_output_size",
            ):
                if getattr(self, field_name) <= 0:
                    raise ValueError(f"{field_name} must be positive when Oscar auxiliary heads are enabled.")

    @classmethod
    def reference(
        cls,
        *,
        projection_hidden_size: int | None = None,
        family_output_size: int = 3,
        section_depth_output_size: int = 5,
        doc_group_output_size: int,
        doc_title_output_size: int,
        section_path_output_size: int,
        concept_output_size: int,
        section_parent_output_size: int,
        workflow_bottleneck_output_size: int,
        workflow_kpi_output_size: int,
        workflow_improvement_output_size: int,
        workflow_motif_output_size: int,
        workflow_reward_bucket_output_size: int,
        workflow_canonical_kpi_output_size: int,
        workflow_canonical_intervention_output_size: int,
        workflow_action_step_output_size: int,
        family_loss_weight: float = 0.05,
        section_depth_loss_weight: float = 0.05,
        doc_group_loss_weight: float = 0.05,
        doc_title_loss_weight: float = 0.1,
        section_path_loss_weight: float = 0.1,
        concept_loss_weight: float = 0.1,
        section_parent_loss_weight: float = 0.1,
        related_doc_loss_weight: float = 0.1,
        workflow_kpi_loss_weight: float = 0.1,
        workflow_improvement_loss_weight: float = 0.1,
        workflow_motif_loss_weight: float = 0.1,
        workflow_reward_bucket_loss_weight: float = 0.1,
        workflow_reward_score_loss_weight: float = 0.05,
        workflow_action_step_loss_weight: float = 0.1,
        workflow_action_kpi_loss_weight: float = 0.15,
        workflow_action_intervention_loss_weight: float = 0.15,
    ) -> "OscarAuxiliaryConfig":
        return cls(
            enabled=True,
            projection_hidden_size=projection_hidden_size,
            family_output_size=family_output_size,
            section_depth_output_size=section_depth_output_size,
            doc_group_output_size=doc_group_output_size,
            doc_title_output_size=doc_title_output_size,
            section_path_output_size=section_path_output_size,
            concept_output_size=concept_output_size,
            section_parent_output_size=section_parent_output_size,
            workflow_bottleneck_output_size=workflow_bottleneck_output_size,
            workflow_kpi_output_size=workflow_kpi_output_size,
            workflow_improvement_output_size=workflow_improvement_output_size,
            workflow_motif_output_size=workflow_motif_output_size,
            workflow_reward_bucket_output_size=workflow_reward_bucket_output_size,
            workflow_canonical_kpi_output_size=workflow_canonical_kpi_output_size,
            workflow_canonical_intervention_output_size=workflow_canonical_intervention_output_size,
            workflow_action_step_output_size=workflow_action_step_output_size,
            family_loss_weight=family_loss_weight,
            section_depth_loss_weight=section_depth_loss_weight,
            doc_group_loss_weight=doc_group_loss_weight,
            doc_title_loss_weight=doc_title_loss_weight,
            section_path_loss_weight=section_path_loss_weight,
            concept_loss_weight=concept_loss_weight,
            section_parent_loss_weight=section_parent_loss_weight,
            related_doc_loss_weight=related_doc_loss_weight,
            workflow_kpi_loss_weight=workflow_kpi_loss_weight,
            workflow_improvement_loss_weight=workflow_improvement_loss_weight,
            workflow_motif_loss_weight=workflow_motif_loss_weight,
            workflow_reward_bucket_loss_weight=workflow_reward_bucket_loss_weight,
            workflow_reward_score_loss_weight=workflow_reward_score_loss_weight,
            workflow_action_step_loss_weight=workflow_action_step_loss_weight,
            workflow_action_kpi_loss_weight=workflow_action_kpi_loss_weight,
            workflow_action_intervention_loss_weight=workflow_action_intervention_loss_weight,
        )

    def resolved_projection_hidden_size(self, model_hidden_size: int) -> int:
        return self.projection_hidden_size or model_hidden_size


@dataclass(frozen=True)
class OscarGraphAuxiliaryConfig:
    enabled: bool = False
    projection_hidden_size: int | None = None
    family_output_size: int = 5
    domain_output_size: int = 0
    motif_output_size: int = 0
    rollout_max_steps: int = 0
    family_loss_weight: float = 0.05
    domain_loss_weight: float = 0.05
    relation_loss_weight: float = 0.1
    neighbor_loss_weight: float = 0.1
    path_via_loss_weight: float = 0.1
    path_target_loss_weight: float = 0.1
    grounding_loss_weight: float = 0.1
    rollout_motif_loss_weight: float = 0.1
    rollout_step_loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.projection_hidden_size is not None and self.projection_hidden_size <= 0:
            raise ValueError("projection_hidden_size must be positive when provided.")
        for field_name in (
            "family_output_size",
            "domain_output_size",
            "motif_output_size",
            "rollout_max_steps",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in (
            "family_loss_weight",
            "domain_loss_weight",
            "relation_loss_weight",
            "neighbor_loss_weight",
            "path_via_loss_weight",
            "path_target_loss_weight",
            "grounding_loss_weight",
            "rollout_motif_loss_weight",
            "rollout_step_loss_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.enabled and self.domain_output_size <= 0:
            raise ValueError("domain_output_size must be positive when Oscar graph auxiliary heads are enabled.")

    @classmethod
    def reference(
        cls,
        *,
        projection_hidden_size: int | None = None,
        family_output_size: int = 5,
        domain_output_size: int,
        motif_output_size: int = 0,
        rollout_max_steps: int = 0,
        family_loss_weight: float = 0.05,
        domain_loss_weight: float = 0.05,
        relation_loss_weight: float = 0.1,
        neighbor_loss_weight: float = 0.1,
        path_via_loss_weight: float = 0.1,
        path_target_loss_weight: float = 0.1,
        grounding_loss_weight: float = 0.1,
        rollout_motif_loss_weight: float = 0.1,
        rollout_step_loss_weight: float = 0.1,
    ) -> "OscarGraphAuxiliaryConfig":
        return cls(
            enabled=True,
            projection_hidden_size=projection_hidden_size,
            family_output_size=family_output_size,
            domain_output_size=domain_output_size,
            motif_output_size=motif_output_size,
            rollout_max_steps=rollout_max_steps,
            family_loss_weight=family_loss_weight,
            domain_loss_weight=domain_loss_weight,
            relation_loss_weight=relation_loss_weight,
            neighbor_loss_weight=neighbor_loss_weight,
            path_via_loss_weight=path_via_loss_weight,
            path_target_loss_weight=path_target_loss_weight,
            grounding_loss_weight=grounding_loss_weight,
            rollout_motif_loss_weight=rollout_motif_loss_weight,
            rollout_step_loss_weight=rollout_step_loss_weight,
        )

    def resolved_projection_hidden_size(self, model_hidden_size: int) -> int:
        return self.projection_hidden_size or model_hidden_size


@dataclass(frozen=True)
class DecisionActionConfig:
    enabled: bool = False
    projection_hidden_size: int | None = None
    action_loss_weight: float = 0.2
    benchmark_adapter_names: tuple[str, ...] = ()
    benchmark_output_sizes: tuple[tuple[str, int], ...] = ()
    benchmark_name_output_sizes: tuple[tuple[str, int], ...] = ()
    benchmark_argument_output_sizes: tuple[tuple[str, int], ...] = ()
    benchmark_loss_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.projection_hidden_size is not None and self.projection_hidden_size <= 0:
            raise ValueError("projection_hidden_size must be positive when provided.")
        if self.action_loss_weight < 0.0:
            raise ValueError("action_loss_weight must be non-negative.")
        for benchmark_name in self.benchmark_adapter_names:
            if not benchmark_name:
                raise ValueError("benchmark_adapter_names entries must be non-empty.")
        for benchmark_name, size in self.benchmark_output_sizes:
            if not benchmark_name:
                raise ValueError("benchmark_output_sizes entries must have a non-empty benchmark name.")
            if size <= 0:
                raise ValueError("benchmark_output_sizes entries must have positive sizes.")
        for benchmark_name, size in self.benchmark_name_output_sizes:
            if not benchmark_name:
                raise ValueError("benchmark_name_output_sizes entries must have a non-empty benchmark name.")
            if size <= 0:
                raise ValueError("benchmark_name_output_sizes entries must have positive sizes.")
        for benchmark_name, size in self.benchmark_argument_output_sizes:
            if not benchmark_name:
                raise ValueError("benchmark_argument_output_sizes entries must have a non-empty benchmark name.")
            if size <= 0:
                raise ValueError("benchmark_argument_output_sizes entries must have positive sizes.")
        for benchmark_name, weight in self.benchmark_loss_weights:
            if not benchmark_name:
                raise ValueError("benchmark_loss_weights entries must have a non-empty benchmark name.")
            if weight < 0.0:
                raise ValueError("benchmark_loss_weights entries must be non-negative.")

    @classmethod
    def reference(
        cls,
        *,
        projection_hidden_size: int | None = None,
        action_loss_weight: float = 0.2,
        benchmark_adapter_names: tuple[str, ...] = (),
        benchmark_output_sizes: tuple[tuple[str, int], ...] = (),
        benchmark_name_output_sizes: tuple[tuple[str, int], ...] = (),
        benchmark_argument_output_sizes: tuple[tuple[str, int], ...] = (),
        benchmark_loss_weights: tuple[tuple[str, float], ...] = (),
    ) -> "DecisionActionConfig":
        return cls(
            enabled=True,
            projection_hidden_size=projection_hidden_size,
            action_loss_weight=action_loss_weight,
            benchmark_adapter_names=benchmark_adapter_names,
            benchmark_output_sizes=benchmark_output_sizes,
            benchmark_name_output_sizes=benchmark_name_output_sizes,
            benchmark_argument_output_sizes=benchmark_argument_output_sizes,
            benchmark_loss_weights=benchmark_loss_weights,
        )

    def resolved_projection_hidden_size(self, model_hidden_size: int) -> int:
        return self.projection_hidden_size or model_hidden_size

    def output_size_map(self) -> dict[str, int]:
        return dict(self.benchmark_output_sizes)

    def name_output_size_map(self) -> dict[str, int]:
        return dict(self.benchmark_name_output_sizes)

    def argument_output_size_map(self) -> dict[str, int]:
        return dict(self.benchmark_argument_output_sizes)

    def loss_weight_map(self) -> dict[str, float]:
        return dict(self.benchmark_loss_weights)


@dataclass(frozen=True)
class MoEConfig:
    enabled: bool = False
    num_experts: int = 4
    experts_per_token: int = 2
    router_jitter_noise: float = 0.0
    auxiliary_loss_weight: float = 1e-2

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if self.experts_per_token <= 0:
            raise ValueError("experts_per_token must be positive.")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token must be less than or equal to num_experts.")
        if self.router_jitter_noise < 0.0:
            raise ValueError("router_jitter_noise must be non-negative.")
        if self.auxiliary_loss_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be non-negative.")

    @classmethod
    def reference(
        cls,
        *,
        num_experts: int = 4,
        experts_per_token: int = 2,
        router_jitter_noise: float = 0.0,
        auxiliary_loss_weight: float = 1e-2,
    ) -> "MoEConfig":
        return cls(
            enabled=True,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            router_jitter_noise=router_jitter_noise,
            auxiliary_loss_weight=auxiliary_loss_weight,
        )


@dataclass(frozen=True)
class AttentionBackendConfig:
    backend: AttentionBackendName = "sdpa"
    dropout: float = 0.0
    sliding_window: int | None = None
    global_token_stride: int | None = None
    sink_token_count: int = 0
    latent_kv_dim: int | None = None
    scale_invariant_tau: float | None = None
    scale_invariant_last_n_layers: int | None = None
    scale_invariant_decode_mode: ScaleInvariantDecodeMode = "all_tokens"

    @classmethod
    def from_preset(
        cls,
        preset_name: AttentionBackendPresetName,
        *,
        latent_kv_dim: int | None = None,
        scale_invariant_tau: float | None = None,
        dropout: float = 0.0,
    ) -> "AttentionBackendConfig":
        if preset_name == "mla_default":
            return cls(
                backend="mla",
                dropout=dropout,
                latent_kv_dim=latent_kv_dim,
            )
        if preset_name == "mla_sia_prefill_l1":
            return cls(
                backend="mla_sia",
                dropout=dropout,
                latent_kv_dim=latent_kv_dim,
                scale_invariant_tau=scale_invariant_tau,
                scale_invariant_last_n_layers=1,
                scale_invariant_decode_mode="prefill_only",
            )
        raise ValueError(f"Unsupported attention preset: {preset_name}")

    def __post_init__(self) -> None:
        if self.backend not in {"eager", "sdpa", "hybrid", "mla", "sia", "sia_hybrid", "mla_sia"}:
            raise ValueError(f"Unsupported attention backend: {self.backend}")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("Attention dropout must be in the range [0, 1).")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when provided.")
        if self.global_token_stride is not None and self.global_token_stride <= 0:
            raise ValueError("global_token_stride must be positive when provided.")
        if self.sink_token_count < 0:
            raise ValueError("sink_token_count must be non-negative.")
        if self.backend == "hybrid" and self.sliding_window is None:
            raise ValueError("The hybrid attention backend requires sliding_window to be set.")
        if self.backend == "sia_hybrid" and self.sliding_window is None:
            raise ValueError("The sia_hybrid attention backend requires sliding_window to be set.")
        if self.latent_kv_dim is not None and self.latent_kv_dim <= 0:
            raise ValueError("latent_kv_dim must be positive when provided.")
        if self.uses_scale_invariant_scores() and self.resolved_scale_invariant_tau() <= 0.0:
            raise ValueError("Scale-invariant attention requires a positive scale_invariant_tau.")
        if self.scale_invariant_last_n_layers is not None and self.scale_invariant_last_n_layers < 0:
            raise ValueError("scale_invariant_last_n_layers must be non-negative when provided.")
        if self.scale_invariant_decode_mode not in {"all_tokens", "prefill_only"}:
            raise ValueError(f"Unsupported scale_invariant_decode_mode: {self.scale_invariant_decode_mode}")

    def uses_hybrid_mask(self) -> bool:
        return self.backend in {"hybrid", "sia_hybrid"}

    def uses_scale_invariant_scores(self) -> bool:
        return self.backend in {"sia", "sia_hybrid", "mla_sia"}

    def uses_latent_kv(self) -> bool:
        return self.backend in {"mla", "mla_sia"}

    def resolved_scale_invariant_tau(self) -> float:
        return self.scale_invariant_tau or 10.0

    def resolved_scale_invariant_last_n_layers(self, total_layers: int) -> int:
        if total_layers <= 0:
            raise ValueError("total_layers must be positive.")
        if not self.uses_scale_invariant_scores():
            return 0
        if self.scale_invariant_last_n_layers is None:
            return total_layers
        return min(self.scale_invariant_last_n_layers, total_layers)

    def uses_scale_invariant_for_layer(self, layer_index: int, total_layers: int) -> bool:
        if not self.uses_scale_invariant_scores():
            return False
        if layer_index < 0 or layer_index >= total_layers:
            raise ValueError("layer_index must be in range for the decoder depth.")
        last_n_layers = self.resolved_scale_invariant_last_n_layers(total_layers)
        return layer_index >= (total_layers - last_n_layers)

    def uses_scale_invariant_for_step(
        self,
        layer_index: int,
        total_layers: int,
        *,
        is_decode_step: bool,
    ) -> bool:
        if self.scale_invariant_decode_mode == "prefill_only" and is_decode_step:
            return False
        return self.uses_scale_invariant_for_layer(layer_index, total_layers)

    def backend_without_scale_invariant(self, backend_name: AttentionBackendName) -> AttentionBackendName:
        fallback = {
            "sia": "sdpa",
            "sia_hybrid": "hybrid",
            "mla_sia": "mla",
        }.get(backend_name, backend_name)
        return cast(AttentionBackendName, fallback)

    def backend_for_layer(self, layer_index: int, total_layers: int) -> AttentionBackendName:
        if self.uses_scale_invariant_for_layer(layer_index, total_layers):
            return self.backend
        return self.backend_without_scale_invariant(self.backend)

    def preset_name(self) -> AttentionBackendPresetName | None:
        if (
            self.backend == "mla"
            and self.scale_invariant_last_n_layers is None
            and self.scale_invariant_decode_mode == "all_tokens"
        ):
            return "mla_default"
        if (
            self.backend == "mla_sia"
            and self.scale_invariant_last_n_layers == 1
            and self.scale_invariant_decode_mode == "prefill_only"
        ):
            return "mla_sia_prefill_l1"
        return None


@dataclass(frozen=True)
class DecoderModelConfig:
    vocab_size: int
    max_position_embeddings: int = 4096
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    rotary_dim: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    attention: AttentionBackendConfig = field(default_factory=AttentionBackendConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    core_auxiliary: CoReAuxiliaryConfig = field(default_factory=CoReAuxiliaryConfig)
    oscar_auxiliary: OscarAuxiliaryConfig = field(default_factory=OscarAuxiliaryConfig)
    oscar_graph_auxiliary: OscarGraphAuxiliaryConfig = field(default_factory=OscarGraphAuxiliaryConfig)
    decision_action: DecisionActionConfig = field(default_factory=DecisionActionConfig)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.resolved_num_key_value_heads() <= 0:
            raise ValueError("num_key_value_heads must be positive.")
        if self.num_attention_heads % self.resolved_num_key_value_heads() != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive when provided.")
        if self.resolved_rotary_dim() <= 0:
            raise ValueError("rotary_dim must resolve to a positive even value.")
        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive.")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive.")
        if self.attention.uses_latent_kv() and self.resolved_latent_kv_dim() <= 0:
            raise ValueError("Latent-KV attention requires a positive latent_kv_dim.")
        if self.moe.enabled and self.moe.num_experts < 2:
            raise ValueError("MoE mode requires at least two experts.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def resolved_num_key_value_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    def resolved_intermediate_size(self) -> int:
        return self.intermediate_size or (4 * self.hidden_size)

    def resolved_rotary_dim(self) -> int:
        rotary_dim = self.rotary_dim if self.rotary_dim is not None else self.head_dim
        rotary_dim = min(rotary_dim, self.head_dim)
        if rotary_dim % 2 != 0:
            rotary_dim -= 1
        return rotary_dim

    def resolved_latent_kv_dim(self) -> int:
        return self.attention.latent_kv_dim or max(self.head_dim, self.hidden_size // 8)


@dataclass(frozen=True)
class ResolvedInferenceBudget:
    max_prompt_tokens: int | None
    max_new_tokens: int
    active_layers: int
    attention_window: int | None
    max_cache_tokens: int | None
    use_kv_cache: bool
    stop_token_ids: Tuple[int, ...]
    temperature: float
    top_k: int | None


@dataclass(frozen=True)
class InferenceBudget:
    max_prompt_tokens: int | None = None
    max_new_tokens: int = 128
    active_layers: int | None = None
    attention_window: int | None = None
    max_cache_tokens: int | None = None
    use_kv_cache: bool = True
    stop_token_ids: Tuple[int, ...] = ()
    temperature: float = 0.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.max_prompt_tokens is not None and self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive when provided.")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if self.active_layers is not None and self.active_layers <= 0:
            raise ValueError("active_layers must be positive when provided.")
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention_window must be positive when provided.")
        if self.max_cache_tokens is not None and self.max_cache_tokens <= 0:
            raise ValueError("max_cache_tokens must be positive when provided.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")

    def resolve(self, config: DecoderModelConfig) -> ResolvedInferenceBudget:
        active_layers = self.active_layers or config.num_hidden_layers
        active_layers = min(active_layers, config.num_hidden_layers)
        attention_window = self.attention_window if self.attention_window is not None else config.attention.sliding_window
        max_cache_tokens = self.max_cache_tokens
        if max_cache_tokens is None and attention_window is not None:
            max_cache_tokens = attention_window
        return ResolvedInferenceBudget(
            max_prompt_tokens=self.max_prompt_tokens,
            max_new_tokens=self.max_new_tokens,
            active_layers=active_layers,
            attention_window=attention_window,
            max_cache_tokens=max_cache_tokens,
            use_kv_cache=self.use_kv_cache,
            stop_token_ids=self.stop_token_ids,
            temperature=self.temperature,
            top_k=self.top_k,
        )
