from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parameter import UninitializedParameter
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "The models.decoder module requires torch. Install requirements-models.txt or use .venv_atari."
    ) from exc

from .attention import AttentionBackend, build_attention_backend
from .cache import DecoderKVCache, LayerKVCache
from .config import DecoderModelConfig, InferenceBudget, ResolvedInferenceBudget
from .core_auxiliary import CoReAuxiliaryHeads
from .decision_auxiliary import DecisionActionAuxiliaryHeads
from .moe import SparseMoEGatedMLP
from .oscar_auxiliary import OscarAuxiliaryHeads
from .oscar_graph_auxiliary import OscarGraphAuxiliaryHeads


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    first = x[..., :half]
    second = x[..., half:]
    return torch.cat((-second, first), dim=-1)


def apply_rotary_embedding(
    tensor: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    rotary_part = tensor[..., :rotary_dim]
    passthrough = tensor[..., rotary_dim:]
    cos = cos.unsqueeze(1).to(dtype=tensor.dtype)
    sin = sin.unsqueeze(1).to(dtype=tensor.dtype)
    rotated = (rotary_part * cos) + (rotate_half(rotary_part) * sin)
    if passthrough.numel() == 0:
        return rotated
    return torch.cat((rotated, passthrough), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_heads, repeats, seq_len, head_dim)
    return expanded.reshape(batch, num_heads * repeats, seq_len, head_dim)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bs,d->bsd", position_ids.to(dtype=self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class GatedMLP(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        intermediate_size = config.resolved_intermediate_size()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


@dataclass
class DecoderBlockOutput:
    hidden_states: torch.Tensor
    auxiliary_loss: torch.Tensor | None = None
    router_entropy: torch.Tensor | None = None
    expert_fraction: torch.Tensor | None = None


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        config: DecoderModelConfig,
        *,
        backend_name: str | None = None,
        fallback_backend_name: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.resolved_num_key_value_heads()
        self.head_dim = config.head_dim
        self.rotary_dim = config.resolved_rotary_dim()
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.rotary_dim, config.rope_theta)
        primary_backend_name = backend_name or config.attention.backend
        self.backend: AttentionBackend = build_attention_backend(
            primary_backend_name,
            global_token_stride=config.attention.global_token_stride,
            sink_token_count=config.attention.sink_token_count,
            scale_invariant_tau=config.attention.scale_invariant_tau,
        )
        self.fallback_backend: AttentionBackend | None = None
        if fallback_backend_name is not None and fallback_backend_name != primary_backend_name:
            self.fallback_backend = build_attention_backend(
                fallback_backend_name,
                global_token_stride=config.attention.global_token_stride,
                sink_token_count=config.attention.sink_token_count,
                scale_invariant_tau=config.attention.scale_invariant_tau,
            )
        self.dropout = config.attention.dropout
        self.kv_repeat = self.num_attention_heads // self.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        layer_cache: LayerKVCache | None,
        max_cache_tokens: int | None,
        attention_window: int | None,
        use_scale_invariant_scores: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(position_ids)
        query = apply_rotary_embedding(query, cos=cos, sin=sin, rotary_dim=self.rotary_dim)
        key = apply_rotary_embedding(key, cos=cos, sin=sin, rotary_dim=self.rotary_dim)
        key_positions = position_ids

        if layer_cache is not None:
            key, value, key_positions = layer_cache.append(
                key,
                value,
                position_ids,
                max_cache_tokens=max_cache_tokens,
            )

        key = repeat_kv(key, self.kv_repeat)
        value = repeat_kv(value, self.kv_repeat)
        backend = self.backend if (use_scale_invariant_scores or self.fallback_backend is None) else self.fallback_backend
        attn_output = backend.forward(
            query,
            key,
            value,
            query_positions=position_ids,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
            window_size=attention_window,
            dropout_p=self.dropout,
            training=self.training,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        config: DecoderModelConfig,
        *,
        use_scale_invariant_scores: bool | None = None,
        fallback_to_plain_scores: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.resolved_num_key_value_heads()
        self.head_dim = config.head_dim
        self.rotary_dim = config.resolved_rotary_dim()
        self.latent_kv_dim = config.resolved_latent_kv_dim()
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.kv_down_proj = nn.Linear(config.hidden_size, self.latent_kv_dim, bias=False)
        self.k_up_proj = nn.Linear(self.latent_kv_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.latent_kv_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.rotary_dim, config.rope_theta)
        if use_scale_invariant_scores is None:
            use_scale_invariant_scores = config.attention.uses_scale_invariant_scores()
        latent_backend_name = "sia" if use_scale_invariant_scores else "sdpa"
        self.backend: AttentionBackend = build_attention_backend(
            latent_backend_name,
            scale_invariant_tau=config.attention.scale_invariant_tau,
        )
        self.fallback_backend: AttentionBackend | None = None
        if fallback_to_plain_scores and latent_backend_name != "sdpa":
            self.fallback_backend = build_attention_backend(
                "sdpa",
                scale_invariant_tau=config.attention.scale_invariant_tau,
            )
        self.dropout = config.attention.dropout
        self.kv_repeat = self.num_attention_heads // self.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        layer_cache: LayerKVCache | None,
        max_cache_tokens: int | None,
        attention_window: int | None,
        use_scale_invariant_scores: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        latent = self.kv_down_proj(hidden_states)

        if layer_cache is not None:
            latent, key_positions = layer_cache.append_latent(
                latent,
                position_ids,
                max_cache_tokens=max_cache_tokens,
            )
        else:
            key_positions = position_ids

        key = self.k_up_proj(latent).view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_up_proj(latent).view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_cos, query_sin = self.rotary(position_ids)
        key_cos, key_sin = self.rotary(key_positions)
        query = apply_rotary_embedding(query, cos=query_cos, sin=query_sin, rotary_dim=self.rotary_dim)
        key = apply_rotary_embedding(key, cos=key_cos, sin=key_sin, rotary_dim=self.rotary_dim)

        key = repeat_kv(key, self.kv_repeat)
        value = repeat_kv(value, self.kv_repeat)
        backend = self.backend if (use_scale_invariant_scores or self.fallback_backend is None) else self.fallback_backend
        attn_output = backend.forward(
            query,
            key,
            value,
            query_positions=position_ids,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
            window_size=attention_window,
            dropout_p=self.dropout,
            training=self.training,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderModelConfig, *, layer_index: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        primary_backend = config.attention.backend_for_layer(layer_index, config.num_hidden_layers)
        fallback_backend = config.attention.backend_without_scale_invariant(primary_backend)
        if primary_backend in {"mla", "mla_sia"}:
            self.attn = MultiHeadLatentAttention(
                config,
                use_scale_invariant_scores=(primary_backend == "mla_sia"),
                fallback_to_plain_scores=(fallback_backend != primary_backend),
            )
        else:
            self.attn = GroupedQueryAttention(
                config,
                backend_name=primary_backend,
                fallback_backend_name=fallback_backend,
            )
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.uses_moe = config.moe.enabled
        self.mlp = SparseMoEGatedMLP(config) if config.moe.enabled else GatedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        layer_cache: LayerKVCache | None,
        max_cache_tokens: int | None,
        attention_window: int | None,
        use_scale_invariant_scores: bool,
    ) -> DecoderBlockOutput:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            position_ids=position_ids,
            key_padding_mask=key_padding_mask,
            layer_cache=layer_cache,
            max_cache_tokens=max_cache_tokens,
            attention_window=attention_window,
            use_scale_invariant_scores=use_scale_invariant_scores,
        )
        mlp_input = self.mlp_norm(hidden_states)
        if self.uses_moe:
            mlp_output, router_metrics = self.mlp(mlp_input)
            hidden_states = hidden_states + mlp_output
            return DecoderBlockOutput(
                hidden_states=hidden_states,
                auxiliary_loss=router_metrics.auxiliary_loss,
                router_entropy=router_metrics.router_entropy,
                expert_fraction=router_metrics.expert_fraction,
            )
        hidden_states = hidden_states + self.mlp(mlp_input)
        return DecoderBlockOutput(hidden_states=hidden_states)


@dataclass
class DecoderModelOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    cache: DecoderKVCache | None = None
    auxiliary_loss: torch.Tensor | None = None
    router_entropy: torch.Tensor | None = None
    expert_fraction: torch.Tensor | None = None
    task_auxiliary_loss: torch.Tensor | None = None
    task_auxiliary_metrics: dict[str, float] | None = None


class DecoderLanguageModel(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderBlock(config, layer_index=layer_index) for layer_index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.core_auxiliary_heads = CoReAuxiliaryHeads(config) if config.core_auxiliary.enabled else None
        self.oscar_auxiliary_heads = OscarAuxiliaryHeads(config) if config.oscar_auxiliary.enabled else None
        self.oscar_graph_auxiliary_heads = (
            OscarGraphAuxiliaryHeads(config) if config.oscar_graph_auxiliary.enabled else None
        )
        self.decision_action_heads = (
            DecisionActionAuxiliaryHeads(config) if config.decision_action.enabled else None
        )
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if isinstance(module.weight, UninitializedParameter):
                return
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                if isinstance(module.bias, UninitializedParameter):
                    return
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _resolve_budget(
        self,
        budget: InferenceBudget | ResolvedInferenceBudget | None,
    ) -> ResolvedInferenceBudget:
        if budget is None:
            return InferenceBudget().resolve(self.config)
        if isinstance(budget, ResolvedInferenceBudget):
            return budget
        return budget.resolve(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        budget: InferenceBudget | ResolvedInferenceBudget | None = None,
        cache: DecoderKVCache | None = None,
        task_name: str | None = None,
        task_auxiliary_labels: dict[str, object] | None = None,
    ) -> DecoderModelOutput:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len] or [seq_len].")

        resolved_budget = self._resolve_budget(budget)
        if resolved_budget.max_prompt_tokens is not None and input_ids.size(1) > resolved_budget.max_prompt_tokens:
            input_ids = input_ids[:, -resolved_budget.max_prompt_tokens :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -resolved_budget.max_prompt_tokens :]

        if cache is not None and attention_mask is not None:
            raise ValueError("attention_mask with cache is not supported in this reference decoder yet.")

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input_ids must contain at least one token.")

        position_offset = 0
        if resolved_budget.use_kv_cache:
            cache = cache or DecoderKVCache.create(self.config.num_hidden_layers)
            cache.ensure_layers(self.config.num_hidden_layers)
            position_offset = cache.tokens_seen
        else:
            cache = None

        position_ids = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        key_padding_mask = attention_mask.to(dtype=torch.bool) if attention_mask is not None else None
        is_decode_step = cache is not None and cache.tokens_seen > 0
        auxiliary_losses: list[torch.Tensor] = []
        router_entropies: list[torch.Tensor] = []
        expert_fractions: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.layers[: resolved_budget.active_layers]):
            layer_cache = cache.layer(layer_index) if cache is not None else None
            block_output = layer(
                hidden_states,
                position_ids=position_ids,
                key_padding_mask=key_padding_mask,
                layer_cache=layer_cache,
                max_cache_tokens=resolved_budget.max_cache_tokens,
                attention_window=resolved_budget.attention_window,
                use_scale_invariant_scores=self.config.attention.uses_scale_invariant_for_step(
                    layer_index,
                    self.config.num_hidden_layers,
                    is_decode_step=is_decode_step,
                ),
            )
            hidden_states = block_output.hidden_states
            if block_output.auxiliary_loss is not None:
                auxiliary_losses.append(block_output.auxiliary_loss)
            if block_output.router_entropy is not None:
                router_entropies.append(block_output.router_entropy)
            if block_output.expert_fraction is not None:
                expert_fractions.append(block_output.expert_fraction)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if cache is not None:
            cache.tokens_seen += seq_len
        auxiliary_loss = None
        router_entropy = None
        expert_fraction = None
        if auxiliary_losses:
            auxiliary_loss = torch.stack(auxiliary_losses).sum() * self.config.moe.auxiliary_loss_weight
        if router_entropies:
            router_entropy = torch.stack(router_entropies).mean()
        if expert_fractions:
            expert_fraction = torch.stack(expert_fractions).mean(dim=0)
        task_auxiliary_loss = None
        task_auxiliary_metrics = None
        if (
            task_name == "core"
            and self.core_auxiliary_heads is not None
            and task_auxiliary_labels is not None
        ):
            auxiliary_output = self.core_auxiliary_heads(
                hidden_states,
                attention_mask=attention_mask,
                labels=task_auxiliary_labels,
            )
            task_auxiliary_loss = auxiliary_output.loss
            task_auxiliary_metrics = auxiliary_output.metrics
        if (
            task_name == "oscar_scope_reasoning"
            and self.oscar_auxiliary_heads is not None
            and task_auxiliary_labels is not None
        ):
            auxiliary_output = self.oscar_auxiliary_heads(
                hidden_states,
                attention_mask=attention_mask,
                labels=task_auxiliary_labels,
            )
            task_auxiliary_loss = auxiliary_output.loss
            task_auxiliary_metrics = auxiliary_output.metrics
        if (
            task_name == "oscar_graph_reasoning"
            and self.oscar_graph_auxiliary_heads is not None
            and task_auxiliary_labels is not None
        ):
            auxiliary_output = self.oscar_graph_auxiliary_heads(
                hidden_states,
                attention_mask=attention_mask,
                labels=task_auxiliary_labels,
            )
            task_auxiliary_loss = auxiliary_output.loss
            task_auxiliary_metrics = auxiliary_output.metrics
        if (
            task_name == "decision_action"
            and self.decision_action_heads is not None
            and task_auxiliary_labels is not None
        ):
            auxiliary_output = self.decision_action_heads(
                hidden_states,
                attention_mask=attention_mask,
                labels=task_auxiliary_labels,
            )
            task_auxiliary_loss = auxiliary_output.loss
            task_auxiliary_metrics = auxiliary_output.metrics
        return DecoderModelOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            cache=cache,
            auxiliary_loss=auxiliary_loss,
            router_entropy=router_entropy,
            expert_fraction=expert_fraction,
            task_auxiliary_loss=task_auxiliary_loss,
            task_auxiliary_metrics=task_auxiliary_metrics,
        )

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        scaled = logits / temperature
        if top_k is not None and top_k < scaled.size(-1):
            values, _ = torch.topk(scaled, k=top_k, dim=-1)
            threshold = values[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < threshold, torch.finfo(scaled.dtype).min)
        probabilities = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        budget: InferenceBudget | ResolvedInferenceBudget | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len] or [seq_len].")

        resolved_budget = self._resolve_budget(budget)
        generated = input_ids.to(device=self.device, dtype=torch.long)
        if resolved_budget.max_prompt_tokens is not None and generated.size(1) > resolved_budget.max_prompt_tokens:
            generated = generated[:, -resolved_budget.max_prompt_tokens :]

        if resolved_budget.max_new_tokens == 0:
            return generated

        cache = DecoderKVCache.create(self.config.num_hidden_layers) if resolved_budget.use_kv_cache else None
        outputs = self.forward(generated, budget=resolved_budget, cache=cache)
        next_logits = outputs.logits[:, -1, :]
        cache = outputs.cache
        stop_ids = (
            torch.as_tensor(resolved_budget.stop_token_ids, device=generated.device, dtype=torch.long)
            if resolved_budget.stop_token_ids
            else None
        )
        finished = torch.zeros(generated.size(0), device=generated.device, dtype=torch.bool)
        stop_fill = resolved_budget.stop_token_ids[0] if resolved_budget.stop_token_ids else 0

        for _ in range(resolved_budget.max_new_tokens):
            next_token = self._sample_next_token(
                next_logits,
                temperature=resolved_budget.temperature,
                top_k=resolved_budget.top_k,
            )
            if stop_ids is not None and torch.any(finished):
                next_token = torch.where(finished, torch.full_like(next_token, stop_fill), next_token)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            if stop_ids is not None:
                finished = finished | torch.isin(next_token, stop_ids)
                if bool(torch.all(finished)):
                    break
            model_input = next_token[:, None] if cache is not None else generated
            outputs = self.forward(model_input, budget=resolved_budget, cache=cache)
            next_logits = outputs.logits[:, -1, :]
            cache = outputs.cache
        return generated
