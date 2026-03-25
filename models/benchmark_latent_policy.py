from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from .config import DecoderModelConfig
from .decoder import DecoderBlock, RMSNorm


class BenchmarkLatentPolicy(nn.Module):
    def __init__(
        self,
        config: DecoderModelConfig,
        *,
        benchmark_names: Sequence[str],
        output_head_sizes: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.benchmark_names = tuple(benchmark_names)
        self.benchmark_to_id = {name: index for index, name in enumerate(self.benchmark_names)}
        self.output_head_sizes = dict(output_head_sizes)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.benchmark_embed = nn.Embedding(len(self.benchmark_names), config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderBlock(config, layer_index=layer_index)
            for layer_index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_heads = nn.ModuleDict(
            {
                head_name: nn.Linear(config.hidden_size, head_size, bias=True)
                for head_name, head_size in sorted(self.output_head_sizes.items())
            }
        )
        self.apply(self._init_weights)

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

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        benchmark_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        if benchmark_ids.dim() != 1 or benchmark_ids.size(0) != input_ids.size(0):
            raise ValueError("benchmark_ids must have shape [batch] and match input_ids batch size.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.dim() != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        benchmark_bias = self.benchmark_embed(benchmark_ids).unsqueeze(1)
        hidden_states = self.embed_tokens(input_ids) + benchmark_bias
        key_padding_mask = attention_mask.to(dtype=torch.bool)

        for layer_index, layer in enumerate(self.layers):
            block_output = layer(
                hidden_states,
                position_ids=position_ids,
                key_padding_mask=key_padding_mask,
                layer_cache=None,
                max_cache_tokens=None,
                attention_window=None,
                use_scale_invariant_scores=self.config.attention.uses_scale_invariant_for_step(
                    layer_index,
                    self.config.num_hidden_layers,
                    is_decode_step=False,
                ),
            )
            hidden_states = block_output.hidden_states

        hidden_states = self.norm(hidden_states)
        lengths = key_padding_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(batch_size, device=input_ids.device)
        return hidden_states[batch_index, lengths]

    def head_logits(self, pooled_states: torch.Tensor, *, output_head: str) -> torch.Tensor:
        if output_head not in self.output_heads:
            raise KeyError(f"Unknown output head: {output_head}")
        return self.output_heads[output_head](pooled_states)
