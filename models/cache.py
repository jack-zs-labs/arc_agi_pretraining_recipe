from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerKVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        *,
        max_cache_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.key is None:
            merged_key = key
            merged_value = value
            merged_positions = positions
        else:
            merged_key = torch.cat((self.key, key), dim=2)
            merged_value = torch.cat((self.value, value), dim=2)
            merged_positions = torch.cat((self.positions, positions), dim=1)
        if max_cache_tokens is not None and merged_key.size(2) > max_cache_tokens:
            merged_key = merged_key[:, :, -max_cache_tokens:, :].contiguous()
            merged_value = merged_value[:, :, -max_cache_tokens:, :].contiguous()
            merged_positions = merged_positions[:, -max_cache_tokens:].contiguous()
        self.key = merged_key
        self.value = merged_value
        self.positions = merged_positions
        return merged_key, merged_value, merged_positions

    def append_latent(
        self,
        latent: torch.Tensor,
        positions: torch.Tensor,
        *,
        max_cache_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent is None:
            merged_latent = latent
            merged_positions = positions
        else:
            merged_latent = torch.cat((self.latent, latent), dim=1)
            merged_positions = torch.cat((self.positions, positions), dim=1)
        if max_cache_tokens is not None and merged_latent.size(1) > max_cache_tokens:
            merged_latent = merged_latent[:, -max_cache_tokens:, :].contiguous()
            merged_positions = merged_positions[:, -max_cache_tokens:].contiguous()
        self.latent = merged_latent
        self.positions = merged_positions
        return merged_latent, merged_positions

    @property
    def current_length(self) -> int:
        if self.key is not None:
            return int(self.key.size(2))
        if self.latent is not None:
            return int(self.latent.size(1))
        return 0


@dataclass
class DecoderKVCache:
    layers: list[LayerKVCache] = field(default_factory=list)
    tokens_seen: int = 0

    @classmethod
    def create(cls, num_layers: int) -> "DecoderKVCache":
        return cls(layers=[LayerKVCache() for _ in range(num_layers)])

    def ensure_layers(self, num_layers: int) -> None:
        if len(self.layers) < num_layers:
            self.layers.extend(LayerKVCache() for _ in range(num_layers - len(self.layers)))

    def layer(self, index: int) -> LayerKVCache:
        self.ensure_layers(index + 1)
        return self.layers[index]

    @property
    def current_length(self) -> int:
        if not self.layers:
            return 0
        return max(layer.current_length for layer in self.layers)
