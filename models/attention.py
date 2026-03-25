from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


def _normalize_positions(positions: torch.Tensor) -> torch.Tensor:
    if positions.dim() == 1:
        return positions.unsqueeze(0)
    if positions.dim() != 2:
        raise ValueError("Position tensors must have shape [seq_len] or [batch, seq_len].")
    return positions


def build_causal_attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    window_size: int | None = None,
) -> torch.Tensor:
    query_positions = _normalize_positions(query_positions)
    key_positions = _normalize_positions(key_positions)
    if key_positions.size(0) == 1 and query_positions.size(0) > 1:
        key_positions = key_positions.expand(query_positions.size(0), -1)
    if query_positions.size(0) != key_positions.size(0):
        raise ValueError("query_positions and key_positions batch dimensions must match.")
    query = query_positions[:, :, None]
    key = key_positions[:, None, :]
    mask = key <= query
    if window_size is not None:
        mask = mask & (key > (query - window_size))
    return mask


def build_hybrid_attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    window_size: int,
    global_token_stride: int | None,
    sink_token_count: int,
) -> torch.Tensor:
    query_positions = _normalize_positions(query_positions)
    key_positions = _normalize_positions(key_positions)
    if key_positions.size(0) == 1 and query_positions.size(0) > 1:
        key_positions = key_positions.expand(query_positions.size(0), -1)
    if query_positions.size(0) != key_positions.size(0):
        raise ValueError("query_positions and key_positions batch dimensions must match.")
    query = query_positions[:, :, None]
    key = key_positions[:, None, :]
    causal = key <= query
    local = key > (query - window_size)
    anchors = torch.zeros_like(causal)
    if sink_token_count > 0:
        anchors = anchors | (key < sink_token_count)
    if global_token_stride is not None:
        anchors = anchors | (torch.remainder(key, global_token_stride) == 0)
    return causal & (local | anchors)


def combine_attention_masks(
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
    window_size: int | None = None,
    global_token_stride: int | None = None,
    sink_token_count: int = 0,
) -> torch.Tensor:
    if global_token_stride is not None or sink_token_count > 0:
        if window_size is None:
            raise ValueError("Hybrid attention requires a positive window_size.")
        mask = build_hybrid_attention_mask(
            query_positions,
            key_positions,
            window_size=window_size,
            global_token_stride=global_token_stride,
            sink_token_count=sink_token_count,
        )
    else:
        mask = build_causal_attention_mask(
            query_positions,
            key_positions,
            window_size=window_size,
        )
    mask = mask.unsqueeze(1)
    if key_padding_mask is None:
        return mask
    if key_padding_mask.dim() != 2:
        raise ValueError("key_padding_mask must have shape [batch, key_length].")
    if key_padding_mask.size(1) != key_positions.size(-1):
        raise ValueError("key_padding_mask length must match key_positions.")
    return mask & key_padding_mask[:, None, None, :].to(dtype=torch.bool)


def mask_to_bias(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    zeros = torch.zeros((), dtype=dtype, device=mask.device)
    masked = torch.full((), min_value, dtype=dtype, device=mask.device)
    return torch.where(mask, zeros, masked)


class AttentionBackend(ABC):
    name: str

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        window_size: int | None,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class EagerAttentionBackend(AttentionBackend):
    name = "eager"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        window_size: int | None,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        mask = combine_attention_masks(
            query_positions=query_positions,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
            window_size=window_size,
        )
        scores = scores + mask_to_bias(mask, dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        if training and dropout_p > 0.0:
            attention = F.dropout(attention, p=dropout_p)
        return torch.matmul(attention, value)


class SdpaAttentionBackend(AttentionBackend):
    name = "sdpa"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        window_size: int | None,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        attn_mask = None
        is_causal = key_padding_mask is None and window_size is None
        if not is_causal:
            mask = combine_attention_masks(
                query_positions=query_positions,
                key_positions=key_positions,
                key_padding_mask=key_padding_mask,
                window_size=window_size,
            )
            attn_mask = mask_to_bias(mask, dtype=query.dtype)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p if training else 0.0,
            is_causal=is_causal,
        )


class HybridAttentionBackend(AttentionBackend):
    name = "hybrid"

    def __init__(self, *, global_token_stride: int | None, sink_token_count: int) -> None:
        self.global_token_stride = global_token_stride
        self.sink_token_count = sink_token_count

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        window_size: int | None,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        if window_size is None:
            raise ValueError("Hybrid attention requires window_size to be set.")
        mask = combine_attention_masks(
            query_positions=query_positions,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
            window_size=window_size,
            global_token_stride=self.global_token_stride,
            sink_token_count=self.sink_token_count,
        )
        attn_bias = mask_to_bias(mask, dtype=query.dtype)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=dropout_p if training else 0.0,
            is_causal=False,
        )


class ScaleInvariantAttentionBackend(AttentionBackend):
    name = "sia"

    def __init__(
        self,
        *,
        tau: float,
        global_token_stride: int | None = None,
        sink_token_count: int = 0,
    ) -> None:
        if tau <= 0.0:
            raise ValueError("Scale-invariant attention requires tau to be positive.")
        self.tau = tau
        self.global_token_stride = global_token_stride
        self.sink_token_count = sink_token_count

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        window_size: int | None,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        mask = combine_attention_masks(
            query_positions=query_positions,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
            window_size=window_size,
            global_token_stride=self.global_token_stride,
            sink_token_count=self.sink_token_count,
        )
        relative_positions = (
            _normalize_positions(query_positions)[:, :, None] - _normalize_positions(key_positions)[:, None, :]
        ).clamp_min(0)
        relative_positions = relative_positions.to(dtype=scores.dtype)
        log_term = torch.log1p(relative_positions / self.tau)
        score_scale = torch.sqrt((2.0 * log_term) + 1.0).unsqueeze(1)
        score_shift = (-2.0 * log_term).unsqueeze(1)
        scores = (scores * score_scale) + score_shift
        scores = scores + mask_to_bias(mask, dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        if training and dropout_p > 0.0:
            attention = F.dropout(attention, p=dropout_p)
        return torch.matmul(attention, value)


def build_attention_backend(
    name: str,
    *,
    global_token_stride: int | None = None,
    sink_token_count: int = 0,
    scale_invariant_tau: float | None = None,
) -> AttentionBackend:
    if name == EagerAttentionBackend.name:
        return EagerAttentionBackend()
    if name == SdpaAttentionBackend.name:
        return SdpaAttentionBackend()
    if name == HybridAttentionBackend.name:
        return HybridAttentionBackend(
            global_token_stride=global_token_stride,
            sink_token_count=sink_token_count,
        )
    if name == ScaleInvariantAttentionBackend.name:
        return ScaleInvariantAttentionBackend(
            tau=scale_invariant_tau or 10.0,
        )
    if name == "sia_hybrid":
        return ScaleInvariantAttentionBackend(
            tau=scale_invariant_tau or 10.0,
            global_token_stride=global_token_stride,
            sink_token_count=sink_token_count,
        )
    supported = ", ".join(("eager", "hybrid", "sdpa", "sia", "sia_hybrid"))
    raise ValueError(f"Unsupported attention backend {name!r}. Supported backends: {supported}.")
