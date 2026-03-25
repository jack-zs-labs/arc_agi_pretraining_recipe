from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "The models.moe module requires torch. Install requirements-models.txt or use .venv_atari."
    ) from exc

from .config import DecoderModelConfig


@dataclass
class MoERouterMetrics:
    auxiliary_loss: torch.Tensor
    expert_fraction: torch.Tensor
    router_prob_fraction: torch.Tensor
    router_entropy: torch.Tensor


class ExpertMLP(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        intermediate_size = config.resolved_intermediate_size()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SparseMoEGatedMLP(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe.num_experts
        self.experts_per_token = config.moe.experts_per_token
        self.router_jitter_noise = config.moe.router_jitter_noise
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(ExpertMLP(config) for _ in range(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, MoERouterMetrics]:
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.router(flat_states)
        if self.training and self.router_jitter_noise > 0.0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise

        router_probs = torch.softmax(router_logits, dim=-1)
        route_weights, route_indices = torch.topk(router_probs, k=self.experts_per_token, dim=-1)
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        mixed_output = torch.zeros_like(flat_states)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slot_indices = torch.where(route_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_inputs = flat_states.index_select(0, token_indices)
            expert_outputs = expert(expert_inputs)
            expert_weights = route_weights[token_indices, slot_indices].unsqueeze(-1).to(dtype=expert_outputs.dtype)
            mixed_output.index_add_(0, token_indices, expert_outputs * expert_weights)

        expert_assignments = torch.zeros_like(router_probs)
        expert_assignments.scatter_(1, route_indices, 1.0 / self.experts_per_token)
        expert_fraction = expert_assignments.mean(dim=0)
        router_prob_fraction = router_probs.mean(dim=0)
        auxiliary_loss = self.num_experts * torch.sum(expert_fraction * router_prob_fraction)
        router_entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()

        metrics = MoERouterMetrics(
            auxiliary_loss=auxiliary_loss,
            expert_fraction=expert_fraction,
            router_prob_fraction=router_prob_fraction,
            router_entropy=router_entropy,
        )
        return mixed_output.reshape_as(hidden_states), metrics
