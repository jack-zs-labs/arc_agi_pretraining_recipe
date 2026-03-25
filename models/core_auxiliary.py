from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoReAuxiliaryConfig, DecoderModelConfig


CORE_DEPENDENCY_KIND_TO_ID = {
    "control": 0,
    "data": 1,
    "infoflow": 2,
}


@dataclass
class CoReAuxiliaryOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class CoReAuxiliaryHeads(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        self.config = config.core_auxiliary
        projection_hidden = self.config.resolved_projection_hidden_size(config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.query_positive_head = nn.Linear(projection_hidden, 2, bias=True)
        self.source_count_head = nn.Linear(projection_hidden, 4, bias=True)
        self.trace_length_head = nn.Linear(projection_hidden, 5, bias=True)
        self.dependency_kind_head = nn.Linear(projection_hidden, len(CORE_DEPENDENCY_KIND_TO_ID), bias=True)
        self.infoflow_data_edge_head = nn.Linear(projection_hidden, 2, bias=True)
        self.candidate_proj = nn.Sequential(
            nn.LazyLinear(projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.source_membership_head = nn.Linear(projection_hidden, 1, bias=True)
        self.direct_edge_head = nn.Linear(projection_hidden, 1, bias=True)

    def pooled_hidden_state(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        weights = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denom

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        labels: dict[str, torch.Tensor],
    ) -> CoReAuxiliaryOutput:
        pooled = self.pooled_hidden_state(hidden_states, attention_mask=attention_mask)
        features = self.proj(pooled)

        losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {}

        query_logits = self.query_positive_head(features)
        query_loss = F.cross_entropy(query_logits, labels["query_positive"])
        losses.append(query_loss * self.config.query_positive_loss_weight)
        metrics["core_query_positive_loss"] = float(query_loss.item())
        metrics["core_query_positive_accuracy"] = float(
            (query_logits.argmax(dim=-1) == labels["query_positive"]).float().mean().item()
        )

        source_logits = self.source_count_head(features)
        source_loss = F.cross_entropy(source_logits, labels["source_count_bucket"])
        losses.append(source_loss * self.config.source_count_loss_weight)
        metrics["core_source_count_loss"] = float(source_loss.item())
        metrics["core_source_count_accuracy"] = float(
            (source_logits.argmax(dim=-1) == labels["source_count_bucket"]).float().mean().item()
        )

        trace_logits = self.trace_length_head(features)
        trace_loss = F.cross_entropy(trace_logits, labels["trace_length_bucket"])
        losses.append(trace_loss * self.config.trace_length_loss_weight)
        metrics["core_trace_length_loss"] = float(trace_loss.item())
        metrics["core_trace_length_accuracy"] = float(
            (trace_logits.argmax(dim=-1) == labels["trace_length_bucket"]).float().mean().item()
        )

        dependency_logits = self.dependency_kind_head(features)
        dependency_loss = F.cross_entropy(dependency_logits, labels["dependency_kind_id"])
        losses.append(dependency_loss * self.config.dependency_kind_loss_weight)
        metrics["core_dependency_kind_loss"] = float(dependency_loss.item())
        metrics["core_dependency_kind_accuracy"] = float(
            (dependency_logits.argmax(dim=-1) == labels["dependency_kind_id"]).float().mean().item()
        )

        infoflow_logits = self.infoflow_data_edge_head(features)
        infoflow_loss = F.cross_entropy(infoflow_logits, labels["infoflow_has_data_edge"])
        losses.append(infoflow_loss * self.config.infoflow_data_edge_loss_weight)
        metrics["core_infoflow_data_edge_loss"] = float(infoflow_loss.item())
        metrics["core_infoflow_data_edge_accuracy"] = float(
            (infoflow_logits.argmax(dim=-1) == labels["infoflow_has_data_edge"]).float().mean().item()
        )

        source_candidate_features = labels.get("source_candidate_features")
        source_candidate_mask = labels.get("source_candidate_mask")
        source_membership_labels = labels.get("source_membership_labels")
        direct_edge_labels = labels.get("direct_edge_to_target_labels")
        if (
            source_candidate_features is not None
            and source_candidate_mask is not None
            and source_membership_labels is not None
            and direct_edge_labels is not None
        ):
            candidate_hidden = self.candidate_proj(
                torch.cat(
                    (
                        features.unsqueeze(1).expand(-1, source_candidate_features.size(1), -1),
                        source_candidate_features.to(dtype=features.dtype),
                    ),
                    dim=-1,
                )
            )
            source_membership_logits = self.source_membership_head(candidate_hidden).squeeze(-1)
            source_loss, source_accuracy = self.masked_binary_objective(
                source_membership_logits,
                source_membership_labels,
                source_candidate_mask,
            )
            if source_loss is not None:
                losses.append(source_loss * self.config.source_membership_loss_weight)
                metrics["core_source_membership_loss"] = float(source_loss.item())
                metrics["core_source_membership_accuracy"] = source_accuracy

            direct_edge_logits = self.direct_edge_head(candidate_hidden).squeeze(-1)
            direct_edge_loss, direct_edge_accuracy = self.masked_binary_objective(
                direct_edge_logits,
                direct_edge_labels,
                source_candidate_mask,
            )
            if direct_edge_loss is not None:
                losses.append(direct_edge_loss * self.config.direct_edge_loss_weight)
                metrics["core_direct_edge_loss"] = float(direct_edge_loss.item())
                metrics["core_direct_edge_accuracy"] = direct_edge_accuracy

        total_loss = torch.stack(losses).sum() if losses else hidden_states.new_zeros(())
        metrics["core_auxiliary_loss"] = float(total_loss.item())
        return CoReAuxiliaryOutput(loss=total_loss, metrics=metrics)

    def masked_binary_objective(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, float]:
        active = mask.to(dtype=logits.dtype)
        denom = active.sum()
        if float(denom.item()) <= 0.0:
            return None, 0.0
        labels = labels.to(dtype=logits.dtype)
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (losses * active).sum() / denom
        predictions = (logits >= 0).to(dtype=labels.dtype)
        accuracy = ((predictions == labels).to(dtype=logits.dtype) * active).sum() / denom
        return loss, float(accuracy.item())
