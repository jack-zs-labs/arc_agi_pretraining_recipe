from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DecoderModelConfig


@dataclass
class OscarGraphAuxiliaryOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class OscarGraphAuxiliaryHeads(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        self.config = config.oscar_graph_auxiliary
        projection_hidden = self.config.resolved_projection_hidden_size(config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.family_head = nn.Linear(projection_hidden, self.config.family_output_size, bias=True)
        self.domain_head = nn.Linear(projection_hidden, self.config.domain_output_size, bias=True)
        self.motif_head = (
            nn.Linear(projection_hidden, self.config.motif_output_size, bias=True)
            if self.config.motif_output_size > 0
            else None
        )
        self.node_candidate_proj = nn.Sequential(
            nn.LazyLinear(projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.relation_candidate_proj = nn.Sequential(
            nn.LazyLinear(projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.relation_head = nn.Linear(projection_hidden, 1, bias=True)
        self.neighbor_head = nn.Linear(projection_hidden, 1, bias=True)
        self.path_via_head = nn.Linear(projection_hidden, 1, bias=True)
        self.path_target_head = nn.Linear(projection_hidden, 1, bias=True)
        self.grounding_head = nn.Linear(projection_hidden, 1, bias=True)
        self.rollout_candidate_proj = nn.Sequential(
            nn.LazyLinear(projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.rollout_step_heads = nn.ModuleList(
            nn.Linear(projection_hidden, 1, bias=True)
            for _ in range(self.config.rollout_max_steps)
        )

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
    ) -> OscarGraphAuxiliaryOutput:
        pooled = self.pooled_hidden_state(hidden_states, attention_mask=attention_mask)
        features = self.proj(pooled)

        losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {}

        family_logits = self.family_head(features)
        family_loss = F.cross_entropy(family_logits, labels["family_id"])
        losses.append(family_loss * self.config.family_loss_weight)
        metrics["oscar_graph_family_loss"] = float(family_loss.item())
        metrics["oscar_graph_family_accuracy"] = float(
            (family_logits.argmax(dim=-1) == labels["family_id"]).float().mean().item()
        )

        domain_logits = self.domain_head(features)
        domain_loss = F.cross_entropy(domain_logits, labels["graph_domain_id"])
        losses.append(domain_loss * self.config.domain_loss_weight)
        metrics["oscar_graph_domain_loss"] = float(domain_loss.item())
        metrics["oscar_graph_domain_accuracy"] = float(
            (domain_logits.argmax(dim=-1) == labels["graph_domain_id"]).float().mean().item()
        )

        if self.motif_head is not None and "rollout_motif_id" in labels:
            motif_logits = self.motif_head(features)
            motif_loss = F.cross_entropy(motif_logits, labels["rollout_motif_id"])
            losses.append(motif_loss * self.config.rollout_motif_loss_weight)
            metrics["oscar_graph_rollout_motif_loss"] = float(motif_loss.item())
            metrics["oscar_graph_rollout_motif_accuracy"] = float(
                (motif_logits.argmax(dim=-1) == labels["rollout_motif_id"]).float().mean().item()
            )

        relation_loss, relation_metrics = self._candidate_binary_head(
            pooled_features=features,
            candidate_features=labels.get("relation_candidate_features"),
            candidate_mask=labels.get("relation_candidate_mask"),
            candidate_labels=labels.get("relation_candidate_labels"),
            candidate_proj=self.relation_candidate_proj,
            candidate_head=self.relation_head,
            top1_metric_name="oscar_graph_relation_accuracy",
            exact_metric_name=None,
        )
        if relation_loss is not None:
            losses.append(relation_loss * self.config.relation_loss_weight)
            metrics.update(relation_metrics)

        neighbor_loss, neighbor_metrics = self._candidate_binary_head(
            pooled_features=features,
            candidate_features=labels.get("neighbor_candidate_features"),
            candidate_mask=labels.get("neighbor_candidate_mask"),
            candidate_labels=labels.get("neighbor_target_labels"),
            candidate_proj=self.node_candidate_proj,
            candidate_head=self.neighbor_head,
            top1_metric_name=None,
            exact_metric_name="oscar_graph_neighbor_exact_match",
            prefix="oscar_graph_neighbor",
        )
        if neighbor_loss is not None:
            losses.append(neighbor_loss * self.config.neighbor_loss_weight)
            metrics.update(neighbor_metrics)

        path_via_loss, path_via_metrics = self._candidate_binary_head(
            pooled_features=features,
            candidate_features=labels.get("path_via_candidate_features"),
            candidate_mask=labels.get("path_via_candidate_mask"),
            candidate_labels=labels.get("path_via_labels"),
            candidate_proj=self.node_candidate_proj,
            candidate_head=self.path_via_head,
            top1_metric_name="oscar_graph_path_via_accuracy",
            exact_metric_name=None,
            prefix="oscar_graph_path_via",
        )
        if path_via_loss is not None:
            losses.append(path_via_loss * self.config.path_via_loss_weight)
            metrics.update(path_via_metrics)

        path_target_loss, path_target_metrics = self._candidate_binary_head(
            pooled_features=features,
            candidate_features=labels.get("path_target_candidate_features"),
            candidate_mask=labels.get("path_target_candidate_mask"),
            candidate_labels=labels.get("path_target_labels"),
            candidate_proj=self.node_candidate_proj,
            candidate_head=self.path_target_head,
            top1_metric_name="oscar_graph_path_target_accuracy",
            exact_metric_name=None,
            prefix="oscar_graph_path_target",
        )
        if path_target_loss is not None:
            losses.append(path_target_loss * self.config.path_target_loss_weight)
            metrics.update(path_target_metrics)

        grounding_loss, grounding_metrics = self._candidate_binary_head(
            pooled_features=features,
            candidate_features=labels.get("grounding_candidate_features"),
            candidate_mask=labels.get("grounding_candidate_mask"),
            candidate_labels=labels.get("grounding_labels"),
            candidate_proj=self.node_candidate_proj,
            candidate_head=self.grounding_head,
            top1_metric_name="oscar_graph_grounding_accuracy",
            exact_metric_name=None,
            prefix="oscar_graph_grounding",
        )
        if grounding_loss is not None:
            losses.append(grounding_loss * self.config.grounding_loss_weight)
            metrics.update(grounding_metrics)

        rollout_loss, rollout_metrics = self._rollout_step_objective(
            pooled_features=features,
            candidate_features=labels.get("rollout_step_candidate_features"),
            candidate_mask=labels.get("rollout_step_candidate_mask"),
            candidate_labels=labels.get("rollout_step_labels"),
            step_active_mask=labels.get("rollout_step_active_mask"),
        )
        if rollout_loss is not None:
            losses.append(rollout_loss * self.config.rollout_step_loss_weight)
            metrics.update(rollout_metrics)

        total_loss = torch.stack(losses).sum() if losses else hidden_states.new_zeros(())
        metrics["oscar_graph_auxiliary_loss"] = float(total_loss.item())
        return OscarGraphAuxiliaryOutput(loss=total_loss, metrics=metrics)

    def _rollout_step_objective(
        self,
        *,
        pooled_features: torch.Tensor,
        candidate_features: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
        candidate_labels: torch.Tensor | None,
        step_active_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if (
            candidate_features is None
            or candidate_mask is None
            or candidate_labels is None
            or step_active_mask is None
            or candidate_features.dim() != 4
        ):
            return None, {}
        if float(step_active_mask.sum().item()) <= 0.0:
            return None, {}
        losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        step_exact_matches: list[torch.Tensor] = []
        batch_size, max_steps, _max_candidates, _feature_dim = candidate_features.shape
        for step_index in range(min(max_steps, len(self.rollout_step_heads))):
            active_rows = step_active_mask[:, step_index] > 0
            if not bool(active_rows.any().item()):
                continue
            step_candidate_features = candidate_features[:, step_index, :, :]
            step_candidate_mask = candidate_mask[:, step_index, :]
            step_candidate_labels = candidate_labels[:, step_index, :]
            if float(step_candidate_mask.sum().item()) <= 0.0:
                continue
            candidate_hidden = self.rollout_candidate_proj(
                torch.cat(
                    (
                        pooled_features.unsqueeze(1).expand(-1, step_candidate_features.size(1), -1),
                        step_candidate_features.to(dtype=pooled_features.dtype),
                    ),
                    dim=-1,
                )
            )
            logits = self.rollout_step_heads[step_index](candidate_hidden).squeeze(-1)
            step_loss, step_candidate_accuracy = self.masked_binary_objective(
                logits,
                step_candidate_labels,
                step_candidate_mask,
            )
            if step_loss is None:
                continue
            losses.append(step_loss)
            metrics[f"oscar_graph_rollout_step_{step_index + 1}_loss"] = float(step_loss.item())
            metrics[f"oscar_graph_rollout_step_{step_index + 1}_candidate_accuracy"] = step_candidate_accuracy
            metrics[f"oscar_graph_rollout_step_{step_index + 1}_accuracy"] = self.masked_top1_accuracy(
                logits,
                step_candidate_labels,
                step_candidate_mask,
            )
            masked_logits = logits.masked_fill(step_candidate_mask <= 0, torch.finfo(logits.dtype).min)
            predicted = masked_logits.argmax(dim=-1)
            gold = step_candidate_labels.argmax(dim=-1)
            step_exact_matches.append((predicted == gold) | ~active_rows)
        if not losses:
            return None, {}
        if step_exact_matches:
            exact_match = torch.stack(step_exact_matches, dim=0).all(dim=0).float().mean()
            metrics["oscar_graph_rollout_exact_match"] = float(exact_match.item())
        return torch.stack(losses).mean(), metrics

    def _candidate_binary_head(
        self,
        *,
        pooled_features: torch.Tensor,
        candidate_features: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
        candidate_labels: torch.Tensor | None,
        candidate_proj: nn.Module,
        candidate_head: nn.Module,
        top1_metric_name: str | None,
        exact_metric_name: str | None,
        prefix: str | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if (
            candidate_features is None
            or candidate_mask is None
            or candidate_labels is None
        ):
            return None, {}
        if float(candidate_mask.sum().item()) <= 0.0:
            return None, {}
        candidate_hidden = candidate_proj(
            torch.cat(
                (
                    pooled_features.unsqueeze(1).expand(-1, candidate_features.size(1), -1),
                    candidate_features.to(dtype=pooled_features.dtype),
                ),
                dim=-1,
            )
        )
        logits = candidate_head(candidate_hidden).squeeze(-1)
        loss, accuracy = self.masked_binary_objective(logits, candidate_labels, candidate_mask)
        if loss is None:
            return None, {}
        metric_prefix = prefix or top1_metric_name or exact_metric_name or "oscar_graph_candidate"
        metrics = {
            f"{metric_prefix}_loss": float(loss.item()),
            f"{metric_prefix}_candidate_accuracy": accuracy,
        }
        if top1_metric_name is not None:
            metrics[top1_metric_name] = self.masked_top1_accuracy(logits, candidate_labels, candidate_mask)
        if exact_metric_name is not None:
            metrics[exact_metric_name] = self.masked_exact_match(logits, candidate_labels, candidate_mask)
        return loss, metrics

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

    def masked_top1_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        valid = (mask > 0) & (labels > 0)
        valid_rows = valid.any(dim=-1)
        if not bool(valid_rows.any().item()):
            return 0.0
        masked_logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        predicted = masked_logits.argmax(dim=-1)
        gold = labels.argmax(dim=-1)
        accuracy = (predicted[valid_rows] == gold[valid_rows]).float().mean()
        return float(accuracy.item())

    def masked_exact_match(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        active = mask > 0
        valid_rows = active.any(dim=-1)
        if not bool(valid_rows.any().item()):
            return 0.0
        predictions = logits >= 0
        exact = ((predictions == (labels > 0)) | ~active).all(dim=-1)
        return float(exact[valid_rows].float().mean().item())
