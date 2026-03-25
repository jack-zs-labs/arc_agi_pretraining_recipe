from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DecoderModelConfig


@dataclass
class OscarAuxiliaryOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class OscarAuxiliaryHeads(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        self.config = config.oscar_auxiliary
        projection_hidden = self.config.resolved_projection_hidden_size(config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.family_head = nn.Linear(projection_hidden, self.config.family_output_size, bias=True)
        self.section_depth_head = nn.Linear(projection_hidden, self.config.section_depth_output_size, bias=True)
        self.doc_group_head = nn.Linear(projection_hidden, self.config.doc_group_output_size, bias=True)
        self.doc_title_head = nn.Linear(projection_hidden, self.config.doc_title_output_size, bias=True)
        self.section_path_head = nn.Linear(projection_hidden, self.config.section_path_output_size, bias=True)
        self.section_parent_head = nn.Linear(projection_hidden, self.config.section_parent_output_size, bias=True)
        self.concept_head = nn.Linear(projection_hidden, self.config.concept_output_size, bias=True)
        self.related_doc_head = nn.Linear(projection_hidden, self.config.doc_title_output_size, bias=True)
        self.workflow_kpi_head = nn.Linear(projection_hidden, self.config.workflow_kpi_output_size, bias=True)
        self.workflow_improvement_head = nn.Linear(
            projection_hidden,
            self.config.workflow_improvement_output_size,
            bias=True,
        )
        self.workflow_motif_head = nn.Linear(
            projection_hidden,
            self.config.workflow_motif_output_size,
            bias=True,
        )
        self.workflow_reward_bucket_head = nn.Linear(
            projection_hidden,
            self.config.workflow_reward_bucket_output_size,
            bias=True,
        )
        self.workflow_reward_score_head = nn.Linear(projection_hidden, 1, bias=True)
        self.workflow_action_step_head = nn.Linear(
            projection_hidden,
            self.config.workflow_action_step_output_size,
            bias=True,
        )
        self.workflow_action_kpi_head = nn.Linear(
            projection_hidden,
            self.config.workflow_canonical_kpi_output_size,
            bias=True,
        )
        self.workflow_action_intervention_head = nn.Linear(
            projection_hidden,
            self.config.workflow_canonical_intervention_output_size,
            bias=True,
        )
        self.workflow_motif_context_embedding = (
            nn.Embedding(self.config.workflow_motif_output_size, projection_hidden)
            if self.config.workflow_motif_output_size > 0
            else None
        )
        self.workflow_canonical_kpi_context_embedding = (
            nn.Embedding(self.config.workflow_canonical_kpi_output_size, projection_hidden)
            if self.config.workflow_canonical_kpi_output_size > 0
            else None
        )
        self.workflow_canonical_intervention_context_embedding = (
            nn.Embedding(self.config.workflow_canonical_intervention_output_size, projection_hidden)
            if self.config.workflow_canonical_intervention_output_size > 0
            else None
        )
        self.workflow_action_step_embedding = (
            nn.Embedding(self.config.workflow_action_step_output_size, projection_hidden)
            if self.config.workflow_action_step_output_size > 0
            else None
        )
        self.workflow_bottleneck_proj = (
            nn.Linear(self.config.workflow_bottleneck_output_size, projection_hidden, bias=False)
            if self.config.workflow_bottleneck_output_size > 0
            else None
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

    def masked_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        active = active_mask > 0
        if not torch.any(active):
            return logits.new_zeros(()), 0.0, 0.0
        active_logits = logits[active]
        active_labels = labels[active]
        loss = F.cross_entropy(active_logits, active_labels)
        accuracy = float((active_logits.argmax(dim=-1) == active_labels).float().mean().item())
        active_fraction = float(active.float().mean().item())
        return loss, accuracy, active_fraction

    def masked_mse(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        active = active_mask > 0
        if not torch.any(active):
            return predictions.new_zeros(()), 0.0, 0.0
        active_predictions = predictions[active]
        active_labels = labels[active]
        loss = F.mse_loss(active_predictions, active_labels)
        active_fraction = float(active.float().mean().item())
        return loss, float(loss.item()), active_fraction

    def masked_candidate_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        active_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, torch.Tensor | None]:
        active = active_mask > 0
        if not torch.any(active):
            return logits.new_zeros(()), 0.0, 0.0, None
        active_logits = logits[active]
        active_labels = labels[active]
        active_candidate_mask = candidate_mask[active].to(dtype=torch.bool)
        valid_rows = torch.any(active_candidate_mask, dim=1)
        if not bool(torch.all(valid_rows)):
            active_candidate_mask = active_candidate_mask.clone()
            active_candidate_mask[~valid_rows] = True
        masked_logits = active_logits.masked_fill(~active_candidate_mask, torch.finfo(active_logits.dtype).min)
        loss = F.cross_entropy(masked_logits, active_labels)
        predictions = masked_logits.argmax(dim=-1)
        accuracy = float((predictions == active_labels).float().mean().item())
        active_fraction = float(active.float().mean().item())
        return loss, accuracy, active_fraction, predictions

    def workflow_structured_features(
        self,
        features: torch.Tensor,
        *,
        labels: dict[str, torch.Tensor],
        include_action_step: bool,
        include_kpi_context: bool,
        include_source_intervention: bool,
    ) -> torch.Tensor:
        structured = features
        motif_embedding = self.workflow_motif_context_embedding
        if motif_embedding is not None:
            motif_mask = labels["workflow_motif_active_mask"].to(dtype=features.dtype).unsqueeze(-1)
            structured = structured + (motif_mask * motif_embedding(labels["workflow_motif_id"]))
        bottleneck_proj = self.workflow_bottleneck_proj
        if bottleneck_proj is not None:
            structured = structured + bottleneck_proj(labels["workflow_active_bottleneck_multihot"].to(dtype=features.dtype))
        if include_action_step and self.workflow_action_step_embedding is not None:
            step_mask = labels["workflow_action_step_active_mask"].to(dtype=features.dtype).unsqueeze(-1)
            structured = structured + (step_mask * self.workflow_action_step_embedding(labels["workflow_action_step_id"]))
        if include_kpi_context and self.workflow_canonical_kpi_context_embedding is not None:
            kpi_mask = labels["workflow_canonical_kpi_context_active_mask"].to(dtype=features.dtype).unsqueeze(-1)
            structured = structured + (
                kpi_mask * self.workflow_canonical_kpi_context_embedding(labels["workflow_canonical_kpi_context_id"])
            )
        if include_source_intervention and self.workflow_canonical_intervention_context_embedding is not None:
            source_mask = labels["workflow_source_canonical_intervention_active_mask"].to(dtype=features.dtype).unsqueeze(-1)
            structured = structured + (
                source_mask
                * self.workflow_canonical_intervention_context_embedding(
                    labels["workflow_source_canonical_intervention_id"]
                )
            )
        return structured

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        labels: dict[str, torch.Tensor],
    ) -> OscarAuxiliaryOutput:
        pooled = self.pooled_hidden_state(hidden_states, attention_mask=attention_mask)
        features = self.proj(pooled)

        losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {}

        family_logits = self.family_head(features)
        family_loss = F.cross_entropy(family_logits, labels["family_id"])
        losses.append(family_loss * self.config.family_loss_weight)
        metrics["oscar_family_loss"] = float(family_loss.item())
        metrics["oscar_family_accuracy"] = float(
            (family_logits.argmax(dim=-1) == labels["family_id"]).float().mean().item()
        )

        section_depth_logits = self.section_depth_head(features)
        section_depth_loss = F.cross_entropy(section_depth_logits, labels["section_depth_bucket"])
        losses.append(section_depth_loss * self.config.section_depth_loss_weight)
        metrics["oscar_section_depth_loss"] = float(section_depth_loss.item())
        metrics["oscar_section_depth_accuracy"] = float(
            (section_depth_logits.argmax(dim=-1) == labels["section_depth_bucket"]).float().mean().item()
        )

        doc_group_logits = self.doc_group_head(features)
        doc_group_loss = F.cross_entropy(doc_group_logits, labels["doc_group_id"])
        losses.append(doc_group_loss * self.config.doc_group_loss_weight)
        metrics["oscar_doc_group_loss"] = float(doc_group_loss.item())
        metrics["oscar_doc_group_accuracy"] = float(
            (doc_group_logits.argmax(dim=-1) == labels["doc_group_id"]).float().mean().item()
        )

        doc_title_logits = self.doc_title_head(features)
        doc_title_loss = F.cross_entropy(doc_title_logits, labels["doc_title_id"])
        losses.append(doc_title_loss * self.config.doc_title_loss_weight)
        metrics["oscar_doc_title_loss"] = float(doc_title_loss.item())
        metrics["oscar_doc_title_accuracy"] = float(
            (doc_title_logits.argmax(dim=-1) == labels["doc_title_id"]).float().mean().item()
        )

        section_path_logits = self.section_path_head(features)
        section_path_loss = F.cross_entropy(section_path_logits, labels["section_path_id"])
        losses.append(section_path_loss * self.config.section_path_loss_weight)
        metrics["oscar_section_path_loss"] = float(section_path_loss.item())
        metrics["oscar_section_path_accuracy"] = float(
            (section_path_logits.argmax(dim=-1) == labels["section_path_id"]).float().mean().item()
        )

        section_parent_logits = self.section_parent_head(features)
        section_parent_loss = F.cross_entropy(section_parent_logits, labels["section_parent_id"])
        losses.append(section_parent_loss * self.config.section_parent_loss_weight)
        metrics["oscar_section_parent_loss"] = float(section_parent_loss.item())
        metrics["oscar_section_parent_accuracy"] = float(
            (section_parent_logits.argmax(dim=-1) == labels["section_parent_id"]).float().mean().item()
        )

        concept_logits = self.concept_head(features)
        concept_labels = labels["concept_multihot"].to(dtype=features.dtype)
        concept_loss = F.binary_cross_entropy_with_logits(concept_logits, concept_labels)
        losses.append(concept_loss * self.config.concept_loss_weight)
        metrics["oscar_concept_loss"] = float(concept_loss.item())
        concept_predictions = (concept_logits >= 0).to(dtype=concept_labels.dtype)
        metrics["oscar_concept_accuracy"] = float((concept_predictions == concept_labels).float().mean().item())
        metrics["oscar_concept_exact_match"] = float(
            (concept_predictions == concept_labels).all(dim=-1).float().mean().item()
        )

        related_doc_logits = self.related_doc_head(features)
        related_doc_labels = labels["related_doc_multihot"].to(dtype=features.dtype)
        related_doc_loss = F.binary_cross_entropy_with_logits(related_doc_logits, related_doc_labels)
        losses.append(related_doc_loss * self.config.related_doc_loss_weight)
        metrics["oscar_related_doc_loss"] = float(related_doc_loss.item())
        related_doc_predictions = (related_doc_logits >= 0).to(dtype=related_doc_labels.dtype)
        metrics["oscar_related_doc_accuracy"] = float(
            (related_doc_predictions == related_doc_labels).float().mean().item()
        )
        metrics["oscar_related_doc_exact_match"] = float(
            (related_doc_predictions == related_doc_labels).all(dim=-1).float().mean().item()
        )

        workflow_kpi_logits = self.workflow_kpi_head(features)
        workflow_kpi_loss, workflow_kpi_accuracy, workflow_kpi_active_fraction = self.masked_cross_entropy(
            workflow_kpi_logits,
            labels["workflow_kpi_id"],
            labels["workflow_kpi_active_mask"],
        )
        losses.append(workflow_kpi_loss * self.config.workflow_kpi_loss_weight)
        metrics["oscar_workflow_kpi_loss"] = float(workflow_kpi_loss.item())
        metrics["oscar_workflow_kpi_accuracy"] = workflow_kpi_accuracy
        metrics["oscar_workflow_kpi_active_fraction"] = workflow_kpi_active_fraction

        workflow_improvement_logits = self.workflow_improvement_head(features)
        workflow_improvement_loss, workflow_improvement_accuracy, workflow_improvement_active_fraction = (
            self.masked_cross_entropy(
                workflow_improvement_logits,
                labels["workflow_improvement_id"],
                labels["workflow_improvement_active_mask"],
            )
        )
        losses.append(workflow_improvement_loss * self.config.workflow_improvement_loss_weight)
        metrics["oscar_workflow_improvement_loss"] = float(workflow_improvement_loss.item())
        metrics["oscar_workflow_improvement_accuracy"] = workflow_improvement_accuracy
        metrics["oscar_workflow_improvement_active_fraction"] = workflow_improvement_active_fraction

        workflow_motif_logits = self.workflow_motif_head(features)
        workflow_motif_loss, workflow_motif_accuracy, workflow_motif_active_fraction = self.masked_cross_entropy(
            workflow_motif_logits,
            labels["workflow_motif_id"],
            labels["workflow_motif_active_mask"],
        )
        losses.append(workflow_motif_loss * self.config.workflow_motif_loss_weight)
        metrics["oscar_workflow_motif_loss"] = float(workflow_motif_loss.item())
        metrics["oscar_workflow_motif_accuracy"] = workflow_motif_accuracy
        metrics["oscar_workflow_motif_active_fraction"] = workflow_motif_active_fraction

        workflow_reward_bucket_logits = self.workflow_reward_bucket_head(features)
        workflow_reward_bucket_loss, workflow_reward_bucket_accuracy, workflow_reward_active_fraction = (
            self.masked_cross_entropy(
                workflow_reward_bucket_logits,
                labels["workflow_reward_bucket_id"],
                labels["workflow_reward_active_mask"],
            )
        )
        losses.append(workflow_reward_bucket_loss * self.config.workflow_reward_bucket_loss_weight)
        metrics["oscar_workflow_reward_bucket_loss"] = float(workflow_reward_bucket_loss.item())
        metrics["oscar_workflow_reward_bucket_accuracy"] = workflow_reward_bucket_accuracy
        metrics["oscar_workflow_reward_active_fraction"] = workflow_reward_active_fraction

        workflow_reward_score_predictions = torch.sigmoid(self.workflow_reward_score_head(features)).squeeze(-1)
        workflow_reward_score_loss, workflow_reward_score_mse, workflow_reward_score_active_fraction = self.masked_mse(
            workflow_reward_score_predictions,
            labels["workflow_reward_score"].to(dtype=features.dtype),
            labels["workflow_reward_active_mask"],
        )
        losses.append(workflow_reward_score_loss * self.config.workflow_reward_score_loss_weight)
        metrics["oscar_workflow_reward_score_loss"] = float(workflow_reward_score_loss.item())
        metrics["oscar_workflow_reward_score_mse"] = workflow_reward_score_mse
        metrics["oscar_workflow_reward_score_active_fraction"] = workflow_reward_score_active_fraction

        step_features = self.workflow_structured_features(
            features,
            labels=labels,
            include_action_step=False,
            include_kpi_context=False,
            include_source_intervention=False,
        )
        workflow_action_step_logits = self.workflow_action_step_head(step_features)
        workflow_action_step_loss, workflow_action_step_accuracy, workflow_action_step_active_fraction, step_predictions = (
            self.masked_candidate_cross_entropy(
                workflow_action_step_logits,
                labels["workflow_action_step_id"],
                labels["workflow_action_step_active_mask"],
                torch.ones_like(workflow_action_step_logits, dtype=features.dtype),
            )
        )
        losses.append(workflow_action_step_loss * self.config.workflow_action_step_loss_weight)
        metrics["oscar_workflow_action_step_loss"] = float(workflow_action_step_loss.item())
        metrics["oscar_workflow_action_step_accuracy"] = workflow_action_step_accuracy
        metrics["oscar_workflow_action_step_active_fraction"] = workflow_action_step_active_fraction

        kpi_action_features = self.workflow_structured_features(
            features,
            labels=labels,
            include_action_step=True,
            include_kpi_context=False,
            include_source_intervention=False,
        )
        workflow_action_kpi_logits = self.workflow_action_kpi_head(kpi_action_features)
        workflow_action_kpi_loss, workflow_action_kpi_accuracy, workflow_action_kpi_active_fraction, kpi_predictions = (
            self.masked_candidate_cross_entropy(
                workflow_action_kpi_logits,
                labels["workflow_action_kpi_family_id"],
                labels["workflow_action_kpi_active_mask"],
                labels["workflow_action_kpi_candidate_mask"],
            )
        )
        losses.append(workflow_action_kpi_loss * self.config.workflow_action_kpi_loss_weight)
        metrics["oscar_workflow_action_kpi_family_loss"] = float(workflow_action_kpi_loss.item())
        metrics["oscar_workflow_action_kpi_family_accuracy"] = workflow_action_kpi_accuracy
        metrics["oscar_workflow_action_kpi_family_active_fraction"] = workflow_action_kpi_active_fraction

        intervention_action_features = self.workflow_structured_features(
            features,
            labels=labels,
            include_action_step=True,
            include_kpi_context=True,
            include_source_intervention=True,
        )
        workflow_action_intervention_logits = self.workflow_action_intervention_head(intervention_action_features)
        (
            workflow_action_intervention_loss,
            workflow_action_intervention_accuracy,
            workflow_action_intervention_active_fraction,
            intervention_predictions,
        ) = self.masked_candidate_cross_entropy(
            workflow_action_intervention_logits,
            labels["workflow_action_intervention_family_id"],
            labels["workflow_action_intervention_active_mask"],
            labels["workflow_action_intervention_candidate_mask"],
        )
        losses.append(
            workflow_action_intervention_loss * self.config.workflow_action_intervention_loss_weight
        )
        metrics["oscar_workflow_action_intervention_family_loss"] = float(
            workflow_action_intervention_loss.item()
        )
        metrics["oscar_workflow_action_intervention_family_accuracy"] = workflow_action_intervention_accuracy
        metrics["oscar_workflow_action_intervention_family_active_fraction"] = (
            workflow_action_intervention_active_fraction
        )

        action_step_active = labels["workflow_action_step_active_mask"] > 0
        action_step_correct = torch.zeros_like(action_step_active, dtype=torch.bool)
        if step_predictions is not None:
            action_step_correct[action_step_active] = step_predictions == labels["workflow_action_step_id"][action_step_active]
        kpi_active = labels["workflow_action_kpi_active_mask"] > 0
        kpi_correct = torch.zeros_like(kpi_active, dtype=torch.bool)
        if kpi_predictions is not None:
            kpi_correct[kpi_active] = (
                kpi_predictions == labels["workflow_action_kpi_family_id"][kpi_active]
            )
        intervention_active = labels["workflow_action_intervention_active_mask"] > 0
        intervention_correct = torch.zeros_like(intervention_active, dtype=torch.bool)
        if intervention_predictions is not None:
            intervention_correct[intervention_active] = (
                intervention_predictions == labels["workflow_action_intervention_family_id"][intervention_active]
            )
        action_exact = torch.zeros_like(action_step_active, dtype=torch.bool)
        action_exact[kpi_active] = action_step_correct[kpi_active] & kpi_correct[kpi_active]
        action_exact[intervention_active] = (
            action_step_correct[intervention_active] & intervention_correct[intervention_active]
        )
        if torch.any(action_step_active):
            metrics["oscar_workflow_action_exact_match"] = float(
                action_exact[action_step_active].float().mean().item()
            )
        else:
            metrics["oscar_workflow_action_exact_match"] = 0.0
        transfer_active = labels["workflow_transfer_action_active_mask"] > 0
        if torch.any(transfer_active):
            metrics["oscar_workflow_transfer_intervention_accuracy"] = float(
                intervention_correct[transfer_active].float().mean().item()
            )
        else:
            metrics["oscar_workflow_transfer_intervention_accuracy"] = 0.0
        intervention_trace_active = labels["workflow_intervention_trace_action_active_mask"] > 0
        if torch.any(intervention_trace_active):
            metrics["oscar_workflow_intervention_trace_action_accuracy"] = float(
                action_exact[intervention_trace_active].float().mean().item()
            )
        else:
            metrics["oscar_workflow_intervention_trace_action_accuracy"] = 0.0

        total_loss = torch.stack(losses).sum() if losses else hidden_states.new_zeros(())
        metrics["oscar_auxiliary_loss"] = float(total_loss.item())
        return OscarAuxiliaryOutput(loss=total_loss, metrics=metrics)
