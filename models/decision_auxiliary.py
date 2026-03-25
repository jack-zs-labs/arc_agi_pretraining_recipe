from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DecisionActionConfig, DecoderModelConfig


LOGIT_FLOOR = -1e9


def argument_head_key(output_head: str, name_id: int) -> str:
    return f"{output_head}|{name_id}"


@dataclass
class DecisionActionAuxiliaryOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class DecisionActionAuxiliaryHeads(nn.Module):
    def __init__(self, config: DecoderModelConfig) -> None:
        super().__init__()
        self.config: DecisionActionConfig = config.decision_action
        projection_hidden = self.config.resolved_projection_hidden_size(config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, projection_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(projection_hidden, projection_hidden, bias=False),
            nn.SiLU(),
        )
        self.full_output_sizes = self.config.output_size_map()
        self.name_output_sizes = self.config.name_output_size_map()
        self.argument_output_sizes = self.config.argument_output_size_map()
        self.loss_weights = self.config.loss_weight_map()
        self.benchmark_adapter_names = tuple(self.config.benchmark_adapter_names)
        self.benchmark_name_to_id = {
            benchmark_name: index for index, benchmark_name in enumerate(self.benchmark_adapter_names)
        }
        self.benchmark_adapter = (
            nn.Embedding(len(self.benchmark_adapter_names), projection_hidden)
            if self.benchmark_adapter_names
            else None
        )
        self.benchmark_adapter_scale = (
            nn.Parameter(torch.zeros(()))
            if self.benchmark_adapter is not None
            else None
        )
        self.name_heads = nn.ModuleDict(
            {
                benchmark_name: nn.Linear(projection_hidden, head_size, bias=True)
                for benchmark_name, head_size in sorted(self.name_output_sizes.items())
            }
        )
        self.argument_heads = nn.ModuleDict(
            {
                head_key: nn.Linear(projection_hidden, head_size, bias=True)
                for head_key, head_size in sorted(self.argument_output_sizes.items())
            }
        )

    def pooled_hidden_state(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_index, lengths]

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        candidate_mask = mask.to(dtype=torch.bool)
        valid_rows = torch.any(candidate_mask, dim=1)
        if not bool(torch.all(valid_rows)):
            candidate_mask = candidate_mask.clone()
            candidate_mask[~valid_rows] = True
        return logits.masked_fill(~candidate_mask, LOGIT_FLOOR)

    def _adapt_features(
        self,
        features: torch.Tensor,
        *,
        benchmark_names: tuple[str, ...],
    ) -> torch.Tensor:
        if self.benchmark_adapter is None or self.benchmark_adapter_scale is None:
            return features
        if not benchmark_names:
            return features
        if len(benchmark_names) != features.size(0):
            raise ValueError("benchmark_names must align with the batch dimension.")
        adapter_ids = torch.tensor(
            [self.benchmark_name_to_id.get(name, -1) for name in benchmark_names],
            dtype=torch.long,
            device=features.device,
        )
        valid_mask = adapter_ids >= 0
        if not bool(torch.any(valid_mask)):
            return features
        adapted = features.clone()
        adapted[valid_mask] = adapted[valid_mask] + (
            self.benchmark_adapter_scale * self.benchmark_adapter(adapter_ids[valid_mask])
        )
        return adapted

    def _predict_exact_action_from_features(
        self,
        features: torch.Tensor,
        *,
        output_heads: tuple[str, ...],
        full_candidate_masks: torch.Tensor,
        full_candidate_name_ids: torch.Tensor,
        full_candidate_argument_ids: torch.Tensor,
    ) -> torch.Tensor:
        predictions = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        for row_index, output_head in enumerate(output_heads):
            if output_head not in self.name_heads:
                continue
            full_size = self.full_output_sizes.get(output_head, 0)
            if full_size <= 0:
                continue
            allowed = full_candidate_masks[row_index, :full_size].to(dtype=torch.bool)
            if not bool(torch.any(allowed)):
                allowed = torch.ones(full_size, dtype=torch.bool, device=features.device)
            candidate_name_ids = full_candidate_name_ids[row_index, :full_size]
            candidate_argument_ids = full_candidate_argument_ids[row_index, :full_size]
            feature_row = features[row_index : row_index + 1]
            name_logits = self.name_heads[output_head](feature_row).squeeze(0)
            candidate_scores = torch.full(
                (full_size,),
                LOGIT_FLOOR,
                dtype=name_logits.dtype,
                device=features.device,
            )
            assigned_scores = torch.zeros(full_size, dtype=torch.bool, device=features.device)
            allowed_indices = allowed.nonzero(as_tuple=False).flatten()
            if allowed_indices.numel() == 0:
                predictions[row_index] = 0
                continue
            unique_name_ids = torch.unique(candidate_name_ids.index_select(0, allowed_indices))
            for name_id_tensor in unique_name_ids:
                name_id = int(name_id_tensor.item())
                arg_key = argument_head_key(output_head, name_id)
                if arg_key not in self.argument_heads:
                    continue
                name_specific_indices = allowed_indices[
                    candidate_name_ids.index_select(0, allowed_indices) == name_id
                ]
                if name_specific_indices.numel() == 0:
                    continue
                arg_logits = self.argument_heads[arg_key](feature_row).squeeze(0)
                arg_ids = candidate_argument_ids.index_select(0, name_specific_indices)
                candidate_scores.index_copy_(
                    0,
                    name_specific_indices,
                    name_logits[name_id] + arg_logits.index_select(0, arg_ids),
                )
                assigned_scores.index_fill_(0, name_specific_indices, True)
            if not bool(torch.any(assigned_scores & allowed)):
                fallback_scores = torch.full_like(candidate_scores, LOGIT_FLOOR)
                fallback_scores[allowed] = name_logits.index_select(0, candidate_name_ids[allowed])
                candidate_scores = fallback_scores
            predictions[row_index] = int(torch.argmax(candidate_scores).item())
        return predictions

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        labels: dict[str, object],
    ) -> DecisionActionAuxiliaryOutput:
        benchmark_names = tuple(str(name) for name in labels.get("benchmark_names", ()))
        output_heads = tuple(str(name) for name in labels["output_heads"])
        target_action_ids = labels["target_action_ids"]
        target_name_ids = labels["target_name_ids"]
        target_argument_output_heads = tuple(str(name) for name in labels["target_argument_output_heads"])
        target_argument_ids = labels["target_argument_ids"]
        name_candidate_masks = labels["name_candidate_masks"]
        target_argument_candidate_masks = labels["target_argument_candidate_masks"]
        full_candidate_masks = labels["full_candidate_masks"]
        full_candidate_name_ids = labels["full_candidate_name_ids"]
        full_candidate_argument_ids = labels["full_candidate_argument_ids"]
        required_tensors = (
            target_action_ids,
            target_name_ids,
            target_argument_ids,
            name_candidate_masks,
            target_argument_candidate_masks,
            full_candidate_masks,
            full_candidate_name_ids,
            full_candidate_argument_ids,
        )
        if not all(isinstance(value, torch.Tensor) for value in required_tensors):
            raise TypeError("Decision action labels must provide tensor candidate and target payloads.")

        pooled = self.pooled_hidden_state(hidden_states, attention_mask=attention_mask)
        features = self.proj(pooled)
        features = self._adapt_features(features, benchmark_names=benchmark_names)

        losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        overall_name_correct = 0
        overall_name_count = 0
        overall_argument_correct = 0
        overall_argument_count = 0

        grouped_name_indices: dict[str, list[int]] = {}
        for index, output_head in enumerate(output_heads):
            grouped_name_indices.setdefault(output_head, []).append(index)
        for output_head, indices in grouped_name_indices.items():
            if output_head not in self.name_heads:
                continue
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
            logits = self.name_heads[output_head](features.index_select(0, index_tensor))
            head_size = self.name_output_sizes[output_head]
            masked_logits = self._masked_logits(
                logits,
                name_candidate_masks.index_select(0, index_tensor)[:, :head_size],
            )
            head_targets = target_name_ids.index_select(0, index_tensor)
            head_loss = F.cross_entropy(masked_logits, head_targets)
            weight = self.loss_weights.get(output_head, 1.0)
            losses.append(head_loss * weight)
            predictions = masked_logits.argmax(dim=-1)
            correct = int((predictions == head_targets).sum().item())
            overall_name_correct += correct
            overall_name_count += len(indices)
            metrics[f"decision_action_name_loss_{output_head}"] = float(head_loss.item())
            metrics[f"decision_action_name_accuracy_{output_head}"] = float(correct / max(len(indices), 1))

        grouped_argument_indices: dict[str, list[int]] = {}
        for index, head_key in enumerate(target_argument_output_heads):
            grouped_argument_indices.setdefault(head_key, []).append(index)
        for head_key, indices in grouped_argument_indices.items():
            if head_key not in self.argument_heads:
                continue
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
            logits = self.argument_heads[head_key](features.index_select(0, index_tensor))
            head_size = self.argument_output_sizes[head_key]
            masked_logits = self._masked_logits(
                logits,
                target_argument_candidate_masks.index_select(0, index_tensor)[:, :head_size],
            )
            head_targets = target_argument_ids.index_select(0, index_tensor)
            head_loss = F.cross_entropy(masked_logits, head_targets)
            output_head = head_key.split("|", 1)[0]
            weight = self.loss_weights.get(output_head, 1.0)
            losses.append(head_loss * weight)
            predictions = masked_logits.argmax(dim=-1)
            correct = int((predictions == head_targets).sum().item())
            overall_argument_correct += correct
            overall_argument_count += len(indices)
            metrics[f"decision_action_argument_loss_{head_key}"] = float(head_loss.item())
            metrics[f"decision_action_argument_accuracy_{head_key}"] = float(correct / max(len(indices), 1))

        exact_predictions = self._predict_exact_action_from_features(
            features,
            output_heads=output_heads,
            full_candidate_masks=full_candidate_masks,
            full_candidate_name_ids=full_candidate_name_ids,
            full_candidate_argument_ids=full_candidate_argument_ids,
        )
        exact_correct = (exact_predictions == target_action_ids).to(dtype=torch.float32)
        for output_head, indices in grouped_name_indices.items():
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
            metrics[f"decision_action_accuracy_{output_head}"] = float(
                exact_correct.index_select(0, index_tensor).mean().item()
            )

        total_loss = torch.stack(losses).sum() * self.config.action_loss_weight if losses else hidden_states.new_zeros(())
        metrics["decision_action_loss"] = float(total_loss.item())
        metrics["decision_action_name_accuracy"] = float(overall_name_correct / max(overall_name_count, 1))
        metrics["decision_action_argument_accuracy"] = float(
            overall_argument_correct / max(overall_argument_count, 1)
        )
        metrics["decision_action_accuracy"] = float(exact_correct.mean().item()) if exact_correct.numel() else 0.0
        return DecisionActionAuxiliaryOutput(loss=total_loss, metrics=metrics)

    @torch.inference_mode()
    def predict(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        benchmark_names: list[str] | tuple[str, ...] | None,
        output_heads: list[str],
        full_candidate_masks: torch.Tensor,
        full_candidate_name_ids: torch.Tensor,
        full_candidate_argument_ids: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pooled_hidden_state(hidden_states, attention_mask=attention_mask)
        features = self.proj(pooled)
        features = self._adapt_features(
            features,
            benchmark_names=tuple(str(name) for name in (benchmark_names or ())),
        )
        return self._predict_exact_action_from_features(
            features,
            output_heads=tuple(output_heads),
            full_candidate_masks=full_candidate_masks,
            full_candidate_name_ids=full_candidate_name_ids,
            full_candidate_argument_ids=full_candidate_argument_ids,
        )
