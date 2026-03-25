from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Literal, Protocol

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arc_trajectory_sampler import sample_episode, sample_latent_rule
from arc_trajectory_sampler.state_adapter import decision_workspace_state, encode_workspace, verifier_targets
from arc_trajectory_sampler.stage1_latent_sampler import Family
from arc_trajectory_sampler.stage2_episode_sampler import count_group_key
from arc_trajectory_sampler.stage3_grid_executor import (
    align_object_to,
    arrange_row,
    connector_object,
    move_until_contact,
    normalized_objects,
    orientation_from_value,
    place_in_container,
    reflect_objects,
    render_scene,
    replace_object,
    scene_from_objects,
    selected_objects,
    stack_next_to,
    stack_side,
    transformed_unary_object,
    execute_episode,
)
from arc_trajectory_sampler.stage4_trajectory_dataset import (
    TrajectoryRecord,
    TrajectoryStep,
    compile_episode_trajectories,
    compile_trajectory,
)
from experiments.arc_sampler_structured_verifier import (
    VerifierFitConfig,
    build_verifier_adapter,
    build_verifier_training_data,
)


MAX_GRID_HEIGHT = 20
MAX_GRID_WIDTH = 29
MAX_TRACE_STEPS = 5
START_ACTION_TOKEN = "<start>"
FAMILIES = tuple(family.value for family in Family)
FLOAT_DTYPE = np.float32
SIGMOID_EPS = FLOAT_DTYPE(1e-4)
GRAD_CLIP = FLOAT_DTYPE(8.0)
ANSWER_STEPS_BY_FAMILY = {
    Family.UNARY_OBJECT.value: frozenset({"transform"}),
    Family.RELATIONAL.value: frozenset({"relate"}),
    Family.COUNT_SELECT.value: frozenset({"group", "reduce", "act"}),
    Family.CONTEXTUAL.value: frozenset({"apply"}),
    Family.SYMBOL_MAP.value: frozenset({"apply"}),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class TrainConfig:
    gamma: float = 1.0
    fqi_iterations: int = 6
    optimizer_steps: int = 24
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    projection_dim: int = 192


@dataclass(frozen=True)
class SamplerEpisode:
    seed: int
    train_trajectories: tuple[TrajectoryRecord, ...]
    test_trajectory: TrajectoryRecord


class ArcFeatureEncoder(Protocol):
    start_action_index: int
    output_dim: int

    def encode(self, trajectory: TrajectoryRecord, step_index: int, previous_action_index: int) -> np.ndarray:
        ...


class RandomFeatureProjector:
    def __init__(self, input_dim: int, output_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(input_dim, 1)),
            size=(input_dim, output_dim),
        ).astype(FLOAT_DTYPE, copy=False)
        self.bias = rng.normal(loc=0.0, scale=0.15, size=output_dim).astype(FLOAT_DTYPE, copy=False)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(features @ self.weights + self.bias).astype(FLOAT_DTYPE, copy=False)


class ArcTraceFeatureEncoder:
    def __init__(
        self,
        *,
        projection_dim: int,
        seed: int,
        action_tokens: tuple[str, ...],
    ):
        self.family_to_index = {family: idx for idx, family in enumerate(FAMILIES)}
        self.action_tokens = action_tokens
        self.action_to_index = {token: idx for idx, token in enumerate(action_tokens)}
        self.start_action_index = self.action_to_index[START_ACTION_TOKEN]
        raw_dim = MAX_GRID_HEIGHT * MAX_GRID_WIDTH * 3 + 20 + 8
        self.projector = RandomFeatureProjector(raw_dim, projection_dim, seed=seed)
        self.output_dim = projection_dim + len(FAMILIES) + MAX_TRACE_STEPS + len(action_tokens) + 5

    def encode(self, trajectory: TrajectoryRecord, step_index: int, previous_action_index: int) -> np.ndarray:
        current_grid = current_workspace_grid(trajectory, step_index)
        input_grid = grid_to_array(trajectory.input_grid)
        current_array = grid_to_array(current_grid)
        delta_mask = (current_array != input_grid).astype(FLOAT_DTYPE)
        raw = np.concatenate(
            [
                (input_grid / 9.0).reshape(-1),
                (current_array / 9.0).reshape(-1),
                delta_mask.reshape(-1),
                color_histogram(input_grid),
                color_histogram(current_array),
                np.asarray(
                    [
                        float(step_index) / max(len(trajectory.steps), 1),
                        float(len(trajectory.steps) - step_index) / max(len(trajectory.steps), 1),
                        float(trajectory.difficulty) / 5.0,
                        float(np.mean(delta_mask)),
                        float(len(trajectory.input_state.get("selected_object_ids", []))) / 8.0,
                        float(len(trajectory.example.input_scene.objects)) / 12.0,
                        float(trajectory.example.input_scene.height) / MAX_GRID_HEIGHT,
                        float(trajectory.example.input_scene.width) / MAX_GRID_WIDTH,
                    ],
                    dtype=FLOAT_DTYPE,
                ),
            ]
        ).astype(FLOAT_DTYPE, copy=False)
        projected = self.projector.transform(raw)

        family_one_hot = np.zeros(len(FAMILIES), dtype=FLOAT_DTYPE)
        family_one_hot[self.family_to_index[trajectory.family]] = 1.0

        step_one_hot = np.zeros(MAX_TRACE_STEPS, dtype=FLOAT_DTYPE)
        step_one_hot[min(step_index, MAX_TRACE_STEPS - 1)] = 1.0

        previous_action = np.zeros(len(self.action_tokens), dtype=FLOAT_DTYPE)
        previous_action[previous_action_index] = 1.0

        scalar_features = np.asarray(
            [
                float(step_index) / max(len(trajectory.steps), 1),
                float(len(trajectory.steps)),
                float(trajectory.total_possible_reward),
                float(len(trajectory.trace_template)),
                1.0,
            ],
            dtype=FLOAT_DTYPE,
        )
        return np.concatenate([projected, family_one_hot, step_one_hot, previous_action, scalar_features]).astype(
            FLOAT_DTYPE,
            copy=False,
        )


FeatureEncoderName = Literal["flat_grid", "structured_oracle", "structured_workspace"]
PolicyEncoderName = Literal["flat_grid", "structured_oracle", "structured_reconstructed"]
StructuredVerifierProvider = Callable[[TrajectoryRecord, int, dict[str, Any], int], dict[str, Any]]


def stable_hash_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def hashed_bag(tokens: list[str], dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=FLOAT_DTYPE)
    if dim <= 0:
        return vector
    for token in tokens:
        hashed = stable_hash_u64(token)
        vector[hashed % dim] += FLOAT_DTYPE(-1.0 if ((hashed >> 63) & 1) else 1.0)
    if tokens:
        vector /= FLOAT_DTYPE(np.sqrt(len(tokens)))
    return vector


class ArcStructuredWorkspaceEncoder:
    """Workspace-only structured encoder with no verifier-side signals."""

    def __init__(
        self,
        *,
        projection_dim: int,
        seed: int,
        action_tokens: tuple[str, ...],
    ):
        self.family_to_index = {family: idx for idx, family in enumerate(FAMILIES)}
        self.action_tokens = action_tokens
        self.action_to_index = {token: idx for idx, token in enumerate(action_tokens)}
        self.start_action_index = self.action_to_index[START_ACTION_TOKEN]
        self.hash_prefix = f"seed={seed}|"
        self.hash_dim = projection_dim
        self.workspace_scalar_dim = 11
        self.output_dim = self.hash_dim + len(FAMILIES) + MAX_TRACE_STEPS + len(action_tokens) + self.workspace_scalar_dim

    def encode(self, trajectory: TrajectoryRecord, step_index: int, previous_action_index: int) -> np.ndarray:
        encoded = encode_workspace(trajectory, step_index, include_verifier=False)
        hashed = hashed_bag([self.hash_prefix + token for token in encoded.tokens], self.hash_dim)

        family_one_hot = np.zeros(len(FAMILIES), dtype=FLOAT_DTYPE)
        family_one_hot[self.family_to_index[trajectory.family]] = 1.0

        step_one_hot = np.zeros(MAX_TRACE_STEPS, dtype=FLOAT_DTYPE)
        step_one_hot[min(step_index, MAX_TRACE_STEPS - 1)] = 1.0

        previous_action = np.zeros(len(self.action_tokens), dtype=FLOAT_DTYPE)
        previous_action[previous_action_index] = 1.0

        workspace_scalars = np.asarray(encoded.scalar_features, dtype=FLOAT_DTYPE)
        return np.concatenate([hashed, family_one_hot, step_one_hot, previous_action, workspace_scalars]).astype(
            FLOAT_DTYPE,
            copy=False,
        )


class ArcStructuredStateEncoder:
    """Object-scene encoder that consumes workspace_state plus verifier labels.

    This mode is intentionally oracle-style: it reads the structured workspace and
    verifier labels already attached to the trajectory record. The benchmark head
    remains unchanged so this can be used as a representation ablation against the
    default flat-grid encoder.
    """

    def __init__(
        self,
        *,
        projection_dim: int,
        seed: int,
        action_tokens: tuple[str, ...],
        verifier_provider: StructuredVerifierProvider | None = None,
    ):
        self.workspace_encoder = ArcStructuredWorkspaceEncoder(
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
        )
        self.action_tokens = action_tokens
        self.start_action_index = self.workspace_encoder.start_action_index
        self.hash_prefix = f"seed={seed}|"
        self.hash_dim = projection_dim
        self.scalar_dim = 17
        self.output_dim = self.hash_dim + len(FAMILIES) + MAX_TRACE_STEPS + len(action_tokens) + self.scalar_dim
        self.verifier_provider = verifier_provider

    def encode(self, trajectory: TrajectoryRecord, step_index: int, previous_action_index: int) -> np.ndarray:
        workspace_state = decision_workspace_state(trajectory, step_index)
        verifier_state = (
            self.verifier_provider(
                trajectory,
                step_index,
                workspace_state,
                previous_action_index,
            )
            if self.verifier_provider is not None
            else verifier_targets(trajectory, step_index)
        )
        encoded = encode_workspace(
            trajectory,
            step_index,
            include_verifier=True,
            verifier_state=verifier_state,
        )
        hashed = hashed_bag([self.hash_prefix + token for token in encoded.tokens], self.hash_dim)

        family_one_hot = np.zeros(len(FAMILIES), dtype=FLOAT_DTYPE)
        family_one_hot[self.workspace_encoder.family_to_index[trajectory.family]] = 1.0

        step_one_hot = np.zeros(MAX_TRACE_STEPS, dtype=FLOAT_DTYPE)
        step_one_hot[min(step_index, MAX_TRACE_STEPS - 1)] = 1.0

        previous_action = np.zeros(len(self.action_tokens), dtype=FLOAT_DTYPE)
        previous_action[previous_action_index] = 1.0

        scalar_features = np.asarray(encoded.scalar_features, dtype=FLOAT_DTYPE)
        return np.concatenate([hashed, family_one_hot, step_one_hot, previous_action, scalar_features]).astype(
            FLOAT_DTYPE,
            copy=False,
        )


def build_feature_encoder(
    *,
    encoder_name: FeatureEncoderName,
    projection_dim: int,
    seed: int,
    action_tokens: tuple[str, ...],
) -> ArcFeatureEncoder:
    if encoder_name == "flat_grid":
        return ArcTraceFeatureEncoder(
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
        )
    if encoder_name == "structured_oracle":
        return ArcStructuredStateEncoder(
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
        )
    if encoder_name == "structured_workspace":
        return ArcStructuredWorkspaceEncoder(
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
        )
    raise ValueError(f"Unsupported encoder: {encoder_name}")


def build_q_encoder(
    *,
    state_encoder: PolicyEncoderName,
    projection_dim: int,
    seed: int,
    action_tokens: tuple[str, ...],
    train_episodes: list[SamplerEpisode],
    action_to_index: dict[str, int],
    verifier_config: VerifierFitConfig,
) -> ArcFeatureEncoder:
    if state_encoder == "structured_reconstructed":
        workspace_encoder = build_feature_encoder(
            encoder_name="structured_workspace",
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
        )
        if not isinstance(workspace_encoder, ArcStructuredWorkspaceEncoder):
            raise TypeError("structured_workspace must build an ArcStructuredWorkspaceEncoder")
        verifier_features, verifier_target_array = build_verifier_training_data(
            train_episodes,
            workspace_encoder=workspace_encoder,
            action_to_index=action_to_index,
            action_labeler=canonical_action_label,
        )
        verifier_adapter = build_verifier_adapter(
            workspace_encoder=workspace_encoder,
            features=verifier_features,
            targets=verifier_target_array,
            config=VerifierFitConfig(
                model_name=verifier_config.model_name,
                ridge_lambda=verifier_config.ridge_lambda,
                random_feature_dim=verifier_config.random_feature_dim,
                seed=verifier_config.seed,
            ),
        )
        return ArcStructuredStateEncoder(
            projection_dim=projection_dim,
            seed=seed,
            action_tokens=action_tokens,
            verifier_provider=verifier_adapter,
        )
    return build_feature_encoder(
        encoder_name=state_encoder,
        projection_dim=projection_dim,
        seed=seed,
        action_tokens=action_tokens,
    )


class SigmoidLinearQ:
    def __init__(self, feature_dim: int, num_actions: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(loc=0.0, scale=0.02, size=(num_actions, feature_dim)).astype(
            FLOAT_DTYPE,
            copy=False,
        )
        self.num_actions = num_actions

    def copy(self) -> "SigmoidLinearQ":
        clone = SigmoidLinearQ(self.weights.shape[1], self.num_actions, seed=0)
        clone.weights = self.weights.copy()
        return clone

    def predict(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(features @ self.weights.T)

    def greedy_action(self, features: np.ndarray, *, allowed_mask: np.ndarray | None = None) -> int:
        values = self.predict(features[None, :])[0]
        if allowed_mask is None:
            return int(np.argmax(values))
        masked_values = values.copy()
        masked_values[~allowed_mask] = -np.inf
        if np.all(~allowed_mask):
            return int(np.argmax(values))
        return int(np.argmax(masked_values))

    def fit(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        *,
        loss_name: str,
        steps: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        num_samples = features.shape[0]
        learning_rate = FLOAT_DTYPE(learning_rate)
        weight_decay = FLOAT_DTYPE(weight_decay)
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1 = FLOAT_DTYPE(0.9)
        beta2 = FLOAT_DTYPE(0.999)
        eps = FLOAT_DTYPE(1e-8)
        scale = FLOAT_DTYPE(1.0 / max(num_samples, 1))
        action_indexer = actions.astype(np.intp, copy=False)
        sort_order = np.argsort(action_indexer, kind="stable")
        sorted_actions = action_indexer[sort_order]
        group_starts = np.concatenate(
            (
                np.asarray([0], dtype=np.intp),
                (np.flatnonzero(np.diff(sorted_actions)) + 1).astype(np.intp, copy=False),
            )
        )
        grouped_actions = sorted_actions[group_starts]
        sorted_features = features[sort_order]

        for step in range(1, steps + 1):
            chosen_logits = np.einsum("nd,nd->n", features, self.weights[action_indexer], optimize=True)
            chosen = np.clip(
                sigmoid(chosen_logits),
                SIGMOID_EPS,
                FLOAT_DTYPE(1.0) - SIGMOID_EPS,
            ).astype(FLOAT_DTYPE, copy=False)
            if loss_name == "log":
                grad_scalar = chosen - targets
            elif loss_name == "sq":
                grad_scalar = 2.0 * (chosen - targets) * chosen * (1.0 - chosen)
            else:
                raise ValueError(loss_name)
            grad_scalar = np.clip(grad_scalar, -GRAD_CLIP, GRAD_CLIP, out=grad_scalar)

            sample_gradient = grad_scalar[sort_order, None] * sorted_features
            gradient = np.zeros_like(self.weights)
            gradient[grouped_actions] = np.add.reduceat(sample_gradient, group_starts, axis=0)
            gradient *= scale
            gradient += weight_decay * self.weights

            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def grid_to_array(grid: tuple[tuple[int, ...], ...] | None) -> np.ndarray:
    array = np.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=FLOAT_DTYPE)
    if grid is None:
        return array
    for row, values in enumerate(grid[:MAX_GRID_HEIGHT]):
        width = min(len(values), MAX_GRID_WIDTH)
        if width:
            array[row, :width] = np.asarray(values[:width], dtype=FLOAT_DTYPE)
    return array


def color_histogram(grid: np.ndarray) -> np.ndarray:
    counts = np.bincount(grid.astype(np.int64).reshape(-1), minlength=10).astype(FLOAT_DTYPE)
    return counts / max(1.0, counts.sum())


def canonical_action_label(step: TrajectoryStep) -> str:
    action = step.action
    if step.name in {"transform", "apply", "relate", "act"}:
        return f"{step.name}:{action.get('action', '?')}"
    if step.name == "read_cue":
        return f"read_cue:{action.get('cue_kind', '?')}:{action.get('cue_value', '?')}"
    if step.name == "branch":
        return f"branch:{action.get('branch', '?')}"
    if step.name == "group":
        return f"group:{action.get('group_by', '?')}"
    if step.name == "reduce":
        return f"reduce:{action.get('reducer', '?')}"
    return step.name


def parse_action_token(token: str) -> tuple[str, tuple[str, ...]]:
    parts = token.split(":")
    return parts[0], tuple(parts[1:])


def choice_from_token(token: str, *, expected_step: str, fallback: str) -> str:
    step_name, details = parse_action_token(token)
    if step_name != expected_step or not details:
        return fallback
    return details[0]


def step_lookup(trajectory: TrajectoryRecord) -> dict[str, TrajectoryStep]:
    return {step.name: step for step in trajectory.steps}


def default_unary_params(
    action_name: str,
    *,
    truth_params: dict[str, object],
    role_bindings: dict[str, object],
) -> dict[str, object]:
    if action_name == "translate":
        return {
            "direction": str(truth_params.get("direction", "right")),
            "distance": int(truth_params.get("distance", role_bindings.get("N_repeat", 1))),
        }
    if action_name in {"scale_up", "scale_down"}:
        return {"factor": int(truth_params.get("factor", 2))}
    if action_name == "recolor":
        fallback_color = role_bindings.get("C_dst", role_bindings.get("C_src", 1))
        return {"to_color": int(truth_params.get("to_color", fallback_color))}
    return {}


def default_relational_params(action_name: str, *, truth_params: dict[str, object]) -> dict[str, object]:
    if action_name == "move_until_contact":
        return {"direction": str(truth_params.get("direction", "right"))}
    if action_name == "align_to":
        return {"axis": str(truth_params.get("axis", "x"))}
    if action_name == "stack_next_to":
        return {"side": str(truth_params.get("side", "right"))}
    if action_name == "draw_connecting_line":
        return {"line_type": str(truth_params.get("line_type", "orthogonal"))}
    if action_name == "place_in_container":
        return {"fit_mode": str(truth_params.get("fit_mode", "center"))}
    return {}


def default_count_action_params(
    action_name: str,
    *,
    truth_params: dict[str, object],
    role_bindings: dict[str, object],
) -> dict[str, object]:
    if action_name == "recolor_selected":
        fallback_color = role_bindings.get("C_dst", 1)
        return {"to_color": int(truth_params.get("to_color", fallback_color))}
    if action_name == "sort_into_row":
        return {"order": str(truth_params.get("order", "ascending"))}
    if action_name == "repeat_selected":
        return {"n": int(truth_params.get("n", role_bindings.get("N_repeat", 2)))}
    return {}


def invalid_prediction_grid() -> tuple[tuple[int, ...], ...]:
    return ((-1,),)


def predicted_final_grid(
    trajectory: TrajectoryRecord,
    *,
    predicted_tokens: tuple[str, ...],
) -> tuple[tuple[int, ...], ...] | None:
    if trajectory.output_grid is None:
        return None
    tokens_by_step = {
        step.name: predicted_tokens[index]
        for index, step in enumerate(trajectory.steps)
    }
    steps_by_name = step_lookup(trajectory)
    example = trajectory.example
    input_scene = example.input_scene

    if trajectory.family == Family.UNARY_OBJECT.value:
        transform_step = steps_by_name["transform"]
        truth_action = str(transform_step.action.get("action", "noop"))
        action_name = choice_from_token(
            tokens_by_step.get("transform", canonical_action_label(transform_step)),
            expected_step="transform",
            fallback=truth_action,
        )
        params = default_unary_params(
            action_name,
            truth_params=dict(transform_step.action.get("params", {})),
            role_bindings=trajectory.role_bindings,
        )
        chosen = selected_objects(input_scene, example.selected_object_ids)
        transformed = tuple(transformed_unary_object(obj, action_name, params) for obj in chosen)
        if action_name == "crop_to_bbox":
            objects = normalized_objects(transformed)
            min_height = 1
            min_width = 1
        else:
            objects = transformed
            min_height = input_scene.height
            min_width = input_scene.width
        return render_scene(
            scene_from_objects(
                objects,
                background_color=0,
                min_height=min_height,
                min_width=min_width,
                attributes={"family": trajectory.family, "benchmark_action": action_name},
            )
        )

    if trajectory.family == Family.RELATIONAL.value:
        relate_step = steps_by_name["relate"]
        truth_action = str(relate_step.action.get("action", "move_until_contact"))
        action_name = choice_from_token(
            tokens_by_step.get("relate", canonical_action_label(relate_step)),
            expected_step="relate",
            fallback=truth_action,
        )
        params = default_relational_params(action_name, truth_params=dict(relate_step.action.get("params", {})))
        source_ids = tuple(example.metadata.get("source_ids", ()))
        target_ids = tuple(example.metadata.get("target_ids", ()))
        sources = list(selected_objects(input_scene, source_ids))
        targets = list(selected_objects(input_scene, target_ids))
        if not sources or not targets:
            chosen = selected_objects(input_scene, example.selected_object_ids)
            midpoint = max(1, len(chosen) // 2)
            sources = list(chosen[:midpoint])
            targets = list(chosen[midpoint:] or chosen[-1:])

        moved_sources = []
        extra_objects = []
        for source in sources:
            anchor = targets[0]
            if action_name == "move_until_contact":
                moved_sources.append(move_until_contact(source, anchor, str(params["direction"])))
            elif action_name == "align_to":
                moved_sources.append(align_object_to(source, anchor, str(params["axis"])))
            elif action_name == "stack_next_to":
                moved_sources.append(stack_next_to(source, anchor, str(params["side"])))
            elif action_name == "draw_connecting_line":
                moved_sources.append(source)
                extra_objects.append(connector_object(source, anchor))
            else:
                moved_sources.append(place_in_container(source, anchor))

        return render_scene(
            scene_from_objects(
                [*moved_sources, *targets, *extra_objects],
                background_color=0,
                min_height=input_scene.height,
                min_width=input_scene.width,
                attributes={"family": trajectory.family, "benchmark_action": action_name},
            )
        )

    if trajectory.family == Family.COUNT_SELECT.value:
        group_step = steps_by_name["group"]
        reduce_step = steps_by_name["reduce"]
        act_step = steps_by_name["act"]
        group_by = choice_from_token(
            tokens_by_step.get("group", canonical_action_label(group_step)),
            expected_step="group",
            fallback=str(group_step.action.get("group_by", example.metadata.get("group_by", "shape"))),
        )
        reducer_name = choice_from_token(
            tokens_by_step.get("reduce", canonical_action_label(reduce_step)),
            expected_step="reduce",
            fallback=str(reduce_step.action.get("reducer", example.metadata.get("reducer", "argmax_count"))),
        )
        action_name = choice_from_token(
            tokens_by_step.get("act", canonical_action_label(act_step)),
            expected_step="act",
            fallback=str(act_step.action.get("action", example.metadata.get("post_action", "copy_selected"))),
        )
        group_summary: dict[str, dict[str, int]] = {}
        for obj in input_scene.objects:
            key_str = str(count_group_key(group_by, obj))
            group_summary.setdefault(key_str, {"count": 0, "max_mass": 0, "min_mass": 10**9})
            group_summary[key_str]["count"] += 1
            group_summary[key_str]["max_mass"] = max(group_summary[key_str]["max_mass"], obj.mass)
            group_summary[key_str]["min_mass"] = min(group_summary[key_str]["min_mass"], obj.mass)
        if not group_summary:
            winner_ids: tuple[str, ...] = ()
        elif reducer_name == "argmax_count":
            winner_key = max(group_summary.items(), key=lambda item: item[1]["count"])[0]
            winner_ids = tuple(obj.object_id for obj in input_scene.objects if str(count_group_key(group_by, obj)) == str(winner_key))
        elif reducer_name == "argmin_count":
            winner_key = min(group_summary.items(), key=lambda item: item[1]["count"])[0]
            winner_ids = tuple(obj.object_id for obj in input_scene.objects if str(count_group_key(group_by, obj)) == str(winner_key))
        elif reducer_name == "argmax_size":
            winner_key = max(group_summary.items(), key=lambda item: item[1]["max_mass"])[0]
            winner_ids = tuple(obj.object_id for obj in input_scene.objects if str(count_group_key(group_by, obj)) == str(winner_key))
        else:
            winner_key = min(group_summary.items(), key=lambda item: item[1]["min_mass"])[0]
            winner_ids = tuple(obj.object_id for obj in input_scene.objects if str(count_group_key(group_by, obj)) == str(winner_key))

        winners = normalized_objects(selected_objects(input_scene, winner_ids))
        params = default_count_action_params(
            action_name,
            truth_params=dict(act_step.action.get("params", {})),
            role_bindings=trajectory.role_bindings,
        )
        if action_name == "copy_selected":
            placed = winners
        elif action_name == "recolor_selected":
            placed = tuple(replace_object(obj, color=int(params["to_color"])) for obj in winners)
        elif action_name == "sort_into_row":
            placed = arrange_row(
                winners,
                gap=1,
                descending=str(params.get("order", "ascending")) == "descending",
            )
        else:
            placed = arrange_row(winners, gap=1, repeat=max(1, int(params.get("n", 2))))

        return render_scene(
            scene_from_objects(
                placed,
                background_color=0,
                min_height=1,
                min_width=1,
                attributes={"family": trajectory.family, "benchmark_action": action_name},
            )
        )

    if trajectory.family == Family.CONTEXTUAL.value:
        branch_step = steps_by_name["branch"]
        apply_step = steps_by_name["apply"]
        predicted_branch = choice_from_token(
            tokens_by_step.get("branch", canonical_action_label(branch_step)),
            expected_step="branch",
            fallback=str(branch_step.action.get("branch", example.metadata.get("branch", "then"))),
        )
        action_name = choice_from_token(
            tokens_by_step.get("apply", canonical_action_label(apply_step)),
            expected_step="apply",
            fallback=str(apply_step.action.get("action", "stack_left")),
        )
        chosen = normalized_objects(selected_objects(input_scene, example.selected_object_ids))
        if action_name == "stack_left":
            placed = stack_side(chosen, side="left", canvas_height=input_scene.height, canvas_width=input_scene.width)
        elif action_name == "stack_right":
            placed = stack_side(chosen, side="right", canvas_height=input_scene.height, canvas_width=input_scene.width)
        elif action_name == "reflect_h":
            placed = reflect_objects(chosen, axis="h", canvas_height=input_scene.height, canvas_width=input_scene.width)
        elif action_name == "reflect_v":
            placed = reflect_objects(chosen, axis="v", canvas_height=input_scene.height, canvas_width=input_scene.width)
        else:
            if predicted_branch == "then":
                to_color = int(trajectory.role_bindings.get("C_dst", apply_step.action.get("params", {}).get("to_color", 1)))
            else:
                to_color = int(trajectory.role_bindings.get("C_src", apply_step.action.get("params", {}).get("to_color", 1)))
            placed = tuple(replace_object(obj, color=to_color) for obj in chosen)

        return render_scene(
            scene_from_objects(
                placed,
                background_color=0,
                min_height=input_scene.height,
                min_width=input_scene.width,
                attributes={
                    "family": trajectory.family,
                    "benchmark_branch": predicted_branch,
                    "benchmark_action": action_name,
                },
            )
        )

    apply_step = steps_by_name["apply"]
    action_name = choice_from_token(
        tokens_by_step.get("apply", canonical_action_label(apply_step)),
        expected_step="apply",
        fallback=str(apply_step.action.get("action", "recolor_targets")),
    )
    expected_apply_mode = str(example.metadata.get("apply_mode", action_name))
    if action_name != expected_apply_mode:
        return invalid_prediction_grid()
    mapping = dict(example.metadata.get("mapping", {}))
    targets = normalized_objects(selected_objects(input_scene, example.selected_object_ids))
    if action_name == "recolor_targets":
        if not all(isinstance(mapping.get(obj.attributes.get("target_key")), int) for obj in targets):
            return invalid_prediction_grid()
        placed = tuple(
            replace_object(obj, color=int(mapping.get(obj.attributes.get("target_key"), obj.color)))
            for obj in targets
        )
    elif action_name == "place_targets":
        placed = arrange_row(targets, gap=1)
    else:
        if not all(
            isinstance(mapping.get(obj.attributes.get("target_key")), (int, str))
            for obj in targets
        ):
            return invalid_prediction_grid()
        placed = tuple(
            replace_object(obj, orientation=orientation_from_value(mapping.get(obj.attributes.get("target_key"))))
            for obj in targets
        )
    return render_scene(
        scene_from_objects(
            placed,
            background_color=0,
            min_height=1,
            min_width=1,
            attributes={"family": trajectory.family, "benchmark_action": action_name},
        )
    )


def current_workspace_grid(trajectory: TrajectoryRecord, step_index: int) -> tuple[tuple[int, ...], ...]:
    if step_index <= 0:
        assert trajectory.input_grid is not None
        return trajectory.input_grid
    return trajectory.steps[step_index - 1].workspace_grid or trajectory.output_grid or trajectory.input_grid


def generate_sampler_episodes(
    *,
    count: int,
    seed_start: int,
    include_train_trajectories: bool = True,
    timing_rows: list[dict[str, object]] | None = None,
    timing_split: str = "train",
    trial_index: int | None = None,
    progress: bool = False,
    progress_desc: str | None = None,
) -> list[SamplerEpisode]:
    episodes = []
    offsets = range(count)
    if progress:
        offsets = tqdm(
            offsets,
            total=count,
            desc=progress_desc or "sample episodes",
            leave=False,
            dynamic_ncols=True,
        )
    for offset in offsets:
        seed = seed_start + offset
        total_start = perf_counter()
        sample_start = total_start
        latent = sample_latent_rule(seed=seed)
        episode = sample_episode(latent, seed=seed)
        sample_s = perf_counter() - sample_start

        execute_start = perf_counter()
        executed = execute_episode(episode)
        execute_s = perf_counter() - execute_start

        compile_train_s = 0.0
        train_trajectories: tuple[TrajectoryRecord, ...] = ()
        if include_train_trajectories:
            compile_train_start = perf_counter()
            train_trajectories = compile_episode_trajectories(executed, include_test=False)
            compile_train_s = perf_counter() - compile_train_start

        compile_test_start = perf_counter()
        test_trajectory = compile_trajectory(
            episode,
            executed.test_example,
            split="test",
            trajectory_index=len(train_trajectories),
            variant_kind="canonical",
        )
        compile_test_s = perf_counter() - compile_test_start
        total_s = perf_counter() - total_start

        if timing_rows is not None:
            rejection_counts = dict(episode.rejection_counts)
            top_rejections = sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            timing_rows.append(
                {
                    "trial": trial_index if trial_index is not None else "",
                    "split": timing_split,
                    "seed": seed,
                    "family": episode.latent.family.value,
                    "difficulty": episode.latent.difficulty,
                    "sampling_attempts": episode.sampling_attempts,
                    "rejection_total": int(sum(rejection_counts.values())),
                    "top_rejection_reasons_json": json.dumps(top_rejections),
                    "train_trajectory_count": len(train_trajectories),
                    "test_trace_steps": len(test_trajectory.steps),
                    "sample_s": round(sample_s, 6),
                    "execute_s": round(execute_s, 6),
                    "compile_train_s": round(compile_train_s, 6),
                    "compile_test_s": round(compile_test_s, 6),
                    "total_s": round(total_s, 6),
                }
            )

        episodes.append(
            SamplerEpisode(
                seed=seed,
                train_trajectories=train_trajectories,
                test_trajectory=test_trajectory,
            )
        )
    return episodes


def summarize_episode_timings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    grouped: defaultdict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["family"]), int(row["difficulty"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (split, family, difficulty), group in sorted(grouped.items()):
        total_s = np.asarray([float(row["total_s"]) for row in group], dtype=np.float64)
        sample_s = np.asarray([float(row["sample_s"]) for row in group], dtype=np.float64)
        execute_s = np.asarray([float(row["execute_s"]) for row in group], dtype=np.float64)
        compile_train_s = np.asarray([float(row["compile_train_s"]) for row in group], dtype=np.float64)
        compile_test_s = np.asarray([float(row["compile_test_s"]) for row in group], dtype=np.float64)
        attempts = np.asarray([float(row["sampling_attempts"]) for row in group], dtype=np.float64)
        train_counts = np.asarray([float(row["train_trajectory_count"]) for row in group], dtype=np.float64)
        test_steps = np.asarray([float(row["test_trace_steps"]) for row in group], dtype=np.float64)
        rejection_totals = np.asarray([float(row["rejection_total"]) for row in group], dtype=np.float64)
        summary_rows.append(
            {
                "split": split,
                "family": family,
                "difficulty": difficulty,
                "count": len(group),
                "mean_total_s": round(float(np.mean(total_s)), 6),
                "p95_total_s": round(float(np.percentile(total_s, 95)), 6),
                "max_total_s": round(float(np.max(total_s)), 6),
                "mean_sample_s": round(float(np.mean(sample_s)), 6),
                "mean_execute_s": round(float(np.mean(execute_s)), 6),
                "mean_compile_train_s": round(float(np.mean(compile_train_s)), 6),
                "mean_compile_test_s": round(float(np.mean(compile_test_s)), 6),
                "mean_sampling_attempts": round(float(np.mean(attempts)), 4),
                "max_sampling_attempts": int(np.max(attempts)),
                "mean_train_trajectory_count": round(float(np.mean(train_counts)), 4),
                "mean_test_trace_steps": round(float(np.mean(test_steps)), 4),
                "mean_rejection_total": round(float(np.mean(rejection_totals)), 4),
                "max_rejection_total": int(np.max(rejection_totals)),
            }
        )
    return summary_rows


def build_data_only_summary_row(
    *,
    seed: int,
    dataset_sizes: list[int],
    train_episodes: list[SamplerEpisode],
    eval_episodes: list[SamplerEpisode],
    vocabulary: tuple[str, ...],
    mask_lookup: dict[tuple[str, int], np.ndarray],
) -> dict[str, object]:
    train_trajectories = [trajectory for episode in train_episodes for trajectory in episode.train_trajectories]
    eval_trajectories = [episode.test_trajectory for episode in eval_episodes]
    families = sorted({trajectory.family for trajectory in train_trajectories + eval_trajectories})
    train_trace_steps = sum(len(trajectory.steps) for trajectory in train_trajectories)
    eval_trace_steps = sum(len(trajectory.steps) for trajectory in eval_trajectories)
    return {
        "seed": seed,
        "dataset_sizes_json": json.dumps(dataset_sizes),
        "train_episode_count": len(train_episodes),
        "eval_episode_count": len(eval_episodes),
        "train_trajectory_count": len(train_trajectories),
        "train_trace_step_count": train_trace_steps,
        "eval_trace_step_count": eval_trace_steps,
        "mean_train_trajectories_per_episode": float(len(train_trajectories) / max(len(train_episodes), 1)),
        "mean_train_steps_per_trajectory": float(train_trace_steps / max(len(train_trajectories), 1)),
        "mean_eval_steps_per_episode": float(eval_trace_steps / max(len(eval_trajectories), 1)),
        "action_vocab_size": len(vocabulary),
        "mask_bucket_count": len(mask_lookup),
        "families_json": json.dumps(families),
    }


def build_data_only_payload(
    *,
    args: argparse.Namespace,
    data_only_rows: list[dict[str, object]],
    timing_summary_rows: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": "data_only",
        "dataset_sizes": args.dataset_sizes,
        "seeds": args.seeds,
        "eval_episodes": args.eval_episodes,
        "data_only_summary_rows": data_only_rows,
        "episode_timing_summary_rows": timing_summary_rows,
    }
    if hasattr(args, "state_encoder"):
        payload["state_encoder"] = args.state_encoder
    if hasattr(args, "verifier_model"):
        payload["verifier_model"] = args.verifier_model
    return payload


def action_vocabulary(episodes: list[SamplerEpisode]) -> tuple[str, ...]:
    labels = {START_ACTION_TOKEN}
    for episode in episodes:
        for trajectory in (*episode.train_trajectories, episode.test_trajectory):
            for step in trajectory.steps:
                labels.add(canonical_action_label(step))
    return tuple(sorted(labels))


def action_masks(
    episodes: list[SamplerEpisode],
    *,
    action_to_index: dict[str, int],
) -> dict[tuple[str, int], np.ndarray]:
    masks: dict[tuple[str, int], np.ndarray] = {}
    for episode in episodes:
        for trajectory in (*episode.train_trajectories, episode.test_trajectory):
            for step_index, step in enumerate(trajectory.steps):
                key = (trajectory.family, step_index)
                if key not in masks:
                    masks[key] = np.zeros(len(action_to_index), dtype=bool)
                masks[key][action_to_index[canonical_action_label(step)]] = True
    return masks


def build_training_prefix_cache(
    episodes: list[SamplerEpisode],
    *,
    encoder: ArcFeatureEncoder,
    action_to_index: dict[str, int],
    mask_lookup: dict[tuple[str, int], np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = []
    actions = []
    rewards = []
    next_features = []
    next_masks = []
    dones = []
    episode_transition_ends = []
    transition_count = 0

    for episode in episodes:
        for trajectory in episode.train_trajectories:
            previous_action_index = encoder.start_action_index
            for step_index, step in enumerate(trajectory.steps):
                action_index = action_to_index[canonical_action_label(step)]
                features.append(encoder.encode(trajectory, step_index, previous_action_index))
                actions.append(action_index)
                rewards.append(step.reward / max(trajectory.total_possible_reward, 1e-8))
                dones.append(float(step.done))
                if step.done:
                    next_features.append(np.zeros(encoder.output_dim, dtype=FLOAT_DTYPE))
                    next_masks.append(np.zeros(len(action_to_index), dtype=bool))
                else:
                    next_features.append(encoder.encode(trajectory, step_index + 1, action_index))
                    next_masks.append(mask_lookup[(trajectory.family, step_index + 1)])
                previous_action_index = action_index
                transition_count += 1
        episode_transition_ends.append(transition_count)

    return {
        "features": np.asarray(features, dtype=FLOAT_DTYPE),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=FLOAT_DTYPE),
        "next_features": np.asarray(next_features, dtype=FLOAT_DTYPE),
        "next_masks": np.asarray(next_masks, dtype=bool),
        "dones": np.asarray(dones, dtype=FLOAT_DTYPE),
    }, np.asarray(episode_transition_ends, dtype=np.int64)


def slice_training_batch(batch: dict[str, np.ndarray], transition_count: int) -> dict[str, np.ndarray]:
    return {name: values[:transition_count] for name, values in batch.items()}


def build_training_batch(
    episodes: list[SamplerEpisode],
    *,
    encoder: ArcFeatureEncoder,
    action_to_index: dict[str, int],
    mask_lookup: dict[tuple[str, int], np.ndarray],
) -> dict[str, np.ndarray]:
    batch, episode_transition_ends = build_training_prefix_cache(
        episodes,
        encoder=encoder,
        action_to_index=action_to_index,
        mask_lookup=mask_lookup,
    )
    transition_count = int(episode_transition_ends[-1]) if len(episode_transition_ends) else 0
    return slice_training_batch(batch, transition_count)


def masked_max(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    masked = values.copy()
    masked[~masks] = -np.inf
    next_values = np.max(masked, axis=1)
    next_values[~np.isfinite(next_values)] = 0.0
    return next_values


def train_fqi(
    batch: dict[str, np.ndarray],
    *,
    num_actions: int,
    train_config: TrainConfig,
    loss_name: str,
    seed: int,
    progress: bool = False,
    progress_desc: str | None = None,
) -> SigmoidLinearQ:
    model = SigmoidLinearQ(batch["features"].shape[1], num_actions=num_actions, seed=seed)
    gamma = FLOAT_DTYPE(train_config.gamma)
    iterations = range(train_config.fqi_iterations)
    if progress:
        iterations = tqdm(
            iterations,
            total=train_config.fqi_iterations,
            desc=progress_desc or f"train {loss_name}",
            leave=False,
            dynamic_ncols=True,
        )
    for _ in iterations:
        next_values = masked_max(model.predict(batch["next_features"]), batch["next_masks"])
        targets = batch["rewards"] + gamma * (FLOAT_DTYPE(1.0) - batch["dones"]) * next_values
        targets = np.clip(targets, FLOAT_DTYPE(0.0), FLOAT_DTYPE(1.0)).astype(FLOAT_DTYPE, copy=False)
        updated = model.copy()
        updated.fit(
            batch["features"],
            batch["actions"],
            targets,
            loss_name=loss_name,
            steps=train_config.optimizer_steps,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        model = updated
    return model


def evaluate_policy(
    trajectories: list[TrajectoryRecord],
    *,
    model: SigmoidLinearQ,
    encoder: ArcFeatureEncoder,
    action_to_index: dict[str, int],
    mask_lookup: dict[tuple[str, int], np.ndarray],
) -> dict[str, float]:
    index_to_action = {index: token for token, index in action_to_index.items()}
    closed_loop_returns = []
    closed_loop_prefix = []
    exact_successes = []
    exact_answer_successes = []
    final_grid_successes = []
    matched_steps = []
    open_loop_correct = 0
    open_loop_total = 0

    family_returns: defaultdict[str, list[float]] = defaultdict(list)
    family_successes: defaultdict[str, list[float]] = defaultdict(list)
    family_answer_successes: defaultdict[str, list[float]] = defaultdict(list)
    family_grid_successes: defaultdict[str, list[float]] = defaultdict(list)

    for trajectory in trajectories:
        previous_action_index = encoder.start_action_index
        normalized_return = 0.0
        matched = 0
        failed = False
        answer_correct = True
        predicted_tokens = []
        answer_steps = ANSWER_STEPS_BY_FAMILY.get(trajectory.family, frozenset())
        for step_index, step in enumerate(trajectory.steps):
            features = encoder.encode(trajectory, step_index, previous_action_index)
            predicted = model.greedy_action(features, allowed_mask=mask_lookup[(trajectory.family, step_index)])
            predicted_tokens.append(index_to_action[predicted])
            target = action_to_index[canonical_action_label(step)]
            if step.name in answer_steps and predicted != target:
                answer_correct = False

            expert_previous_action = encoder.start_action_index if step_index == 0 else action_to_index[
                canonical_action_label(trajectory.steps[step_index - 1])
            ]
            expert_features = encoder.encode(trajectory, step_index, expert_previous_action)
            expert_prediction = model.greedy_action(
                expert_features,
                allowed_mask=mask_lookup[(trajectory.family, step_index)],
            )
            open_loop_correct += int(expert_prediction == target)
            open_loop_total += 1

            if not failed:
                if predicted != target:
                    failed = True
                else:
                    normalized_return += step.reward / max(trajectory.total_possible_reward, 1e-8)
                    matched += 1
            previous_action_index = predicted

        prefix = matched / max(len(trajectory.steps), 1)
        exact = 0.0 if failed else 1.0
        exact_answer = 1.0 if answer_correct else 0.0
        predicted_grid = predicted_final_grid(
            trajectory,
            predicted_tokens=tuple(predicted_tokens),
        )
        final_grid_exact = float(predicted_grid == trajectory.output_grid) if predicted_grid is not None else float("nan")
        closed_loop_returns.append(normalized_return)
        closed_loop_prefix.append(prefix)
        exact_successes.append(exact)
        exact_answer_successes.append(exact_answer)
        final_grid_successes.append(final_grid_exact)
        matched_steps.append(float(matched))
        family_returns[trajectory.family].append(normalized_return)
        family_successes[trajectory.family].append(exact)
        family_answer_successes[trajectory.family].append(exact_answer)
        family_grid_successes[trajectory.family].append(final_grid_exact)

    metrics: dict[str, float] = {
        "mean_normalized_return": float(np.mean(closed_loop_returns)),
        "mean_prefix_accuracy": float(np.mean(closed_loop_prefix)),
        "exact_trace_success": float(np.mean(exact_successes)),
        "exact_answer_success": float(np.mean(exact_answer_successes)),
        "final_grid_exact_match": float(np.nanmean(np.asarray(final_grid_successes, dtype=np.float64))),
        "mean_matched_steps": float(np.mean(matched_steps)),
        "open_loop_action_accuracy": float(open_loop_correct / max(open_loop_total, 1)),
    }
    for family in sorted(family_returns):
        metrics[f"return_{family}"] = float(np.mean(family_returns[family]))
        metrics[f"success_{family}"] = float(np.mean(family_successes[family]))
        metrics[f"answer_success_{family}"] = float(np.mean(family_answer_successes[family]))
        metrics[f"grid_success_{family}"] = float(np.nanmean(np.asarray(family_grid_successes[family], dtype=np.float64)))
    for family in FAMILIES:
        metrics.setdefault(f"return_{family}", float("nan"))
        metrics.setdefault(f"success_{family}", float("nan"))
        metrics.setdefault(f"answer_success_{family}", float("nan"))
        metrics.setdefault(f"grid_success_{family}", float("nan"))
    return metrics


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[int, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["dataset_episodes"]),
                str(row["loss"]),
                str(row.get("state_encoder", "flat_grid")),
                str(row.get("verifier_model", "n/a")),
            )
        ].append(row)

    metric_names = [
        "mean_normalized_return",
        "mean_prefix_accuracy",
        "exact_trace_success",
        "exact_answer_success",
        "final_grid_exact_match",
        "mean_matched_steps",
        "open_loop_action_accuracy",
    ]
    for family in FAMILIES:
        metric_names.extend([f"return_{family}", f"success_{family}", f"answer_success_{family}", f"grid_success_{family}"])

    summary_rows: list[dict[str, object]] = []
    for (dataset_episodes, loss_name, state_encoder, verifier_model), group in sorted(grouped.items()):
        summary_row: dict[str, object] = {
            "dataset_episodes": dataset_episodes,
            "loss": loss_name,
            "state_encoder": state_encoder,
            "verifier_model": verifier_model,
        }
        for metric_name in metric_names:
            values = np.asarray([float(item[metric_name]) for item in group], dtype=np.float64)
            valid_values = values[np.isfinite(values)]
            if len(valid_values) == 0:
                summary_row[metric_name] = float("nan")
                summary_row[f"stderr_{metric_name}"] = float("nan")
                continue
            summary_row[metric_name] = float(valid_values.mean())
            summary_row[f"stderr_{metric_name}"] = float(valid_values.std(ddof=0) / np.sqrt(len(valid_values)))
        summary_rows.append(summary_row)
    return summary_rows


def plot_summary(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        return
    dataset_sizes = sorted({int(row["dataset_episodes"]) for row in summary_rows})
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    metric_specs = [
        ("mean_normalized_return", "Normalized Return"),
        ("exact_trace_success", "Exact Trace Success"),
        ("exact_answer_success", "Exact Answer (Abstract)"),
        ("final_grid_exact_match", "Final Grid Exact"),
        ("open_loop_action_accuracy", "Open-Loop Accuracy"),
    ]
    colors = {"log": "#0f766e", "sq": "#b45309"}
    state_encoder = str(summary_rows[0].get("state_encoder", "flat_grid"))
    verifier_model = str(summary_rows[0].get("verifier_model", "n/a"))
    if state_encoder == "structured_oracle":
        labels = {"log": "FQI-LOG (structured-oracle)", "sq": "FQI-SQ (structured-oracle)"}
    elif state_encoder == "structured_reconstructed":
        labels = {
            "log": f"FQI-LOG (structured-reconstructed/{verifier_model})",
            "sq": f"FQI-SQ (structured-reconstructed/{verifier_model})",
        }
    else:
        labels = {"log": "FQI-LOG", "sq": "FQI-SQ"}

    for axis, (metric_name, title) in zip(axes, metric_specs, strict=True):
        for loss_name in ("log", "sq"):
            loss_rows = {int(row["dataset_episodes"]): row for row in summary_rows if row["loss"] == loss_name}
            y = [float(loss_rows[size][metric_name]) for size in dataset_sizes]
            yerr = [float(loss_rows[size][f"stderr_{metric_name}"]) for size in dataset_sizes]
            axis.errorbar(
                dataset_sizes,
                y,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                capsize=4,
                color=colors[loss_name],
                label=labels[loss_name],
            )
        axis.set_title(title)
        axis.set_xlabel("Training Episodes")
        axis.set_ylabel(title)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25, linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    train_config = TrainConfig(
        gamma=args.gamma,
        fqi_iterations=args.fqi_iterations,
        optimizer_steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
    )
    verifier_config = VerifierFitConfig(
        model_name=args.verifier_model,
        ridge_lambda=args.verifier_ridge_lambda,
        random_feature_dim=args.verifier_random_feature_dim,
        seed=args.seed + 913,
    )

    raw_rows: list[dict[str, object]] = []
    data_only_rows: list[dict[str, object]] = []
    episode_timing_rows: list[dict[str, object]] = []
    seed_iterator = range(args.seeds)
    if args.progress:
        seed_iterator = tqdm(seed_iterator, total=args.seeds, desc="seed sweeps", dynamic_ncols=True)

    for trial in seed_iterator:
        train_seed_start = args.seed + trial * 10_000
        eval_seed_start = train_seed_start + max(args.dataset_sizes) + 1_000
        max_train_episodes = max(args.dataset_sizes)

        train_episodes = generate_sampler_episodes(
            count=max_train_episodes,
            seed_start=train_seed_start,
            include_train_trajectories=True,
            timing_rows=episode_timing_rows,
            timing_split="train",
            trial_index=trial + 1,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} train episodes",
        )
        eval_episodes = generate_sampler_episodes(
            count=args.eval_episodes,
            seed_start=eval_seed_start,
            include_train_trajectories=False,
            timing_rows=episode_timing_rows,
            timing_split="eval",
            trial_index=trial + 1,
            progress=args.progress,
            progress_desc=f"seed {trial + 1} eval episodes",
        )

        vocabulary = action_vocabulary(train_episodes + eval_episodes)
        action_to_index = {token: idx for idx, token in enumerate(vocabulary)}
        mask_lookup = action_masks(train_episodes + eval_episodes, action_to_index=action_to_index)
        if args.data_only:
            data_only_rows.append(
                build_data_only_summary_row(
                    seed=train_seed_start,
                    dataset_sizes=args.dataset_sizes,
                    train_episodes=train_episodes,
                    eval_episodes=eval_episodes,
                    vocabulary=vocabulary,
                    mask_lookup=mask_lookup,
                )
            )
            continue
        base_encoder: ArcFeatureEncoder | None = None
        full_batch: dict[str, np.ndarray] | None = None
        episode_transition_ends: np.ndarray | None = None
        if args.state_encoder != "structured_reconstructed":
            base_encoder = build_q_encoder(
                state_encoder=args.state_encoder,
                projection_dim=args.projection_dim,
                seed=train_seed_start + 31,
                action_tokens=vocabulary,
                train_episodes=train_episodes,
                action_to_index=action_to_index,
                verifier_config=VerifierFitConfig(
                    model_name=verifier_config.model_name,
                    ridge_lambda=verifier_config.ridge_lambda,
                    random_feature_dim=verifier_config.random_feature_dim,
                    seed=train_seed_start + 913,
                ),
            )
            full_batch, episode_transition_ends = build_training_prefix_cache(
                train_episodes,
                encoder=base_encoder,
                action_to_index=action_to_index,
                mask_lookup=mask_lookup,
            )

        eval_trajectories = [episode.test_trajectory for episode in eval_episodes]
        fit_total = len(args.dataset_sizes) * 2
        fit_bar = None
        if args.progress:
            fit_bar = tqdm(
                total=fit_total,
                desc=f"seed {trial + 1} fits",
                leave=False,
                dynamic_ncols=True,
            )
        for dataset_episodes in args.dataset_sizes:
            subset_train_episodes = train_episodes[:dataset_episodes]
            if args.state_encoder == "structured_reconstructed":
                encoder = build_q_encoder(
                    state_encoder=args.state_encoder,
                    projection_dim=args.projection_dim,
                    seed=train_seed_start + 31,
                    action_tokens=vocabulary,
                    train_episodes=subset_train_episodes,
                    action_to_index=action_to_index,
                    verifier_config=VerifierFitConfig(
                        model_name=verifier_config.model_name,
                        ridge_lambda=verifier_config.ridge_lambda,
                        random_feature_dim=verifier_config.random_feature_dim,
                        seed=train_seed_start + dataset_episodes + 913,
                    ),
                )
                batch = build_training_batch(
                    subset_train_episodes,
                    encoder=encoder,
                    action_to_index=action_to_index,
                    mask_lookup=mask_lookup,
                )
            else:
                assert base_encoder is not None
                assert full_batch is not None
                assert episode_transition_ends is not None
                encoder = base_encoder
                batch = slice_training_batch(full_batch, int(episode_transition_ends[dataset_episodes - 1]))
            for loss_name in ("log", "sq"):
                model = train_fqi(
                    batch,
                    num_actions=len(vocabulary),
                    train_config=train_config,
                    loss_name=loss_name,
                    seed=train_seed_start + dataset_episodes + (0 if loss_name == "log" else 50_000),
                    progress=args.progress,
                    progress_desc=f"seed {trial + 1} {dataset_episodes} {loss_name}",
                )
                metrics = evaluate_policy(
                    eval_trajectories,
                    model=model,
                    encoder=encoder,
                    action_to_index=action_to_index,
                    mask_lookup=mask_lookup,
                )
                raw_rows.append(
                    {
                        "seed": train_seed_start,
                        "dataset_episodes": dataset_episodes,
                        "loss": loss_name,
                        "state_encoder": args.state_encoder,
                        "verifier_model": args.verifier_model if args.state_encoder == "structured_reconstructed" else "n/a",
                        "action_vocab_size": len(vocabulary),
                        **metrics,
                    }
                )
                if fit_bar is not None:
                    fit_bar.update(1)
        if fit_bar is not None:
            fit_bar.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_summary_rows = summarize_episode_timings(episode_timing_rows)
    save_csv(episode_timing_rows, output_dir / "episode_timings.csv")
    save_csv(timing_summary_rows, output_dir / "episode_timing_summary.csv")
    if args.data_only:
        save_csv(data_only_rows, output_dir / "data_only_summary.csv")
        payload = build_data_only_payload(
            args=args,
            data_only_rows=data_only_rows,
            timing_summary_rows=timing_summary_rows,
        )
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    summary_rows = aggregate(raw_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    plot_summary(summary_rows, output_dir / "comparison.png")
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "dataset_sizes": args.dataset_sizes,
                "seeds": args.seeds,
                "eval_episodes": args.eval_episodes,
                "state_encoder": args.state_encoder,
                "verifier_model": args.verifier_model,
                "train_config": train_config.__dict__,
                "summary_rows": summary_rows,
                "episode_timing_summary_rows": timing_summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows, "episode_timing_summary_rows": timing_summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FQI-LOG vs FQI-SQ on ARC sampler trajectories.")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fqi-iterations", type=int, default=6)
    parser.add_argument("--optimizer-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=192)
    parser.add_argument("--verifier-ridge-lambda", type=float, default=1e-3)
    parser.add_argument(
        "--verifier-model",
        choices=("linear", "random_features"),
        default="random_features",
        help="Model used to reconstruct verifier signals from workspace-only structured features.",
    )
    parser.add_argument(
        "--verifier-random-feature-dim",
        type=int,
        default=128,
        help="Hidden width for the random-feature verifier when --verifier-model=random_features.",
    )
    parser.add_argument(
        "--state-encoder",
        choices=("flat_grid", "structured_oracle", "structured_reconstructed"),
        default="flat_grid",
        help=(
            "Feature encoder for ARC states. "
            "'structured_oracle' reads workspace_state plus verifier labels from the trajectory records; "
            "'structured_reconstructed' fits a lightweight verifier over workspace-only structured features "
            "and feeds its predictions back into the same encoder surface."
        ),
    )
    parser.add_argument("--output-dir", default="results/arc_sampler_fqi")
    parser.add_argument("--data-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    if args.data_only:
        print(json.dumps(payload["data_only_summary_rows"], indent=2))
    else:
        print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
