from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

import numpy as np

from arc_trajectory_sampler.state_adapter import verifier_targets


FLOAT_DTYPE = np.float32
VerifierModelName = Literal["linear", "random_features"]


class WorkspaceEncoder(Protocol):
    start_action_index: int

    def encode(self, trajectory: Any, step_index: int, previous_action_index: int) -> np.ndarray:
        ...


ActionLabeler = Callable[[Any], str]


@dataclass(frozen=True)
class VerifierFitConfig:
    model_name: VerifierModelName = "random_features"
    ridge_lambda: float = 1e-3
    random_feature_dim: int = 128
    seed: int = 0


def _feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(FLOAT_DTYPE, copy=False)
    scale = features.std(axis=0, dtype=np.float64).astype(FLOAT_DTYPE, copy=False)
    scale = np.where(scale < FLOAT_DTYPE(1e-5), FLOAT_DTYPE(1.0), scale)
    return mean, scale


def _normalize_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features.astype(FLOAT_DTYPE, copy=False) - mean) / scale).astype(FLOAT_DTYPE, copy=False)


def _solve_ridge(design: np.ndarray, targets: np.ndarray, *, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(ridge_lambda)
    regularizer[-1, -1] = 0.0
    rhs = design.T @ targets.astype(np.float64, copy=False)
    solution = np.linalg.solve(gram + regularizer, rhs)
    return (
        solution[:-1].astype(FLOAT_DTYPE, copy=False),
        solution[-1].astype(FLOAT_DTYPE, copy=False),
    )


def _prediction_to_verifier_state(prediction: np.ndarray, workspace_state: dict[str, object]) -> dict[str, object]:
    completed = list(workspace_state.get("completed_subgoals", []))
    remaining = list(workspace_state.get("remaining_subgoals", []))
    exact_grid_match = bool(prediction[2] >= 0.5)
    exact_scene_match = bool(prediction[3] >= 0.5)
    should_stop = bool(not remaining and prediction[4] >= 0.5 and exact_grid_match)
    next_subgoal = remaining[0] if remaining else None
    if next_subgoal is not None:
        non_terminal_reason = f"remaining_subgoal:{next_subgoal}"
    elif should_stop:
        non_terminal_reason = None
    else:
        non_terminal_reason = "workspace_does_not_match_target_output"
    return {
        "exact_match": should_stop,
        "exact_grid_match": exact_grid_match,
        "exact_scene_match": exact_scene_match,
        "should_stop": should_stop,
        "grid_match": float(prediction[0]),
        "scene_match": float(prediction[1]),
        "resolved_subgoal_count": len(completed),
        "unresolved_subgoal_count": len(remaining),
        "next_subgoal": next_subgoal,
        "non_terminal_reason": non_terminal_reason,
    }


class LinearVerifierAdapter:
    """Standardized linear ridge fit over workspace-only structured features."""

    def __init__(
        self,
        *,
        workspace_encoder: WorkspaceEncoder,
        ridge_lambda: float,
    ):
        self.workspace_encoder = workspace_encoder
        self.ridge_lambda = FLOAT_DTYPE(ridge_lambda)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.size == 0:
            raise ValueError("Verifier training features are empty.")
        self.feature_mean, self.feature_scale = _feature_stats(features)
        normalized = _normalize_features(features, self.feature_mean, self.feature_scale).astype(np.float64, copy=False)
        design = np.concatenate([normalized, np.ones((normalized.shape[0], 1), dtype=np.float64)], axis=1)
        self.weights, self.bias = _solve_ridge(design, targets, ridge_lambda=float(self.ridge_lambda))

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Verifier adapter must be fit before prediction.")
        normalized = _normalize_features(features, self.feature_mean, self.feature_scale)
        outputs = normalized @ self.weights + self.bias
        return np.clip(outputs, FLOAT_DTYPE(0.0), FLOAT_DTYPE(1.0)).astype(FLOAT_DTYPE, copy=False)

    def __call__(
        self,
        trajectory: Any,
        step_index: int,
        workspace_state: dict[str, object],
        previous_action_index: int,
    ) -> dict[str, object]:
        features = self.workspace_encoder.encode(trajectory, step_index, previous_action_index)
        prediction = self.predict_features(features[None, :])[0]
        return _prediction_to_verifier_state(prediction, workspace_state)


class RandomFeatureVerifierAdapter:
    """Nonlinear ridge verifier using fixed random tanh features plus a linear skip."""

    def __init__(
        self,
        *,
        workspace_encoder: WorkspaceEncoder,
        ridge_lambda: float,
        random_feature_dim: int,
        seed: int,
    ):
        self.workspace_encoder = workspace_encoder
        self.ridge_lambda = FLOAT_DTYPE(ridge_lambda)
        self.random_feature_dim = max(8, int(random_feature_dim))
        self.seed = int(seed)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.random_weights: np.ndarray | None = None
        self.random_bias: np.ndarray | None = None
        self.output_weights: np.ndarray | None = None
        self.output_bias: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.size == 0:
            raise ValueError("Verifier training features are empty.")
        self.feature_mean, self.feature_scale = _feature_stats(features)
        normalized = _normalize_features(features, self.feature_mean, self.feature_scale)
        rng = np.random.default_rng(self.seed)
        input_dim = normalized.shape[1]
        self.random_weights = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(input_dim, 1)),
            size=(input_dim, self.random_feature_dim),
        ).astype(FLOAT_DTYPE, copy=False)
        self.random_bias = rng.normal(loc=0.0, scale=0.35, size=self.random_feature_dim).astype(FLOAT_DTYPE, copy=False)
        hidden = np.tanh(normalized @ self.random_weights + self.random_bias).astype(FLOAT_DTYPE, copy=False)
        design = np.concatenate(
            [
                normalized.astype(np.float64, copy=False),
                hidden.astype(np.float64, copy=False),
                np.ones((normalized.shape[0], 1), dtype=np.float64),
            ],
            axis=1,
        )
        self.output_weights, self.output_bias = _solve_ridge(design, targets, ridge_lambda=float(self.ridge_lambda))

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        if (
            self.feature_mean is None
            or self.feature_scale is None
            or self.random_weights is None
            or self.random_bias is None
            or self.output_weights is None
            or self.output_bias is None
        ):
            raise RuntimeError("Verifier adapter must be fit before prediction.")
        normalized = _normalize_features(features, self.feature_mean, self.feature_scale)
        hidden = np.tanh(normalized @ self.random_weights + self.random_bias).astype(FLOAT_DTYPE, copy=False)
        design = np.concatenate([normalized, hidden], axis=1).astype(FLOAT_DTYPE, copy=False)
        outputs = design @ self.output_weights + self.output_bias
        return np.clip(outputs, FLOAT_DTYPE(0.0), FLOAT_DTYPE(1.0)).astype(FLOAT_DTYPE, copy=False)

    def __call__(
        self,
        trajectory: Any,
        step_index: int,
        workspace_state: dict[str, object],
        previous_action_index: int,
    ) -> dict[str, object]:
        features = self.workspace_encoder.encode(trajectory, step_index, previous_action_index)
        prediction = self.predict_features(features[None, :])[0]
        return _prediction_to_verifier_state(prediction, workspace_state)


def build_verifier_adapter(
    *,
    workspace_encoder: WorkspaceEncoder,
    features: np.ndarray,
    targets: np.ndarray,
    config: VerifierFitConfig,
) -> LinearVerifierAdapter | RandomFeatureVerifierAdapter:
    if config.model_name == "linear":
        adapter = LinearVerifierAdapter(
            workspace_encoder=workspace_encoder,
            ridge_lambda=config.ridge_lambda,
        )
    elif config.model_name == "random_features":
        adapter = RandomFeatureVerifierAdapter(
            workspace_encoder=workspace_encoder,
            ridge_lambda=config.ridge_lambda,
            random_feature_dim=config.random_feature_dim,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unsupported verifier model: {config.model_name}")
    adapter.fit(features, targets)
    return adapter


def build_verifier_training_data(
    episodes: list[Any],
    *,
    workspace_encoder: WorkspaceEncoder,
    action_to_index: dict[str, int],
    action_labeler: ActionLabeler,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for episode in episodes:
        for trajectory in episode.train_trajectories:
            previous_action_index = workspace_encoder.start_action_index
            for step_index, step in enumerate(trajectory.steps):
                features.append(workspace_encoder.encode(trajectory, step_index, previous_action_index))
                verifier_state = verifier_targets(trajectory, step_index)
                targets.append(
                    [
                        float(verifier_state.get("grid_match", 0.0)),
                        float(verifier_state.get("scene_match", 0.0)),
                        float(bool(verifier_state.get("exact_grid_match"))),
                        float(bool(verifier_state.get("exact_scene_match"))),
                        float(bool(verifier_state.get("should_stop"))),
                    ]
                )
                previous_action_index = action_to_index[action_labeler(step)]
    return (
        np.asarray(features, dtype=FLOAT_DTYPE),
        np.asarray(targets, dtype=FLOAT_DTYPE),
    )
