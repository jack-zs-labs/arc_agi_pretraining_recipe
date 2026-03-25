from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACTIONS = ("up", "down", "left", "right")
ACTION_TO_FORCE = {
    0: np.array([0.0, 1.0]),
    1: np.array([0.0, -1.0]),
    2: np.array([-1.0, 0.0]),
    3: np.array([1.0, 0.0]),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 30
    grid_size: int = 7
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 7.5
    tau: float = 0.02
    x_threshold: float = 2.4
    y_threshold: float = 2.4
    theta_threshold_radians: float = np.pi / 10.0
    init_position_scale: float = 0.05
    init_velocity_scale: float = 0.05
    init_theta_scale: float = 0.06
    init_theta_dot_scale: float = 0.08
    heatmap_sigma: float = 0.55


@dataclass(frozen=True)
class TrainConfig:
    fqi_iterations: int = 18
    optimizer_steps: int = 45
    learning_rate: float = 0.04
    weight_decay: float = 1e-4
    gamma: float = 1.0
    feature_dim: int = 48


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    costs: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    success: bool
    length: int


@dataclass
class ProjectedEpisode:
    features: np.ndarray
    actions: np.ndarray
    costs: np.ndarray
    next_features: np.ndarray
    dones: np.ndarray
    success: bool
    length: int


class ThreeDCartPoleEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.total_mass = config.masscart + config.masspole
        self.polemass_length = config.masspole * config.length
        self.state = np.zeros(8, dtype=np.float64)
        self.steps = 0

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        self.state = np.array(
            [
                rng.uniform(-cfg.init_position_scale, cfg.init_position_scale),
                rng.uniform(-cfg.init_position_scale, cfg.init_position_scale),
                rng.uniform(-cfg.init_velocity_scale, cfg.init_velocity_scale),
                rng.uniform(-cfg.init_velocity_scale, cfg.init_velocity_scale),
                rng.uniform(-cfg.init_theta_scale, cfg.init_theta_scale),
                rng.uniform(-cfg.init_theta_scale, cfg.init_theta_scale),
                rng.uniform(-cfg.init_theta_dot_scale, cfg.init_theta_dot_scale),
                rng.uniform(-cfg.init_theta_dot_scale, cfg.init_theta_dot_scale),
            ],
            dtype=np.float64,
        )
        self.steps = 0
        return self._encode_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, bool]]:
        force = ACTION_TO_FORCE[action] * self.config.force_mag
        x, x_dot, theta_x, theta_dot_x = self._step_axis(
            self.state[0],
            self.state[2],
            self.state[4],
            self.state[6],
            force[0],
        )
        y, y_dot, theta_y, theta_dot_y = self._step_axis(
            self.state[1],
            self.state[3],
            self.state[5],
            self.state[7],
            force[1],
        )
        self.state = np.array(
            [x, y, x_dot, y_dot, theta_x, theta_y, theta_dot_x, theta_dot_y],
            dtype=np.float64,
        )
        self.steps += 1

        failed = (
            abs(x) > self.config.x_threshold
            or abs(y) > self.config.y_threshold
            or abs(theta_x) > self.config.theta_threshold_radians
            or abs(theta_y) > self.config.theta_threshold_radians
        )
        timed_out = self.steps >= self.config.max_steps
        done = failed or timed_out
        cost = 1.0 if failed else 0.0
        observation = self._encode_observation()
        return observation, cost, done, {"failed": failed, "timed_out": timed_out}

    def _step_axis(
        self,
        position: float,
        velocity: float,
        angle: float,
        angle_velocity: float,
        force: float,
    ) -> tuple[float, float, float, float]:
        cfg = self.config
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        temp = (force + self.polemass_length * angle_velocity**2 * sin_theta) / self.total_mass
        theta_acc = (
            cfg.gravity * sin_theta - cos_theta * temp
        ) / (cfg.length * (4.0 / 3.0 - cfg.masspole * cos_theta**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        position = position + cfg.tau * velocity
        velocity = velocity + cfg.tau * x_acc
        angle = angle + cfg.tau * angle_velocity
        angle_velocity = angle_velocity + cfg.tau * theta_acc
        return position, velocity, angle, angle_velocity

    def _encode_observation(self) -> np.ndarray:
        cfg = self.config
        size = cfg.grid_size
        rows = np.linspace(-cfg.y_threshold, cfg.y_threshold, size)
        cols = np.linspace(-cfg.x_threshold, cfg.x_threshold, size)
        grid_y, grid_x = np.meshgrid(rows, cols, indexing="ij")

        x, y, x_dot, y_dot, theta_x, theta_y, theta_dot_x, theta_dot_y = self.state
        tip_x = x + cfg.length * np.sin(theta_x)
        tip_y = y + cfg.length * np.sin(theta_y)

        cart = np.exp(
            -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * cfg.heatmap_sigma**2)
        )
        tip = np.exp(
            -((grid_x - tip_x) ** 2 + (grid_y - tip_y) ** 2) / (2.0 * cfg.heatmap_sigma**2)
        )

        def scalar_channel(value: float, scale: float) -> np.ndarray:
            normalized = np.clip(0.5 + 0.5 * value / scale, 0.0, 1.0)
            return np.full((size, size), normalized, dtype=np.float64)

        observation = np.stack(
            [
                cart,
                tip,
                scalar_channel(x_dot, 2.5),
                scalar_channel(y_dot, 2.5),
                scalar_channel(theta_dot_x, 3.0),
                scalar_channel(theta_dot_y, 3.0),
                scalar_channel(1.0 - self.steps / cfg.max_steps, 1.0),
            ],
            axis=-1,
        )
        return observation.astype(np.float32)


class RandomFeatureProjector:
    def __init__(self, input_dim: int, output_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(input_dim),
            size=(input_dim, output_dim),
        )
        self.bias = rng.normal(loc=0.0, scale=0.15, size=output_dim)

    def transform(self, observations: np.ndarray) -> np.ndarray:
        flat = observations.reshape(observations.shape[0], -1).astype(np.float64)
        projected = np.tanh(flat @ self.weights + self.bias)
        bias = np.ones((projected.shape[0], 1), dtype=np.float64)
        return np.concatenate([projected, bias], axis=1)


class TensorFeatureEncoder:
    def __init__(self, grid_size: int):
        coords = np.linspace(-1.0, 1.0, grid_size)
        self.grid_y, self.grid_x = np.meshgrid(coords, coords, indexing="ij")

    def transform(self, observations: np.ndarray) -> np.ndarray:
        observations = observations.astype(np.float64)
        cart = observations[..., 0]
        tip = observations[..., 1]
        scalar_channels = observations[..., 2:]

        def center_of_mass(heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            total = heatmap.sum(axis=(1, 2)) + 1e-8
            x_center = (heatmap * self.grid_x[None, :, :]).sum(axis=(1, 2)) / total
            y_center = (heatmap * self.grid_y[None, :, :]).sum(axis=(1, 2)) / total
            return x_center, y_center

        cart_x, cart_y = center_of_mass(cart)
        tip_x, tip_y = center_of_mass(tip)
        scalar_means = scalar_channels.mean(axis=(1, 2))
        signed_scalars = 2.0 * scalar_means[:, :4] - 1.0
        remaining_time = scalar_means[:, 4:5]
        dx = (tip_x - cart_x)[:, None]
        dy = (tip_y - cart_y)[:, None]
        base = np.column_stack(
            [
                cart_x,
                cart_y,
                tip_x,
                tip_y,
                signed_scalars[:, 0],
                signed_scalars[:, 1],
                signed_scalars[:, 2],
                signed_scalars[:, 3],
                remaining_time[:, 0],
            ]
        )
        quadratic = base**2
        interactions = np.column_stack(
            [
                cart_x * dx[:, 0],
                cart_y * dy[:, 0],
                dx[:, 0],
                dy[:, 0],
                dx[:, 0] * signed_scalars[:, 0],
                dy[:, 0] * signed_scalars[:, 1],
                dx[:, 0] * signed_scalars[:, 2],
                dy[:, 0] * signed_scalars[:, 3],
            ]
        )
        bias = np.ones((observations.shape[0], 1), dtype=np.float64)
        return np.concatenate([base, quadratic, interactions, bias], axis=1)


class SigmoidLinearQ:
    def __init__(self, feature_dim: int, num_actions: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            loc=-0.05,
            scale=0.02,
            size=(num_actions, feature_dim),
        )
        self.num_actions = num_actions

    def copy(self) -> "SigmoidLinearQ":
        clone = SigmoidLinearQ(self.weights.shape[1], self.num_actions, seed=0)
        clone.weights = self.weights.copy()
        return clone

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights.T
        return sigmoid(logits)

    def greedy_action(self, features: np.ndarray) -> int:
        predictions = self.predict(features[None, :])[0]
        return int(np.argmin(predictions))

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
        action_mask = np.zeros((num_samples, self.num_actions), dtype=np.float64)
        action_mask[np.arange(num_samples), actions] = 1.0

        first_moment = np.zeros_like(self.weights)
        second_moment = np.zeros_like(self.weights)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for step in range(1, steps + 1):
            logits = features @ self.weights.T
            predictions = sigmoid(logits)
            chosen_predictions = predictions[np.arange(num_samples), actions]

            if loss_name == "log":
                grad_scalar = chosen_predictions - targets
            elif loss_name == "sq":
                grad_scalar = (
                    2.0
                    * (chosen_predictions - targets)
                    * chosen_predictions
                    * (1.0 - chosen_predictions)
                )
            else:
                raise ValueError(f"unknown loss {loss_name}")

            output_grad = action_mask * grad_scalar[:, None]
            gradient = output_grad.T @ features / num_samples
            gradient += weight_decay * self.weights

            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
            corrected_first = first_moment / (1.0 - beta1**step)
            corrected_second = second_moment / (1.0 - beta2**step)
            self.weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + eps)


def heuristic_action(state: np.ndarray) -> int:
    x, y, x_dot, y_dot, theta_x, theta_y, theta_dot_x, theta_dot_y = state
    x_score = theta_x + 0.35 * theta_dot_x + 0.08 * x + 0.04 * x_dot
    y_score = theta_y + 0.35 * theta_dot_y + 0.08 * y + 0.04 * y_dot
    if abs(x_score) >= abs(y_score):
        return 3 if x_score >= 0.0 else 2
    return 0 if y_score >= 0.0 else 1


def behavior_action(
    state: np.ndarray,
    rng: np.random.Generator,
    heuristic_probability: float,
) -> int:
    if rng.random() < heuristic_probability:
        return heuristic_action(state)
    return int(rng.integers(0, len(ACTIONS)))


def rollout_episode(
    env: ThreeDCartPoleEnv,
    rng: np.random.Generator,
    heuristic_probability: float,
) -> Episode:
    observations = []
    actions = []
    costs = []
    next_observations = []
    dones = []

    observation = env.reset(rng)
    done = False
    failed = False

    while not done:
        state = env.state.copy()
        action = behavior_action(state, rng, heuristic_probability)
        next_observation, cost, done, info = env.step(action)
        observations.append(observation)
        actions.append(action)
        costs.append(cost)
        next_observations.append(next_observation)
        dones.append(done)
        observation = next_observation
        failed = info["failed"]

    return Episode(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        costs=np.asarray(costs, dtype=np.float64),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float64),
        success=not failed,
        length=len(actions),
    )


def project_episode(episode: Episode, projector: RandomFeatureProjector) -> ProjectedEpisode:
    return ProjectedEpisode(
        features=projector.transform(episode.observations),
        actions=episode.actions,
        costs=episode.costs,
        next_features=projector.transform(episode.next_observations),
        dones=episode.dones,
        success=episode.success,
        length=episode.length,
    )


def concatenate_episodes(episodes: list[ProjectedEpisode]) -> dict[str, np.ndarray]:
    return {
        "features": np.concatenate([episode.features for episode in episodes], axis=0),
        "actions": np.concatenate([episode.actions for episode in episodes], axis=0),
        "costs": np.concatenate([episode.costs for episode in episodes], axis=0),
        "next_features": np.concatenate([episode.next_features for episode in episodes], axis=0),
        "dones": np.concatenate([episode.dones for episode in episodes], axis=0),
    }


def train_fqi(
    batch: dict[str, np.ndarray],
    train_config: TrainConfig,
    *,
    loss_name: str,
    seed: int,
) -> SigmoidLinearQ:
    model = SigmoidLinearQ(
        feature_dim=batch["features"].shape[1],
        num_actions=len(ACTIONS),
        seed=seed,
    )
    for iteration in range(train_config.fqi_iterations):
        next_predictions = model.predict(batch["next_features"])
        targets = batch["costs"] + train_config.gamma * (1.0 - batch["dones"]) * np.min(
            next_predictions,
            axis=1,
        )
        targets = np.clip(targets, 0.0, 1.0)

        next_model = model.copy()
        next_model.fit(
            batch["features"],
            batch["actions"],
            targets,
            loss_name=loss_name,
            steps=train_config.optimizer_steps,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        model = next_model
        if iteration == train_config.fqi_iterations - 1:
            break
    return model


def evaluate_policy(
    env_config: EnvConfig,
    projector: RandomFeatureProjector,
    model: SigmoidLinearQ,
    *,
    seed: int,
    episodes: int,
) -> dict[str, float]:
    env = ThreeDCartPoleEnv(env_config)
    rng = np.random.default_rng(seed)
    failure_costs = []
    lengths = []

    for _ in range(episodes):
        observation = env.reset(rng)
        done = False
        total_cost = 0.0
        while not done:
            features = projector.transform(observation[None, ...])[0]
            action = model.greedy_action(features)
            observation, cost, done, _ = env.step(action)
            total_cost += cost
        failure_costs.append(total_cost)
        lengths.append(env.steps)

    failure_costs = np.asarray(failure_costs, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    return {
        "failure_rate": float(failure_costs.mean()),
        "success_rate": float(1.0 - failure_costs.mean()),
        "mean_length": float(lengths.mean()),
    }


def collect_dataset(
    env_config: EnvConfig,
    *,
    episodes: int,
    seed: int,
    heuristic_probability: float,
    frontload_successes: bool,
) -> list[Episode]:
    env = ThreeDCartPoleEnv(env_config)
    rng = np.random.default_rng(seed)
    collected = [rollout_episode(env, rng, heuristic_probability) for _ in range(episodes)]
    if frontload_successes:
        collected.sort(key=lambda episode: (not episode.success, episode.length))
    return collected


def summarize_dataset(episodes: list[Episode]) -> dict[str, float]:
    successes = sum(episode.success for episode in episodes)
    lengths = np.asarray([episode.length for episode in episodes], dtype=np.float64)
    return {
        "episodes": float(len(episodes)),
        "successes": float(successes),
        "success_rate": float(successes / max(len(episodes), 1)),
        "mean_length": float(lengths.mean()),
    }


def aggregate_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, float]]] = {}
    for row in rows:
        key = (row["loss"], int(row["dataset_episodes"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (loss_name, dataset_episodes), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        failure_rates = np.asarray([item["failure_rate"] for item in group], dtype=np.float64)
        mean_lengths = np.asarray([item["mean_length"] for item in group], dtype=np.float64)
        dataset_successes = np.asarray([item["dataset_successes"] for item in group], dtype=np.float64)
        summary_rows.append(
            {
                "loss": loss_name,
                "dataset_episodes": dataset_episodes,
                "mean_failure_rate": float(failure_rates.mean()),
                "stderr_failure_rate": float(failure_rates.std(ddof=0) / np.sqrt(len(failure_rates))),
                "mean_success_rate": float(1.0 - failure_rates.mean()),
                "mean_episode_length": float(mean_lengths.mean()),
                "mean_dataset_successes": float(dataset_successes.mean()),
            }
        )
    return summary_rows


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary_rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for loss_name, color in (("log", "#006d77"), ("sq", "#ae2012")):
        subset = [row for row in summary_rows if row["loss"] == loss_name]
        xs = [row["dataset_episodes"] for row in subset]
        ys = [row["mean_failure_rate"] for row in subset]
        errs = [row["stderr_failure_rate"] for row in subset]
        label = "FQI-LOG" if loss_name == "log" else "FQI-SQ"
        plt.plot(xs, ys, marker="o", color=color, label=label)
        plt.fill_between(
            xs,
            np.asarray(ys) - np.asarray(errs),
            np.asarray(ys) + np.asarray(errs),
            color=color,
            alpha=0.18,
        )
    plt.xlabel("Offline training episodes")
    plt.ylabel("Failure rate")
    plt.title("3D cart-pole offline RL: log-loss vs square loss")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    env_config = EnvConfig(max_steps=args.max_steps, grid_size=args.grid_size, force_mag=args.force_mag)
    train_config = TrainConfig(
        fqi_iterations=args.fqi_iterations,
        optimizer_steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        feature_dim=args.feature_dim,
    )

    output_dir = Path(args.output_dir)
    dataset_sizes = sorted(args.dataset_sizes)

    all_rows: list[dict[str, float]] = []
    dataset_summaries: list[dict[str, float]] = []

    for trial in range(args.seeds):
        seed = args.seed + trial
        episodes = collect_dataset(
            env_config,
            episodes=max(dataset_sizes),
            seed=seed,
            heuristic_probability=args.heuristic_probability,
            frontload_successes=args.frontload_successes,
        )
        dataset_summary = summarize_dataset(episodes)
        dataset_summary["seed"] = float(seed)
        dataset_summaries.append(dataset_summary)

        if args.feature_mode == "random":
            projector = RandomFeatureProjector(
                input_dim=int(np.prod(episodes[0].observations.shape[1:])),
                output_dim=train_config.feature_dim,
                seed=seed + 10_000,
            )
        else:
            projector = TensorFeatureEncoder(grid_size=env_config.grid_size)
        projected = [project_episode(episode, projector) for episode in episodes]

        for dataset_episodes in dataset_sizes:
            prefix = projected[:dataset_episodes]
            batch = concatenate_episodes(prefix)
            dataset_successes = float(sum(episode.success for episode in episodes[:dataset_episodes]))

            for loss_name in ("log", "sq"):
                model = train_fqi(
                    batch,
                    train_config,
                    loss_name=loss_name,
                    seed=seed + (0 if loss_name == "log" else 100_000) + dataset_episodes,
                )
                metrics = evaluate_policy(
                    env_config,
                    projector,
                    model,
                    seed=seed + 50_000,
                    episodes=args.eval_episodes,
                )
                all_rows.append(
                    {
                        "seed": float(seed),
                        "dataset_episodes": float(dataset_episodes),
                        "dataset_successes": dataset_successes,
                        "loss": loss_name,
                        "failure_rate": metrics["failure_rate"],
                        "success_rate": metrics["success_rate"],
                        "mean_length": metrics["mean_length"],
                    }
                )

    summary_rows = aggregate_rows(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(all_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    save_plot(summary_rows, output_dir / "comparison.png")

    summary_payload = {
        "env_config": env_config.__dict__,
        "train_config": train_config.__dict__,
        "dataset_sizes": dataset_sizes,
        "dataset_summaries": dataset_summaries,
        "summary_rows": summary_rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary_payload, handle, indent=2)
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3D cart-pole FQI benchmark.")
    parser.add_argument("--output-dir", default="results/three_d_cartpole")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--heuristic-probability", type=float, default=0.15)
    parser.add_argument("--frontload-successes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--force-mag", type=float, default=7.5)
    parser.add_argument("--fqi-iterations", type=int, default=18)
    parser.add_argument("--optimizer-steps", type=int, default=45)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--feature-dim", type=int, default=48)
    parser.add_argument("--feature-mode", choices=("summary", "random"), default="summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_experiment(args)
    print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
