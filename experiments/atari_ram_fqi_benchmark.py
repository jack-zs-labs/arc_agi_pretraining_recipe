from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    import ale_py
except ImportError as exc:
    raise SystemExit(
        "This benchmark requires the Atari venv. Activate .venv_atari and install gymnasium + ale-py."
    ) from exc


try:
    gym.register_envs(ale_py)
except Exception:
    pass


GAME_CONFIGS = {
    "Asterix": {"env_id": "ALE/Asterix-v5", "max_steps": 1500},
    "Seaquest": {"env_id": "ALE/Seaquest-v5", "max_steps": 1500},
    "Breakout": {"env_id": "ALE/Breakout-v5", "max_steps": 1500},
}

FIRE_BOOTSTRAP_STEPS = 3


@dataclass(frozen=True)
class TrainConfig:
    gamma: float = 0.99
    fqi_iterations: int = 12
    optimizer_steps: int = 30
    learning_rate: float = 0.03
    weight_decay: float = 1e-4


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    signals: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    length: int
    total_reward: float
    positive_events: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class RamFeatureEncoder:
    def transform(self, observations: np.ndarray) -> np.ndarray:
        observations = observations.astype(np.float64)
        scaled = 2.0 * (observations / 255.0) - 1.0
        squared = scaled * scaled
        first_order_deltas = np.diff(scaled[:, :32], axis=1, prepend=scaled[:, :1])
        chunk_means = scaled.reshape(scaled.shape[0], 16, 8).mean(axis=2)
        bias = np.ones((scaled.shape[0], 1), dtype=np.float64)
        return np.concatenate([scaled, squared, first_order_deltas, chunk_means, bias], axis=1)


class SigmoidLinearQ:
    def __init__(self, feature_dim: int, num_actions: int, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(loc=0.0, scale=0.02, size=(num_actions, feature_dim))
        self.num_actions = num_actions

    def copy(self) -> "SigmoidLinearQ":
        clone = SigmoidLinearQ(self.weights.shape[1], self.num_actions, seed=0)
        clone.weights = self.weights.copy()
        return clone

    def predict(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(features @ self.weights.T)

    def greedy_action(self, features: np.ndarray) -> int:
        return int(np.argmax(self.predict(features[None, :])[0]))

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
        mask = np.zeros((num_samples, self.num_actions), dtype=np.float64)
        mask[np.arange(num_samples), actions] = 1.0

        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for step in range(1, steps + 1):
            predictions = self.predict(features)
            chosen = predictions[np.arange(num_samples), actions]
            if loss_name == "log":
                grad_scalar = chosen - targets
            elif loss_name == "sq":
                grad_scalar = 2.0 * (chosen - targets) * chosen * (1.0 - chosen)
            else:
                raise ValueError(loss_name)

            output_grad = mask * grad_scalar[:, None]
            gradient = output_grad.T @ features / num_samples
            gradient += weight_decay * self.weights

            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def behavior_policy(
    rng: np.random.Generator,
    action_meanings: list[str],
    last_action: int | None,
    *,
    sticky_prob: float,
) -> int:
    if last_action is not None and rng.random() < sticky_prob:
        return last_action

    non_noop = [i for i, meaning in enumerate(action_meanings) if meaning != "NOOP"]
    if not non_noop:
        non_noop = list(range(len(action_meanings)))
    return int(rng.choice(non_noop))


def fire_action_index(action_meanings: list[str]) -> int | None:
    if "FIRE" not in action_meanings:
        return None
    return action_meanings.index("FIRE")


def collect_episode(
    game_name: str,
    *,
    seed: int,
    sticky_prob: float,
) -> Episode:
    config = GAME_CONFIGS[game_name]
    env = gym.make(
        config["env_id"],
        obs_type="ram",
        frameskip=4,
        repeat_action_probability=0.0,
    )
    action_meanings = env.unwrapped.get_action_meanings()
    fire_action = fire_action_index(action_meanings)
    rng = np.random.default_rng(seed)

    observation, info = env.reset(seed=seed)
    lives = info.get("lives")
    fire_steps_remaining = FIRE_BOOTSTRAP_STEPS if fire_action is not None else 0
    last_action = None
    observations = []
    actions = []
    signals = []
    rewards = []
    next_observations = []
    dones = []
    total_reward = 0.0
    positive_events = 0.0

    for step in range(config["max_steps"]):
        if fire_steps_remaining > 0 and fire_action is not None:
            action = fire_action
            fire_steps_remaining -= 1
        else:
            action = behavior_policy(
                rng,
                action_meanings,
                last_action,
                sticky_prob=sticky_prob,
            )
        next_observation, reward, terminated, truncated, info = env.step(action)
        signal = 1.0 if reward > 0 else 0.0
        done = terminated or truncated

        observations.append(observation)
        actions.append(action)
        signals.append(signal)
        rewards.append(reward)
        next_observations.append(next_observation)
        dones.append(done)

        total_reward += reward
        positive_events += signal
        observation = next_observation
        next_lives = info.get("lives", lives)
        if fire_action is not None and lives is not None and next_lives is not None and next_lives < lives:
            fire_steps_remaining = FIRE_BOOTSTRAP_STEPS
        lives = next_lives
        last_action = action
        if done:
            break

    env.close()
    return Episode(
        observations=np.asarray(observations, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        signals=np.asarray(signals, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        next_observations=np.asarray(next_observations, dtype=np.uint8),
        dones=np.asarray(dones, dtype=np.float64),
        length=len(actions),
        total_reward=float(total_reward),
        positive_events=float(positive_events),
    )


def concatenate_episodes(episodes: list[Episode], encoder: RamFeatureEncoder) -> dict[str, np.ndarray]:
    return {
        "features": np.concatenate([encoder.transform(ep.observations) for ep in episodes], axis=0),
        "actions": np.concatenate([ep.actions for ep in episodes], axis=0),
        "signals": np.concatenate([ep.signals for ep in episodes], axis=0),
        "rewards": np.concatenate([ep.rewards for ep in episodes], axis=0),
        "next_features": np.concatenate([encoder.transform(ep.next_observations) for ep in episodes], axis=0),
        "dones": np.concatenate([ep.dones for ep in episodes], axis=0),
    }


def train_fqi(
    batch: dict[str, np.ndarray],
    num_actions: int,
    train_config: TrainConfig,
    *,
    loss_name: str,
    seed: int,
) -> SigmoidLinearQ:
    model = SigmoidLinearQ(batch["features"].shape[1], num_actions=num_actions, seed=seed)
    for _ in range(train_config.fqi_iterations):
        next_values = np.max(model.predict(batch["next_features"]), axis=1)
        targets = (1.0 - train_config.gamma) * batch["signals"] + train_config.gamma * (1.0 - batch["dones"]) * next_values
        targets = np.clip(targets, 0.0, 1.0)
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
    game_name: str,
    model: SigmoidLinearQ,
    encoder: RamFeatureEncoder,
    *,
    seed: int,
    episodes: int,
) -> dict[str, float]:
    config = GAME_CONFIGS[game_name]
    env = gym.make(
        config["env_id"],
        obs_type="ram",
        frameskip=4,
        repeat_action_probability=0.0,
    )
    rewards = []
    positive_events = []
    lengths = []
    action_meanings = env.unwrapped.get_action_meanings()
    fire_action = fire_action_index(action_meanings)

    for ep in range(episodes):
        observation, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        signal_count = 0.0
        steps = 0
        done = False
        lives = info.get("lives")
        fire_steps_remaining = FIRE_BOOTSTRAP_STEPS if fire_action is not None else 0
        while not done and steps < config["max_steps"]:
            if fire_steps_remaining > 0 and fire_action is not None:
                action = fire_action
                fire_steps_remaining -= 1
            else:
                features = encoder.transform(observation[None, :])[0]
                action = model.greedy_action(features)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            signal_count += 1.0 if reward > 0 else 0.0
            done = terminated or truncated
            steps += 1
            next_lives = info.get("lives", lives)
            if fire_action is not None and lives is not None and next_lives is not None and next_lives < lives:
                fire_steps_remaining = FIRE_BOOTSTRAP_STEPS
            lives = next_lives
        rewards.append(total_reward)
        positive_events.append(signal_count)
        lengths.append(steps)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_positive_events": float(np.mean(positive_events)),
        "mean_length": float(np.mean(lengths)),
    }


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["game"]), str(row["loss"]), int(row["dataset_episodes"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (game, loss_name, dataset_episodes), group in sorted(grouped.items()):
        mean_reward = np.asarray([float(item["mean_reward"]) for item in group], dtype=np.float64)
        mean_positive_events = np.asarray([float(item["mean_positive_events"]) for item in group], dtype=np.float64)
        mean_length = np.asarray([float(item["mean_length"]) for item in group], dtype=np.float64)
        summary_rows.append(
            {
                "game": game,
                "loss": loss_name,
                "dataset_episodes": dataset_episodes,
                "mean_reward": float(mean_reward.mean()),
                "stderr_reward": float(mean_reward.std(ddof=0) / np.sqrt(len(mean_reward))),
                "mean_positive_events": float(mean_positive_events.mean()),
                "mean_length": float(mean_length.mean()),
            }
        )
    return summary_rows


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    train_config = TrainConfig(
        gamma=args.gamma,
        fqi_iterations=args.fqi_iterations,
        optimizer_steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    encoder = RamFeatureEncoder()

    raw_rows: list[dict[str, object]] = []
    summary_payload: dict[str, object] = {
        "games": args.games,
        "dataset_sizes": args.dataset_sizes,
        "train_config": train_config.__dict__,
    }

    for game in args.games:
        env = gym.make(GAME_CONFIGS[game]["env_id"], obs_type="ram", frameskip=4, repeat_action_probability=0.0)
        num_actions = env.action_space.n
        env.close()

        for trial in range(args.seeds):
            seed = args.seed + trial
            episodes = [
                collect_episode(game, seed=seed * 100_000 + idx, sticky_prob=args.sticky_prob)
                for idx in range(max(args.dataset_sizes))
            ]
            for dataset_episodes in args.dataset_sizes:
                batch = concatenate_episodes(episodes[:dataset_episodes], encoder)
                for loss_name in ("log", "sq"):
                    model = train_fqi(
                        batch,
                        num_actions,
                        train_config,
                        loss_name=loss_name,
                        seed=seed + dataset_episodes + (0 if loss_name == "log" else 10_000),
                    )
                    metrics = evaluate_policy(
                        game,
                        model,
                        encoder,
                        seed=seed + 50_000,
                        episodes=args.eval_episodes,
                    )
                    raw_rows.append(
                        {
                            "game": game,
                            "seed": seed,
                            "dataset_episodes": dataset_episodes,
                            "loss": loss_name,
                            "mean_reward": metrics["mean_reward"],
                            "mean_positive_events": metrics["mean_positive_events"],
                            "mean_length": metrics["mean_length"],
                        }
                    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate(raw_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    with (output_dir / "summary.json").open("w") as handle:
        json.dump({"config": summary_payload, "summary_rows": summary_rows}, handle, indent=2)
    return {"summary_rows": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FQI-LOG vs FQI-SQ on Atari RAM.")
    parser.add_argument("--games", nargs="+", default=["Asterix", "Seaquest"])
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--sticky-prob", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fqi-iterations", type=int, default=12)
    parser.add_argument("--optimizer-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results/atari_ram_fqi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
