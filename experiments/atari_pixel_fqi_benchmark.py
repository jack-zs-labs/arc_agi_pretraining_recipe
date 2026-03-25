from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    import ale_py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from tqdm.auto import tqdm
except ImportError as exc:
    raise SystemExit(
        "This benchmark requires the pixel Atari venv. Activate .venv_atari and install requirements-atari-pixel.txt."
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
    fqi_iterations: int = 6
    optimizer_epochs: int = 2
    batch_size: int = 128
    target_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0


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


@dataclass
class RolloutTrace:
    total_reward: float
    positive_events: float
    length: int
    frames: list[np.ndarray]
    action_names: list[str]


@dataclass
class TensorBatch:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    signals: torch.Tensor
    dones: torch.Tensor
    cached_on_device: bool


def progress_enabled(enabled: bool) -> bool:
    return enabled and sys.stderr.isatty()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def auto_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but MPS is not available in this environment.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda, but CUDA is not available in this environment.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cpu":
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(min(cpu_count, 8))
    # MPS fallback avoids hard failures if an unsupported op slips in.
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(frame[34:194]).to(dtype=torch.float32)
    grayscale = 0.299 * tensor[..., 0] + 0.587 * tensor[..., 1] + 0.114 * tensor[..., 2]
    grayscale = grayscale.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(grayscale, size=(84, 84), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).round().clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()


def build_initial_stack(frame: np.ndarray) -> np.ndarray:
    processed = preprocess_frame(frame)
    return np.repeat(processed[None, :, :], 4, axis=0)


def push_frame(stack: np.ndarray, next_processed: np.ndarray) -> np.ndarray:
    stack[:-1] = stack[1:]
    stack[-1] = next_processed
    return stack


class PixelQNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, num_actions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.to(dtype=torch.float32) / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return self.head(x)

    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(observations))


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


def non_noop_actions(action_meanings: list[str]) -> list[int]:
    actions = [idx for idx, meaning in enumerate(action_meanings) if meaning != "NOOP"]
    return actions if actions else list(range(len(action_meanings)))


def select_policy_action(
    model: PixelQNetwork,
    observation_stack: np.ndarray,
    *,
    device: torch.device,
    action_meanings: list[str],
    rng: np.random.Generator,
    epsilon: float,
) -> int:
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.choice(non_noop_actions(action_meanings)))
    obs_tensor = torch.from_numpy(observation_stack[None, ...]).to(device)
    logits = model.forward(obs_tensor).squeeze(0)
    max_logit = torch.max(logits)
    candidates = torch.nonzero(logits >= max_logit - 1e-6, as_tuple=False).flatten().cpu().numpy()
    if len(candidates) > 1:
        return int(rng.choice(candidates))
    return int(torch.argmax(logits).item())


def dominant_action_name(action_names: list[str]) -> str | None:
    if not action_names:
        return None
    counts: dict[str, int] = {}
    for action_name in action_names:
        counts[action_name] = counts.get(action_name, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def collect_episode(
    game_name: str,
    *,
    seed: int,
    sticky_prob: float,
) -> Episode:
    config = GAME_CONFIGS[game_name]
    env = gym.make(
        config["env_id"],
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0.0,
    )
    action_meanings = env.unwrapped.get_action_meanings()
    fire_action = fire_action_index(action_meanings)
    rng = np.random.default_rng(seed)

    frame, info = env.reset(seed=seed)
    current_stack = build_initial_stack(frame)
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
        next_frame, reward, terminated, truncated, info = env.step(action)
        next_processed = preprocess_frame(next_frame)

        signal = 1.0 if reward > 0 else 0.0
        done = terminated or truncated

        observations.append(current_stack.copy())
        actions.append(action)
        signals.append(signal)
        rewards.append(reward)
        push_frame(current_stack, next_processed)
        next_observations.append(current_stack.copy())
        dones.append(done)

        total_reward += reward
        positive_events += signal
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
        signals=np.asarray(signals, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.uint8),
        dones=np.asarray(dones, dtype=np.float32),
        length=len(actions),
        total_reward=float(total_reward),
        positive_events=float(positive_events),
    )


def concatenate_episodes(episodes: list[Episode]) -> dict[str, np.ndarray]:
    return {
        "observations": np.concatenate([ep.observations for ep in episodes], axis=0),
        "actions": np.concatenate([ep.actions for ep in episodes], axis=0),
        "signals": np.concatenate([ep.signals for ep in episodes], axis=0),
        "rewards": np.concatenate([ep.rewards for ep in episodes], axis=0),
        "next_observations": np.concatenate([ep.next_observations for ep in episodes], axis=0),
        "dones": np.concatenate([ep.dones for ep in episodes], axis=0),
    }


def prefix_batch(batch: dict[str, np.ndarray], end_index: int) -> dict[str, np.ndarray]:
    return {key: values[:end_index] for key, values in batch.items()}


def collect_episodes(
    game_name: str,
    *,
    seed: int,
    count: int,
    sticky_prob: float,
    workers: int,
    show_progress: bool,
) -> list[Episode]:
    progress_bar = tqdm(
        range(count),
        desc=f"collect {game_name} seed={seed}",
        leave=False,
        dynamic_ncols=True,
        disable=not progress_enabled(show_progress),
    )
    if workers <= 1:
        episodes = []
        for idx in progress_bar:
            episodes.append(collect_episode(game_name, seed=seed * 100_000 + idx, sticky_prob=sticky_prob))
        progress_bar.close()
        return episodes

    def collect_one(idx: int) -> Episode:
        return collect_episode(game_name, seed=seed * 100_000 + idx, sticky_prob=sticky_prob)

    episodes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for episode in executor.map(collect_one, range(count)):
            episodes.append(episode)
            progress_bar.update(1)
    progress_bar.close()
    return episodes


def should_cache_pixels_on_device(batch: dict[str, np.ndarray], device: torch.device, max_bytes: int) -> bool:
    if device.type == "cpu":
        return True
    pixel_bytes = batch["observations"].nbytes + batch["next_observations"].nbytes
    return pixel_bytes <= max_bytes


def make_tensor_batch(
    batch: dict[str, np.ndarray],
    *,
    device: torch.device,
    cache_pixels_on_device: bool,
) -> TensorBatch:
    observations = torch.from_numpy(batch["observations"])
    next_observations = torch.from_numpy(batch["next_observations"])
    if cache_pixels_on_device:
        observations = observations.to(device)
        next_observations = next_observations.to(device)
    return TensorBatch(
        observations=observations,
        next_observations=next_observations,
        actions=torch.from_numpy(batch["actions"]).to(device=device, dtype=torch.long),
        signals=torch.from_numpy(batch["signals"]).to(device=device, dtype=torch.float32),
        dones=torch.from_numpy(batch["dones"]).to(device=device, dtype=torch.float32),
        cached_on_device=cache_pixels_on_device,
    )


def select_observation_batch(observations: torch.Tensor, indices_cpu: torch.Tensor, *, device: torch.device, cached_on_device: bool) -> torch.Tensor:
    if cached_on_device:
        return observations.index_select(0, indices_cpu.to(device=device))
    return observations.index_select(0, indices_cpu).to(device)


def predict_max_values(
    model: PixelQNetwork,
    observations: torch.Tensor,
    device: torch.device,
    batch_size: int,
    cached_on_device: bool,
) -> torch.Tensor:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, observations.shape[0], batch_size):
            batch = observations[start : start + batch_size]
            if not cached_on_device:
                batch = batch.to(device)
            logits = model.forward(batch)
            values.append(torch.sigmoid(logits.max(dim=1).values))
    return torch.cat(values, dim=0)


def clone_model(model: PixelQNetwork, device: torch.device) -> PixelQNetwork:
    clone = PixelQNetwork(model.head.out_features).to(device)
    clone.load_state_dict(model.state_dict())
    return clone


def choose_panel_grid(num_items: int, preferred_columns: int) -> tuple[int, int]:
    columns = max(1, min(preferred_columns, num_items))
    rows = int(math.ceil(num_items / columns))
    return rows, columns


def rollout_policy_episode(
    game_name: str,
    model: PixelQNetwork,
    *,
    seed: int,
    device: torch.device,
    epsilon: float = 0.0,
    max_steps: int | None = None,
    capture_frames: bool = False,
    frame_stride: int = 4,
    max_frames: int = 80,
    preview_callback=None,
) -> RolloutTrace:
    config = GAME_CONFIGS[game_name]
    env = gym.make(
        config["env_id"],
        obs_type="rgb",
        frameskip=4,
        repeat_action_probability=0.0,
    )
    frame, info = env.reset(seed=seed)
    action_meanings = env.unwrapped.get_action_meanings()
    fire_action = fire_action_index(action_meanings)
    rng = np.random.default_rng(seed)
    current_stack = build_initial_stack(frame)
    total_reward = 0.0
    positive_events = 0.0
    steps = 0
    done = False
    lives = info.get("lives")
    fire_steps_remaining = FIRE_BOOTSTRAP_STEPS if fire_action is not None else 0
    frames: list[np.ndarray] = []
    action_trace: list[str] = []
    if capture_frames:
        frames.append(frame)
        if preview_callback is not None:
            preview_callback(
                frame,
                action_name=None,
                steps=steps,
                total_reward=total_reward,
                positive_events=positive_events,
                captured_frames=len(frames),
                done=done,
            )
    step_limit = config["max_steps"] if max_steps is None else min(config["max_steps"], max_steps)

    model.eval()
    with torch.inference_mode():
        while not done and steps < step_limit:
            if fire_steps_remaining > 0 and fire_action is not None:
                action = fire_action
                fire_steps_remaining -= 1
            else:
                action = select_policy_action(
                    model,
                    current_stack,
                    device=device,
                    action_meanings=action_meanings,
                    rng=rng,
                    epsilon=epsilon,
                )
            action_name = action_meanings[action]
            action_trace.append(action_name)
            next_frame, reward, terminated, truncated, info = env.step(action)
            next_processed = preprocess_frame(next_frame)
            current_stack = push_frame(current_stack, next_processed)
            total_reward += reward
            positive_events += 1.0 if reward > 0 else 0.0
            done = terminated or truncated
            steps += 1
            next_lives = info.get("lives", lives)
            if fire_action is not None and lives is not None and next_lives is not None and next_lives < lives:
                fire_steps_remaining = FIRE_BOOTSTRAP_STEPS
            lives = next_lives
            if capture_frames and steps % frame_stride == 0 and len(frames) < max_frames:
                frames.append(next_frame)
                if preview_callback is not None:
                    preview_callback(
                        next_frame,
                        action_name=action_name,
                        steps=steps,
                        total_reward=total_reward,
                        positive_events=positive_events,
                        captured_frames=len(frames),
                        done=done,
                    )
            elif preview_callback is not None and done:
                preview_callback(
                    next_frame,
                    action_name=action_name,
                    steps=steps,
                    total_reward=total_reward,
                    positive_events=positive_events,
                    captured_frames=len(frames),
                    done=done,
                )

    env.close()
    return RolloutTrace(
        total_reward=float(total_reward),
        positive_events=float(positive_events),
        length=steps,
        frames=frames,
        action_names=action_trace,
    )


def annotate_frame(frame: np.ndarray, label: str, footer_height: int = 24) -> np.ndarray:
    image = Image.fromarray(frame)
    canvas = Image.new("RGB", (image.width, image.height + footer_height), color=(16, 16, 16))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, image.height + 4), label, fill=(240, 240, 240))
    return np.asarray(canvas, dtype=np.uint8)


def reset_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def save_jpg(frame: np.ndarray, path: Path, *, quality: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path, format="JPEG", quality=quality, optimize=False)


def compose_panel_snapshot(
    frames: list[np.ndarray | None],
    labels: list[str],
    *,
    preferred_columns: int,
) -> np.ndarray | None:
    non_empty = [frame for frame in frames if frame is not None]
    if not non_empty:
        return None
    rows, columns = choose_panel_grid(len(frames), preferred_columns)
    annotated_tiles = []
    for frame, label in zip(frames, labels, strict=True):
        if frame is None:
            placeholder = np.zeros_like(non_empty[0], dtype=np.uint8)
            placeholder[:] = 12
            annotated_tiles.append(annotate_frame(placeholder, label))
        else:
            annotated_tiles.append(annotate_frame(frame, label))
    cell_height = max(tile.shape[0] for tile in annotated_tiles)
    cell_width = max(tile.shape[1] for tile in annotated_tiles)
    panel = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for idx, tile in enumerate(annotated_tiles):
        row = idx // columns
        col = idx % columns
        top = row * cell_height
        left = col * cell_width
        panel[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
    return panel


def compose_trace_panel(
    traces: list[RolloutTrace],
    labels: list[str],
    *,
    preferred_columns: int,
) -> list[np.ndarray]:
    if not traces:
        return []
    rows, columns = choose_panel_grid(len(traces), preferred_columns)
    labeled_sequences = []
    max_length = 0
    cell_height = 0
    cell_width = 0
    for trace, label in zip(traces, labels, strict=True):
        if not trace.frames:
            continue
        labeled_frames = [annotate_frame(frame, label) for frame in trace.frames]
        labeled_sequences.append(labeled_frames)
        max_length = max(max_length, len(labeled_frames))
        cell_height = max(cell_height, labeled_frames[0].shape[0])
        cell_width = max(cell_width, labeled_frames[0].shape[1])
    if not labeled_sequences:
        return []

    panel_frames = []
    background = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    for frame_idx in range(max_length):
        panel = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
        for seq_idx, sequence in enumerate(labeled_sequences):
            row = seq_idx // columns
            col = seq_idx % columns
            tile = sequence[min(frame_idx, len(sequence) - 1)] if sequence else background
            top = row * cell_height
            left = col * cell_width
            panel[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
        panel_frames.append(panel)
    return panel_frames


class LiveTracePreviewWriter:
    def __init__(
        self,
        trace_root: Path,
        *,
        iteration_index: int,
        panel_rollouts: int,
        panel_columns: int,
        frame_duration_ms: int,
        refresh_ms: int,
        jpeg_quality: int,
    ):
        self.trace_root = trace_root
        self.live_dir = trace_root / "live"
        self.iteration_index = iteration_index
        self.panel_rollouts = panel_rollouts
        self.panel_columns = panel_columns
        self.frame_duration_ms = frame_duration_ms
        self.refresh_ms = max(1, refresh_ms)
        self.jpeg_quality = jpeg_quality
        self.frames: list[np.ndarray | None] = [None] * panel_rollouts
        self.labels: list[str] = [f"slot {idx + 1}" for idx in range(panel_rollouts)]
        self.rollout_metrics: list[dict[str, object]] = [
            {
                "slot": idx,
                "status": "pending",
            }
            for idx in range(panel_rollouts)
        ]
        self._status = "running"
        self._final_metrics: dict[str, object] | None = None
        self._last_write_at = 0.0
        self._dirty = True
        self._version = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-trace-preview", daemon=True)
        reset_directory(self.live_dir)
        self._thread.start()

    def update_slot(
        self,
        slot: int,
        *,
        frame: np.ndarray,
        rollout_seed: int,
        label: str,
        action_name: str | None,
        reward: float,
        positive_events: float,
        length: int,
        captured_frames: int,
        done: bool,
    ) -> None:
        with self._lock:
            self.frames[slot] = frame.copy()
            self.labels[slot] = label
            self.rollout_metrics[slot] = {
                "slot": slot,
                "rollout_index": slot + 1,
                "rollout_total": self.panel_rollouts,
                "rollout_seed": rollout_seed,
                "status": "complete" if done else "running",
                "label": label,
                "action": action_name,
                "reward": reward,
                "positive_events": positive_events,
                "length": length,
                "captured_frames": captured_frames,
            }
            self._dirty = True
            self._version += 1
        self._wake.set()

    def finish(self, metrics_payload: dict[str, object]) -> None:
        with self._lock:
            self._status = "complete"
            self._final_metrics = metrics_payload
            self._dirty = True
            self._version += 1
        self._wake.set()
        self._thread.join()

    def write_manifest(
        self,
        *,
        status: str,
        refresh_ms: int,
        labels: list[str],
        rollout_metrics: list[dict[str, object]],
        metrics_payload: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "status": status,
            "iteration": self.iteration_index,
            "panel_rollouts": self.panel_rollouts,
            "observed_rollouts": sum(item.get("status") != "pending" for item in rollout_metrics),
            "completed_rollouts": sum(item.get("status") == "complete" for item in rollout_metrics),
            "frame_duration_ms": self.frame_duration_ms,
            "refresh_ms": refresh_ms,
            "current_image": "current.jpg",
            "labels": labels,
            "rollouts": rollout_metrics,
            "updated_at": time.time(),
        }
        if metrics_payload is not None:
            payload["final_metrics"] = metrics_payload
        (self.live_dir / "manifest.json").write_text(json.dumps(payload, indent=2))

    def _snapshot(self) -> tuple[bool, int, list[np.ndarray | None], list[str], list[dict[str, object]], str, dict[str, object] | None]:
        with self._lock:
            return (
                self._dirty,
                self._version,
                [frame.copy() if frame is not None else None for frame in self.frames],
                list(self.labels),
                [dict(item) for item in self.rollout_metrics],
                self._status,
                None if self._final_metrics is None else dict(self._final_metrics),
            )

    def _run(self) -> None:
        while True:
            self._wake.wait(timeout=self.refresh_ms / 1000.0)
            self._wake.clear()
            dirty, version, frames, labels, rollout_metrics, status, final_metrics = self._snapshot()
            should_force = status == "complete"
            if not should_force and not dirty:
                continue
            if not should_force and (not any(frame is not None for frame in frames)) and version == 0:
                continue
            now = time.time()
            if not should_force and now - self._last_write_at < self.refresh_ms / 1000.0:
                continue
            panel = compose_panel_snapshot(frames, labels, preferred_columns=self.panel_columns)
            if panel is not None:
                save_jpg(panel, self.live_dir / "current.jpg", quality=self.jpeg_quality)
            self.write_manifest(
                status=status,
                refresh_ms=self.refresh_ms,
                labels=labels,
                rollout_metrics=rollout_metrics,
                metrics_payload=final_metrics,
            )
            self._last_write_at = time.time()
            with self._lock:
                if self._version == version:
                    self._dirty = False
                has_more_work = self._dirty
                final_status = self._status
            if final_status == "complete" and not has_more_work:
                break


def save_gif(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def save_mp4(frames: list[np.ndarray], path: Path, duration_ms: int) -> None:
    if not frames:
        return
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for MP4 trajectory traces but was not found in PATH.")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fps = max(1.0, 1000.0 / max(duration_ms, 1))
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while writing {path}: {stderr}")


def save_trace_media(frames: list[np.ndarray], path: Path, *, format_name: str, duration_ms: int) -> None:
    if format_name == "mp4":
        save_mp4(frames, path.with_suffix(".mp4"), duration_ms)
    elif format_name == "gif":
        save_gif(frames, path.with_suffix(".gif"), duration_ms)
    else:
        raise ValueError(format_name)


def write_trace_dashboard(
    trace_root: Path,
    *,
    iteration_index: int,
    media_name: str,
    metrics: dict[str, object],
) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="4">
  <title>Trajectory Trace</title>
  <style>
    body {{ background: #0b0f14; color: #edf2f7; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    video {{ width: 100%; border: 1px solid #334155; border-radius: 12px; background: #000; }}
    pre {{ background: #111827; padding: 16px; border-radius: 12px; overflow-x: auto; }}
    a {{ color: #7dd3fc; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Trajectory Trace</h1>
    <p>Iteration {iteration_index:02d}. This page refreshes automatically while training.</p>
    <video controls autoplay muted loop src="{media_name}?v={iteration_index}"></video>
    <p><a href="{media_name}?v={iteration_index}">Open current video</a></p>
    <pre>{metrics_json}</pre>
  </div>
</body>
</html>
"""
    (trace_root / "index.html").write_text(html)


def write_trace_artifacts(
    trace_root: Path,
    *,
    iteration_index: int,
    rollout_traces: list[RolloutTrace],
    rollout_labels: list[str],
    rollout_metrics: list[dict[str, object]],
    format_name: str,
    duration_ms: int,
    panel_columns: int,
    panel_rollouts: int,
) -> None:
    panel_frames = compose_trace_panel(
        rollout_traces,
        rollout_labels,
        preferred_columns=panel_columns,
    )
    iteration_dir = trace_root / f"iter_{iteration_index:02d}"
    media_stem = iteration_dir / "episode"
    save_trace_media(
        panel_frames,
        media_stem,
        format_name=format_name,
        duration_ms=duration_ms,
    )
    iteration_media = media_stem.with_suffix(f".{format_name}")
    metrics_payload = {
        "iteration": iteration_index,
        "panel_rollouts": panel_rollouts,
        "panel_columns": panel_columns,
        "format": format_name,
        "rollouts": rollout_metrics,
    }
    with (iteration_dir / "metrics.json").open("w") as handle:
        json.dump(metrics_payload, handle, indent=2)
    latest_media = trace_root / f"latest.{format_name}"
    latest_media.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(iteration_media, latest_media)
    with (trace_root / "latest_metrics.json").open("w") as handle:
        json.dump(metrics_payload, handle, indent=2)
    write_trace_dashboard(
        trace_root,
        iteration_index=iteration_index,
        media_name=latest_media.name,
        metrics=metrics_payload,
    )


class AsyncTraceWriter:
    def __init__(self, *, workers: int, max_pending_jobs: int):
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
        self.max_pending_jobs = max_pending_jobs
        self.pending: deque[Future[None]] = deque()

    def submit(self, *args, **kwargs) -> None:
        if self.executor is None:
            write_trace_artifacts(*args, **kwargs)
            return
        while len(self.pending) >= self.max_pending_jobs:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(write_trace_artifacts, *args, **kwargs))

    def close(self) -> None:
        while self.pending:
            self.pending.popleft().result()
        if self.executor is not None:
            self.executor.shutdown(wait=True)


def train_fqi(
    batch: TensorBatch,
    num_actions: int,
    train_config: TrainConfig,
    *,
    loss_name: str,
    seed: int,
    device: torch.device,
    iteration_callback=None,
    show_progress: bool = False,
    progress_desc: str = "fqi",
) -> PixelQNetwork:
    torch.manual_seed(seed)
    model = PixelQNetwork(num_actions).to(device)
    np_rng = np.random.default_rng(seed)
    num_samples = batch.observations.shape[0]

    iteration_bar = tqdm(
        range(train_config.fqi_iterations),
        desc=progress_desc,
        leave=False,
        dynamic_ncols=True,
        disable=not progress_enabled(show_progress),
    )
    for iteration_idx in iteration_bar:
        next_values = predict_max_values(
            model,
            batch.next_observations,
            device,
            train_config.target_batch_size,
            batch.cached_on_device,
        ).to(dtype=torch.float32)
        targets = (1.0 - train_config.gamma) * batch.signals + train_config.gamma * (1.0 - batch.dones) * next_values
        targets = targets.clamp_(0.0, 1.0)

        updated = clone_model(model, device)
        optimizer = torch.optim.Adam(
            updated.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        updated.train()
        for _epoch in range(train_config.optimizer_epochs):
            indices = torch.as_tensor(np_rng.permutation(num_samples), dtype=torch.long)
            for start in range(0, num_samples, train_config.batch_size):
                idx_cpu = indices[start : start + train_config.batch_size]
                idx_device = idx_cpu.to(device=device)
                obs_batch = select_observation_batch(
                    batch.observations,
                    idx_cpu,
                    device=device,
                    cached_on_device=batch.cached_on_device,
                )
                action_batch = batch.actions.index_select(0, idx_device)
                target_batch = targets.index_select(0, idx_device)

                logits = updated.forward(obs_batch)
                chosen_logits = logits.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                if loss_name == "log":
                    loss = F.binary_cross_entropy_with_logits(chosen_logits, target_batch)
                elif loss_name == "sq":
                    chosen = torch.sigmoid(chosen_logits)
                    loss = F.mse_loss(chosen, target_batch)
                else:
                    raise ValueError(loss_name)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(updated.parameters(), train_config.grad_clip)
                optimizer.step()
        model = updated
        iteration_bar.set_postfix(iteration=iteration_idx + 1, samples=num_samples)
        if device.type == "mps":
            torch.mps.empty_cache()
        if iteration_callback is not None:
            iteration_callback(model, iteration_idx + 1)
    iteration_bar.close()
    return model


def evaluate_policy(
    game_name: str,
    model: PixelQNetwork,
    *,
    seed: int,
    episodes: int,
    device: torch.device,
    epsilon: float = 0.0,
) -> dict[str, float]:
    rewards = []
    positive_events = []
    lengths = []
    for ep in range(episodes):
        trace = rollout_policy_episode(game_name, model, seed=seed + ep, device=device, epsilon=epsilon)
        rewards.append(trace.total_reward)
        positive_events.append(trace.positive_events)
        lengths.append(trace.length)
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
        rewards = np.asarray([float(item["mean_reward"]) for item in group], dtype=np.float64)
        positive_events = np.asarray([float(item["mean_positive_events"]) for item in group], dtype=np.float64)
        lengths = np.asarray([float(item["mean_length"]) for item in group], dtype=np.float64)
        summary_rows.append(
            {
                "game": game,
                "loss": loss_name,
                "dataset_episodes": dataset_episodes,
                "mean_reward": float(rewards.mean()),
                "stderr_reward": float(rewards.std(ddof=0) / np.sqrt(len(rewards))),
                "mean_positive_events": float(positive_events.mean()),
                "mean_length": float(lengths.mean()),
            }
        )
    return summary_rows


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.trajectory_panel_rollouts < 1:
        raise SystemExit("--trajectory-panel-rollouts must be at least 1.")
    if args.trajectory_panel_columns < 1:
        raise SystemExit("--trajectory-panel-columns must be at least 1.")
    if args.collect_workers < 1:
        raise SystemExit("--collect-workers must be at least 1.")
    if args.trace_write_workers < 0:
        raise SystemExit("--trace-write-workers must be at least 0.")
    if args.trace_write_queue < 1:
        raise SystemExit("--trace-write-queue must be at least 1.")
    if args.live_preview_refresh_ms < 1:
        raise SystemExit("--live-preview-refresh-ms must be at least 1.")
    if not 1 <= args.live_preview_jpeg_quality <= 100:
        raise SystemExit("--live-preview-jpeg-quality must be between 1 and 100.")
    if not 0.0 <= args.eval_epsilon <= 1.0:
        raise SystemExit("--eval-epsilon must be between 0 and 1.")
    if not 0.0 <= args.trace_epsilon <= 1.0:
        raise SystemExit("--trace-epsilon must be between 0 and 1.")
    if args.trace_trajectories and args.trajectory_format == "mp4" and not ffmpeg_available():
        raise SystemExit("Trajectory video export requires ffmpeg in PATH. Install ffmpeg or use --trajectory-format gif.")
    device = auto_device(args.device)
    configure_runtime(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = TrainConfig(
        gamma=args.gamma,
        fqi_iterations=args.fqi_iterations,
        optimizer_epochs=args.optimizer_epochs,
        batch_size=args.batch_size,
        target_batch_size=args.target_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    raw_rows: list[dict[str, object]] = []
    total_jobs = len(args.games) * args.seeds * len(args.dataset_sizes) * 2
    benchmark_bar = tqdm(
        total=total_jobs,
        desc="benchmark",
        dynamic_ncols=True,
        disable=not progress_enabled(args.progress),
    )
    trace_writer = AsyncTraceWriter(
        workers=args.trace_write_workers if args.trace_trajectories else 0,
        max_pending_jobs=args.trace_write_queue,
    )

    try:
        for game in args.games:
            env = gym.make(GAME_CONFIGS[game]["env_id"], obs_type="rgb", frameskip=4, repeat_action_probability=0.0)
            num_actions = env.action_space.n
            env.close()

            for trial in range(args.seeds):
                seed = args.seed + trial
                episodes = collect_episodes(
                    game,
                    seed=seed,
                    count=max(args.dataset_sizes),
                    sticky_prob=args.sticky_prob,
                    workers=args.collect_workers,
                    show_progress=args.progress,
                )
                full_batch = concatenate_episodes(episodes)
                episode_boundaries = np.cumsum([ep.length for ep in episodes], dtype=np.int64)
                for dataset_episodes in args.dataset_sizes:
                    end_index = int(episode_boundaries[dataset_episodes - 1])
                    batch = prefix_batch(full_batch, end_index)
                    tensor_batch = make_tensor_batch(
                        batch,
                        device=device,
                        cache_pixels_on_device=should_cache_pixels_on_device(
                            batch,
                            device,
                            args.device_cache_max_mb * 1024 * 1024,
                        ),
                    )
                    for loss_name in ("log", "sq"):
                        trace_root = (
                            output_dir
                            / "traces"
                            / game
                            / loss_name
                            / f"episodes_{dataset_episodes}"
                            / f"seed_{seed}"
                        )

                        def iteration_callback(
                            current_model: PixelQNetwork,
                            iteration_index: int,
                            *,
                            _trace_root=trace_root,
                            _seed=seed,
                            _game=game,
                        ) -> None:
                            if not args.trace_trajectories:
                                return
                            preview_writer = LiveTracePreviewWriter(
                                _trace_root,
                                iteration_index=iteration_index,
                                panel_rollouts=args.trajectory_panel_rollouts,
                                panel_columns=args.trajectory_panel_columns,
                                frame_duration_ms=args.trajectory_frame_duration_ms,
                                refresh_ms=args.live_preview_refresh_ms,
                                jpeg_quality=args.live_preview_jpeg_quality,
                            )
                            rollout_traces = []
                            rollout_labels = []
                            rollout_metrics = []
                            for offset in range(args.trajectory_panel_rollouts):
                                rollout_seed = _seed + 90_000 + iteration_index * 100 + offset
                                def preview_callback(
                                    current_frame: np.ndarray,
                                    *,
                                    action_name: str | None,
                                    steps: int,
                                    total_reward: float,
                                    positive_events: float,
                                    captured_frames: int,
                                    done: bool,
                                    _slot=offset,
                                    _rollout_seed=rollout_seed,
                                ) -> None:
                                    preview_writer.update_slot(
                                        _slot,
                                        frame=current_frame,
                                        rollout_seed=_rollout_seed,
                                        label=(
                                            f"seed={_rollout_seed} reward={total_reward:.0f} "
                                            f"len={steps} action={action_name or 'start'}"
                                        ),
                                        action_name=action_name,
                                        reward=total_reward,
                                        positive_events=positive_events,
                                        length=steps,
                                        captured_frames=captured_frames,
                                        done=done,
                                    )
                                trace = rollout_policy_episode(
                                    _game,
                                    current_model,
                                    seed=rollout_seed,
                                    device=device,
                                    epsilon=args.trace_epsilon,
                                    max_steps=args.trajectory_max_steps,
                                    capture_frames=True,
                                    frame_stride=args.trajectory_frame_stride,
                                    max_frames=args.trajectory_max_frames,
                                    preview_callback=preview_callback,
                                )
                                rollout_traces.append(trace)
                                rollout_labels.append(
                                    f"seed={rollout_seed} reward={trace.total_reward:.0f} "
                                    f"len={trace.length} action={dominant_action_name(trace.action_names) or 'n/a'}"
                                )
                                rollout_metrics.append(
                                    {
                                        "rollout_index": offset + 1,
                                        "rollout_total": args.trajectory_panel_rollouts,
                                        "rollout_seed": rollout_seed,
                                        "reward": trace.total_reward,
                                        "positive_events": trace.positive_events,
                                        "length": trace.length,
                                        "captured_frames": len(trace.frames),
                                        "dominant_action": dominant_action_name(trace.action_names),
                                    }
                                )
                            metrics_payload = {
                                "iteration": iteration_index,
                                "panel_rollouts": args.trajectory_panel_rollouts,
                                "panel_columns": args.trajectory_panel_columns,
                                "format": args.trajectory_format,
                                "rollouts": rollout_metrics,
                            }
                            preview_writer.finish(metrics_payload)
                            trace_writer.submit(
                                _trace_root,
                                iteration_index=iteration_index,
                                rollout_traces=rollout_traces,
                                rollout_labels=rollout_labels,
                                rollout_metrics=rollout_metrics,
                                format_name=args.trajectory_format,
                                duration_ms=args.trajectory_frame_duration_ms,
                                panel_columns=args.trajectory_panel_columns,
                                panel_rollouts=args.trajectory_panel_rollouts,
                            )

                        model = train_fqi(
                            tensor_batch,
                            num_actions,
                            train_config,
                            loss_name=loss_name,
                            seed=seed + dataset_episodes + (0 if loss_name == "log" else 10_000),
                            device=device,
                            iteration_callback=iteration_callback,
                            show_progress=args.progress,
                            progress_desc=f"{game} {loss_name} n={dataset_episodes}",
                        )
                        metrics = evaluate_policy(
                            game,
                            model,
                            seed=seed + 50_000,
                            episodes=args.eval_episodes,
                            device=device,
                            epsilon=args.eval_epsilon,
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
                        benchmark_bar.update(1)
                        benchmark_bar.set_postfix(
                            game=game,
                            loss=loss_name,
                            n=dataset_episodes,
                            cache="dev" if tensor_batch.cached_on_device else "cpu",
                        )
    finally:
        trace_writer.close()
        benchmark_bar.close()
    summary_rows = aggregate(raw_rows)
    save_csv(raw_rows, output_dir / "raw_results.csv")
    save_csv(summary_rows, output_dir / "summary_results.csv")
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "games": args.games,
                "dataset_sizes": args.dataset_sizes,
                "device": str(device),
                "trace_trajectories": args.trace_trajectories,
                "trajectory_format": args.trajectory_format,
                "trajectory_panel_rollouts": args.trajectory_panel_rollouts,
                "collect_workers": args.collect_workers,
                "trace_write_workers": args.trace_write_workers,
                "device_cache_max_mb": args.device_cache_max_mb,
                "eval_epsilon": args.eval_epsilon,
                "trace_epsilon": args.trace_epsilon,
                "train_config": train_config.__dict__,
                "summary_rows": summary_rows,
            },
            handle,
            indent=2,
        )
    return {"summary_rows": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FQI-LOG vs FQI-SQ on Atari pixels.")
    parser.add_argument("--games", nargs="+", default=["Asterix"])
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--sticky-prob", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fqi-iterations", type=int, default=8)
    parser.add_argument("--optimizer-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--trace-epsilon", type=float, default=0.10)
    parser.add_argument("--trace-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--trajectory-max-steps", type=int, default=300)
    parser.add_argument("--trajectory-frame-stride", type=int, default=4)
    parser.add_argument("--trajectory-max-frames", type=int, default=80)
    parser.add_argument("--trajectory-frame-duration-ms", type=int, default=80)
    parser.add_argument("--trajectory-panel-rollouts", type=int, default=2)
    parser.add_argument("--trajectory-panel-columns", type=int, default=2)
    parser.add_argument("--live-preview-refresh-ms", type=int, default=250)
    parser.add_argument("--live-preview-jpeg-quality", type=int, default=65)
    parser.add_argument("--collect-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--trace-write-workers", type=int, default=1)
    parser.add_argument("--trace-write-queue", type=int, default=2)
    parser.add_argument("--device-cache-max-mb", type=int, default=768)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="results/atari_pixel_fqi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(args)
    print(json.dumps(payload["summary_rows"], indent=2))


if __name__ == "__main__":
    main()
