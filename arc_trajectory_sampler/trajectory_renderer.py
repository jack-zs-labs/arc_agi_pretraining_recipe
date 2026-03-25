from __future__ import annotations

"""Render projected Stage 2 ARC episodes as trajectory storyboards.

The current sampler emits object-scene blueprints rather than exact cell-level
trajectories. This renderer visualizes those blueprints as a 2D storyboard:

1. full input scene projection
2. selected-object highlight projection
3. selected-object focus projection
4. metadata panel

This keeps the Stage 2 / Stage 3 boundary intact while still making the episode
structure easy to inspect. The same layout can later be fed exact trajectory
states once Stage 4 exists.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

try:
    from .stage1_latent_sampler import TaskLatent, sample_latent_rule
    from .stage2_episode_sampler import EpisodeSpec, ExampleSpec, ObjectSpec, SceneSpec, sample_episode
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import TaskLatent, sample_latent_rule  # type: ignore
    from stage2_episode_sampler import EpisodeSpec, ExampleSpec, ObjectSpec, SceneSpec, sample_episode  # type: ignore


ARC_COLORS = {
    0: "#111111",
    1: "#1f77b4",
    2: "#d62728",
    3: "#2ca02c",
    4: "#f1c40f",
    5: "#7f7f7f",
    6: "#e377c2",
    7: "#ff7f0e",
    8: "#17becf",
    9: "#8c564b",
}

EDGE_COLOR = "#f5f5f5"
DIM_ALPHA = 0.28


@dataclass(frozen=True)
class ProjectionFrame:
    title: str
    scene: SceneSpec
    selected_ids: Tuple[str, ...]
    caption: str = ""
    dim_unselected: bool = False
    show_context: bool = True


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def color_for(value: int) -> str:
    return ARC_COLORS.get(value, "#cccccc")


def normalize_objects(objects: Sequence[ObjectSpec]) -> Tuple[Tuple[ObjectSpec, ...], int, int]:
    if not objects:
        return (tuple(), 1, 1)
    min_top = min(obj.top for obj in objects)
    min_left = min(obj.left for obj in objects)
    max_bottom = max(obj.bottom for obj in objects)
    max_right = max(obj.right for obj in objects)
    normalized = tuple(
        ObjectSpec(
            object_id=obj.object_id,
            shape=obj.shape,
            color=obj.color,
            top=obj.top - min_top,
            left=obj.left - min_left,
            height=obj.height,
            width=obj.width,
            mass=obj.mass,
            orientation=obj.orientation,
            holes=obj.holes,
            is_container=obj.is_container,
            tags=obj.tags,
            attributes=obj.attributes,
        )
        for obj in objects
    )
    return (normalized, max_bottom - min_top + 1, max_right - min_left + 1)


def selected_focus_scene(example: ExampleSpec) -> SceneSpec:
    selected = [obj for obj in example.input_scene.objects if obj.object_id in set(example.selected_object_ids)]
    normalized, height, width = normalize_objects(selected)
    return SceneSpec(
        height=height,
        width=width,
        background_color=0,
        border_color=None,
        outline_color=None,
        marker_position=None,
        objects=normalized,
        legend=tuple(),
        attributes={"focus_only": True},
    )


def example_to_frames(example: ExampleSpec) -> Tuple[ProjectionFrame, ...]:
    return (
        ProjectionFrame(
            title="Input",
            scene=example.input_scene,
            selected_ids=tuple(),
            caption=example.example_id,
            dim_unselected=False,
        ),
        ProjectionFrame(
            title="Selected",
            scene=example.input_scene,
            selected_ids=example.selected_object_ids,
            caption=f"{len(example.selected_object_ids)} selected",
            dim_unselected=True,
        ),
        ProjectionFrame(
            title="Focus",
            scene=selected_focus_scene(example),
            selected_ids=example.selected_object_ids,
            caption="normalized selected cluster",
            dim_unselected=False,
            show_context=False,
        ),
    )


def draw_holes(ax: plt.Axes, obj: ObjectSpec) -> None:
    if obj.holes <= 0:
        return
    hole_color = "#ffffff"
    if obj.holes == 1:
        hole_w = max(0.4, obj.width * 0.34)
        hole_h = max(0.4, obj.height * 0.34)
        hole_x = obj.left + (obj.width - hole_w) / 2
        hole_y = obj.top + (obj.height - hole_h) / 2
        ax.add_patch(Rectangle((hole_x, hole_y), hole_w, hole_h, facecolor=hole_color, edgecolor="none", zorder=4))
        return
    gap = max(0.1, obj.width * 0.08)
    hole_w = max(0.3, (obj.width - 3 * gap) / 2)
    hole_h = max(0.35, obj.height * 0.32)
    hole_y = obj.top + (obj.height - hole_h) / 2
    hole_x1 = obj.left + gap
    hole_x2 = obj.left + obj.width - gap - hole_w
    for hole_x in (hole_x1, hole_x2):
        ax.add_patch(Rectangle((hole_x, hole_y), hole_w, hole_h, facecolor=hole_color, edgecolor="none", zorder=4))


def draw_shape(ax: plt.Axes, obj: ObjectSpec, *, alpha: float, highlight: Optional[str]) -> None:
    face = color_for(obj.color)
    edge = highlight or EDGE_COLOR
    linewidth = 2.6 if highlight else 1.0

    if obj.is_container or "frame" in obj.tags:
        ax.add_patch(
            Rectangle(
                (obj.left, obj.top),
                obj.width,
                obj.height,
                facecolor="none",
                edgecolor=edge,
                linewidth=max(linewidth, 2.0),
                alpha=1.0,
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle(
                (obj.left + 0.08, obj.top + 0.08),
                max(0.2, obj.width - 0.16),
                max(0.2, obj.height - 0.16),
                facecolor=face,
                edgecolor="none",
                alpha=0.18 * alpha,
                zorder=1,
            )
        )
        return

    if obj.shape == "l_shape":
        arm = max(0.35, min(obj.width, obj.height) * 0.34)
        ax.add_patch(Rectangle((obj.left, obj.top), arm, obj.height, facecolor=face, edgecolor=edge, linewidth=linewidth, alpha=alpha, zorder=3))
        ax.add_patch(Rectangle((obj.left, obj.top + obj.height - arm), obj.width, arm, facecolor=face, edgecolor=edge, linewidth=linewidth, alpha=alpha, zorder=3))
        draw_holes(ax, obj)
        return

    if obj.shape == "cross":
        bar_w = max(0.35, obj.width * 0.34)
        bar_h = max(0.35, obj.height * 0.34)
        x = obj.left + (obj.width - bar_w) / 2
        y = obj.top + (obj.height - bar_h) / 2
        ax.add_patch(Rectangle((x, obj.top), bar_w, obj.height, facecolor=face, edgecolor=edge, linewidth=linewidth, alpha=alpha, zorder=3))
        ax.add_patch(Rectangle((obj.left, y), obj.width, bar_h, facecolor=face, edgecolor=edge, linewidth=linewidth, alpha=alpha, zorder=3))
        draw_holes(ax, obj)
        return

    if obj.shape == "dot":
        ax.add_patch(
            Rectangle(
                (obj.left + 0.12, obj.top + 0.12),
                max(0.55, obj.width - 0.24),
                max(0.55, obj.height - 0.24),
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=alpha,
                zorder=3,
            )
        )
        return

    ax.add_patch(
        Rectangle(
            (obj.left, obj.top),
            obj.width,
            obj.height,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            zorder=3,
        )
    )
    draw_holes(ax, obj)


def highlight_color_for(obj: ObjectSpec, example: ExampleSpec) -> Optional[str]:
    metadata = example.metadata
    source_ids = set(metadata.get("source_ids", ()))
    target_ids = set(metadata.get("target_ids", ()))
    if obj.object_id in source_ids:
        return "#80ed99"
    if obj.object_id in target_ids:
        return "#ffd166"
    if obj.object_id in set(example.selected_object_ids):
        return "#f8f9fa"
    return None


def draw_scene(ax: plt.Axes, frame: ProjectionFrame, example: ExampleSpec) -> None:
    scene = frame.scene
    ax.set_xlim(0, scene.width)
    ax.set_ylim(scene.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(scene.width + 1))
    ax.set_yticks(range(scene.height + 1))
    ax.grid(color="#2e2e2e", linewidth=0.55, alpha=0.4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor(color_for(scene.background_color))

    if frame.show_context and scene.border_color is not None:
        ax.add_patch(
            Rectangle(
                (0, 0),
                scene.width,
                scene.height,
                fill=False,
                edgecolor=color_for(scene.border_color),
                linewidth=4.0,
                zorder=5,
            )
        )

    objects = sorted(scene.objects, key=lambda obj: (0 if (obj.is_container or "frame" in obj.tags) else 1, obj.mass))
    selected_set = set(frame.selected_ids)
    for obj in objects:
        is_selected = obj.object_id in selected_set
        alpha = 1.0 if (is_selected or not frame.dim_unselected) else DIM_ALPHA
        highlight = highlight_color_for(obj, example) if is_selected else None
        draw_shape(ax, obj, alpha=alpha, highlight=highlight)
        label_color = "#111111" if obj.color in (4, 8) else "#f8f9fa"
        ax.text(
            obj.left + 0.08,
            obj.top + 0.35,
            obj.object_id,
            color=label_color if alpha >= 0.5 else "#d0d0d0",
            fontsize=7.5,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "#00000088", "edgecolor": "none"},
            zorder=6,
        )

    if frame.show_context and scene.marker_position is not None:
        marker_center = {
            "top_left": (0.5, 0.5),
            "top_right": (scene.width - 0.5, 0.5),
            "bottom_left": (0.5, scene.height - 0.5),
            "bottom_right": (scene.width - 0.5, scene.height - 0.5),
        }[scene.marker_position]
        ax.add_patch(Circle(marker_center, radius=0.32, facecolor="#ffffff", edgecolor="#111111", linewidth=1.2, zorder=7))

    ax.set_title(f"{frame.title}\n{frame.caption}", fontsize=10, pad=6)


def compact_json(value: Any, *, max_len: int = 110) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def example_text_lines(example: ExampleSpec) -> List[str]:
    lines = [
        f"id: {example.example_id}",
        f"selected: {', '.join(example.selected_object_ids) if example.selected_object_ids else 'none'}",
    ]
    if example.notes:
        lines.append(f"notes: {'; '.join(example.notes)}")

    key_order = [
        "branch",
        "cue_kind",
        "cue_value",
        "action",
        "relation_meta",
        "source_ids",
        "target_ids",
        "group_by",
        "winner_key",
        "reducer",
        "post_action",
        "key_type",
        "value_type",
        "apply_mode",
        "legend_keys",
        "mapping",
    ]
    seen: set[str] = set()
    for key in key_order:
        if key in example.metadata:
            lines.append(f"{key}: {compact_json(example.metadata[key])}")
            seen.add(key)
    for key, value in sorted(example.metadata.items()):
        if key in seen or key == "diversity_token":
            continue
        lines.append(f"{key}: {compact_json(value)}")
    return lines


def format_program_summary(latent: TaskLatent) -> str:
    program = latent.to_jsonable()["program"]
    return compact_json(program, max_len=180)


def format_bindings(bindings: Dict[str, Any]) -> str:
    pieces = [f"{key}={value}" for key, value in sorted(bindings.items())]
    return ", ".join(pieces)


def render_metadata_panel(ax: plt.Axes, example: ExampleSpec, label: str) -> None:
    ax.axis("off")
    lines = [label, ""] + example_text_lines(example)
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )


def render_episode_storyboard(episode: EpisodeSpec, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_parent(output_path)

    examples = [(f"Train {index}", example) for index, example in enumerate(episode.train_examples)]
    examples.append(("Test", episode.test_example))
    rows = len(examples) + 1
    fig = plt.figure(figsize=(16, max(5.2, 3.15 * rows)), constrained_layout=False)
    grid = fig.add_gridspec(rows, 4, height_ratios=[0.9] + [2.2] * len(examples), width_ratios=[1.0, 1.0, 0.95, 1.2])

    header_ax = fig.add_subplot(grid[0, :])
    header_ax.axis("off")
    trace = " -> ".join(step.name for step in episode.latent.trace_template)
    header_lines = [
        f"Family: {episode.latent.family.value}    Difficulty: {episode.latent.difficulty}    Attempts: {episode.sampling_attempts}",
        f"Concept tags: {', '.join(episode.latent.concept_tags)}",
        f"Program: {format_program_summary(episode.latent)}",
        f"Trace template: {trace}",
        f"Role bindings: {format_bindings(episode.role_bindings)}",
    ]
    header_ax.text(0.0, 1.0, "\n".join(header_lines), ha="left", va="top", fontsize=10, family="monospace")

    for row_index, (label, example) in enumerate(examples, start=1):
        frames = example_to_frames(example)
        for col_index, frame in enumerate(frames):
            ax = fig.add_subplot(grid[row_index, col_index])
            draw_scene(ax, frame, example)
        meta_ax = fig.add_subplot(grid[row_index, 3])
        render_metadata_panel(meta_ax, example, label)

    fig.suptitle("ARC Stage 2 Projected Trajectory Storyboard", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def sample_and_render_episode(
    *,
    seed: int,
    output_path: str | Path,
    num_train: int = 3,
    max_attempts: int = 256,
) -> Path:
    latent = sample_latent_rule(seed=seed)
    episode = sample_episode(latent, seed=seed, num_train=num_train, max_attempts=max_attempts)
    return render_episode_storyboard(episode, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a projected ARC Stage 2 trajectory storyboard.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for latent and episode sampling.")
    parser.add_argument("--num-train", type=int, default=3, help="Number of train examples in the sampled episode.")
    parser.add_argument("--max-attempts", type=int, default=256, help="Reject-sampling budget for episode generation.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path. Defaults to arc_trajectory_sampler/results/trajectory_seed_<seed>.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else Path("arc_trajectory_sampler/results") / f"trajectory_seed_{args.seed}.png"
    result = sample_and_render_episode(
        seed=args.seed,
        output_path=output,
        num_train=args.num_train,
        max_attempts=args.max_attempts,
    )
    print(f"Wrote trajectory storyboard to {result}")


if __name__ == "__main__":
    main()
