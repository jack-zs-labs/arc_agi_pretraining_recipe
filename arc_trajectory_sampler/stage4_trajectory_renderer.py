from __future__ import annotations

"""Render real Stage 4 trajectory variants as a 2D storyboard."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    from .stage1_latent_sampler import sample_latent_rule
    from .stage2_episode_sampler import sample_episode
    from .stage4_trajectory_dataset import build_trajectories
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import sample_latent_rule  # type: ignore
    from stage2_episode_sampler import sample_episode  # type: ignore
    from stage4_trajectory_dataset import build_trajectories  # type: ignore


ARC_COLORS = [
    "#111111",
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#f1c40f",
    "#7f7f7f",
    "#e377c2",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
]
ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), ARC_CMAP.N)
DIFF_COLORS = [
    "#161616",  # match
    "#ff9f1c",  # extra cell in current
    "#3dd5f3",  # missing cell from current
    "#ff4d6d",  # wrong color
]
DIFF_CMAP = ListedColormap(DIFF_COLORS)
DIFF_NORM = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), DIFF_CMAP.N)

VARIANT_PRIORITY = {
    "canonical": 0,
    "swap_pick_order": 1,
    "swap_bind_match": 2,
    "one_step_short": 90,
    "wrong_final_state": 91,
}
EDGE_DEFAULT = "#4b4b4b"
EDGE_REORDER = "#f4d35e"
EDGE_SUCCESS = "#80ed99"
EDGE_FAILURE = "#ff6b6b"
EDGE_MISSING = "#9e9e9e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 4 trajectory variants for one ARC example.")
    parser.add_argument("--seed", type=int, default=9, help="Seed used when sampling an episode.")
    parser.add_argument(
        "--example-id",
        type=str,
        default="train_0",
        help="Example id to render when sampling or reading a JSONL file.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Optional JSONL file of trajectory records. If omitted, a fresh episode is sampled.",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=None,
        help="Optional parent trajectory id to render from a JSONL file or sampled episode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/stage4_trajectory_variants.png",
        help="Destination PNG path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def sampled_records(seed: int) -> List[Dict[str, Any]]:
    latent = sample_latent_rule(seed=seed)
    episode = sample_episode(latent, seed=seed)
    return [record.to_jsonable() for record in build_trajectories(episode)]


def record_group_id(record: Dict[str, Any]) -> str:
    parent = record.get("parent_trajectory_id")
    if isinstance(parent, str) and parent:
        return parent
    return str(record["trajectory_id"])


def variant_sort_key(record: Dict[str, Any]) -> Tuple[int, int, str]:
    role_rank = 0 if record.get("trajectory_role") == "positive" else 1
    variant_kind = str(record.get("variant_kind", "canonical"))
    return (role_rank, VARIANT_PRIORITY.get(variant_kind, 50), variant_kind)


def select_group(
    records: Sequence[Dict[str, Any]],
    *,
    example_id: Optional[str],
    group_id: Optional[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        gid = record_group_id(record)
        grouped.setdefault(gid, []).append(record)

    if group_id is not None:
        if group_id not in grouped:
            raise ValueError(f"group id {group_id!r} not found")
        return sorted(grouped[group_id], key=variant_sort_key)

    candidate_groups = list(grouped.items())
    if example_id is not None:
        candidate_groups = [
            (gid, group)
            for gid, group in candidate_groups
            if str(group[0].get("example", {}).get("example_id")) == example_id
        ]
        if not candidate_groups:
            raise ValueError(f"example id {example_id!r} not found")

    candidate_groups.sort(key=lambda item: item[0])
    return sorted(candidate_groups[0][1], key=variant_sort_key)


def step_grid(step: Dict[str, Any]) -> Optional[np.ndarray]:
    grid = step.get("workspace_grid")
    if grid is None:
        grid = step.get("workspace_state", {}).get("workspace_grid")
    if grid is None:
        return None
    return np.asarray(grid, dtype=np.uint8)


def draw_grid(ax: plt.Axes, grid: Optional[np.ndarray], *, title: str, subtitle: str = "") -> None:
    if grid is None:
        ax.axis("off")
        return
    height, width = grid.shape
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#2f2f2f", linewidth=0.45, alpha=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor(ARC_COLORS[0])
    ax.set_title(title, fontsize=9, pad=4)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=7, labelpad=2)


def style_axis_border(ax: plt.Axes, *, color: str, linewidth: float = 2.2, linestyle: str = "-") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)
        spine.set_linestyle(linestyle)


def draw_placeholder(ax: plt.Axes, *, title: str, subtitle: str = "", edge_color: str = EDGE_MISSING) -> None:
    ax.set_facecolor("#202020")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(0.5, 0.58, title, ha="center", va="center", color="#f0f0f0", fontsize=10, weight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.38, subtitle, ha="center", va="center", color="#cfcfcf", fontsize=8, transform=ax.transAxes)
    style_axis_border(ax, color=edge_color, linewidth=2.0, linestyle="--")


def compact_trace(trace: Sequence[str]) -> str:
    return " -> ".join(trace)


def mismatch_cells(first: Sequence[Sequence[int]], second: Sequence[Sequence[int]]) -> int:
    height = max(len(first), len(second))
    width = max(len(first[0]) if first else 0, len(second[0]) if second else 0)
    mismatches = 0
    for row in range(height):
        for col in range(width):
            lhs = first[row][col] if row < len(first) and col < len(first[row]) else 0
            rhs = second[row][col] if row < len(second) and col < len(second[row]) else 0
            if lhs != rhs:
                mismatches += 1
    return mismatches


def diff_grid(first: Sequence[Sequence[int]], second: Sequence[Sequence[int]]) -> np.ndarray:
    height = max(len(first), len(second), 1)
    width = max(len(first[0]) if first else 0, len(second[0]) if second else 0, 1)
    diff = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            lhs = first[row][col] if row < len(first) and col < len(first[row]) else 0
            rhs = second[row][col] if row < len(second) and col < len(second[row]) else 0
            if lhs == rhs:
                continue
            if lhs != 0 and rhs == 0:
                diff[row, col] = 1
            elif lhs == 0 and rhs != 0:
                diff[row, col] = 2
            else:
                diff[row, col] = 3
    return diff


def diff_counts(first: Sequence[Sequence[int]], second: Sequence[Sequence[int]]) -> Dict[str, int]:
    grid = diff_grid(first, second)
    return {
        "extra": int(np.sum(grid == 1)),
        "missing": int(np.sum(grid == 2)),
        "wrong_color": int(np.sum(grid == 3)),
    }


def reordered_steps(record: Dict[str, Any]) -> List[str]:
    canonical = list(record.get("canonical_trace_template") or record["trace_template"])
    current = list(record["trace_template"])
    moved: List[str] = []
    for index, step_name in enumerate(current):
        if index >= len(canonical) or canonical[index] != step_name:
            moved.append(step_name)
    return moved


def meta_lines(record: Dict[str, Any]) -> List[str]:
    final_step = record["steps"][-1]
    verifier = final_step["verifier"]
    terminal_grid = final_step.get("workspace_grid") or final_step.get("workspace_state", {}).get("workspace_grid") or []
    diffs = diff_counts(terminal_grid, record["output_grid"])
    mismatch_count = sum(diffs.values())
    moved = reordered_steps(record)
    status = "exact / stop" if verifier["should_stop"] else "keep going"
    lines = [
        f"{record['variant_kind']} ({record['trajectory_role']})",
        f"{record['split']} / {record['example']['example_id']}",
        f"trace: {compact_trace(record['trace_template'])}",
        f"status: {status}",
        f"grid cells off: {mismatch_count}",
        f"diff: +{diffs['extra']} -{diffs['missing']} recolor={diffs['wrong_color']}",
        f"next: {verifier['next_subgoal']}",
        f"reordered: {', '.join(moved) if moved else 'none'}",
        f"reason: {verifier['non_terminal_reason'] or 'none'}",
    ]
    return lines


def draw_meta(ax: plt.Axes, record: Dict[str, Any]) -> None:
    ax.axis("off")
    face = "#1d3b2a" if record.get("trajectory_role") == "positive" else "#4a2323"
    ax.set_facecolor(face)
    lines = meta_lines(record)
    y = 0.98
    for index, line in enumerate(lines):
        ax.text(
            0.03,
            y,
            line,
            transform=ax.transAxes,
            fontsize=9 if index == 0 else 8.2,
            family="monospace" if index > 1 else None,
            weight="bold" if index == 0 else None,
            color="#f5f5f5",
            ha="left",
            va="top",
        )
        y -= 0.11 if index == 0 else 0.092
    style_axis_border(ax, color="#d9d9d9", linewidth=1.4)


def draw_header_meta(ax: plt.Axes, records: Sequence[Dict[str, Any]]) -> None:
    ax.axis("off")
    ax.set_facecolor("#202020")
    canonical = next((record for record in records if record.get("variant_kind") == "canonical"), records[0])
    positive_count = sum(1 for record in records if record.get("trajectory_role") == "positive")
    negative_count = sum(1 for record in records if record.get("trajectory_role") == "negative")
    lines = [
        f"{canonical['family']} / {canonical['example']['example_id']}",
        f"canonical trace: {compact_trace(canonical['canonical_trace_template'] or canonical['trace_template'])}",
        f"variants: {len(records)} total ({positive_count} positive, {negative_count} negative)",
        "rows show executed workspaces only",
        "shared input/target are in this header row",
    ]
    y = 0.94
    for index, line in enumerate(lines):
        ax.text(
            0.04,
            y,
            line,
            transform=ax.transAxes,
            fontsize=10 if index == 0 else 8.5,
            color="#f5f5f5",
            ha="left",
            va="top",
            family="monospace" if index > 0 else None,
            weight="bold" if index == 0 else None,
        )
        y -= 0.16 if index == 0 else 0.13
    style_axis_border(ax, color="#d9d9d9", linewidth=1.4)


def step_border_style(record: Dict[str, Any], step_index: int, step: Dict[str, Any]) -> Tuple[str, float, str]:
    canonical = list(record.get("canonical_trace_template") or record["trace_template"])
    is_reordered = step_index < len(canonical) and canonical[step_index] != step["name"]
    is_final = step_index == len(record["steps"]) - 1
    verifier = step["verifier"]
    if is_final and verifier["should_stop"]:
        return (EDGE_SUCCESS, 2.6, "-")
    if is_final and not verifier["should_stop"]:
        return (EDGE_FAILURE, 2.6, "-")
    if is_reordered:
        return (EDGE_REORDER, 2.4, "-")
    return (EDGE_DEFAULT, 1.4, "-")


def draw_diff_grid(
    ax: plt.Axes,
    current: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    *,
    title: str = "diff",
) -> None:
    diff = diff_grid(current, target)
    counts = diff_counts(current, target)
    height, width = diff.shape
    ax.imshow(diff, cmap=DIFF_CMAP, norm=DIFF_NORM, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#2f2f2f", linewidth=0.45, alpha=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor(DIFF_COLORS[0])
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(
        f"+{counts['extra']}  -{counts['missing']}  recolor {counts['wrong_color']}",
        fontsize=7,
        labelpad=2,
    )
    border = EDGE_SUCCESS if sum(counts.values()) == 0 else EDGE_FAILURE
    style_axis_border(ax, color=border, linewidth=2.2)


def render_variant_storyboard(records: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    if not records:
        raise ValueError("no records to render")

    max_steps = max(len(record["steps"]) for record in records)
    rows = len(records)
    cols = max_steps + 2
    width_ratios = [1.9] + [1.0] * max_steps + [1.0]
    fig = plt.figure(figsize=(2.9 * cols, 2.45 * rows + 2.0))
    gs = fig.add_gridspec(
        rows + 1,
        cols,
        width_ratios=width_ratios,
        height_ratios=[1.05] + [1.0] * rows,
    )
    fig.patch.set_facecolor("#161616")

    example = records[0]["example"]["example_id"]
    family = records[0]["family"]
    fig.suptitle(
        f"Stage 4 Trajectory Variants: {family} / {example}",
        fontsize=15,
        color="#f5f5f5",
        y=0.995,
    )
    fig.text(
        0.5,
        0.968,
        "Green = exact terminal, gold = reordered step, red = wrong final state, dashed = missing subgoal, diff colors: orange extra / cyan missing / pink wrong color",
        ha="center",
        va="top",
        color="#d7d7d7",
        fontsize=9,
    )

    header_meta_ax = fig.add_subplot(gs[0, 0])
    draw_header_meta(header_meta_ax, records)

    shared_input = np.asarray(records[0]["input_grid"], dtype=np.uint8)
    shared_target = np.asarray(records[0]["output_grid"], dtype=np.uint8)
    header_input_ax = fig.add_subplot(gs[0, 1 : 1 + max(1, max_steps // 2)])
    draw_grid(
        header_input_ax,
        shared_input,
        title="shared input",
        subtitle=f"{shared_input.shape[0]}x{shared_input.shape[1]}",
    )
    style_axis_border(header_input_ax, color="#6c757d", linewidth=1.8)

    header_target_ax = fig.add_subplot(gs[0, 1 + max(1, max_steps // 2) : cols])
    draw_grid(
        header_target_ax,
        shared_target,
        title="shared target",
        subtitle=f"{shared_target.shape[0]}x{shared_target.shape[1]}",
    )
    style_axis_border(header_target_ax, color="#6c757d", linewidth=1.8)

    for row_index, record in enumerate(records):
        grid_row = row_index + 1
        meta_ax = fig.add_subplot(gs[grid_row, 0])
        draw_meta(meta_ax, record)

        canonical = list(record.get("canonical_trace_template") or record["trace_template"])
        for step_index in range(max_steps):
            ax = fig.add_subplot(gs[grid_row, 1 + step_index])
            if step_index >= len(record["steps"]):
                missing_name = canonical[step_index] if step_index < len(canonical) else "not executed"
                draw_placeholder(ax, title=f"missing", subtitle=missing_name)
                continue
            step = record["steps"][step_index]
            verifier = step["verifier"]
            title = f"{step_index + 1}. {step['name']}"
            subtitle = f"p={step['progress']:.2f} stop={int(step['stop_target'])}"
            if step_index == len(record["steps"]) - 1:
                subtitle += f"\nmatch={verifier['grid_match']} off={mismatch_cells(step.get('workspace_grid') or [], record['output_grid'])}"
            draw_grid(ax, step_grid(step), title=title, subtitle=subtitle)
            border_color, border_width, border_style = step_border_style(record, step_index, step)
            style_axis_border(ax, color=border_color, linewidth=border_width, linestyle=border_style)

        final_workspace = record["steps"][-1].get("workspace_grid") or record["steps"][-1].get("workspace_state", {}).get("workspace_grid") or []
        diff_ax = fig.add_subplot(gs[grid_row, -1])
        draw_diff_grid(diff_ax, final_workspace, record["output_grid"], title="diff")

    fig.subplots_adjust(left=0.02, right=0.995, top=0.94, bottom=0.03, wspace=0.16, hspace=0.34)
    ensure_parent(output_path)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.input_jsonl:
        records = load_records(Path(args.input_jsonl))
    else:
        records = sampled_records(args.seed)
    selected = select_group(records, example_id=args.example_id, group_id=args.group_id)
    output_path = render_variant_storyboard(selected, Path(args.output))
    print(f"Wrote Stage 4 trajectory render to {output_path}")


if __name__ == "__main__":
    main()
