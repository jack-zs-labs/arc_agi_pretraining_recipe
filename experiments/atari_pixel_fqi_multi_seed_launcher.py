from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import atari_pixel_fqi_benchmark as benchmark
from tqdm.auto import tqdm


LOSSES = ("log", "sq")


@dataclass
class WorkerState:
    seed: int
    output_dir: Path
    log_path: Path
    command: list[str]
    status: str = "queued"
    process: subprocess.Popen[str] | None = None
    log_handle: object | None = None
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def runtime_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end_time = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end_time - self.started_at)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Launch Atari pixel FQI workers in parallel across seeds.")
    parser.add_argument("--games", nargs="+", default=["Asterix"])
    parser.add_argument("--dataset-sizes", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--trace-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--output-dir", default="results/atari_pixel_fqi_parallel")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    parser.add_argument("--log-tail-lines", type=int, default=12)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-script", default=str(Path(__file__).with_name("atari_pixel_fqi_benchmark.py")))
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "queued"
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {remainder:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {remainder:02d}s"
    return f"{remainder:d}s"


def status_badge(status: str) -> str:
    palette = {
        "queued": "#475569",
        "running": "#0f766e",
        "completed": "#166534",
        "failed": "#991b1b",
    }
    color = palette.get(status, "#475569")
    return f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{color};color:#f8fafc;font-size:12px;text-transform:uppercase;letter-spacing:0.04em">{html.escape(status)}</span>'


def tail_lines(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def read_json_file(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def format_value(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def load_summary_rows(path: Path) -> list[dict[str, object]]:
    payload = read_json_file(path)
    if not payload:
        return []
    rows = payload.get("summary_rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def render_summary_table(title: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    grouped: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        game = str(row.get("game", "?"))
        dataset_episodes = int(row.get("dataset_episodes", 0))
        loss_name = str(row.get("loss", "?"))
        grouped.setdefault((game, dataset_episodes), {})[loss_name] = row
    table_rows = []
    for (game, dataset_episodes), losses in sorted(grouped.items()):
        log_row = losses.get("log")
        sq_row = losses.get("sq")
        log_reward = float(log_row["mean_reward"]) if log_row else None
        sq_reward = float(sq_row["mean_reward"]) if sq_row else None
        delta_reward = (log_reward - sq_reward) if (log_reward is not None and sq_reward is not None) else None
        delta_class = ""
        if delta_reward is not None:
            delta_class = "delta-up" if delta_reward > 0 else "delta-down" if delta_reward < 0 else "delta-flat"
        table_rows.append(
            f"""
<tr>
  <td>{html.escape(game)}</td>
  <td>{dataset_episodes}</td>
  <td>{html.escape(format_value(log_reward))}</td>
  <td>{html.escape(format_value(sq_reward))}</td>
  <td class="{delta_class}">{html.escape(format_value(delta_reward))}</td>
</tr>
"""
        )
    return f"""
<section class="summary-table-card">
  <div class="section-head">
    <h3>{html.escape(title)}</h3>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Game</th>
          <th>Episodes</th>
          <th>Log Reward</th>
          <th>Square Reward</th>
          <th>Delta</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</section>
"""


def build_worker_command(
    *,
    python_executable: str,
    worker_script: Path,
    games: list[str],
    dataset_sizes: list[int],
    seed: int,
    device: str,
    trace_trajectories: bool,
    trajectory_format: str,
    output_dir: Path,
    passthrough: list[str],
) -> list[str]:
    command = [
        python_executable,
        str(worker_script),
        "--games",
        *games,
        "--dataset-sizes",
        *[str(size) for size in dataset_sizes],
        "--seeds",
        "1",
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        "--trajectory-format",
        trajectory_format,
    ]
    command.append("--trace-trajectories" if trace_trajectories else "--no-trace-trajectories")
    command.extend(passthrough)
    command.append("--no-progress")
    return command


def relative_to_root(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def render_metric_chips(metrics_payload: dict[str, object] | None) -> str:
    if not metrics_payload:
        return '<div class="chip-row"><span class="chip chip-muted">waiting for metrics</span></div>'
    chips = []
    iteration = metrics_payload.get("iteration")
    if isinstance(iteration, int):
        chips.append(f'<span class="chip chip-strong">iter {iteration}</span>')
    rollouts = metrics_payload.get("rollouts")
    panel_rollouts = metrics_payload.get("panel_rollouts")
    observed_rollouts = metrics_payload.get("observed_rollouts")
    completed_rollouts = metrics_payload.get("completed_rollouts")
    if isinstance(panel_rollouts, (int, float)):
        current_rollouts = completed_rollouts if isinstance(completed_rollouts, (int, float)) else observed_rollouts
        if not isinstance(current_rollouts, (int, float)) and isinstance(rollouts, list):
            current_rollouts = len(rollouts)
        if isinstance(current_rollouts, (int, float)):
            chips.append(f'<span class="chip chip-strong">traj {int(current_rollouts)}/{int(panel_rollouts)}</span>')
    if isinstance(rollouts, list):
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                continue
            seed = rollout.get("rollout_seed")
            slot = rollout.get("slot")
            rollout_index = rollout.get("rollout_index")
            rollout_total = rollout.get("rollout_total")
            status = rollout.get("status")
            reward = rollout.get("reward")
            length = rollout.get("length")
            captured_frames = rollout.get("captured_frames")
            action_name = rollout.get("action") or rollout.get("dominant_action")
            traj_prefix = ""
            if isinstance(rollout_index, (int, float)) and isinstance(rollout_total, (int, float)):
                traj_prefix = f"t{int(rollout_index)}/{int(rollout_total)} "
            if isinstance(seed, (int, float)):
                identity = f"{traj_prefix}s{int(seed)}".strip()
            elif isinstance(slot, (int, float)):
                identity = f"{traj_prefix}slot {int(slot) + 1}".strip()
            else:
                identity = traj_prefix.strip() or "rollout"
            status_prefix = f"{status} " if isinstance(status, str) else ""
            frame_suffix = ""
            if isinstance(captured_frames, (int, float)):
                frame_suffix = f" f{format_value(captured_frames)}"
            action_suffix = f" a{action_name}" if isinstance(action_name, str) and action_name else ""
            chips.append(
                f'<span class="chip">{html.escape(identity)} {html.escape(status_prefix)}r{html.escape(format_value(reward if isinstance(reward, (int, float)) else None))} l{html.escape(format_value(length if isinstance(length, (int, float)) else None))}{html.escape(frame_suffix)}{html.escape(action_suffix)}</span>'
            )
    return f'<div class="chip-row">{"".join(chips)}</div>' if chips else '<div class="chip-row"><span class="chip chip-muted">metrics pending</span></div>'


def render_live_preview(
    *,
    rel_current_image: str,
    refresh_ms: int,
) -> str:
    return (
        '<div class="live-shell">'
        '<div class="live-badge">LIVE</div>'
        f'<img class="live-refresh" data-live-src="{html.escape(rel_current_image)}" data-refresh-ms="{refresh_ms}" src="{html.escape(rel_current_image)}?live=0" alt="live trajectory preview">'
        '</div>'
    )


def render_trace_card(
    *,
    root_dir: Path,
    worker: WorkerState,
    game: str,
    dataset_size: int,
    loss_name: str,
    trajectory_format: str,
    refresh_token: int,
) -> str:
    trace_dir = worker.output_dir / "traces" / game / loss_name / f"episodes_{dataset_size}" / f"seed_{worker.seed}"
    media_path = trace_dir / f"latest.{trajectory_format}"
    dashboard_path = trace_dir / "index.html"
    metrics_path = trace_dir / "latest_metrics.json"
    live_manifest_path = trace_dir / "live" / "manifest.json"
    live_current = trace_dir / "live" / "current.jpg"
    live_manifest = read_json_file(live_manifest_path)
    final_live_metrics = None
    if isinstance(live_manifest, dict):
        candidate = live_manifest.get("final_metrics")
        if isinstance(candidate, dict):
            final_live_metrics = candidate
    live_status = live_manifest.get("status") if isinstance(live_manifest, dict) else None
    live_active = worker.status == "running" and live_current.exists() and live_manifest and live_status != "complete"
    metrics_payload = live_manifest if live_active else final_live_metrics or read_json_file(metrics_path) or live_manifest
    title = loss_name.upper()
    if live_active:
        rel_live_current = relative_to_root(root_dir, live_current)
        rel_dashboard = relative_to_root(root_dir, dashboard_path) if dashboard_path.exists() else rel_live_current
        rel_metrics = relative_to_root(root_dir, live_manifest_path)
        refresh_ms = int(live_manifest.get("refresh_ms", 250)) if isinstance(live_manifest.get("refresh_ms"), (int, float)) else 250
        return f"""
<article class="trace-card">
  <div class="trace-card-head">
    <h4>{html.escape(title)}</h4>
    <span class="trace-card-subtle">seed {worker.seed}</span>
  </div>
  {render_live_preview(rel_current_image=rel_live_current, refresh_ms=max(1, refresh_ms))}
  {render_metric_chips(metrics_payload)}
  <p><a href="{html.escape(rel_dashboard)}">trace view</a> | <a href="{html.escape(rel_metrics)}">live manifest</a></p>
</article>
"""
    if media_path.exists():
        rel_media = relative_to_root(root_dir, media_path)
        rel_dashboard = relative_to_root(root_dir, dashboard_path) if dashboard_path.exists() else rel_media
        rel_metrics = relative_to_root(root_dir, metrics_path) if metrics_path.exists() else rel_media
        media_tag = (
            f'<video controls autoplay muted loop src="{html.escape(rel_media)}?v={refresh_token}"></video>'
            if trajectory_format == "mp4"
            else f'<img alt="{html.escape(title)}" src="{html.escape(rel_media)}?v={refresh_token}">'
        )
        return f"""
<article class="trace-card">
  <div class="trace-card-head">
    <h4>{html.escape(title)}</h4>
    <span class="trace-card-subtle">seed {worker.seed}</span>
  </div>
  {media_tag}
  {render_metric_chips(metrics_payload)}
  <p><a href="{html.escape(rel_dashboard)}">trace view</a> | <a href="{html.escape(rel_metrics)}">metrics</a></p>
</article>
"""
    return f"""
<article class="trace-card trace-missing">
  <div class="trace-card-head">
    <h4>{html.escape(title)}</h4>
    <span class="trace-card-subtle">seed {worker.seed}</span>
  </div>
  <p>waiting for first trajectory...</p>
</article>
"""


def render_dashboard(
    *,
    output_dir: Path,
    workers: list[WorkerState],
    games: list[str],
    dataset_sizes: list[int],
    trajectory_format: str,
    refresh_seconds: float,
    log_tail_lines: int,
    updated_at: float,
) -> None:
    completed = sum(worker.status == "completed" for worker in workers)
    failed = sum(worker.status == "failed" for worker in workers)
    running = sum(worker.status == "running" for worker in workers)
    queued = sum(worker.status == "queued" for worker in workers)
    refresh_token = int(updated_at)
    aggregate_summary_html = render_summary_table(
        "Aggregate Results",
        load_summary_rows(output_dir / "summary.json"),
    )
    worker_sections = []
    for worker in workers:
        rel_log = relative_to_root(output_dir, worker.log_path)
        rel_worker_dir = relative_to_root(output_dir, worker.output_dir)
        summary_html = render_summary_table(
            f"Seed {worker.seed} Results",
            load_summary_rows(worker.output_dir / "summary.json"),
        )
        dataset_sections = []
        for game in games:
            for dataset_size in dataset_sizes:
                trace_cards = []
                for loss_name in LOSSES:
                    trace_cards.append(
                        render_trace_card(
                            root_dir=output_dir,
                            worker=worker,
                            game=game,
                            dataset_size=dataset_size,
                            loss_name=loss_name,
                            trajectory_format=trajectory_format,
                            refresh_token=refresh_token,
                        )
                    )
                dataset_sections.append(
                    f"""
<section class="dataset-card">
  <div class="section-head">
    <h3>{html.escape(game)}</h3>
    <span class="dataset-pill">{dataset_size} episodes</span>
  </div>
  <div class="trace-grid">
    {''.join(trace_cards)}
  </div>
</section>
"""
                )
        log_tail = html.escape(tail_lines(worker.log_path, log_tail_lines))
        worker_sections.append(
            f"""
<section class="worker-card">
  <div class="worker-header">
    <div class="worker-title-block">
      <div class="eyebrow">worker</div>
      <h2>Seed {worker.seed}</h2>
      <p class="worker-meta">{status_badge(worker.status)} runtime: {html.escape(format_seconds(worker.runtime_seconds()))}</p>
    </div>
    <div class="worker-links">
      <a href="{html.escape(rel_worker_dir)}">worker dir</a>
      <a href="{html.escape(rel_log)}">worker.log</a>
    </div>
  </div>
  {summary_html}
  <div class="dataset-stack">
    {''.join(dataset_sections)}
  </div>
  <details class="log-details">
    <summary>Recent log tail</summary>
    <pre>{log_tail}</pre>
  </details>
</section>
"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{max(1, int(refresh_seconds))}">
  <title>Parallel Atari FQI Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1016;
      --bg-2: #121b24;
      --panel: rgba(245, 240, 229, 0.08);
      --panel-strong: rgba(245, 240, 229, 0.12);
      --text: #f7f4ed;
      --muted: #c2baa9;
      --accent: #ffb347;
      --accent-2: #79d2de;
      --border: rgba(255, 244, 224, 0.12);
      --good: #66d18f;
      --bad: #ff8a80;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 15% 20%, rgba(255, 179, 71, 0.16), transparent 32%),
        radial-gradient(circle at 85% 10%, rgba(121, 210, 222, 0.16), transparent 28%),
        linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 48%, #06090d 100%);
    }}
    main {{ max-width: 1850px; margin: 0 auto; padding: 28px 22px 48px; }}
    h1, h2, h3, h4, p {{ margin: 0; }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .hero {{
      position: sticky;
      top: 0;
      z-index: 20;
      margin-bottom: 22px;
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: rgba(11, 16, 22, 0.82);
      backdrop-filter: blur(14px);
      box-shadow: 0 28px 70px rgba(0, 0, 0, 0.32);
    }}
    .hero-top {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; }}
    .hero h1 {{ font-size: clamp(28px, 4vw, 42px); letter-spacing: -0.04em; }}
    .meta {{ color: var(--muted); margin-top: 8px; max-width: 70ch; }}
    .hero-links {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .hero-links a {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--text);
      background: rgba(255, 255, 255, 0.04);
    }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .summary-card, .worker-card, .summary-table-card, .dataset-card {{
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--panel);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22);
    }}
    .summary-card {{
      padding: 14px 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    }}
    .summary-card span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .summary-card strong {{ display: block; font-size: 30px; margin-top: 8px; letter-spacing: -0.03em; }}
    .worker-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; align-items: start; }}
    .worker-card {{ padding: 18px; }}
    .worker-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 14px; }}
    .eyebrow {{ color: var(--accent); font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }}
    .worker-title-block h2 {{ font-size: 30px; letter-spacing: -0.04em; }}
    .worker-meta {{ color: var(--muted); margin-top: 10px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .worker-links {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: end; }}
    .worker-links a {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.04);
      color: var(--text);
      white-space: nowrap;
    }}
    .dataset-stack {{ display: grid; gap: 14px; margin-top: 14px; }}
    .dataset-card {{ padding: 14px; background: rgba(6, 9, 13, 0.38); }}
    .section-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .section-head h3 {{ font-size: 18px; letter-spacing: -0.03em; }}
    .dataset-pill {{
      border-radius: 999px;
      padding: 7px 11px;
      background: rgba(255, 179, 71, 0.16);
      color: #ffd69a;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .trace-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }}
    .trace-card {{
      background: rgba(255,255,255,0.035);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 12px;
    }}
    .trace-card-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 10px; }}
    .trace-card h4 {{ font-size: 14px; letter-spacing: 0.12em; text-transform: uppercase; }}
    .trace-card-subtle {{ font-size: 12px; color: var(--muted); }}
    .trace-card p {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
    .trace-card video, .trace-card img {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      border: 1px solid rgba(255,255,255,0.08);
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: cover;
    }}
    .live-shell {{ position: relative; }}
    .live-badge {{
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 2;
      border-radius: 999px;
      padding: 5px 9px;
      background: rgba(255, 83, 83, 0.92);
      color: #fff4f4;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      box-shadow: 0 10px 24px rgba(255, 83, 83, 0.28);
    }}
    .trace-missing {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      min-height: 220px;
      border-style: dashed;
      color: var(--muted);
    }}
    .chip-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
    .chip {{
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.06);
      color: var(--text);
      font-size: 12px;
    }}
    .chip-strong {{ background: rgba(121, 210, 222, 0.18); color: #c8f3f7; }}
    .chip-muted {{ color: var(--muted); }}
    .summary-table-card {{
      padding: 14px;
      margin-top: 16px;
      background: rgba(255,255,255,0.03);
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 13px; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .delta-up {{ color: var(--good); }}
    .delta-down {{ color: var(--bad); }}
    .delta-flat {{ color: var(--muted); }}
    .log-details {{
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(6, 9, 13, 0.4);
      overflow: hidden;
    }}
    .log-details summary {{
      cursor: pointer;
      list-style: none;
      padding: 12px 14px;
      font-weight: 600;
      color: var(--muted);
    }}
    .log-details summary::-webkit-details-marker {{ display: none; }}
    pre {{
      margin: 0;
      padding: 14px;
      background: rgba(2, 6, 23, 0.72);
      border-top: 1px solid rgba(255,255,255,0.06);
      overflow-x: auto;
      color: #d7ddce;
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .hero-top {{ flex-direction: column; align-items: start; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .worker-list {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      .hero {{ position: static; }}
      .summary {{ grid-template-columns: 1fr; }}
      .trace-grid {{ grid-template-columns: 1fr; }}
      .worker-header {{ flex-direction: column; }}
      .worker-links {{ justify-content: start; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1>Parallel Atari FQI Dashboard</h1>
          <p class="meta">Updated {html.escape(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_at)))}. The page refreshes automatically while workers run, and each worker card is grouped by game and dataset size instead of dumping every artifact in one grid.</p>
        </div>
        <div class="hero-links">
          <a href="summary_results.csv">summary csv</a>
          <a href="summary.json">summary json</a>
          <a href="launcher_status.json">launcher status</a>
        </div>
      </div>
      <div class="summary">
        <div class="summary-card"><span>Running</span><strong>{running}</strong></div>
        <div class="summary-card"><span>Queued</span><strong>{queued}</strong></div>
        <div class="summary-card"><span>Completed</span><strong>{completed}</strong></div>
        <div class="summary-card"><span>Failed</span><strong>{failed}</strong></div>
      </div>
      {aggregate_summary_html}
    </section>
    <section class="worker-list">
      {''.join(worker_sections)}
    </section>
  </main>
</body>
<script>
  document.querySelectorAll('.live-refresh').forEach((image) => {{
    const baseSrc = image.dataset.liveSrc;
    const refreshMs = Number.parseInt(image.dataset.refreshMs || '250', 10);
    if (!baseSrc || !Number.isFinite(refreshMs) || refreshMs <= 0) {{
      return;
    }}
    let tick = 0;
    window.setInterval(() => {{
      tick += 1;
      image.src = `${{baseSrc}}?live=${{Date.now()}}_${{tick}}`;
    }}, refreshMs);
  }});
</script>
</html>
"""
    (output_dir / "index.html").write_text(html_text)


def write_status_json(output_dir: Path, workers: list[WorkerState]) -> None:
    payload = {
        "workers": [
            {
                "seed": worker.seed,
                "status": worker.status,
                "exit_code": worker.exit_code,
                "runtime_seconds": worker.runtime_seconds(),
                "output_dir": str(worker.output_dir),
                "log_path": str(worker.log_path),
            }
            for worker in workers
        ]
    }
    (output_dir / "launcher_status.json").write_text(json.dumps(payload, indent=2))


def aggregate_completed_results(output_dir: Path, workers: list[WorkerState]) -> None:
    raw_rows: list[dict[str, object]] = []
    for worker in workers:
        raw_csv = worker.output_dir / "raw_results.csv"
        if not raw_csv.exists():
            continue
        with raw_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_rows.append(
                    {
                        "game": row["game"],
                        "seed": int(row["seed"]),
                        "dataset_episodes": int(row["dataset_episodes"]),
                        "loss": row["loss"],
                        "mean_reward": float(row["mean_reward"]),
                        "mean_positive_events": float(row["mean_positive_events"]),
                        "mean_length": float(row["mean_length"]),
                    }
                )
    if not raw_rows:
        return
    summary_rows = benchmark.aggregate(raw_rows)
    benchmark.save_csv(raw_rows, output_dir / "raw_results.csv")
    benchmark.save_csv(summary_rows, output_dir / "summary_results.csv")
    payload = {
        "workers": [worker.seed for worker in workers],
        "summary_rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def start_worker(worker: WorkerState) -> None:
    worker.output_dir.mkdir(parents=True, exist_ok=True)
    worker.log_handle = worker.log_path.open("w")
    worker.process = subprocess.Popen(
        worker.command,
        stdout=worker.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    worker.status = "running"
    worker.started_at = time.time()


def close_worker_log(worker: WorkerState) -> None:
    if worker.log_handle is not None:
        worker.log_handle.close()
        worker.log_handle = None


def update_worker_states(workers: list[WorkerState]) -> None:
    for worker in workers:
        if worker.process is None:
            continue
        exit_code = worker.process.poll()
        if exit_code is None:
            continue
        worker.exit_code = exit_code
        worker.finished_at = time.time()
        worker.status = "completed" if exit_code == 0 else "failed"
        close_worker_log(worker)
        worker.process = None


def main() -> None:
    args, passthrough = parse_args()
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be at least 1.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_script = Path(args.worker_script)
    seed_values = [args.seed + idx for idx in range(args.seeds)]
    workers = [
        WorkerState(
            seed=seed,
            output_dir=output_dir / "seed_runs" / f"seed_{seed}",
            log_path=output_dir / "seed_runs" / f"seed_{seed}" / "worker.log",
            command=build_worker_command(
                python_executable=sys.executable,
                worker_script=worker_script,
                games=args.games,
                dataset_sizes=args.dataset_sizes,
                seed=seed,
                device=args.device,
                trace_trajectories=args.trace_trajectories,
                trajectory_format=args.trajectory_format,
                output_dir=output_dir / "seed_runs" / f"seed_{seed}",
                passthrough=passthrough,
            ),
        )
        for seed in seed_values
    ]
    progress_bar = tqdm(
        total=len(workers),
        desc="seed workers",
        dynamic_ncols=True,
        disable=not (args.progress and sys.stderr.isatty()),
    )

    try:
        while True:
            update_worker_states(workers)
            running_count = sum(worker.status == "running" for worker in workers)
            queued_count = sum(worker.status == "queued" for worker in workers)
            completed_count = sum(worker.status in {"completed", "failed"} for worker in workers)
            progress_bar.n = completed_count
            progress_bar.set_postfix(running=running_count, queued=queued_count)
            progress_bar.refresh()
            for worker in workers:
                if running_count >= args.max_parallel:
                    break
                if worker.status == "queued":
                    start_worker(worker)
                    running_count += 1

            write_status_json(output_dir, workers)
            aggregate_completed_results(output_dir, workers)
            render_dashboard(
                output_dir=output_dir,
                workers=workers,
                games=args.games,
                dataset_sizes=args.dataset_sizes,
                trajectory_format=args.trajectory_format,
                refresh_seconds=args.refresh_seconds,
                log_tail_lines=args.log_tail_lines,
                updated_at=time.time(),
            )

            if all(worker.status in {"completed", "failed"} for worker in workers):
                break
            time.sleep(args.refresh_seconds)
    finally:
        for worker in workers:
            if worker.process is not None and worker.process.poll() is None:
                worker.process.terminate()
                try:
                    worker.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    worker.process.wait()
            close_worker_log(worker)
        progress_bar.close()

    aggregate_completed_results(output_dir, workers)
    write_status_json(output_dir, workers)
    render_dashboard(
        output_dir=output_dir,
        workers=workers,
        games=args.games,
        dataset_sizes=args.dataset_sizes,
        trajectory_format=args.trajectory_format,
        refresh_seconds=args.refresh_seconds,
        log_tail_lines=args.log_tail_lines,
        updated_at=time.time(),
    )

    if any(worker.status == "failed" for worker in workers):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
