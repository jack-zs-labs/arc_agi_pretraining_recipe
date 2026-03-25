from __future__ import annotations

"""Render plots and a Markdown brief from sampler evaluation JSON."""

from pathlib import Path
import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def top_items(mapping: Dict[str, Any], n: int) -> List[Tuple[str, Any]]:
    return list(mapping.items())[:n]


def sorted_numeric_items(mapping: Dict[str, Any], n: Optional[int] = None) -> List[Tuple[str, float]]:
    items = [(key, float(value)) for key, value in mapping.items()]
    items.sort(key=lambda item: item[1], reverse=True)
    if n is not None:
        return items[:n]
    return items


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def plot_bar_chart(
    items: Sequence[Tuple[str, float]],
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str,
    xlim: Optional[Tuple[float, float]] = None,
) -> Optional[Path]:
    if plt is None or not items:
        return None

    ensure_dir(output_path.parent)
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    height = max(3.2, 0.45 * len(items) + 1.4)
    fig, ax = plt.subplots(figsize=(9, height))
    positions = range(len(items))
    ax.barh(list(positions), values, color=color)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    for pos, value in enumerate(values):
        label = f"{value:.4f}" if isinstance(value, float) and value <= 1.0 else f"{value:.2f}"
        ax.text(value, pos, f" {label}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_summary(summary: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    success_rates = summary["coverage"].get("success_rate_by_family", {})
    success_plot = plot_bar_chart(
        sorted_numeric_items(success_rates),
        title="Success Rate by Family",
        xlabel="Success rate",
        output_path=output_dir / "success_rate_by_family.png",
        color="#2b8a3e",
        xlim=(0.0, 1.0),
    )
    if success_plot is not None:
        plots["success_rate_by_family"] = success_plot.name

    attempts = {
        family: metrics["mean_sampling_attempts"]
        for family, metrics in summary.get("per_family", {}).items()
        if metrics.get("mean_sampling_attempts") is not None
    }
    attempts_plot = plot_bar_chart(
        sorted_numeric_items(attempts),
        title="Mean Sampling Attempts by Family",
        xlabel="Mean attempts",
        output_path=output_dir / "mean_attempts_by_family.png",
        color="#c77d00",
    )
    if attempts_plot is not None:
        plots["mean_attempts_by_family"] = attempts_plot.name

    rejections = summary.get("near_miss_rejections", {})
    rejection_plot = plot_bar_chart(
        sorted_numeric_items(rejections, n=8),
        title="Top Near-Miss Rejections",
        xlabel="Count",
        output_path=output_dir / "top_near_miss_rejections.png",
        color="#d94841",
    )
    if rejection_plot is not None:
        plots["top_near_miss_rejections"] = rejection_plot.name

    leakage = {
        target: probe["lift"]
        for target, probe in summary.get("leakage_probes", {}).items()
        if probe.get("lift") is not None
    }
    leakage_plot = plot_bar_chart(
        sorted_numeric_items(leakage, n=8),
        title="Strongest Leakage Probe Lifts",
        xlabel="Accuracy lift over majority baseline",
        output_path=output_dir / "top_leakage_lifts.png",
        color="#4263eb",
    )
    if leakage_plot is not None:
        plots["top_leakage_lifts"] = leakage_plot.name

    return plots


def findings(summary: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    overall = summary.get("overall", {})
    success_rate = overall.get("success_rate")
    mean_attempts = overall.get("mean_sampling_attempts")
    hard_failures = summary.get("hard_failures", {})
    leakage = summary.get("leakage_probes", {})
    family_rates = summary.get("coverage", {}).get("success_rate_by_family", {})

    if success_rate is not None and success_rate < 0.95:
        notes.append(f"Overall success rate is {fmt(success_rate)}, below a 0.95 reliability target.")
    if mean_attempts is not None and mean_attempts > 10:
        notes.append(f"Mean sampling attempts is {fmt(mean_attempts)}, which suggests reject-sampling is still expensive.")
    if hard_failures:
        top_failure = next(iter(hard_failures.items()))
        notes.append(f"Hard failures are still present; most frequent: `{top_failure[0]}` ({top_failure[1]} cases).")
    weak_families = [family for family, rate in family_rates.items() if rate < 0.9]
    if weak_families:
        notes.append("Lowest-yield families: " + ", ".join(weak_families) + ".")
    strong_leak = [(target, probe) for target, probe in leakage.items() if probe.get("lift", 0.0) >= 0.15]
    if strong_leak:
        target, probe = max(strong_leak, key=lambda item: item[1]["lift"])
        notes.append(
            f"Potential shortcut leakage: `{target}` is predictable from `{probe['feature']}` "
            f"with lift {fmt(probe['lift'])}."
        )
    if not notes:
        notes.append("No obvious red flags in this summary; the remaining work is mostly about broadening coverage and tightening costs.")
    return notes


def render_overview(summary: Dict[str, Any]) -> List[str]:
    overall = summary["overall"]
    return [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Requested samples | {overall['requested_samples']} |",
        f"| Successful episodes | {overall['successful_episodes']} |",
        f"| Failed episodes | {overall['failed_episodes']} |",
        f"| Success rate | {fmt(overall['success_rate'])} |",
        f"| Mean sampling attempts | {fmt(overall['mean_sampling_attempts'])} |",
        f"| P95 sampling attempts | {fmt(overall['p95_sampling_attempts'])} |",
        f"| Max sampling attempts | {fmt(overall['max_sampling_attempts'])} |",
    ]


def render_family_table(summary: Dict[str, Any]) -> List[str]:
    per_family = summary.get("per_family", {})
    family_rates = summary.get("coverage", {}).get("success_rate_by_family", {})
    lines = [
        "| Family | Success Rate | Mean Attempts | P95 Attempts | Grid Varies | Object Count Varies |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in sorted(per_family):
        metrics = per_family[family]
        lines.append(
            f"| {family} | {fmt(family_rates.get(family))} | {fmt(metrics.get('mean_sampling_attempts'))} | "
            f"{fmt(metrics.get('p95_sampling_attempts'))} | {fmt(metrics.get('share_train_grid_size_varies'))} | "
            f"{fmt(metrics.get('share_train_object_count_varies'))} |"
        )
    return lines


def render_diversity_table(summary: Dict[str, Any]) -> List[str]:
    diversity = summary.get("diversity", {})
    return [
        "| Diversity Metric | Value |",
        "| --- | ---: |",
        f"| Share train grid size varies | {fmt(diversity.get('share_train_grid_size_varies'))} |",
        f"| Share train object count varies | {fmt(diversity.get('share_train_object_count_varies'))} |",
        f"| Share train selected count varies | {fmt(diversity.get('share_train_selected_count_varies'))} |",
        f"| Mean train diversity token count | {fmt(diversity.get('mean_train_diversity_token_count'))} |",
        f"| Mean train color union count | {fmt(diversity.get('mean_train_color_union_count'))} |",
        f"| Mean train shape union count | {fmt(diversity.get('mean_train_shape_union_count'))} |",
        f"| Mean train selected fraction | {fmt(diversity.get('mean_train_selected_fraction'))} |",
    ]


def render_top_mapping(mapping: Dict[str, Any], *, limit: int, header: str, value_label: str) -> List[str]:
    lines = [
        f"| {header} | {value_label} |",
        "| --- | ---: |",
    ]
    for key, value in top_items(mapping, limit):
        lines.append(f"| `{key}` | {fmt(value)} |")
    if len(lines) == 2:
        lines.append("| none | 0 |")
    return lines


def render_leakage_table(summary: Dict[str, Any]) -> List[str]:
    probes = summary.get("leakage_probes", {})
    lines = [
        "| Target | Best Feature | Accuracy | Baseline | Lift |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for target, probe in sorted(probes.items(), key=lambda item: item[1].get("lift", 0.0), reverse=True)[:10]:
        lines.append(
            f"| `{target}` | `{probe['feature']}` | {fmt(probe.get('accuracy'))} | "
            f"{fmt(probe.get('baseline_accuracy'))} | {fmt(probe.get('lift'))} |"
        )
    if len(lines) == 2:
        lines.append("| none | n/a | n/a | n/a | n/a |")
    return lines


def render_plot_links(plots: Dict[str, str]) -> List[str]:
    if not plots:
        return ["Plot generation skipped because `matplotlib` is unavailable."]
    lines: List[str] = []
    for title, filename in plots.items():
        label = title.replace("_", " ").title()
        lines.append(f"- [{label}]({filename})")
    return lines


def build_report(summary: Dict[str, Any], plots: Dict[str, str]) -> str:
    lines = [
        "# ARC Trajectory Sampler Evaluation Report",
        "",
        "## Findings",
    ]
    for note in findings(summary):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Overview",
            "",
            *render_overview(summary),
            "",
            "## Family Summary",
            "",
            *render_family_table(summary),
            "",
            "## Diversity",
            "",
            *render_diversity_table(summary),
            "",
            "## Top Near-Miss Rejections",
            "",
            *render_top_mapping(summary.get("near_miss_rejections", {}), limit=10, header="Rejection", value_label="Count"),
            "",
            "## Hard Failures",
            "",
            *render_top_mapping(summary.get("hard_failures", {}), limit=10, header="Failure", value_label="Count"),
            "",
            "## Leakage Probes",
            "",
            *render_leakage_table(summary),
            "",
            "## Plots",
            "",
            *render_plot_links(plots),
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Markdown report from sampler evaluation JSON.")
    parser.add_argument("--input", required=True, help="Path to evaluation summary JSON.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for the Markdown report and plots. Defaults to <input_dir>/report.",
    )
    parser.add_argument(
        "--report-name",
        default="report.md",
        help="Markdown filename to write inside the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    summary = load_summary(input_path)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "report"
    ensure_dir(output_dir)

    plots = plot_summary(summary, output_dir)
    report = build_report(summary, plots)
    report_path = output_dir / args.report_name
    write_text(report_path, report)

    print(f"Wrote report to {report_path}")
    if plots:
        print(json.dumps({"plots": plots}, indent=2))
    else:
        print("Plot generation skipped")


if __name__ == "__main__":
    main()
