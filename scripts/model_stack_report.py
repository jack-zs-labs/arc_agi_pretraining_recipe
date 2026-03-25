from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-stack inference and training sweeps, then generate combined tradeoff reports."
    )
    parser.add_argument("--output-dir", type=str, default="results/model_stack_report")
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 384])
    parser.add_argument("--mla-latent-kv-dims", nargs="+", type=int, default=[16, 24, 32, 48, 64])
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--sliding-window", type=int, default=64)
    parser.add_argument("--global-token-stride", type=int, default=8)
    parser.add_argument("--sink-token-count", type=int, default=2)
    parser.add_argument("--scale-invariant-tau", type=float, default=10.0)
    parser.add_argument(
        "--scale-invariant-last-n-layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional sweep of suffix-layer counts for SIA backends. Omit to use the full decoder depth.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timing-runs", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            sys.stderr.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_latent_kv_dim(value: object) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def normalize_scale_invariant_last_n_layers(value: object) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def config_key(
    *,
    backend: str,
    length: int,
    latent_kv_dim: object,
    scale_invariant_last_n_layers: object,
) -> tuple[str, int, int | None, int | None]:
    return (
        backend,
        length,
        normalize_latent_kv_dim(latent_kv_dim),
        normalize_scale_invariant_last_n_layers(scale_invariant_last_n_layers),
    )


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def summarize_tradeoffs(
    inference_results: dict[str, object],
    training_results: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    inference_runs = inference_results["runs"]
    training_runs = training_results["runs"]
    inference_by_key = {
        config_key(
            backend=run["backend"],
            length=int(run["prompt_length"]),
            latent_kv_dim=run["latent_kv_dim"],
            scale_invariant_last_n_layers=run.get("scale_invariant_last_n_layers"),
        ): run
        for run in inference_runs
    }
    raw_rows: list[dict[str, object]] = []
    grouped_training: dict[tuple[str, int, int | None, int | None], list[dict[str, object]]] = {}
    for run in training_runs:
        key = config_key(
            backend=run["backend"],
            length=int(run["seq_len"]),
            latent_kv_dim=run["latent_kv_dim"],
            scale_invariant_last_n_layers=run.get("scale_invariant_last_n_layers"),
        )
        inference_run = inference_by_key.get(key)
        if inference_run is None:
            raise KeyError(f"Missing inference run for config {key}.")
        row = {
            "backend": run["backend"],
            "length": int(run["seq_len"]),
            "latent_kv_dim": normalize_latent_kv_dim(run["latent_kv_dim"]) or "",
            "scale_invariant_last_n_layers": (
                normalize_scale_invariant_last_n_layers(run.get("scale_invariant_last_n_layers")) or ""
            ),
            "seed": int(run["seed"]),
            "initial_val_loss": float(run["initial_val_loss"]),
            "final_val_loss": float(run["final_val_loss"]),
            "train_tokens_per_second": float(run["train_tokens_per_second"]),
            "train_elapsed_seconds": float(run["train_elapsed_seconds"]),
            "parameter_count": int(run["parameter_count"]),
            "train_cache_numel_for_probe": int(run["cache_numel_for_probe"]),
            "inference_cache_bytes_total": int(inference_run["cache"]["bytes_total"]),
            "inference_cache_numel_total": int(inference_run["cache"]["numel_total"]),
            "forward_latency_median_ms": float(inference_run["forward_latency"]["median_ms"]),
            "generate_latency_median_ms": float(inference_run["generate_latency"]["median_ms"]),
            "forward_tokens_per_second_median": float(inference_run["forward_tokens_per_second_median"]),
            "generate_new_tokens_per_second_median": float(inference_run["generate_new_tokens_per_second_median"]),
        }
        raw_rows.append(row)
        grouped_training.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for key, rows in sorted(
        grouped_training.items(),
        key=lambda item: (item[0][1], item[0][0], item[0][2] or -1, item[0][3] or -1),
    ):
        backend, length, latent_kv_dim, scale_invariant_last_n_layers = key
        initial_losses = [float(row["initial_val_loss"]) for row in rows]
        final_losses = [float(row["final_val_loss"]) for row in rows]
        train_tps = [float(row["train_tokens_per_second"]) for row in rows]
        row0 = rows[0]
        summary_rows.append(
            {
                "backend": backend,
                "length": length,
                "latent_kv_dim": latent_kv_dim if latent_kv_dim is not None else "",
                "scale_invariant_last_n_layers": (
                    scale_invariant_last_n_layers if scale_invariant_last_n_layers is not None else ""
                ),
                "seed_count": len(rows),
                "initial_val_loss_mean": mean(initial_losses),
                "initial_val_loss_std": stddev(initial_losses),
                "final_val_loss_mean": mean(final_losses),
                "final_val_loss_std": stddev(final_losses),
                "train_tokens_per_second_mean": mean(train_tps),
                "train_tokens_per_second_std": stddev(train_tps),
                "parameter_count": int(row0["parameter_count"]),
                "train_cache_numel_for_probe": int(row0["train_cache_numel_for_probe"]),
                "inference_cache_bytes_total": int(row0["inference_cache_bytes_total"]),
                "inference_cache_numel_total": int(row0["inference_cache_numel_total"]),
                "forward_latency_median_ms": float(row0["forward_latency_median_ms"]),
                "generate_latency_median_ms": float(row0["generate_latency_median_ms"]),
                "forward_tokens_per_second_median": float(row0["forward_tokens_per_second_median"]),
                "generate_new_tokens_per_second_median": float(row0["generate_new_tokens_per_second_median"]),
            }
        )

    best_by_length: list[dict[str, object]] = []
    for length in sorted({int(row["length"]) for row in summary_rows}):
        candidates = [row for row in summary_rows if int(row["length"]) == length]
        best = min(candidates, key=lambda row: (float(row["final_val_loss_mean"]), float(row["inference_cache_bytes_total"])))
        best_by_length.append(best)

    summary = {
        "lengths": sorted({int(row["length"]) for row in summary_rows}),
        "seed_count": len(training_results["seeds"]),
        "best_by_length": best_by_length,
        "global_best": min(
            summary_rows,
            key=lambda row: (float(row["final_val_loss_mean"]), float(row["inference_cache_bytes_total"])),
        ),
    }
    return raw_rows, summary_rows, summary


def label_for_row(row: dict[str, object]) -> str:
    base = str(row["backend"])
    if row["backend"] in {"mla", "mla_sia"}:
        base = f"{base}-{row['latent_kv_dim']}"
    layer_count = row.get("scale_invariant_last_n_layers")
    if layer_count not in ("", None):
        base = f"{base}-L{layer_count}"
    return base


def plot_tradeoff(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    lengths = sorted({int(row["length"]) for row in summary_rows})
    cols = min(2, len(lengths))
    rows = math.ceil(len(lengths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    all_losses = [float(row["final_val_loss_mean"]) for row in summary_rows]
    vmin = min(all_losses)
    vmax = max(all_losses)
    markers = {"sdpa": "o", "hybrid": "s", "sia": "D", "sia_hybrid": "P", "mla": "^", "mla_sia": "X"}
    scatter = None
    for index, length in enumerate(lengths):
        axis = axes[index // cols][index % cols]
        axis.set_title(f"Length {length}")
        axis.set_xlabel("Inference cache bytes")
        axis.set_ylabel("Forward latency median (ms)")
        axis.grid(alpha=0.2, linestyle=":")
        for row in [item for item in summary_rows if int(item["length"]) == length]:
            scatter = axis.scatter(
                float(row["inference_cache_bytes_total"]),
                float(row["forward_latency_median_ms"]),
                c=[float(row["final_val_loss_mean"])],
                cmap="viridis_r",
                vmin=vmin,
                vmax=vmax,
                s=110,
                marker=markers.get(str(row["backend"]), "o"),
                edgecolors="black",
                linewidths=0.6,
            )
            axis.annotate(
                label_for_row(row),
                (
                    float(row["inference_cache_bytes_total"]),
                    float(row["forward_latency_median_ms"]),
                ),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
        axis.ticklabel_format(style="plain", axis="x")
    for index in range(len(lengths), rows * cols):
        axes[index // cols][index % cols].axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Final validation loss (mean)")
    fig.suptitle("Model Stack Tradeoff: Cache vs Latency vs Loss")
    fig.subplots_adjust(top=0.88, right=0.9, wspace=0.28, hspace=0.32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_ablation_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        args.python,
        "scripts/model_stack_ablation.py",
        "--batch-size",
        str(args.inference_batch_size),
        "--prompt-lengths",
        *[str(length) for length in args.lengths],
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--num-kv-heads",
        str(args.num_kv_heads),
        "--intermediate-size",
        str(args.intermediate_size),
        "--vocab-size",
        str(args.vocab_size),
        "--sliding-window",
        str(args.sliding_window),
        "--global-token-stride",
        str(args.global_token_stride),
        "--sink-token-count",
        str(args.sink_token_count),
        "--scale-invariant-tau",
        str(args.scale_invariant_tau),
        "--mla-latent-kv-dims",
        *[str(value) for value in args.mla_latent_kv_dims],
        "--warmup-runs",
        str(args.warmup_runs),
        "--timing-runs",
        str(args.timing_runs),
        "--output",
        str(output_dir / "inference.json"),
        "--csv-output",
        str(output_dir / "inference.csv"),
    ]
    if args.scale_invariant_last_n_layers is not None:
        command.extend(
            [
                "--scale-invariant-last-n-layers",
                *[str(value) for value in args.scale_invariant_last_n_layers],
            ]
        )
    return command


def build_training_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        args.python,
        "scripts/model_stack_training_ablation.py",
        "--train-episodes",
        str(args.train_episodes),
        "--val-episodes",
        str(args.val_episodes),
        "--seq-lens",
        *[str(length) for length in args.lengths],
        "--batch-size",
        str(args.train_batch_size),
        "--steps",
        str(args.steps),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--num-kv-heads",
        str(args.num_kv_heads),
        "--intermediate-size",
        str(args.intermediate_size),
        "--sliding-window",
        str(args.sliding_window),
        "--global-token-stride",
        str(args.global_token_stride),
        "--sink-token-count",
        str(args.sink_token_count),
        "--scale-invariant-tau",
        str(args.scale_invariant_tau),
        "--mla-latent-kv-dims",
        *[str(value) for value in args.mla_latent_kv_dims],
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--output",
        str(output_dir / "training.json"),
        "--csv-output",
        str(output_dir / "training.csv"),
    ]
    if args.scale_invariant_last_n_layers is not None:
        command.extend(
            [
                "--scale-invariant-last-n-layers",
                *[str(value) for value in args.scale_invariant_last_n_layers],
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Running inference sweep...")
    run_command(build_ablation_command(args, output_dir))
    print("Running training sweep...")
    run_command(build_training_command(args, output_dir))

    inference_results = read_json(output_dir / "inference.json")
    training_results = read_json(output_dir / "training.json")
    raw_rows, summary_rows, summary = summarize_tradeoffs(inference_results, training_results)

    print("Writing combined tradeoff artifacts...")
    write_csv(output_dir / "tradeoff_raw.csv", raw_rows)
    write_csv(output_dir / "tradeoff_summary.csv", summary_rows)
    plot_tradeoff(summary_rows, output_dir / "tradeoff_scatter.png")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "inference": {
                    "device": inference_results["device"],
                    "prompt_lengths": inference_results["prompt_lengths"],
                    "mla_latent_kv_dims": inference_results["mla_latent_kv_dims"],
                    "scale_invariant_last_n_layers": inference_results.get("scale_invariant_last_n_layers", []),
                    "scale_invariant_tau": inference_results["scale_invariant_tau"],
                },
                "training": {
                    "device": training_results["device"],
                    "seeds": training_results["seeds"],
                    "seq_lens": training_results["seq_lens"],
                    "mla_latent_kv_dims": training_results["mla_latent_kv_dims"],
                    "scale_invariant_last_n_layers": training_results.get("scale_invariant_last_n_layers", []),
                    "scale_invariant_tau": training_results["scale_invariant_tau"],
                },
                "tradeoff": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
