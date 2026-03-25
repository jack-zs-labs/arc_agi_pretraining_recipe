#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    version: int | None
    size_bytes: int | None
    note: str = ""


DATASETS = (
    DatasetSpec(
        slug="sorokin/nvarc-artifacts-puzzles",
        title="NVARC Artifacts Puzzles",
        version=1,
        size_bytes=42_008_423_274,
        note="Generated summaries, mixtures, input code, and output code.",
    ),
    DatasetSpec(
        slug="sorokin/nvarc-synthetic-puzzles",
        title="NVARC Synthetic Puzzles",
        version=1,
        size_bytes=338_160_847,
        note="The 103k synthetic puzzles described in the NVARC solution README.",
    ),
    DatasetSpec(
        slug="sorokin/nvarc-augmented-puzzles",
        title="NVARC Augmented Puzzles",
        version=1,
        size_bytes=1_322_144_874,
        note="Augmented subsets used by the ARChitects training pipeline.",
    ),
    DatasetSpec(
        slug="cpmpml/arc-prize-trm-training-data",
        title="arc-prize-trm-training-data",
        version=1,
        size_bytes=2_218_651_916,
        note="TRM pretraining data referenced by external/NVARC/TRM/README.md.",
    ),
    DatasetSpec(
        slug="cpmpml/arc-prize-trm-evaluation-data",
        title="arc-prize-trm-evaluation-data",
        version=None,
        size_bytes=None,
        note=(
            "Referenced by external/NVARC/TRM/README.md. "
            "The public dataset page returned HTTP 404 on 2026-03-23 from this environment."
        ),
    ),
)


def format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def has_kaggle_auth() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def print_manifest() -> None:
    print("NVARC Kaggle datasets")
    for spec in DATASETS:
        print(f"- {spec.slug}")
        print(f"  title: {spec.title}")
        print(f"  version: {spec.version if spec.version is not None else 'unknown'}")
        print(f"  size: {format_size(spec.size_bytes)}")
        if spec.note:
            print(f"  note: {spec.note}")


def dataset_target(root: Path, spec: DatasetSpec) -> Path:
    owner, dataset = spec.slug.split("/", 1)
    return root / owner / dataset


def download_dataset(spec: DatasetSpec, root: Path, force: bool) -> int:
    target = dataset_target(root, spec)
    target.mkdir(parents=True, exist_ok=True)

    if not force and any(target.iterdir()):
        print(f"skip {spec.slug}: {target} is not empty")
        return 0

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        spec.slug,
        "-p",
        str(target),
        "--unzip",
    ]
    if force:
        cmd.append("--force")

    print(f"download {spec.slug} -> {target}")
    completed = subprocess.run(cmd)
    if completed.returncode == 0:
        return 0

    print(f"failed {spec.slug} with exit code {completed.returncode}", file=sys.stderr)
    if spec.note:
        print(f"note: {spec.note}", file=sys.stderr)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle-hosted datasets referenced by the NVARC ARC Prize 2025 solution."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "nvarc" / "kaggle",
        help="Destination root for downloaded datasets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the dataset manifest and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload datasets even if the destination directory is not empty.",
    )
    args = parser.parse_args()

    if args.list:
        print_manifest()
        return 0

    if shutil.which("kaggle") is None:
        print("kaggle CLI is not installed.", file=sys.stderr)
        print("Install it with: python3 -m pip install --user kaggle", file=sys.stderr)
        return 2

    if not has_kaggle_auth():
        print("Kaggle credentials are not configured.", file=sys.stderr)
        print(
            "Set KAGGLE_USERNAME and KAGGLE_KEY, or create ~/.kaggle/kaggle.json before rerunning.",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for spec in DATASETS:
        failures += 1 if download_dataset(spec, args.root, args.force) != 0 else 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
