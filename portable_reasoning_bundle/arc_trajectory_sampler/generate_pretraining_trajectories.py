from __future__ import annotations

"""Generate reward-shaped ARC trajectories for pretraining."""

import argparse
import json
from pathlib import Path
from typing import Dict

try:
    from .stage1_latent_sampler import sample_latent_rule
    from .stage2_episode_sampler import sample_episode
    from .stage4_trajectory_dataset import build_trajectories, write_jsonl
    from .trm_dataset_export import episode_is_trm_compatible, write_trm_dataset
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import sample_latent_rule  # type: ignore
    from stage2_episode_sampler import sample_episode  # type: ignore
    from stage4_trajectory_dataset import build_trajectories, write_jsonl  # type: ignore
    from trm_dataset_export import episode_is_trm_compatible, write_trm_dataset  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense-reward ARC trajectory data.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of sampled episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--num-train", type=int, default=3, help="Number of train examples per episode.")
    parser.add_argument("--max-attempts", type=int, default=256, help="Reject-sampling budget per episode.")
    parser.add_argument(
        "--max-sample-multiplier",
        type=int,
        default=20,
        help="Maximum outer resampling multiplier when TRM compatibility filtering is enabled.",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "trm", "both"],
        default="both",
        help="Output dense-reward JSONL, TinyRecursiveModels dataset format, or both.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/pretraining_trajectories.jsonl",
        help="Destination JSONL path when --format includes jsonl.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="arc_trajectory_sampler/results/pretraining_trm_dataset",
        help="Destination directory when --format includes trm.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=0,
        help="Number of ARC-style dihedral/color augmentations per sampled episode in TRM export mode.",
    )
    parser.add_argument(
        "--include-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the held-out test example trajectory for each sampled episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    all_records = []
    episodes = []
    family_counts: Dict[str, int] = {}
    rejected_incompatible = 0
    candidate_index = 0
    max_candidates = max(args.num_episodes, 1) * max(1, args.max_sample_multiplier)

    while len(episodes) < args.num_episodes:
        if candidate_index >= max_candidates:
            raise RuntimeError(
                f"failed to collect {args.num_episodes} compatible episodes after {max_candidates} candidates; "
                f"rejected {rejected_incompatible} for TRM size constraints"
            )
        seed = args.seed + candidate_index
        candidate_index += 1
        latent = sample_latent_rule(seed=seed)
        episode = sample_episode(
            latent,
            seed=seed,
            num_train=args.num_train,
            max_attempts=args.max_attempts,
        )
        if args.format in {"trm", "both"} and not episode_is_trm_compatible(episode):
            rejected_incompatible += 1
            continue
        episodes.append(episode)
        trajectory_count = len(episode.train_examples) + (1 if args.include_test else 0)
        family_counts[latent.family.value] = family_counts.get(latent.family.value, 0) + trajectory_count
        if args.format in {"jsonl", "both"}:
            all_records.extend(build_trajectories(episode, include_test=args.include_test))

    jsonl_path = None
    if args.format in {"jsonl", "both"}:
        jsonl_path = write_jsonl(output_path, all_records)

    trm_summary = None
    if args.format in {"trm", "both"}:
        trm_summary = write_trm_dataset(
            episodes,
            args.dataset_dir,
            seed=args.seed,
            include_test=args.include_test,
            num_aug=args.num_aug,
        )

    summary = {
        "episodes": args.num_episodes,
        "seed": args.seed,
        "include_test": args.include_test,
        "family_counts": family_counts,
        "format": args.format,
        "sampled_candidates": candidate_index,
    }
    if args.format in {"trm", "both"}:
        summary["rejected_incompatible"] = rejected_incompatible
    if jsonl_path is not None:
        summary["records"] = len(all_records)
        summary["jsonl_output"] = str(jsonl_path)
    if trm_summary is not None:
        summary["trm_output_dir"] = trm_summary.output_dir
        summary["trm_train_puzzles"] = trm_summary.train_puzzles
        summary["trm_test_puzzles"] = trm_summary.test_puzzles
        summary["trm_num_puzzle_identifiers"] = trm_summary.num_puzzle_identifiers
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
