from __future__ import annotations

"""Evaluate ARC pretraining trajectory quality."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .stage1_latent_sampler import sample_latent_rule
    from .stage2_episode_sampler import sample_episode
    from .stage3_grid_executor import execute_episode
    from .stage4_trajectory_dataset import compile_episode_trajectories
    from .trm_dataset_export import episode_is_trm_compatible
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import sample_latent_rule  # type: ignore
    from stage2_episode_sampler import sample_episode  # type: ignore
    from stage3_grid_executor import execute_episode  # type: ignore
    from stage4_trajectory_dataset import compile_episode_trajectories  # type: ignore
    from trm_dataset_export import episode_is_trm_compatible  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic ARC pretraining data quality.")
    parser.add_argument("--num-seeds", type=int, default=300, help="Number of seed attempts to evaluate.")
    parser.add_argument(
        "--output",
        type=str,
        default="arc_trajectory_sampler/results/quality_eval_summary.json",
        help="Destination summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_attempts = Counter()
    family_success = Counter()
    family_trm = Counter()
    failures = Counter()
    step_reward_values = Counter()
    total_rewards = []
    total_possible = []
    progress_terminal = []
    example_hashes = Counter()
    unique_by_family = defaultdict(set)
    changed_fraction = []
    trace_lengths = Counter()
    local_progress = []
    output_progress = []
    state_delta = []
    step_examples = []
    variant_kinds = Counter()
    trajectory_roles = Counter()

    for seed in range(args.num_seeds):
        latent = sample_latent_rule(seed=seed)
        family = latent.family.value
        family_attempts[family] += 1
        try:
            episode = sample_episode(latent, seed=seed)
            executed = execute_episode(episode)
            trajectories = compile_episode_trajectories(executed)
        except Exception as exc:  # pragma: no cover - evaluation summary should include failures
            failures[f"{family}: {type(exc).__name__}: {exc}"] += 1
            continue

        family_success[family] += 1
        if episode_is_trm_compatible(episode):
            family_trm[family] += 1

        for trajectory in trajectories:
            total_rewards.append(trajectory.total_reward)
            total_possible.append(trajectory.total_possible_reward)
            progress_terminal.append(trajectory.steps[-1].progress if trajectory.steps else 0.0)
            trace_lengths[len(trajectory.steps)] += 1
            variant_kinds[trajectory.variant_kind] += 1
            trajectory_roles[trajectory.trajectory_role] += 1
            for step in trajectory.steps:
                step_reward_values[round(step.reward, 6)] += 1
                local_progress.append(float(step.reward_terms.get("local_progress", 0.0)))
                output_progress.append(float(step.reward_terms.get("output_progress", 0.0)))
                state_delta.append(float(step.reward_terms.get("state_delta", 0.0)))
                if "changed_cell_fraction" in step.action:
                    changed_fraction.append(float(step.action["changed_cell_fraction"]))
                if len(step_examples) < 8:
                    step_examples.append(
                        {
                            "family": trajectory.family,
                            "step": step.name,
                            "reward": step.reward,
                            "reward_terms": step.reward_terms,
                        }
                    )
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "family": trajectory.family,
                        "variant_kind": trajectory.variant_kind,
                        "input": trajectory.input_grid,
                        "output": trajectory.output_grid,
                        "trace": trajectory.trace_template,
                        "trajectory_role": trajectory.trajectory_role,
                        "parent_trajectory_id": trajectory.parent_trajectory_id,
                        "final_workspace": trajectory.steps[-1].workspace_state if trajectory.steps else None,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            example_hashes[digest] += 1
            unique_by_family[trajectory.family].add(digest)

    summary = {
        "num_candidate_seeds": args.num_seeds,
        "family_attempts": dict(family_attempts),
        "family_success": dict(family_success),
        "family_trm_compatible": dict(family_trm),
        "success_rate_by_family": {
            family: family_success[family] / family_attempts[family] for family in family_attempts
        },
        "trm_compat_rate_by_family": {
            family: (family_trm[family] / family_success[family]) if family_success[family] else 0.0
            for family in family_attempts
        },
        "failure_counts": dict(failures.most_common(10)),
        "trajectory_count": len(total_rewards),
        "reward_min": min(total_rewards) if total_rewards else None,
        "reward_max": max(total_rewards) if total_rewards else None,
        "reward_mean": statistics.mean(total_rewards) if total_rewards else None,
        "possible_unique": sorted(set(total_possible)),
        "all_total_reward_equal_possible": all(abs(a - b) < 1e-9 for a, b in zip(total_rewards, total_possible)),
        "terminal_progress_min": min(progress_terminal) if progress_terminal else None,
        "terminal_progress_max": max(progress_terminal) if progress_terminal else None,
        "terminal_progress_mean": statistics.mean(progress_terminal) if progress_terminal else None,
        "unique_step_reward_values_count": len(step_reward_values),
        "unique_step_reward_values_sample": sorted(step_reward_values)[:30],
        "trace_length_histogram": dict(trace_lengths),
        "variant_kind_counts": dict(variant_kinds),
        "trajectory_role_counts": dict(trajectory_roles),
        "duplicate_trajectory_records": sum(count - 1 for count in example_hashes.values() if count > 1),
        "unique_trajectory_records": len(example_hashes),
        "unique_records_by_family": {family: len(hashes) for family, hashes in unique_by_family.items()},
        "changed_cell_fraction": {
            "min": min(changed_fraction) if changed_fraction else None,
            "max": max(changed_fraction) if changed_fraction else None,
            "mean": statistics.mean(changed_fraction) if changed_fraction else None,
        },
        "local_progress": {
            "min": min(local_progress) if local_progress else None,
            "max": max(local_progress) if local_progress else None,
            "mean": statistics.mean(local_progress) if local_progress else None,
        },
        "output_progress": {
            "min": min(output_progress) if output_progress else None,
            "max": max(output_progress) if output_progress else None,
            "mean": statistics.mean(output_progress) if output_progress else None,
        },
        "state_delta": {
            "min": min(state_delta) if state_delta else None,
            "max": max(state_delta) if state_delta else None,
            "mean": statistics.mean(state_delta) if state_delta else None,
        },
        "step_examples": step_examples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
