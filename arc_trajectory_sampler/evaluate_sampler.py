from __future__ import annotations

"""Evaluate the Stage 1 + Stage 2 ARC trajectory sampler."""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .stage1_latent_sampler import Family, TaskLatent, sample_latent_rule
    from .stage2_episode_sampler import EpisodeSpec, ExampleSpec, SceneSpec, sample_episode
except ImportError:  # pragma: no cover - direct script execution
    from stage1_latent_sampler import Family, TaskLatent, sample_latent_rule  # type: ignore
    from stage2_episode_sampler import EpisodeSpec, ExampleSpec, SceneSpec, sample_episode  # type: ignore


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def p95_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 4)


def counter_to_dict(counter: Counter[str]) -> Dict[str, int]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return {key: value for key, value in ordered}


def bool_rate(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def area_bucket(area: float) -> str:
    if area <= 64:
        return "small"
    if area <= 144:
        return "medium"
    return "large"


def count_bucket(count: float) -> str:
    if count <= 3:
        return "low"
    if count <= 6:
        return "medium"
    return "high"


def vocab_bucket(size: int) -> str:
    if size <= 3:
        return "low"
    if size <= 5:
        return "medium"
    return "high"


def fraction_bucket(value: float) -> str:
    if value <= 0.25:
        return "low"
    if value <= 0.6:
        return "medium"
    return "high"


def scene_colors(scene: SceneSpec) -> set[int]:
    colors = {scene.background_color}
    if scene.border_color is not None:
        colors.add(scene.border_color)
    if scene.outline_color is not None:
        colors.add(scene.outline_color)
    for obj in scene.objects:
        colors.add(obj.color)
    return colors


def scene_shapes(scene: SceneSpec) -> set[str]:
    return {obj.shape for obj in scene.objects}


def latent_targets(latent: TaskLatent) -> Dict[str, str]:
    targets = {
        "family": latent.family.value,
        "difficulty": str(latent.difficulty),
        "program_op": latent.program.op,
    }
    if latent.family == Family.UNARY_OBJECT:
        selector, action = latent.program.args
        targets["family_detail"] = f"{selector.kind.value}->{action.name}"
        targets["unary_transform"] = action.name
        targets["unary_selector"] = selector.kind.value
    elif latent.family == Family.RELATIONAL:
        src_selector, dst_selector, action = latent.program.args
        targets["family_detail"] = f"{src_selector.kind.value}:{dst_selector.kind.value}->{action.name}"
        targets["relational_action"] = action.name
        targets["relational_src_selector"] = src_selector.kind.value
        targets["relational_dst_selector"] = dst_selector.kind.value
    elif latent.family == Family.COUNT_SELECT:
        group_by, reducer, post_action = latent.program.args
        targets["family_detail"] = f"{group_by}:{reducer.kind.value}:{post_action.name}"
        targets["count_group_by"] = str(group_by)
        targets["count_reducer"] = reducer.kind.value
        targets["count_post"] = post_action.name
    elif latent.family == Family.CONTEXTUAL:
        cue, selector, then_action, else_action = latent.program.args
        targets["family_detail"] = f"{cue.value}:{selector.kind.value}:{then_action.name}:{else_action.name}"
        targets["contextual_cue"] = cue.value
        targets["contextual_selector"] = selector.kind.value
        targets["contextual_then_action"] = then_action.name
        targets["contextual_else_action"] = else_action.name
    elif latent.family == Family.SYMBOL_MAP:
        key_type, value_type, apply_mode = latent.program.args
        targets["family_detail"] = f"{key_type}:{value_type}:{apply_mode.name}"
        targets["symbol_key_type"] = str(key_type)
        targets["symbol_value_type"] = str(value_type)
        targets["symbol_apply_mode"] = apply_mode.name
    return targets


def episode_features(episode: EpisodeSpec) -> Dict[str, Any]:
    train_examples = episode.train_examples
    train_scenes = [example.input_scene for example in train_examples]
    test_scene = episode.test_example.input_scene

    train_areas = [scene.height * scene.width for scene in train_scenes]
    train_object_counts = [len(scene.objects) for scene in train_scenes]
    train_selected_counts = [len(example.selected_object_ids) for example in train_examples]
    train_selected_fractions = [
        len(example.selected_object_ids) / max(1, len(example.input_scene.objects)) for example in train_examples
    ]
    train_color_union = set().union(*(scene_colors(scene) for scene in train_scenes))
    train_shape_union = set().union(*(scene_shapes(scene) for scene in train_scenes))
    train_grid_sizes = {(scene.height, scene.width) for scene in train_scenes}
    train_diversity_tokens = {example.metadata["diversity_token"] for example in train_examples}

    return {
        "train_mean_area_bucket": area_bucket(sum(train_areas) / len(train_areas)),
        "train_mean_object_count_bucket": count_bucket(sum(train_object_counts) / len(train_object_counts)),
        "train_mean_selected_count_bucket": count_bucket(sum(train_selected_counts) / len(train_selected_counts)),
        "train_color_union_bucket": vocab_bucket(len(train_color_union)),
        "train_shape_union_bucket": vocab_bucket(len(train_shape_union)),
        "train_selected_fraction_bucket": fraction_bucket(sum(train_selected_fractions) / len(train_selected_fractions)),
        "test_area_bucket": area_bucket(test_scene.height * test_scene.width),
        "test_object_count_bucket": count_bucket(len(test_scene.objects)),
        "train_grid_size_varies": len(train_grid_sizes) > 1,
        "train_object_count_varies": len(set(train_object_counts)) > 1,
        "train_selected_count_varies": len(set(train_selected_counts)) > 1,
        "has_border_context": any(scene.border_color is not None for scene in train_scenes + [test_scene]),
        "has_outline_context": any(scene.outline_color is not None for scene in train_scenes + [test_scene]),
        "has_marker_context": any(scene.marker_position is not None for scene in train_scenes + [test_scene]),
        "has_legend": any(bool(scene.legend) for scene in train_scenes + [test_scene]),
        "train_diversity_token_bucket": count_bucket(len(train_diversity_tokens)),
    }


def episode_metrics(episode: EpisodeSpec) -> Dict[str, Any]:
    train_examples = episode.train_examples
    train_scenes = [example.input_scene for example in train_examples]
    test_scene = episode.test_example.input_scene

    train_areas = [scene.height * scene.width for scene in train_scenes]
    train_object_counts = [len(scene.objects) for scene in train_scenes]
    train_selected_counts = [len(example.selected_object_ids) for example in train_examples]
    train_selected_fractions = [
        len(example.selected_object_ids) / max(1, len(example.input_scene.objects)) for example in train_examples
    ]
    train_color_union = set().union(*(scene_colors(scene) for scene in train_scenes))
    train_shape_union = set().union(*(scene_shapes(scene) for scene in train_scenes))
    train_diversity_tokens = {example.metadata["diversity_token"] for example in train_examples}

    metrics = {
        "sampling_attempts": episode.sampling_attempts,
        "train_grid_size_varies": len({(scene.height, scene.width) for scene in train_scenes}) > 1,
        "train_object_count_varies": len(set(train_object_counts)) > 1,
        "train_selected_count_varies": len(set(train_selected_counts)) > 1,
        "mean_train_area": round(sum(train_areas) / len(train_areas), 4),
        "mean_train_object_count": round(sum(train_object_counts) / len(train_object_counts), 4),
        "mean_train_selected_count": round(sum(train_selected_counts) / len(train_selected_counts), 4),
        "mean_train_selected_fraction": round(sum(train_selected_fractions) / len(train_selected_fractions), 4),
        "train_color_union_count": len(train_color_union),
        "train_shape_union_count": len(train_shape_union),
        "train_diversity_token_count": len(train_diversity_tokens),
        "test_object_count": len(test_scene.objects),
        "test_area": test_scene.height * test_scene.width,
    }

    if episode.latent.family == Family.COUNT_SELECT:
        winner_keys = {str(example.metadata["winner_key"]) for example in train_examples}
        metrics["train_winner_cardinality"] = len(winner_keys)
        metrics["winner_varies_across_train"] = len(winner_keys) > 1
    if episode.latent.family == Family.CONTEXTUAL:
        branches = [str(example.metadata["branch"]) for example in train_examples]
        metrics["train_branch_cardinality"] = len(set(branches))
        metrics["both_branches_realized"] = len(set(branches)) == 2
    if episode.latent.family == Family.SYMBOL_MAP:
        legend_sizes = [len(scene.legend) for scene in train_scenes]
        target_counts = [
            sum(1 for obj in scene.objects if "target" in obj.tags) for scene in train_scenes + [test_scene]
        ]
        metrics["mean_legend_size"] = round(sum(legend_sizes) / len(legend_sizes), 4)
        metrics["mean_target_count"] = round(sum(target_counts) / len(target_counts), 4)

    return metrics


def aggregate_metric(records: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [record["metrics"][key] for record in records if key in record["metrics"]]
    if not values:
        return None
    if isinstance(values[0], bool):
        return bool_rate(values)
    return mean_or_none(values)


def split_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(records) < 6:
        return ([], [])
    cutoff = max(1, int(len(records) * 0.8))
    if cutoff >= len(records):
        cutoff = len(records) - 1
    return (list(records[:cutoff]), list(records[cutoff:]))


def evaluate_lookup_probe(
    records: Sequence[Dict[str, Any]],
    feature_name: str,
    target_name: str,
) -> Optional[Dict[str, Any]]:
    train_records, test_records = split_records(records)
    if not train_records or not test_records:
        return None

    label_counts = Counter(record["targets"][target_name] for record in train_records)
    if len(label_counts) < 2:
        return None

    feature_map: Dict[Any, Counter[str]] = defaultdict(Counter)
    for record in train_records:
        feature_map[record["features"][feature_name]][record["targets"][target_name]] += 1
    lookup = {feature: counts.most_common(1)[0][0] for feature, counts in feature_map.items()}
    global_majority = label_counts.most_common(1)[0][0]

    correct = 0
    baseline_correct = 0
    for record in test_records:
        prediction = lookup.get(record["features"][feature_name], global_majority)
        label = record["targets"][target_name]
        correct += int(prediction == label)
        baseline_correct += int(global_majority == label)

    accuracy = correct / len(test_records)
    baseline = baseline_correct / len(test_records)
    return {
        "feature": feature_name,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline, 4),
        "lift": round(accuracy - baseline, 4),
        "num_train": len(train_records),
        "num_test": len(test_records),
        "num_labels": len(label_counts),
    }


def build_leakage_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    feature_names = list(records[0]["features"].keys())
    all_target_names = sorted({key for record in records for key in record["targets"].keys()})
    probes: Dict[str, Any] = {}

    for target_name in all_target_names:
        target_records = [record for record in records if target_name in record["targets"]]
        if len(target_records) < 6:
            continue
        best_probe = None
        for feature_name in feature_names:
            probe = evaluate_lookup_probe(target_records, feature_name, target_name)
            if probe is None:
                continue
            if best_probe is None or probe["accuracy"] > best_probe["accuracy"]:
                best_probe = probe
        if best_probe is None:
            continue
        label_counts = Counter(record["targets"][target_name] for record in target_records)
        probes[target_name] = {
            **best_probe,
            "label_distribution": counter_to_dict(label_counts),
        }

    return probes


def summarize_family(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "count": len(records),
        "mean_sampling_attempts": aggregate_metric(records, "sampling_attempts"),
        "p95_sampling_attempts": p95_or_none(
            [record["metrics"]["sampling_attempts"] for record in records if "sampling_attempts" in record["metrics"]]
        ),
        "share_train_grid_size_varies": aggregate_metric(records, "train_grid_size_varies"),
        "share_train_object_count_varies": aggregate_metric(records, "train_object_count_varies"),
        "share_train_selected_count_varies": aggregate_metric(records, "train_selected_count_varies"),
        "mean_train_diversity_token_count": aggregate_metric(records, "train_diversity_token_count"),
        "mean_train_color_union_count": aggregate_metric(records, "train_color_union_count"),
        "mean_train_shape_union_count": aggregate_metric(records, "train_shape_union_count"),
        "mean_train_selected_fraction": aggregate_metric(records, "mean_train_selected_fraction"),
    }

    family = records[0]["targets"]["family"] if records else None
    if family == Family.COUNT_SELECT.value:
        summary["share_winner_varies_across_train"] = aggregate_metric(records, "winner_varies_across_train")
        summary["mean_train_winner_cardinality"] = aggregate_metric(records, "train_winner_cardinality")
    if family == Family.CONTEXTUAL.value:
        summary["share_both_branches_realized"] = aggregate_metric(records, "both_branches_realized")
        summary["mean_train_branch_cardinality"] = aggregate_metric(records, "train_branch_cardinality")
    if family == Family.SYMBOL_MAP.value:
        summary["mean_legend_size"] = aggregate_metric(records, "mean_legend_size")
        summary["mean_target_count"] = aggregate_metric(records, "mean_target_count")
    return summary


def evaluate_sampler(
    *,
    num_samples: int = 250,
    start_seed: int = 0,
    num_train: int = 3,
    max_attempts: int = 256,
) -> Dict[str, Any]:
    latent_counter: Counter[str] = Counter()
    success_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    concept_counter: Counter[str] = Counter()
    near_miss_rejections: Counter[str] = Counter()
    hard_failures: Counter[str] = Counter()
    successful_records: List[Dict[str, Any]] = []
    failure_examples: List[Dict[str, Any]] = []
    latent_target_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    success_target_counter: Dict[str, Counter[str]] = defaultdict(Counter)

    for index in range(num_samples):
        latent_seed = start_seed + index
        episode_seed = start_seed + 100_000 + index
        latent = sample_latent_rule(seed=latent_seed)
        targets = latent_targets(latent)

        latent_counter[targets["family"]] += 1
        difficulty_counter[targets["difficulty"]] += 1
        for tag in latent.concept_tags:
            concept_counter[tag] += 1
        for key, value in targets.items():
            latent_target_counter[key][value] += 1

        try:
            episode = sample_episode(latent, seed=episode_seed, num_train=num_train, max_attempts=max_attempts)
        except RuntimeError as exc:
            failure_key = f"{targets['family']}::{exc}"
            hard_failures[failure_key] += 1
            if len(failure_examples) < 12:
                failure_examples.append(
                    {
                        "seed": latent_seed,
                        "family": targets["family"],
                        "error": str(exc),
                    }
                )
            continue

        success_counter[targets["family"]] += 1
        near_miss_rejections.update(episode.rejection_counts)
        for key, value in targets.items():
            success_target_counter[key][value] += 1

        successful_records.append(
            {
                "seed": latent_seed,
                "targets": targets,
                "features": episode_features(episode),
                "metrics": episode_metrics(episode),
            }
        )

    success_attempts = [record["metrics"]["sampling_attempts"] for record in successful_records]
    overall = {
        "requested_samples": num_samples,
        "successful_episodes": len(successful_records),
        "failed_episodes": num_samples - len(successful_records),
        "success_rate": round(len(successful_records) / max(1, num_samples), 4),
        "mean_sampling_attempts": mean_or_none(success_attempts),
        "p95_sampling_attempts": p95_or_none(success_attempts),
        "max_sampling_attempts": max(success_attempts) if success_attempts else None,
    }

    success_rate_by_family = {}
    for family_name, total in latent_counter.items():
        success_rate_by_family[family_name] = round(success_counter[family_name] / max(1, total), 4)

    coverage = {
        "latent_family_counts": counter_to_dict(latent_counter),
        "successful_family_counts": counter_to_dict(success_counter),
        "success_rate_by_family": dict(sorted(success_rate_by_family.items())),
        "difficulty_counts": counter_to_dict(difficulty_counter),
        "concept_tag_counts": counter_to_dict(concept_counter),
        "latent_target_counts": {key: counter_to_dict(counter) for key, counter in sorted(latent_target_counter.items())},
        "successful_target_counts": {
            key: counter_to_dict(counter) for key, counter in sorted(success_target_counter.items())
        },
    }

    diversity = {
        "share_train_grid_size_varies": aggregate_metric(successful_records, "train_grid_size_varies"),
        "share_train_object_count_varies": aggregate_metric(successful_records, "train_object_count_varies"),
        "share_train_selected_count_varies": aggregate_metric(successful_records, "train_selected_count_varies"),
        "mean_train_diversity_token_count": aggregate_metric(successful_records, "train_diversity_token_count"),
        "mean_train_color_union_count": aggregate_metric(successful_records, "train_color_union_count"),
        "mean_train_shape_union_count": aggregate_metric(successful_records, "train_shape_union_count"),
        "mean_train_selected_fraction": aggregate_metric(successful_records, "mean_train_selected_fraction"),
    }

    per_family = {}
    for family_name in sorted(success_counter):
        family_records = [record for record in successful_records if record["targets"]["family"] == family_name]
        per_family[family_name] = summarize_family(family_records)

    return {
        "parameters": {
            "num_samples": num_samples,
            "start_seed": start_seed,
            "num_train": num_train,
            "max_attempts": max_attempts,
        },
        "overall": overall,
        "coverage": coverage,
        "diversity": diversity,
        "per_family": per_family,
        "near_miss_rejections": counter_to_dict(near_miss_rejections),
        "hard_failures": counter_to_dict(hard_failures),
        "leakage_probes": build_leakage_summary(successful_records),
        "failure_examples": failure_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ARC trajectory sampler.")
    parser.add_argument("--num-samples", type=int, default=250, help="Number of latent + episode samples to evaluate.")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed for deterministic evaluation.")
    parser.add_argument("--num-train", type=int, default=3, help="Train examples per sampled episode.")
    parser.add_argument("--max-attempts", type=int, default=256, help="Reject-sampling budget per episode.")
    parser.add_argument("--output", type=str, default="", help="Optional path to write the JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_sampler(
        num_samples=args.num_samples,
        start_seed=args.start_seed,
        num_train=args.num_train,
        max_attempts=args.max_attempts,
    )
    payload = json.dumps(summary, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
        print(f"Wrote evaluation summary to {output_path}")
        print(
            json.dumps(
                {
                    "success_rate": summary["overall"]["success_rate"],
                    "successful_episodes": summary["overall"]["successful_episodes"],
                    "mean_sampling_attempts": summary["overall"]["mean_sampling_attempts"],
                },
                indent=2,
            )
        )
        return
    print(payload)


if __name__ == "__main__":
    main()
