# ARC Trajectory Sampler

This folder is the isolated workspace for building a synthetic trajectory
sampler for the ARC-AGI-2 challenge.

Current scope:

1. `stage1_latent_sampler.py`: sample a typed latent rule and task family.
2. `stage2_episode_sampler.py`: build train/test episode blueprints with
   resolved role bindings, object-scene specs, and reject-sampling checks.
3. `stage3_grid_executor.py`: execute the latent program into concrete input and
   output grids using the Stage 2 object abstractions, while preserving
   per-step object-scene workspaces.
4. `stage4_trajectory_dataset.py`: compile object-level trajectories with
   subgoal state, verifier labels, dense local rewards, alternate valid paths,
   and near-miss negatives.
5. `generate_pretraining_trajectories.py`: export JSONL trajectories for
   pretraining.
6. `trm_dataset_export.py`: write the exact TinyRecursiveModels `.npy` dataset
   layout consumed by `external/NVARC/external/TinyRecursiveModels/pretrain.py`.
7. `evaluate_pretraining_quality.py`: measure execution success, reward
   variability, family balance, duplicates, and TRM compatibility.
8. `reasoning_ir.py`: define a modality-neutral intermediate representation for
   abstract reasoning tasks.
9. `word_problem_translation_dataset.py`: generate synthetic arithmetic word
   problems paired with abstract IR targets and canonical trajectories.
10. `evaluate_word_problem_translation_quality.py`: measure template balance,
    duplicates, reward variability, and symbolic answer consistency for the
    word-problem dataset.
11. `analyze_gsm8k_template_fit.py`: download GSM8K and estimate exact or loose
    fit against the current word-problem templates.
12. `recommend_gsm8k_arc_families.py`: propose the smallest ARC-style numeric
    family set that covers a useful slice of GSM8K.
13. `gsm8k_reasoning_parser.py`: parse GSM8K worked solutions into the
    canonical `compose_total` / `compose_difference` IR and trajectories.
14. `evaluate_gsm8k_parser.py`: audit the heuristic classifier, the
    supervision-time GSM8K parser, template fit, and failure modes.
15. `mmlu_parser.py`: parse official MMLU multiple-choice rows into the
    canonical IR for the `quantitative_choice`, `factual_choice`,
    `case_application_choice`, `passage_reference_choice`,
    `negation_exception_choice`, `completion_choice`,
    `rule_application_choice`, `statement_evaluation_choice`,
    `descriptor_match_choice`, `comparative_inference_choice`, and
    `concept_identification_choice` families.
16. `evaluate_mmlu_parser.py`: measure MMLU parser coverage, family balance,
    and multiple-choice trajectory consistency.

The Stage 1 sampler is intentionally narrow. It samples symbolic roles and
typed programs, not finished grids.

Quick start:

```bash
python3 arc_trajectory_sampler/stage1_latent_sampler.py
python3 arc_trajectory_sampler/stage2_episode_sampler.py
python3 arc_trajectory_sampler/stage3_grid_executor.py
python3 arc_trajectory_sampler/stage4_trajectory_dataset.py
python3 arc_trajectory_sampler/evaluate_sampler.py --num-samples 100 --output arc_trajectory_sampler/results/summary.json
python3 arc_trajectory_sampler/report_evaluation.py --input arc_trajectory_sampler/results/summary.json --output-dir arc_trajectory_sampler/results/report
python3 arc_trajectory_sampler/trajectory_renderer.py --seed 0 --output arc_trajectory_sampler/results/trajectory_seed_0.png
python3 arc_trajectory_sampler/stage4_trajectory_renderer.py --seed 9 --example-id train_0 --output arc_trajectory_sampler/results/stage4_variants_seed_9.png
python3 arc_trajectory_sampler/generate_pretraining_trajectories.py --num-episodes 100 --format both --output arc_trajectory_sampler/results/pretraining_trajectories.jsonl --dataset-dir arc_trajectory_sampler/results/pretraining_trm_dataset
python3 arc_trajectory_sampler/evaluate_pretraining_quality.py --num-seeds 300 --output arc_trajectory_sampler/results/quality_eval_summary.json
python3 arc_trajectory_sampler/word_problem_translation_dataset.py --num-examples 32 --num-test 4 --translation-output arc_trajectory_sampler/results/word_problem_translation.jsonl --trajectory-output arc_trajectory_sampler/results/word_problem_translation_trajectories.jsonl
python3 arc_trajectory_sampler/evaluate_word_problem_translation_quality.py --num-examples 500 --num-test 50 --output arc_trajectory_sampler/results/word_problem_quality_summary.json
python3 arc_trajectory_sampler/analyze_gsm8k_template_fit.py --data-dir arc_trajectory_sampler/data/gsm8k --splits train test --output arc_trajectory_sampler/results/gsm8k_template_fit_summary.json
python3 arc_trajectory_sampler/recommend_gsm8k_arc_families.py --data-dir arc_trajectory_sampler/data/gsm8k --output arc_trajectory_sampler/results/gsm8k_arc_family_recommendations.json
python3 arc_trajectory_sampler/gsm8k_reasoning_parser.py --data-dir arc_trajectory_sampler/data/gsm8k --splits train --translation-output arc_trajectory_sampler/results/gsm8k_reasoning_examples.jsonl --trajectory-output arc_trajectory_sampler/results/gsm8k_reasoning_trajectories.jsonl
python3 arc_trajectory_sampler/evaluate_gsm8k_parser.py --data-dir arc_trajectory_sampler/data/gsm8k --splits train test --output arc_trajectory_sampler/results/gsm8k_parser_eval_summary.json
python3 arc_trajectory_sampler/mmlu_parser.py --data-dir arc_trajectory_sampler/data/mmlu --splits auxiliary_train --translation-output arc_trajectory_sampler/results/mmlu_reasoning_examples.jsonl --trajectory-output arc_trajectory_sampler/results/mmlu_reasoning_trajectories.jsonl
python3 arc_trajectory_sampler/evaluate_mmlu_parser.py --data-dir arc_trajectory_sampler/data/mmlu --splits dev val test --output arc_trajectory_sampler/results/mmlu_parser_eval_summary.json
python3 arc_trajectory_sampler/evaluate_arc_agi2_representation.py --dataset-root arc-agi-2 --output arc_trajectory_sampler/results/arc_agi2_representation_summary.json
python3 arc_trajectory_sampler/evaluate_arc_agi2_representation.py --dataset-root external/NVARC/external/ConceptARC/corpus --dataset-format task_directory --max-tasks 25 --output arc_trajectory_sampler/results/conceptarc_subset_representation_summary.json
```

Or from Python:

```python
from arc_trajectory_sampler import (
    build_trajectories,
    build_word_problem_trajectories,
    execute_episode,
    sample_episode,
    sample_latent_rule,
)
from arc_trajectory_sampler.evaluate_sampler import evaluate_sampler

latent = sample_latent_rule(seed=0)
episode = sample_episode(latent, seed=0)
executed = execute_episode(episode)
trajectories = build_trajectories(episode)
word_trajectories = build_word_problem_trajectories(num_examples=4, seed_start=0, num_test=1)
print(episode.to_jsonable())
print(executed.to_jsonable()["train_examples"][0]["output_grid"])
print(trajectories[0].to_jsonable()["steps"][0]["reward"])
print(word_trajectories[0].to_jsonable()["output_state"]["program"]["op"])
print(evaluate_sampler(num_samples=25)["overall"])
```

The report generator reads an evaluation summary and writes:

1. `report.md`: compact Markdown brief with findings and tables.
2. `*.png`: bar charts for success rate, sampling cost, rejection modes, and leakage lift.

The trajectory renderer projects the current Stage 2 object blueprint into a
storyboard image. It is not an exact cell-level executor; it visualizes:

1. the full input scene
2. selected objects highlighted in-context
3. the normalized selected-object focus panel
4. per-example metadata in a side panel

The Stage 4 trajectory renderer consumes real compiled trajectories and shows:

1. one row per canonical, alternate, or negative variant
2. the shared input grid and target output grid
3. each intermediate workspace grid in order
4. verifier and stop-target metadata for the final step

The pretraining exporter writes one JSONL record per train/test example. Each
record contains:

1. the latent family, bindings, and trace template
2. the executed input/output grids
3. per-step local rewards aligned with the Stage 1 trace steps
4. progress scalars so a learner gets credit for intermediate subgoals, not just
   final rendering

The canonical trajectory schema is modality-neutral:

1. `input_state`, `output_state`, and `workspace_state` are the primary fields
2. `input_grid`, `output_grid`, and `workspace_grid` remain for ARC
   compatibility
3. text tasks such as `word_problem_translation` reuse the same record shape
   without a separate export ontology

Benchmark hygiene for benchmark-targeting corpora is strict by default:

1. `gsm8k_reasoning_parser.py` defaults to `--splits train`
2. official GSM8K `test` rows require `--allow-eval-splits`
3. `mmlu_parser.py` defaults to `--splits auxiliary_train`
4. official MMLU `dev` / `val` / `test` rows require `--allow-eval-splits`
5. the evaluator scripts remain the audit path and can scan benchmark splits explicitly

The ARC-AGI-2 representation evaluator is stricter than the synthetic sampler
checks. It parses raw ARC grids into connected components, fits each component
to the current object vocabulary, re-renders the scene, and reports exact-grid
plus cell-level reconstruction accuracy. The current representation includes an
exact `bitmap` fallback for arbitrary connected components, so this benchmark
measures whether the scene schema is lossless enough to encode ARC examples; it
does not search over latent rules or recover executable programs. The same
script can also score ARC-style per-task JSON directories, including
`train`/`test` corpora such as ConceptARC and `examples` corpora such as BARC
synthetic problems.

When `--format` includes `trm`, the generator also writes the exact directory
shape expected by TinyRecursiveModels:

1. `train/all__inputs.npy`
2. `train/all__labels.npy`
3. `train/all__puzzle_identifiers.npy`
4. `train/all__puzzle_indices.npy`
5. `train/all__group_indices.npy`
6. `train/dataset.json`
7. matching `test/` files when `--include-test` is enabled

The word-problem generator writes:

1. a translation JSONL of `source_text -> abstract_task`
2. a trajectory JSONL where each workspace is a partial symbolic state and the
   final target is the full typed IR
3. simple translation families (`add_change`, `subtract_change`,
   `compare_difference`, `multiply_groups`) plus ARC-like compose families
   (`compose_total`, `compose_difference`)
4. compose families use the same short latent-trace style as ARC:
   `segment -> bind -> apply -> reduce -> render`
5. GSM8K-close variants now include chained-scale weekly total problems and
   fractional budget-gap problems without changing that trace shape

The GSM8K parser is supervision-time rather than question-only:

1. it reads the `<<expr=result>>` worked-solution tags already present in GSM8K
2. it maps `+` / `-` / `*` / `/` finals into the ARC-style families
   `compose_total`, `compose_difference`, `rate_scale`, and
   `partition_inverse`
3. when tags are missing or end in alias-only restatements such as `<<6=6>>`,
   it falls back to plain `=` equations in the worked solution text
4. it turns earlier worked equations into local derivation rules
5. it preserves exact rational literals such as `45/55` in quantity metadata so
   symbolic replay does not lose precision
6. trainable exports default to `train`, and official `test` rows require
   explicit opt-in via `--allow-eval-splits`

The MMLU parser is intentionally narrower:

1. it reads the official MMLU `dev/`, `val/`, and `test/` CSV layout plus the
   `auxiliary_train/` rows from the Hendrycks test archive
2. it currently targets eleven ARC-style multiple-choice families:
   `quantitative_choice`, `factual_choice`, `case_application_choice`,
   `passage_reference_choice`, `negation_exception_choice`,
   `completion_choice`, `rule_application_choice`,
   `statement_evaluation_choice`, `descriptor_match_choice`,
   `comparative_inference_choice`, and `concept_identification_choice`
3. it maps each row into the same `AbstractReasoningTask` schema by using
   `answer_format="multiple_choice"` plus typed `choices`
4. it keeps the traces short and ARC-like:
   `segment -> bind -> match -> select -> render` for quantitative, factual,
   completion, descriptor-match, and concept-identification rows, and
   `segment -> bind -> eliminate -> select -> render` for the
   elimination-oriented MMLU subfamilies
5. it compiles MMLU rows into the same canonical `TrajectoryRecord` structure
   as ARC and GSM8K, with local rewards on evidence matching and final option
   selection
6. trainable exports default to `auxiliary_train`, and official benchmark rows
   require explicit opt-in via `--allow-eval-splits`
