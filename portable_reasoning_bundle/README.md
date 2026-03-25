# Portable Reasoning Bundle

Vendored subset of the repo's reasoning-data layer for quick reuse in another
project.

Included:

- `arc_trajectory_sampler/`: ARC stage 1-4 latent/episode/execution/trajectory
  pipeline
- `arc_trajectory_sampler/state_adapter.py`: structured workspace and verifier
  serialization
- `arc_trajectory_sampler/reasoning_ir.py`: shared reasoning IR
- `arc_trajectory_sampler/gsm8k_reasoning_parser.py`: GSM8K-to-trajectory
  adapter
- `arc_trajectory_sampler/mmlu_parser.py`: MMLU-to-trajectory adapter
- `arc_trajectory_sampler/mixed_reasoning_dataset.py`: mixed ARC/GSM8K/MMLU
  text example builder
- `arc_trajectory_sampler/word_problem_translation_dataset.py`: synthetic word
  problem augmentation helpers
- `arc_trajectory_sampler/trm_dataset_export.py`: simple dataset export helpers

Not included:

- repo-local benchmark runners in `experiments/`
- evaluation scripts, plots, and cached `results/`
- benchmark datasets themselves

## Install

From the other project:

```bash
pip install -e /path/to/portable_reasoning_bundle
```

Or copy the `arc_trajectory_sampler/` package directory directly into that
project and add it to `PYTHONPATH`.

## Quick Use

ARC trajectory generation:

```python
from arc_trajectory_sampler import sample_latent_rule, sample_episode, execute_episode, build_trajectories

latent = sample_latent_rule(seed=0)
episode = sample_episode(latent, seed=0)
executed = execute_episode(episode)
trajectories = build_trajectories(executed)
```

Structured ARC state text:

```python
from arc_trajectory_sampler import encode_workspace, serialize_workspace_text, verifier_targets

trajectory = trajectories[0]
encoded = encode_workspace(trajectory, step_index=0, include_verifier=True)
text = serialize_workspace_text(encoded)
targets = verifier_targets(trajectory, step_index=0)
```

Mixed benchmark examples:

```python
from arc_trajectory_sampler import (
    build_arc_reasoning_examples,
    build_gsm8k_reasoning_examples,
    build_mmlu_reasoning_examples,
)
```

`build_gsm8k_reasoning_examples(...)` and `build_mmlu_reasoning_examples(...)`
expect benchmark data directories from the destination project. This bundle does
not ship benchmark data.
