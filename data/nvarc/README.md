# NVARC Data

This workspace now includes the public NVARC solution repo and its source-data submodules under `external/NVARC`.

Cloned public sources:

- `external/NVARC`
- `external/NVARC/external/ARC-AGI-2`
- `external/NVARC/external/h-arc`
- `external/NVARC/external/BARC`
- `external/NVARC/external/MINI-ARC`
- `external/NVARC/external/ConceptARC`
- `external/NVARC/external/re-arc`
- `external/NVARC/external/TinyRecursiveModels`

Kaggle-hosted datasets referenced by the solution can be downloaded into `data/nvarc/kaggle` with:

```bash
python3 scripts/download_nvarc_kaggle_data.py
```

Requirements:

- `kaggle` CLI installed
- Kaggle credentials in `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`

Referenced Kaggle datasets:

- `sorokin/nvarc-artifacts-puzzles`
- `sorokin/nvarc-synthetic-puzzles`
- `sorokin/nvarc-augmented-puzzles`
- `cpmpml/arc-prize-trm-training-data`
- `cpmpml/arc-prize-trm-evaluation-data`

Note:

- The public page for `cpmpml/arc-prize-trm-evaluation-data` returned `HTTP 404` when checked from this environment on 2026-03-23.
