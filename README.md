# ARC-AGI Reasoning Workspace

This repository is currently a benchmark and synthetic-supervision workspace for building toward a frontier reasoning system. The implemented pieces today are:

- `arc_trajectory_sampler/`: synthetic ARC trajectories, a shared reasoning IR, dataset adapters for ARC, GSM8K, MMLU, MMLU-Pro, MMLU-Redux, OlymMATH, and a structured CoRe ingestion path with an explicit graph extractor, a Python-AST parser-backed graph path with heuristic fallback, trace/source supervision, a native Oscar-scope corpus path for process-intelligence specs and meeting notes, and a first-class canonical Oscar graph reasoning benchmark
- `experiments/`: lightweight offline RL and ARC-sequence baselines used to test training loops and supervision formats
- `models/`: a reference decoder-only transformer stack with pluggable attention backends, including a local-global hybrid path, reference scale-invariant attention variants, an MLA-inspired latent-KV path, an MLA+SIA hybrid path, selective suffix-layer SIA routing, prefill-only SIA decode mode, a reference sparse MoE feedforward path, and a shared benchmark-aware reasoning-budget policy layer above the low-level inference budget interface
- `external/NVARC/` and `arc-agi-2/`: public ARC references and evaluation data

The repo still does not implement most of the frontier stack from the target spec: there is no production MLA kernel, no lightning attention backend, no multi-token prediction path, and no mixed-domain RL post-training pipeline. The current codebase now has a reference decoder substrate, a reference sparse MoE option, a benchmark-aware reasoning-budget policy scaffold, a mixed-benchmark supervised training path, MMLU-Pro and MMLU-Redux evaluation loaders, and a structured CoRe path with explicit graph extraction plus trace/source supervision, but it is still best understood as the benchmark and supervision layer around the future model rather than the finished frontier stack.

Read `DEVELOPMENT_MANIFESTO.md` first if you are deciding what to build next.

See `docs/current_system_spec.md` for the implemented system spec, and `docs/spec_alignment.md` for the explicit gap analysis and recommended build sequence.

For the current single-node overnight language-model recipe, start with `docs/overnight_8h100_epiplex_recipe.md` and use `scripts/run_epiplex_generic_8h100_overnight.sh`.

## Setup

For the base ARC sampler, parser, and cart-pole work:

```bash
python3 -m pip install -r requirements.txt
```

For Atari experiments, use the separate extras in `requirements-atari.txt` and `requirements-atari-pixel.txt`.
For the reference decoder stack, install `requirements-models.txt`.

## Smoke Check

Run the fast repo health check from the root:

```bash
bash scripts/smoke_check.sh
```

To verify the repo is ready for a packed pretraining run end to end:

```bash
bash scripts/pretraining_ready_check.sh
```

This builds a compact large-profile corpus with byte tokenization, runs the
document and packed-manifest preflights, packs the shards, and finishes with a
1-step packed-LM training smoke on the same artifact chain the larger run uses.

For the optional Torch-backed model smoke check:

```bash
python3 scripts/model_stack_smoke.py
```

For a focused OlymMATH parser audit over the official benchmark rows:

```bash
python3 arc_trajectory_sampler/evaluate_olympiad_math_parser.py \
  --configs en-easy en-hard \
  --max-rows 100
```

This evaluates the official `RUC-AIBOX/OlymMATH` rows, canonicalizes open-answer scalar and interval targets with SymPy, and emits a coverage summary plus sample Olympiad trace scaffolds.

To audit the `lean` config and informal/formal concordance path:

```bash
python3 arc_trajectory_sampler/evaluate_olympiad_math_parser.py \
  --configs lean \
  --max-rows 20
```

`lean` rows are treated as formal-proof concordance tasks: the parser keeps the bilingual informal theorem text, normalizes the Lean theorem statement, preserves the full Lean proof artifact, and emits concordance metadata instead of SymPy answer metadata.

To export OlymMATH as a structured pretraining trace corpus:

```bash
python3 scripts/export_olympiad_math_trace_corpus.py \
  --configs en-easy en-hard lean \
  --max-rows 100
```

This writes `REASONING_STATE_V1` trajectory text as JSONL and splits the source labels into `olympiad_math_open` and `olympiad_math_lean`, so the corpus can be mixed as two separate trace bands during continued pretraining.

To export a broader mixed `Ptraj` corpus across ARC, GSM8K, benchmark-safe MMLU, OlymMATH open/lean, and CoRe:

```bash
python3 scripts/export_ptraj_corpus.py \
  --arc-episodes 32 \
  --gsm8k-max-rows 96 \
  --mmlu-max-rows 96 \
  --olympiad-math-configs en-easy en-hard lean \
  --olympiad-math-max-rows 96 \
  --core-max-rows 96 \
  --shuffle-seed 0
```

This writes one JSONL manifest under `results/ptraj_corpus/` with per-step `REASONING_STATE_V1` text plus a summary JSON. The default MMLU source is `auxiliary_train`, so the export stays on the benchmark-safe side of the current repo policy.

The mixed `Ptraj` export now includes Oscar doc-grounded reasoning tasks and canonical Oscar graph reasoning tasks by default when a local `oscar_design_docs/` workspace is available. Use `--oscar-scope-reasoning-max-examples 0` and `--oscar-graph-reasoning-max-examples 0` to disable those slices.

To export the Oscar process-intelligence spec and meeting notes as a native domain corpus:

```bash
python3 scripts/export_oscar_scope_corpus.py \
  --max-documents 16 \
  --max-chunks 256
```

When a sibling `oscar_design_docs/` workspace is present, this exporter auto-discovers it and emits native text chunks plus section-outline records under `results/oscar_scope_corpus/`. This path is intentionally separate from `Ptraj`: it is the domain `Pgen` side for training the model inside the Oscar scope, not the structured reasoning-trace side.

To export deterministic Oscar doc-grounded reasoning traces:

```bash
python3 scripts/export_oscar_scope_reasoning_corpus.py \
  --max-documents 16 \
  --max-examples 256
```

This writes `REASONING_STATE_V1` terminal-answer tasks under `results/oscar_scope_reasoning_corpus/`. The current task families are chunk-to-section grounding, outline-next-heading prediction, and canonical concept tagging. Each trace now also carries local Oscar graph metadata over document nodes, section nodes, concept nodes, section-parent edges, and related-document edges so the same export can serve as both structured `Ptraj` text and graph-conditioned auxiliary supervision.

The same benchmark now also includes workflow-focused families derived from business case-study and workflow docs: environment anchoring, KPI tags, bottleneck tags, improvement tags, KPI-to-improvement supervision, multi-step intervention traces that sequence toward better KPI outcomes, and cross-case analogy/transfer tasks that abstract shared workflow motifs across different business environments. To export only those workflow traces:

```bash
python3 scripts/export_oscar_workflow_reasoning_corpus.py \
  --max-examples 512
```

This writes workflow-heavy `REASONING_STATE_V1` traces under `results/oscar_workflow_reasoning_corpus/`, interleaving the currently supported real environments: GenAI customer support, CPG order-to-cash, VC portfolio support, and the private-equity thesis-to-value lifecycle. The workflow export now includes `oscar_workflow_intervention_trace`, a three-step `decision_action` family that picks a focus KPI, applies a primary intervention, then applies a follow-up intervention, with projected reward score/bucket labels tied to KPI movement and bottleneck relief. It also includes `oscar_workflow_case_analogy` and `oscar_workflow_transfer`, which pair different business cases through shared workflow motifs and ask the model to transfer interventions across domains. Those action traces now use a canonical workflow action surface, with shared KPI-family and intervention-family labels across environments, so held-out transfer evaluation is not blocked by purely domain-specific action IDs. Larger exports will naturally skew toward whichever source documents have the most usable chunks, and capped exports may land slightly below `--max-examples` because intervention trajectories are kept complete instead of being cut mid-sequence.

To measure whether those workflow abstractions survive a strict held-out business environment split:

```bash
python3 scripts/evaluate_oscar_workflow_holdout.py \
  --tokenizer byte \
  --holdout-environments all \
  --max-examples 256 \
  --steps 20
```

This trains only on workflow traces from non-heldout environments, then evaluates the held-out environment on three surfaces at once: language-model loss over the structured traces, Oscar workflow auxiliary metrics such as motif/KPI/reward prediction, and strict `decision_action` coverage/accuracy for intervention transfer. The important strictness detail is that the decision-action vocabulary is built from the non-heldout environments only, so the summary exposes whether the held-out workflow still maps into the learned intervention/action surface or falls outside it.

To export the canonical Oscar process graph reasoning benchmark directly:

```bash
python3 scripts/export_oscar_graph_reasoning_corpus.py \
  --max-examples 256
```

This writes `REASONING_STATE_V1` traces under `results/oscar_graph_reasoning_corpus/`. The current task families are relation classification, neighbor enumeration, path completion, process-to-graph grounding, executor-style rollout tasks over canonical Oscar process motifs, and branching executor-trace `decision_action` records with frontier updates. The rollout and trace families tie graph-native paths back to matched Oscar excerpts when possible, so the abstraction path is grounded in surrounding process prose rather than only in bare node labels.

To build the unified pretraining corpus manifest across native text (`Pgen`),
reasoning traces (`Ptraj`), and held-out benchmark probes (`Pbench`):

For the shortest tech-heavy path, use the built-in preset:

```bash
python3 scripts/build_pretraining_manifest.py \
  --output-dir results/unified_manifest_tech \
  --preset tech_reasoning \
  --preset-scale pilot
```

That preset turns on benchmark-safe train `pgen`, mixed `ptraj`, Oscar native
and reasoning slices, CoRe, and the held-out benchmark probe groups. Use
`--preset-scale smoke` for a fast validation build or `--preset-scale large`
for a much larger technical mix.

```bash
python3 scripts/build_pretraining_manifest.py \
  --output-dir results/unified_manifest \
  --include-dclm \
  --dclm-max-documents 256 \
  --include-benchmark-train-pgen \
  --include-ptraj \
  --arc-episodes 16 \
  --gsm8k-max-rows 48 \
  --mmlu-max-rows 48 \
  --olympiad-math-max-rows 48 \
  --core-max-rows 48 \
  --include-gsm8k-bench \
  --gsm8k-bench-max-rows 48 \
  --include-mmlu-bench \
  --mmlu-bench-max-rows 48 \
  --include-mmlu-pro-bench \
  --mmlu-pro-max-rows 48 \
  --include-mmlu-redux-bench \
  --mmlu-redux-max-rows 48 \
  --include-olympiad-math-bench \
  --olympiad-math-bench-max-rows 48
```

This writes one central `manifest.json` with trainable `document_files` for the
packed-LM path plus separate `holdout_files` for benchmark-only probe bands.
With `--include-benchmark-train-pgen`, benchmark-safe training rows are also
included as native `pgen` documents:

- GSM8K `train`
- MMLU `auxiliary_train`
- OlymMATH train-side pretraining configs already used by the repo

Held-out benchmark material still stays isolated in `pbench`, so official test
or audit rows are not mixed into trainable `document_files`. Each row now
carries an explicit `band` field, so the same manifest can express `pgen`,
`ptraj`, and `pbench` without needing separate corpus formats.

Before using that manifest for a larger run, preflight the current design
target:

```bash
python3 scripts/preflight_large_pretraining_run.py \
  --manifest results/unified_manifest/manifest.json
```

This checks that:

- `pgen` dominates train mass
- `ptraj` is present but bounded
- `pbench` stays holdout-only

For a deployment-oriented corpus profile that sizes `pgen` and `ptraj`
independently, auto-runs the large-run preflight, optionally packs the corpus,
and emits a ready-to-run 8xH100 launch command:

```bash
python3 scripts/build_large_pretraining_profile.py \
  --output-dir results/large_pretraining_profile
```

This is the safer path when you are moving toward a real pretraining run
instead of doing corpus-mix experiments. The profile is local-first by default:

- benchmark-safe native train text for `pgen`
- local Oscar scope native text when `oscar_design_docs/` is present
- bounded ARC/GSM8K/MMLU/CoRe/Oscar reasoning traces for `ptraj`
- held-out GSM8K and MMLU audit probes for `pbench`

The builder writes:

- `manifest/manifest.json`
- `manifest_preflight.json`
- `packed/packed_manifest.json` when packing is enabled
- `packed_preflight.json` when packing is enabled
- `summary.json` with a recommended `launch_pretraining_lm_8h100.sh` command

To push checkpoints and final training artifacts to Hugging Face during the
packed-LM run, set these env vars before launch:

- `HF_UPLOAD_MODE=best_effort` or `required`
- `HF_UPLOAD_REPO_ID=yourname/your-model-or-dataset`
- optional `HF_UPLOAD_REPO_TYPE=model|dataset|space`
- optional `HF_UPLOAD_PATH_PREFIX=runs/<run_name>`
- optional `HF_UPLOAD_PRIVATE=1`

The trainer uploads each saved checkpoint plus final `summary.json`,
`progress.json`, and `summary.csv` from rank 0. Use `required` only when you
want the run to fail on upload errors.

For node-side launches, the pretraining launchers now auto-load repo-local env
files before applying defaults. The simplest deployment path is:

- copy [config/runtime/hf_upload.env.example](/Users/jjmayo/projects/demo_day/arc-agi/config/runtime/hf_upload.env.example) to `config/runtime/hf_upload.env.local`
- put your `HF_UPLOAD_*` values and `HF_TOKEN` there
- launch as usual; [launch_pretraining_lm_8h100.sh](/Users/jjmayo/projects/demo_day/arc-agi/scripts/launch_pretraining_lm_8h100.sh) and [launch_pretraining_lm_48h100.sh](/Users/jjmayo/projects/demo_day/arc-agi/scripts/launch_pretraining_lm_48h100.sh) source it automatically

`config/runtime/*.env.local` is gitignored, so the live token stays in your
working tree and off the tracked repo history.

If you need more native text than the local corpus can provide, opt into DCLM
explicitly:

```bash
python3 scripts/build_large_pretraining_profile.py \
  --output-dir results/large_pretraining_profile \
  --include-dclm \
  --dclm-max-documents 200000
```

For the fastest Oscar-specific bridge into the packed-LM path, build an
Oscar workflow adapter bundle:

```bash
python3 scripts/build_oscar_workflow_adapter_bundle.py \
  --output-dir results/oscar_workflow_adapter_bundle
```

This creates a workflow-heavy Oscar manifest plus packed token shards under one
directory and writes a `summary.json` with a ready-to-run
`train_pretraining_lm.py` command. The default bundle mixes:

- native Oscar scope text (`pgen`)
- workflow-focused Oscar reasoning traces (`ptraj`)
- canonical Oscar graph reasoning traces (`ptraj`)

The workflow reasoning slice is intentionally biased toward KPI tags,
bottleneck/improvement supervision, intervention traces, and cross-case
analogy/transfer so a larger model can absorb the Oscar workflow surface
tonight without needing the full integrated trainer path.

The current held-out `pbench` groups supported by the unified builder are:

- `gsm8k_test`
- `mmlu_audit`
- `mmlu_pro`
- `mmlu_redux`
- `olympiad_math_eval`

To pack that manifest into fixed-length token shards:

```bash
python3 scripts/pack_pretraining_corpus.py \
  --document-manifest results/unified_manifest/manifest.json \
  --output-dir results/unified_packed \
  --seq-len 1024
```

You can run the same preflight after packing, now against token counts:

```bash
python3 scripts/preflight_large_pretraining_run.py \
  --manifest results/unified_packed/packed_manifest.json
```

For cache-footprint and backend ablations across `sdpa`, `hybrid`, `sia`, `sia_hybrid`, `mla`, and `mla_sia`:

```bash
python3 scripts/model_stack_ablation.py
```

The model stack now has two named long-context presets:

- `mla_default`
- `mla_sia_prefill_l1`

To compare full-cache `mla_sia` against `prefill_only` decoding on long prompts:

```bash
python3 scripts/model_stack_ablation.py \
  --prompt-lengths 2048 4096 \
  --mla-latent-kv-dims 24 \
  --scale-invariant-last-n-layers 1 6 \
  --scale-invariant-decode-modes all_tokens prefill_only
```

For a tiny token-level training comparison on ARC-native text:

```bash
python3 scripts/model_stack_training_ablation.py
```

That now fits the shared Epiplex tokenizer on the ARC training texts before the
decoder sweep, so the ablation is no longer tied to the old byte vocabulary.
For an LM-facing structured ARC serialization built from the same
`workspace_state` and optional verifier targets used by the ARC sampler
benchmarks:

```bash
python3 scripts/model_stack_training_ablation.py \
  --arc-serialization structured_workspace \
  --arc-include-verifier-targets
```

`structured_workspace` emits one text record per ARC decision step, so its
example counts and token budgets are not directly comparable to the old
per-trajectory compact JSON mode. You can still force the legacy byte path with
`--tokenizer byte`, or persist / reuse fitted Epiplex vocabularies with
`--tokenizer-save PATH` and `--tokenizer-load PATH`.

For a combined report with JSON/CSV artifacts plus a cache-latency-loss plot:

```bash
python3 scripts/model_stack_report.py
```

By default this writes inference and training sweeps under `results/model_stack_report/`, along with `tradeoff_raw.csv`, `tradeoff_summary.csv`, `summary.json`, and `tradeoff_scatter.png`.

To sweep suffix-only scale-invariant attention on the SIA backends:

```bash
python3 scripts/model_stack_report.py \
  --lengths 512 1024 \
  --mla-latent-kv-dims 24 \
  --scale-invariant-last-n-layers 1 2 3 6
```

For a decode-aware ARC evaluation that trains small models, then scores exact-match and token accuracy with `prefill -> cached decode`:

```bash
python3 scripts/model_stack_decode_eval.py \
  --presets mla_default mla_sia_prefill_l1 \
  --decode-modes policy_default greedy context_trie schema_json context_then_schema \
  --reasoning-effort deep \
  --train-episodes 24 \
  --val-episodes 8
```

`policy_default` resolves to the ARC budget policy's structured-decode hint, which is currently `context_trie`. `context_trie` constrains cached decode to train-split `target_action=` candidates matched by prompt context. `schema_json` unions that with prompt-grounded action-schema candidates for `segment`, `pick_*`, `select`, and `render`. `context_then_schema` uses the exact context bucket when it exists and only falls back to schema candidates when it does not. These are benchmark-side structured decoding sidecars, not backbone changes.

For a focused summary of the CoRe graph extractor and query anchoring coverage:

```bash
python3 scripts/core_graph_probe.py \
  --dependency-kinds infoflow \
  --categories trace \
  --graph-backend auto \
  --max-examples 256
```

`auto` currently uses a parser-backed Python AST graph extractor when the snippet parses cleanly, and falls back to the older heuristic extractor for unsupported languages or parse failures. You can force `heuristic` or `python_ast` with `--graph-backend`.

For an integrated supervised training pass across ARC, GSM8K, and base MMLU, with optional MMLU-Pro / MMLU-Redux eval rows, OlymMATH evaluation rows, and a structured CoRe supervision slice, using either a dense or reference sparse-MoE decoder:

```bash
python3 scripts/train_integrated_reasoning_stack.py \
  --architectures dense moe \
  --attention-preset mla_default \
  --train-reasoning-effort fast \
  --eval-reasoning-effort deep \
  --arc-train-episodes 16 \
  --arc-val-episodes 4 \
  --gsm8k-max-rows 48 \
  --mmlu-max-rows 48 \
  --mmlu-pro-max-rows 48 \
  --mmlu-redux-max-rows 48 \
  --olympiad-math-max-rows 48 \
  --core-max-rows 48
```

This uses the shared `REASONING_STATE_V1` text interface across ARC, GSM8K, MMLU, and the new benchmark adapters, reports per-benchmark validation loss, exposes MoE router diagnostics when the sparse path is enabled, and attaches benchmark-specific effort presets that control active layers, prompt limits, cache sizing, and ARC's default structured decode mode. `MMLU-Pro`, `MMLU-Redux`, and `OlymMATH` remain evaluation-only in this scaffold. For corpus building, `OlymMATH` is better understood as a structured pretraining trace source with two variants: open-answer Olympiad rows and `lean` formal-proof concordance rows. `CoRe` now contributes query-level `trace` and `list_source` tasks with prompt-extracted code lines, explicit graph views, reconstructed traces, aggregated source sets, example-level auxiliary heads, and candidate-wise source-membership / direct-edge supervision; it is still not a full compiler-grade IR or a true node-span/edge-span supervision stack. When local Oscar docs are available, the trainer now treats both `oscar_scope_reasoning` and `oscar_graph_reasoning` as first-class training benchmarks rather than folding them into the native-text Oscar bucket. `oscar_scope_reasoning` now carries both doc-grounded section/document/concept supervision and workflow-heavy business-environment traces for KPI, bottleneck, improvement, intervention sequencing, and cross-case workflow abstraction, including projected reward labels. The Oscar auxiliary stack now supervises not just section/document metadata but also workflow KPI IDs, intervention IDs, workflow motif IDs, and workflow reward bucket/score targets. `oscar_graph_reasoning` trains directly on canonical process-graph relations, neighbors, paths, process-to-graph grounding, executor-style motif rollouts, and branching executor-trace `decision_action` updates. The Oscar graph auxiliary stack now includes candidate-wise graph heads for relation selection, neighbor-set prediction, path via/target prediction, canonical-node grounding, rollout motif prediction, and stepwise rollout-node prediction, while the shared decision-action head now also sees Oscar graph executor traces with explicit frontier state. It is a supervised integration scaffold, not the mixed-domain RL stack from the target spec.

Important distinction: the architecture supports both `mla_default` and the
research preset `mla_sia_prefill_l1`, but the integrated trainer default and
the saved integrated benchmark runs in this repo use `mla_default`. The
`mla_sia_prefill_l1` preset is mainly used for explicit long-context/decode
comparisons; it is not the current standard integrated operating point.

If a sibling `oscar_design_docs/` workspace is present, the trainer now auto-discovers three Oscar benchmarks by default: native-domain `oscar_scope` text, structured `oscar_scope_reasoning` traces, and canonical `oscar_graph_reasoning` traces. Use `--no-include-oscar-scope` to disable the native text path, `--no-include-oscar-scope-reasoning` to disable the doc-grounded reasoning path, `--no-include-oscar-graph-reasoning` to disable the canonical graph path, or adjust the slices with `--oscar-scope-max-documents`, `--oscar-scope-max-chunks`, `--oscar-scope-roots`, `--oscar-scope-paths`, `--oscar-scope-reasoning-max-examples`, `--oscar-scope-reasoning-families`, `--oscar-graph-reasoning-max-examples`, and `--oscar-graph-reasoning-families`.

Progress bars are enabled by default with `tqdm`. If you pass `--output` or `--csv-output`, the trainer also writes `progress.json` plus rolling `partial_summary.json` / `partial_summary.csv` siblings beside the final summary files, so long integrated runs can be monitored before every architecture finishes. You can tune snapshot frequency with `--checkpoint-every N` or disable terminal bars with `--no-progress`.

The integrated trainer now defaults to the shared Epiplex tokenizer path rather
than raw UTF-8 bytes. Use `--tokenizer-vocab-size`, `--tokenizer-save`, and
`--tokenizer-load` to control corpus-scale tokenizer fitting, or
`--tokenizer byte` only for quick compatibility checks.

You can switch CoRe graph extraction with `--core-graph-backend auto|heuristic|python_ast`. `auto` is the recommended setting; it uses the parser-backed Python path where possible and keeps the heuristic fallback for Java, C, and unparsable snippets.

The variant loaders fetch from official Hugging Face-hosted sources:

- `MMLU-Pro`: `TIGER-Lab/MMLU-Pro`
- `MMLU-Redux`: `edinburgh-dawg/mmlu-redux-2.0`
- `CoRe`: `lt-asset/CoRe`
- `OlymMATH`: `RUC-AIBOX/OlymMATH`

Those paths require network access on first use unless the corresponding data has already been cached locally under `arc_trajectory_sampler/data/`.

To smoke-test the same combined data path without training a model:

```bash
python3 scripts/train_integrated_reasoning_stack.py \
  --data-only \
  --arc-train-episodes 16 \
  --arc-val-episodes 4 \
  --gsm8k-max-rows 48 \
  --mmlu-max-rows 48 \
  --mmlu-pro-max-rows 48 \
  --mmlu-redux-max-rows 48 \
  --core-max-rows 48 \
  --output results/integrated_reasoning_data_only/summary.json \
  --csv-output results/integrated_reasoning_data_only/summary.csv
```

This builds the real mixed ARC/GSM8K/MMLU/OlymMATH/CoRe text datasets, chunks them into token windows, writes example/window counts, and exits before any device or model setup.

## 3D Cart-Pole Offline RL Benchmark

The repo also contains a self-contained NumPy benchmark for testing the adapted FQI setup on a 3D cart-pole with four directional actions:

- `up`
- `down`
- `left`
- `right`

The environment models a cart moving on a plane while balancing a pole that can fall along both horizontal axes. Observations are 3D tensors with spatial heatmaps plus scalar channels. The benchmark compares:

- `FQI-LOG`: sigmoid Q-head trained with log-loss
- `FQI-SQ`: the same sigmoid Q-head trained with square loss

## Run

```bash
python3 experiments/three_d_cartpole_experiment.py
```

The default configuration runs a sweep over `32`, `64`, and `128` offline training episodes, averages across `4` random seeds, and writes outputs to:

- `results/three_d_cartpole/raw_results.csv`
- `results/three_d_cartpole/summary_results.csv`
- `results/three_d_cartpole/summary.json`
- `results/three_d_cartpole/comparison.png`

For a larger sweep that extends to `1000` offline episodes:

```bash
python3 experiments/three_d_cartpole_experiment.py \
  --dataset-sizes 128 256 512 1000 \
  --eval-episodes 500 \
  --output-dir results/three_d_cartpole_long_run
```

## Notes

- The benchmark is dependency-light and uses only `numpy` and `matplotlib`.
- The default encoder is a deterministic summary encoder computed from the tensor observation, which keeps the comparison focused on the loss function rather than on a heavy neural network stack.
- You can switch to a random feature encoder with `--feature-mode random`.

## Atari RAM Benchmark

There is also an Atari benchmark that compares `FQI-LOG` and `FQI-SQ` on ALE Atari `RAM` observations for `Asterix` and `Seaquest`.

Setup:

```bash
python3 -m venv .venv_atari
. .venv_atari/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-atari.txt
```

Run:

```bash
. .venv_atari/bin/activate
python experiments/atari_ram_fqi_benchmark.py
```

This benchmark uses:

- real ALE Atari environments
- `RAM` observations instead of pixels, so the comparison stays lightweight
- normalized discounted positive-reward return as the FQI target
- the same sigmoid Q-head trained with either log-loss or square loss

## Atari Pixel Benchmark

There is also a pixel-based Atari benchmark with a small convolutional Q-network and stacked grayscale `84x84` inputs.

Setup:

```bash
python3 -m venv .venv_atari
. .venv_atari/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-atari-pixel.txt
```

Run:

```bash
. .venv_atari/bin/activate
python experiments/atari_pixel_fqi_benchmark.py
```

Run explicitly on Apple Metal (`mps`):

```bash
. .venv_atari/bin/activate
python experiments/atari_pixel_fqi_benchmark.py --device mps
```

Each training run can also save one short evaluation trajectory panel per FQI iteration as an `mp4` under `results/.../traces/`. By default the panel shows `2` rollout seeds side by side, updates a stable `latest.mp4`, and writes an auto-refresh `index.html` you can keep open while training. This is enabled by default and can be disabled with `--no-trace-trajectories`.
Progress bars for dataset collection, benchmark jobs, and FQI iterations are also enabled by default and can be disabled with `--no-progress`.
`mp4` export requires `ffmpeg` in `PATH`; use `--trajectory-format gif` if needed.
The current benchmark defaults are also more aggressive about throughput: threaded episode collection, async trace encoding, adaptive device-side pixel caching, and a rate-limited live preview path. You can tune those with `--collect-workers`, `--trace-write-workers`, `--trace-write-queue`, `--device-cache-max-mb`, `--live-preview-refresh-ms`, and `--live-preview-jpeg-quality`.

For a longer traced run on Apple Metal:

```bash
. .venv_atari/bin/activate
python experiments/atari_pixel_fqi_benchmark.py \
  --games Asterix \
  --dataset-sizes 50 100 \
  --seeds 2 \
  --eval-episodes 10 \
  --device mps \
  --fqi-iterations 10 \
  --optimizer-epochs 4 \
  --collect-workers 4 \
  --trace-write-workers 1 \
  --trace-write-queue 2 \
  --device-cache-max-mb 768 \
  --trajectory-panel-rollouts 2 \
  --trajectory-max-steps 600 \
  --trajectory-frame-stride 2 \
  --trajectory-max-frames 180 \
  --output-dir results/atari_pixel_fqi_mps_long
```

For a true multi-process seed launch with one aggregate dashboard:

```bash
. .venv_atari/bin/activate
python experiments/atari_pixel_fqi_multi_seed_launcher.py \
  --games Asterix \
  --dataset-sizes 50 100 \
  --seeds 2 \
  --max-parallel 2 \
  --device mps \
  --fqi-iterations 10 \
  --optimizer-epochs 4 \
  --collect-workers 4 \
  --trace-write-workers 1 \
  --trace-write-queue 2 \
  --device-cache-max-mb 768 \
  --trajectory-panel-rollouts 2 \
  --trajectory-max-steps 600 \
  --trajectory-frame-stride 2 \
  --trajectory-max-frames 180 \
  --output-dir results/atari_pixel_fqi_parallel
```

This launcher spawns one worker process per seed, keeps each worker under `results/.../seed_runs/seed_<N>/`, and writes a root auto-refresh dashboard at `results/.../index.html` that embeds each seed's current trajectory view. While an eval rollout is in progress, the card switches to a live-updating preview from `live/current.jpg`; once the rollout settles, it falls back to the final `latest.mp4`.

If you want lower overhead while keeping the live feel, increase `--live-preview-refresh-ms` or lower `--live-preview-jpeg-quality`. The terminal benchmark progress bars are still available when running `experiments/atari_pixel_fqi_benchmark.py` directly, and the multi-seed launcher now shows a parent worker progress bar as well.

This variant uses:

- raw ALE pixel observations
- grayscale resize and 4-frame stacking
- a compact convolutional Q-network implemented in `torch`
- explicit `mps` support on Apple Silicon
- per-iteration trajectory MP4 panels plus a live trace dashboard
- the same `log-loss` vs `square-loss` comparison on a bounded positive-reward objective

## ARC Sampler FQI Benchmark

There is also a lightweight ARC-trajectory benchmark that applies the same `FQI-LOG` vs `FQI-SQ` comparison to the synthetic ARC sampler in `arc_trajectory_sampler/`.

It trains on the sampler's `train` trajectories, evaluates closed-loop on held-out sampler `test` trajectories, and treats the action space as a canonicalized ARC reasoning-operator vocabulary such as `segment`, `select`, `transform:rotate_90`, `relate:align_to`, and `branch:then`.

Run:

```bash
python3 experiments/arc_sampler_fqi_benchmark.py
```

Progress bars are enabled by default with `tqdm` and can be disabled with `--no-progress`.

To smoke-test the same ARC FQI data-generation path without fitting either loss:

```bash
python3 experiments/arc_sampler_fqi_benchmark.py \
  --dataset-sizes 8 16 \
  --seeds 1 \
  --eval-episodes 16 \
  --data-only \
  --output-dir results/arc_sampler_fqi_data_only
```

This writes `data_only_summary.csv`, `episode_timings.csv`, `episode_timing_summary.csv`, and `summary.json`, then exits before the FQI `log` / `sq` fit loops.

The default configuration sweeps `32`, `64`, and `128` sampled training episodes, averages across `2` seeds, and writes:

- `results/arc_sampler_fqi/raw_results.csv`
- `results/arc_sampler_fqi/summary_results.csv`
- `results/arc_sampler_fqi/summary.json`
- `results/arc_sampler_fqi/comparison.png`

This ARC benchmark uses a phase-aware action mask in the Bellman backup and evaluation policy, keyed by `(family, trace step index)`, so FQI compares plausible ARC reasoning operators instead of arbitrary out-of-support actions.
The summary plot now also includes `Exact Answer (Abstract)`, which measures whether the answer-determining abstract ARC choices match on the held-out test trajectory, and `Final Grid Exact`, which re-executes the predicted abstract choices through the family-specific ARC simulator and checks the resulting output grid. `Final Grid Exact` is meaningfully stronger than the old abstract-answer metric, but it still lives inside the current abstract operator vocabulary rather than a fully open-ended ARC program search.

For a longer ARC sweep:

```bash
python3 experiments/arc_sampler_fqi_benchmark.py \
  --dataset-sizes 128 256 512 1024 \
  --seeds 4 \
  --eval-episodes 256 \
  --fqi-iterations 10 \
  --optimizer-steps 40 \
  --projection-dim 256 \
  --output-dir results/arc_sampler_fqi_longer
```

For a structured-state FQI ablation that uses the object-scene workspace plus a
reconstructed verifier signal instead of the flat-grid encoder:

```bash
python3 experiments/arc_sampler_fqi_benchmark.py \
  --dataset-sizes 128 256 512 1024 \
  --seeds 4 \
  --eval-episodes 256 \
  --fqi-iterations 10 \
  --optimizer-steps 40 \
  --projection-dim 256 \
  --state-encoder structured_reconstructed \
  --output-dir results/arc_sampler_fqi_structured_reconstructed
```

As in the BC benchmark, `structured_reconstructed` keeps the policy/Q head
unchanged and fits the verifier only on the current training subset for each
dataset-size point. The default verifier model is now `random_features`, which
adds a lightweight nonlinear ridge layer over standardized workspace features.
Use `--verifier-model linear` to recover the older linear sidecar.

For a higher-capacity ARC sweep with a Torch MLP Q head:

```bash
.venv_atari/bin/python experiments/arc_sampler_fqi_capacity_benchmark.py \
  --dataset-sizes 128 256 512 \
  --seeds 4 \
  --eval-episodes 256 \
  --fqi-iterations 8 \
  --optimizer-epochs 6 \
  --batch-size 256 \
  --target-batch-size 2048 \
  --projection-dim 256 \
  --hidden-dims 512 256 \
  --device auto \
  --output-dir results/arc_sampler_fqi_capacity
```

The capacity benchmark also supports the same structured ARC state ablation as
the linear FQI benchmark:

```bash
.venv_atari/bin/python experiments/arc_sampler_fqi_capacity_benchmark.py \
  --dataset-sizes 128 256 512 \
  --seeds 4 \
  --eval-episodes 256 \
  --fqi-iterations 8 \
  --optimizer-epochs 6 \
  --batch-size 256 \
  --target-batch-size 2048 \
  --projection-dim 256 \
  --hidden-dims 512 256 \
  --state-encoder structured_reconstructed \
  --output-dir results/arc_sampler_fqi_capacity_structured
```

For a masked behavior-cloning baseline on the same ARC sampler traces:

```bash
python3 experiments/arc_sampler_bc_benchmark.py \
  --dataset-sizes 128 256 512 1024 \
  --seeds 4 \
  --eval-episodes 256 \
  --epochs 40 \
  --batch-size 256 \
  --projection-dim 256 \
  --output-dir results/arc_sampler_bc
```

This benchmark keeps the same ARC state encoder, phase-aware action masks, and held-out evaluation as the FQI benchmark, including the `Final Grid Exact` metric, but replaces Bellman updates with direct masked next-action prediction from expert traces.

To smoke-test the same BC data-generation path without fitting the policy:

```bash
python3 experiments/arc_sampler_bc_benchmark.py \
  --dataset-sizes 8 16 \
  --seeds 1 \
  --eval-episodes 16 \
  --data-only \
  --output-dir results/arc_sampler_bc_data_only
```

For a structured-state oracle ablation that reads `workspace_state` plus
trajectory-side verifier labels instead of the flat grid encoder:

```bash
python3 experiments/arc_sampler_bc_benchmark.py \
  --dataset-sizes 128 256 512 1024 \
  --seeds 4 \
  --eval-episodes 256 \
  --epochs 40 \
  --batch-size 256 \
  --projection-dim 256 \
  --state-encoder structured_oracle \
  --output-dir results/arc_sampler_bc_structured
```

`structured_oracle` is intentionally an encoder ablation, not a deployable test-time state estimator: it consumes the object-scene `workspace_state` and verifier fields already attached to the synthetic trajectories.

For the verifier-reconstructed variant, which first fits a lightweight
auxiliary verifier on top of workspace-only structured features and then feeds
those predictions back into the same structured encoder surface:

```bash
python3 experiments/arc_sampler_bc_benchmark.py \
  --dataset-sizes 128 256 512 1024 \
  --seeds 4 \
  --eval-episodes 256 \
  --epochs 40 \
  --batch-size 256 \
  --projection-dim 256 \
  --state-encoder structured_reconstructed \
  --output-dir results/arc_sampler_bc_structured_reconstructed
```

`structured_reconstructed` keeps the policy head unchanged and avoids reading the stored verifier labels directly at policy time. The default verifier model is `random_features`, a lightweight nonlinear ridge sidecar over standardized workspace features; use `--verifier-model linear` to recover the older linear fit.

For a higher-capacity masked behavior-cloning baseline with a Torch MLP policy:

```bash
.venv_atari/bin/python experiments/arc_sampler_bc_capacity_benchmark.py \
  --dataset-sizes 128 256 512 1024 \
  --seeds 4 \
  --eval-episodes 256 \
  --epochs 24 \
  --batch-size 256 \
  --projection-dim 256 \
  --hidden-dims 512 256 \
  --device auto \
  --output-dir results/arc_sampler_bc_capacity
```

For a mixed trace behavior-cloning benchmark over the action-bearing corpora in
the shared `REASONING_STATE_V1` format, combining ARC, GSM8K, and MMLU, with
optional MMLU-Pro / MMLU-Redux eval rows:

```bash
python3 experiments/mixed_trace_bc_benchmark.py \
  --arc-train-episodes 16 \
  --arc-val-episodes 4 \
  --gsm8k-max-rows 48 \
  --mmlu-max-rows 48 \
  --mmlu-pro-max-rows 48 \
  --mmlu-redux-max-rows 48 \
  --seeds 2 \
  --output-dir results/mixed_trace_bc
```

This benchmark trains a lightweight masked linear policy over hashed text
features from the serialized reasoning traces and reports per-benchmark action
accuracy plus exact-trace success. `CoRe` can be included with
`--core-max-rows`, but its records are counted and skipped from the action loss
because they currently expose `target_answer` rather than stepwise
`target_action`.

To smoke-test the same mixed data path without fitting:

```bash
python3 experiments/mixed_trace_bc_benchmark.py \
  --arc-train-episodes 2 \
  --arc-val-episodes 1 \
  --gsm8k-max-rows 4 \
  --mmlu-max-rows 4 \
  --core-max-rows 2 \
  --seeds 1 \
  --data-only \
  --output-dir results/mixed_trace_bc_data_only
```

For a higher-capacity version of the same mixed trace benchmark, using a Torch
MLP policy over the same hashed `REASONING_STATE_V1` text features:

```bash
.venv_atari/bin/python experiments/mixed_trace_bc_capacity_benchmark.py \
  --arc-train-episodes 16 \
  --arc-val-episodes 4 \
  --gsm8k-max-rows 48 \
  --mmlu-max-rows 48 \
  --mmlu-pro-max-rows 48 \
  --mmlu-redux-max-rows 48 \
  --seeds 2 \
  --epochs 24 \
  --batch-size 256 \
  --feature-dim 4096 \
  --hidden-dims 1024 512 \
  --device auto \
  --output-dir results/mixed_trace_bc_capacity
```

This keeps the same data path and metrics as the linear mixed trace baseline,
but gives the policy a substantially larger nonlinear head. `CoRe` is still
excluded from the action loss unless it eventually exposes stepwise
`target_action` supervision.

For a version that is closer to the integrated stack, using byte-token inputs,
the shared multi-head latent attention decoder blocks, a benchmark embedding,
and per-benchmark-family output heads:

```bash
.venv_atari/bin/python experiments/mixed_trace_latent_attention_benchmark.py \
  --arc-train-episodes 16 \
  --arc-val-episodes 4 \
  --gsm8k-max-rows 48 \
  --mmlu-max-rows 48 \
  --mmlu-pro-max-rows 48 \
  --mmlu-redux-max-rows 48 \
  --olympiad-math-max-rows 32 \
  --olympiad-math-configs en-easy en-hard \
  --seeds 2 \
  --epochs 12 \
  --batch-size 32 \
  --eval-batch-size 128 \
  --hidden-size 128 \
  --num-layers 3 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --intermediate-size 256 \
  --latent-kv-dim 32 \
  --max-seq-len 320 \
  --device auto \
  --output-dir results/mixed_trace_latent_attention
```

This benchmark shares the same mixed ARC/GSM8K/MMLU action-trace setup as the
BC baselines, but replaces the hashed-text encoder with the repo's latent-KV
attention stack and uses separate output heads for `arc`, `gsm8k`,
`olympiad_math`, and the shared `mmlu` family (`mmlu`, `mmlu_pro`,
`mmlu_redux`).
