# Development Manifesto

This is the central build-order document for the repository.

If you are deciding what to implement next, use this file before adding new
benchmarks, new model variants, or new training scripts.

## What The Repo Is Now

The repo is now a real research pretraining harness, but not yet a
production-scale one.

What is already real:

- a shared structured reasoning interface centered on `REASONING_STATE_V1`
- benchmark and corpus adapters for ARC, GSM8K, MMLU-family, OlymMATH, CoRe,
  and Oscar-derived corpora
- an Epiplex-aligned tokenizer path in `models/reasoning_tokenizer.py`
- a packed-LM path:
  `build_pretraining_manifest.py -> pack_pretraining_corpus.py -> train_pretraining_lm.py`
- a mixed structured-supervision path in
  `scripts/train_integrated_reasoning_stack.py`
- resumable distributed LM training, packed shards, progress files, and launch
  scripts

What it is not yet:

- a unified large-scale pretraining pipeline
- a production backbone stack
- a final post-training / RL stack
- a fully frozen, protocolized scaling environment

## The Main Problem

The repo still has two partially separate training worlds:

1. packed autoregressive LM pretraining over frozen document manifests
2. integrated structured supervision over reasoning traces and auxiliary heads

That split is acceptable for pilot work, but it is the main reason the repo is
not yet a clean large-scale harness.

## The Goal

Build one reproducible pretraining system with:

- one corpus plane
- one tokenizer artifact
- one experiment protocol
- one scalable launch/resume/checkpoint story
- one shared model stack that can absorb both plain LM and structured
  reasoning-side supervision

## Non-Negotiable Principles

### 1. Freeze Inputs Before Training

Serious runs must train from frozen manifests and frozen tokenizer artifacts.

Runtime fitting, ad hoc example sampling, and hidden data drift are acceptable
for smoke tests, but not for large experiments.

### 2. Prefer One Canonical Path Over Many Clever Paths

If two scripts solve the same pretraining problem in different ways, the repo is
getting weaker, not stronger.

The harness should converge toward one default corpus path and one default
training path.

### 3. Keep Research Ablations Separate From The Mainline

`mla_default` is the current mainline preset.

`mla_sia_prefill_l1` and other attention variants are research comparison paths.
They should remain easy to run, but they should not blur the default operating
point.

### 4. Make Scaling Reproducible

A run does not count as large-scale-ready unless:

- the corpus manifest is frozen
- the tokenizer is frozen
- resume works
- checkpointing works
- validation cadence is explicit
- throughput and data-wait metrics are recorded

### 5. Unify Corpora By Interface, Not By Hand-Written Exceptions

The repo should mix `Pgen`, `Ptraj`, and benchmark-style held-out sets through a
common manifest surface, not through one-off glue code in each trainer.

## Build Order

### Priority 1: Unify The Corpus Plane

Create one canonical corpus-manifest builder that can express:

- `Pgen`: native text
- `Ptraj`: reasoning traces
- `Pbench`: held-out probes and benchmark eval sets

This should replace the current split where the packed-LM path is strongest for
document-like corpora while the integrated trainer remains the natural home for
many reasoning traces.

Done means:

- one manifest schema for trainable and held-out corpora
- the packed path can ingest structured trace text as first-class training data
- benchmark probes are recorded in the same frozen run protocol

### Priority 2: Freeze Tokenization As An Artifact

The Epiplex path is the right direction. Now make it strict:

- fit once
- save once
- record the tokenizer artifact in the manifest
- require load-only for serious packing and training runs

Done means:

- no large run depends on runtime tokenizer fitting
- token counts are stable across reruns
- manifest and summary outputs always identify the tokenizer artifact used

### Priority 3: Keep A Two-Stage Training Protocol Until The Unified Path Is Stable

Near term, the simplest reliable setup is:

1. packed LM pretraining on frozen corpora
2. structured decision and auxiliary training on frozen reasoning corpora

Do not prematurely collapse everything into one trainer unless the resulting
path is actually simpler and more reproducible.

Done means:

- the two stages share tokenizer and corpus artifacts cleanly
- transferring from stage 1 to stage 2 is scripted and reproducible
- evaluation is comparable across runs

### Priority 4: Harden The Scale Path

The pretraining trainer already has the basic pieces. The next work is
operational:

- restart testing
- checkpoint retention policy
- divergence / NaN detection
- stable multi-node launch conventions
- explicit throughput and data-wait reporting
- one canonical launch path for burn-in and multi-node runs

Done means:

- single-node and multi-node resumes are routine
- failed runs can be restarted without manual surgery
- progress files are enough to diagnose slowdowns and stalls

### Priority 5: Freeze A Real Experiment Protocol

Before calling anything a large-scale experiment, define:

- frozen corpus manifest
- frozen tokenizer artifact
- fixed token budget
- fixed validation cadence
- fixed checkpoint cadence
- fixed probe suite

Done means:

- experiments are comparable by design
- scaling claims are attached to a concrete protocol, not a moving target

### Priority 6: Run A Scaling Ladder

The repo should advance through a fixed ladder:

1. single-node burn-in
2. `8xH100` stability run
3. `48xH100` pilot

Advance only if:

- resume works
- throughput is stable
- validation behaves normally
- checkpoints are usable

## Immediate Default Direction

Until the corpus plane is unified, the default development stance should be:

- use the packed-LM path for serious pretraining infrastructure work
- use the integrated trainer for structured reasoning-side supervision work
- avoid adding parallel new training entrypoints unless they clearly replace an
  existing one

## What Should Not Happen Next

Do not spend the next phase on:

- adding more isolated benchmark-specific trainers
- adding more model variants before the corpus and protocol surfaces are stable
- claiming large-scale readiness from smoke tests or pilot runs

The next bottlenecks are infrastructure and protocol, not a lack of benchmark
coverage.

## Working Definition Of Success

The repo becomes a strong large-scale pretraining harness when:

- one frozen manifest can describe the train and held-out corpus mixture
- one frozen tokenizer artifact is used end to end
- one mainline trainer can run packed pretraining reproducibly at scale
- structured reasoning supervision plugs into that path cleanly
- launch, checkpoint, resume, and evaluation are routine rather than bespoke

Until then, treat the repo as a strong research harness in transition, not a
finished large-scale system.
