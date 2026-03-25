# Full-Scale Reasoning Architecture Review

This document reviews the current design against the actual target:

- a full-scale LLM with strong reasoning performance
- strong benchmark performance on GSM8K, MMLU, and ARC-AGI-2
- retained small-model language ability in the style of NanoChat
- a specialized-pretraining and late-injection schedule informed by
  `trajectory_spt_protocol.pdf`

It is intentionally critical. The goal is to identify what should be preserved,
what should be changed, and what is currently misaligned with the end state.

## Bottom Line

The current system is good as a reasoning-data compiler and benchmark-side
substrate.

It is not yet the right end-to-end design for a full-scale reasoning LLM.

The strongest parts are:

- the shared typed reasoning IR
- the shared trajectory schema
- the GSM8K and MMLU supervision adapters
- the action-centric decode-aware evaluation loop

The weakest parts are:

- operational benchmark/train separation
- incomplete migration onto the new reasoning-state serializer
- the continued use of byte-level training text in the model ablations
- the absence of a real ARC-AGI-2 task-level parser / solver path
- over-indexing on `target_action=` as if it should replace all other training
  views

## Findings

### 1. Benchmark contamination defaults are fixed, but the separation still matters

This was the biggest architectural risk, and the default parser/export paths are
now guarded.

The trainable builders now do the right thing by default:

- MMLU trainable exports now default to `auxiliary_train`, while official
  benchmark splits require explicit opt-in via `--allow-eval-splits` in
  [mmlu_parser.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/mmlu_parser.py#L1043).
- GSM8K trainable exports now default to `train`, while official `test`
  requires explicit opt-in via `--allow-eval-splits` in
  [gsm8k_reasoning_parser.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/gsm8k_reasoning_parser.py#L202).

That fixes the default-footgun. The remaining requirement is operational:
trainable builders and audit builders still need to stay distinct in the
training pipeline and experiment tracking.

Implication:

- official GSM8K test, MMLU test, and ARC-AGI-2 evaluation tasks must be hard
  excluded from any trainable corpus path
- MMLU `dev` and `val` should also be treated as benchmark-only for model
  development, not generic pretraining text
- the repo should continue distinguishing `audit/eval` builders from
  `trainable` builders

### 2. The LM interface boundary is now correct, but the script stack is lagging

The serializer boundary now emits `REASONING_STATE_V1` from the modality-neutral
adapter in
[state_adapter.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/state_adapter.py).

That fixes the prompt-family problem at the adapter layer. The remaining gap is
that the training and evaluation scripts are still partly named and scoped
around ARC experiments, and the production tokenizer path is not wired through
yet.

What should survive:

- short structured prompts
- explicit `target_action=`
- verifier context

What should change:

- the training and decode scripts should move fully onto the shared adapter
- the Epiplex tokenizer path should replace byte-level text in the actual model
  stack

### 3. The current model ablation path is not aligned with the tokenizer decision

The training ablation still uses:

- a byte vocabulary of size 257 in
  [model_stack_training_ablation.py](/Users/jjmayo/projects/demo_day/arc-agi/scripts/model_stack_training_ablation.py#L27)
- raw UTF-8 byte encoding in
  [model_stack_training_ablation.py](/Users/jjmayo/projects/demo_day/arc-agi/scripts/model_stack_training_ablation.py#L205)

That was fine for quick decoder experiments, but it is not aligned with the
Epiplex tokenizer decision and should not become the long-term training path.

Implication:

- the current decode-aware results are useful for attention-backend comparison
- they are not yet a faithful readout of the future large-model data interface

### 4. The design is in danger of over-unifying the prompt space

The new `REASONING_STATE_V1` direction is correct for structured reasoning
supervision.

It is not correct to force all pretraining data into that format.

If the model is supposed to retain general language ability in the style of
NanoChat, then the corpus must remain multi-view:

- native language modeling and chat text for broad language competence
- structured reasoning-state text for trajectory supervision
- benchmark-native prompt/answer text for interface alignment

If everything is serialized as structured state, the model will become better at
policy-style decoding and worse at plain language use.

### 5. `target_action=` should be primary, but not exclusive

The current decode-aware experiments correctly suggest that `target_action=` is
the best short cached-decode target for the reasoning-policy objective.

That does not mean it should be the only target family.

A strong full-scale model should train on at least three target types:

- `decision_action`
- `decision_stop`
- `terminal_answer`

And for benchmark alignment it should also see benchmark-native targets:

- GSM8K question -> answer
- MMLU question + choices -> letter
- ARC task -> solver-side action traces or final outputs via a dedicated solver

Otherwise the model may become good at internal policy serialization but weaker
at the actual benchmark interfaces.

### 6. ARC-AGI-2 is still only represented, not parsed into reasoning tasks

The ARC-AGI-2 path is explicitly representation-level in
[evaluate_arc_agi2_representation.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/evaluate_arc_agi2_representation.py#L3).

That means:

- scene fitting exists
- object vocabulary stress-testing exists
- task-level program recovery does not
- solver-side training data from real ARC tasks does not yet

If ARC-AGI-2 is a primary target, this is a major missing piece.

Synthetic ARC trajectories are useful, but they are not a substitute for a real
ARC train-task adapter and a test-time adaptation loop.

### 7. MMLU and GSM8K are strong supervision sources, but only on one side of the problem

GSM8K parsing is currently supervision-time because it reads worked solutions in
[gsm8k_reasoning_parser.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/gsm8k_reasoning_parser.py#L45).

MMLU parsing is structural and heuristic in
[mmlu_parser.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/mmlu_parser.py#L551).

These are good teacher-data pipelines.

They are not themselves inference-time solvers.

That is fine, but the architecture should acknowledge the distinction:

- teacher-side parser/compiler
- student-side benchmark prompt interface

### 8. The post-training injection idea is good, but should remain an experiment, not an assumption

The attached protocol is sensible:

- fixed total token budget
- small early trajectory mixture
- optional late specialized continued pretraining when repetition gets high
- retention measurement on general capability

That is exactly the right way to test trajectory SPT.

What should not happen:

- hard-coding late trajectory injection into the architecture without measuring
  retention and benchmark lift

The protocol should be treated as an evaluation program:

- NPT baseline
- early SPT
- optional late SCPT
- optional replay timing control

## What Should Be Preserved

### Shared typed task IR

[reasoning_ir.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/reasoning_ir.py#L48)
is the right abstraction boundary.

Keep:

- typed entities, quantities, relations, goals
- explicit program
- explicit trace template
- support for both open-form and multiple-choice answers

### Shared trajectory schema

[stage4_trajectory_dataset.py](/Users/jjmayo/projects/demo_day/arc-agi/arc_trajectory_sampler/stage4_trajectory_dataset.py#L67)
is also the right core representation.

Keep:

- `TrajectoryRecord`
- `workspace_state`
- local reward terms
- verifier metadata

### Action-centric short decode targets

The current decode-aware setup is valuable.

Preserve:

- short cached continuations
- constrained candidate buckets where appropriate
- explicit `target_action=` supervision

## What Should Change

### 1. Separate corpus roles clearly

Use four data bands, not one monolithic corpus.

#### A. General pretraining corpus `Pgen`

Purpose:

- language fluency
- world knowledge
- NanoChat-like small-model competence

Examples:

- plain language modeling text
- dialogue / instruction data
- code if desired

This corpus should remain dominant by token count.

#### B. Structured reasoning pretraining corpus `Ptraj`

Purpose:

- action-conditioned reasoning
- local subgoal completion
- verifier-aware policy shaping

Examples:

- ARC synthetic trajectories
- GSM8K parsed trajectories from train only
- MMLU structured trajectories from non-benchmark-train sources or held-safe
  subsets only
- OlymMATH open-answer traces and Lean concordance traces as separate mixture
  sources
- later ARC train-task trajectories once available

This is where `REASONING_STATE_V1` belongs.

#### C. Benchmark-native alignment corpus `Pbench`

Purpose:

- make the model good at the actual surface form of target benchmarks

Examples:

- GSM8K question -> final numeric answer
- MMLU question + choices -> letter
- ARC train-task solver prompts -> output format expected by the solver stack

This prevents the model from becoming only a structured-policy model.

#### D. Post-training reasoning corpus `Ppost`

Purpose:

- final reasoning specialization
- late injection or SFT-style sharpening

This is where the attached SPT protocol should be applied.

### 2. Treat `REASONING_STATE_V1` as a reasoning-view format, not a universal format

The right design is:

- general LM text stays in normal text/chat format
- structured reasoning data uses `REASONING_STATE_V1`
- benchmark-native alignment data uses native prompt/answer format

One model, multiple input views.

### 3. Introduce trainable-builders vs audit-builders

The current builder APIs are too permissive for benchmark-targeted work.

Needed split:

- `build_*_examples_for_audit(...)`
- `build_*_examples_for_training(...)`

Training builders should default to safe splits only.

### 4. Add an ARC train-task reasoning adapter

For ARC-AGI-2 performance, the repo needs more than representation fit.

Needed next:

- parse ARC train tasks into reusable object-scene reasoning tasks
- derive or synthesize teacher traces from those tasks
- train a policy / search sidecar on real ARC tasks, not only synthetic ones

### 5. Keep two evaluation axes

For every training-stage change, measure both:

- reasoning quality
- general-language retention

The attached protocol is correct to insist on a retention metric.

Suggested standing metrics:

- GSM8K exact match
- MMLU accuracy
- ARC-AGI-2 solve rate at fixed adaptation/search budget
- one general small-model language probe
- one decode-policy metric on `decision_action`

## Recommended Training Schedule

### Stage 0: Infrastructure cleanup

Before scaling:

1. make benchmark-safe builders the default
2. replace byte-level training text with the chosen Epiplex tokenizer path
3. finish migrating the training stack onto the now-modality-neutral `state_adapter.py`

### Stage 1: Base LM pretraining

Train mostly on `Pgen`, with either:

- no structured trajectory mix
- or a very small early reasoning mix

Use the attached protocol logic:

- hold total token budget fixed
- keep the reasoning fraction small
- log effective repetition count

### Stage 2: Specialized continued pretraining experiment

If the trajectory corpus is small enough that repetition becomes high, run:

- early SPT arm
- late SCPT arm
- baseline NPT arm

Measure:

- benchmark lift
- retention cost

### Stage 3: Post-training

Use a mixture of:

- benchmark-native QA / MCQ examples
- structured reasoning-state examples
- chat / instruction data for general language behavior

### Stage 4: Benchmark-specific sidecars

Needed especially for ARC:

- constrained decoding
- search / adaptation
- possibly per-task refinement

## Practical Recommendations Right Now

1. Do not train on official MMLU test or GSM8K test under any default path.
2. Do not force general language data into `REASONING_STATE_V1`.
3. Keep `decision_action` as the main structured decode target.
4. Add `decision_stop` and `terminal_answer` as separate short-target examples.
5. Build a real ARC train-task adapter next if ARC-AGI-2 matters as much as GSM8K and MMLU.
6. Move the model stack off byte-level text before drawing large architectural conclusions from the current ablations.
7. Use the SPT protocol as an experiment grid, not as a fixed training dogma.

## Overall Assessment

The project is pointed in the right direction, but the current design is still
best understood as:

- a strong reasoning-supervision compiler
- a promising structured-decode substrate
- an incomplete full-scale LLM training recipe

To become the intended system, it needs:

- benchmark-safe data boundaries
- a multi-view corpus strategy
- a modality-neutral reasoning adapter
- a tokenizer-aligned training path
- a real ARC task-level solver / adaptation path

That is a tractable next phase, but those gaps are real and should be treated as
first-order design work rather than cleanup.
