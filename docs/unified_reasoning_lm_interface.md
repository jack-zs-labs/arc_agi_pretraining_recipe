# Unified Reasoning LM Interface

This document defines the LM-facing example format for the shared ARC /
GSM8K / MMLU reasoning stack.

The core serializer boundary in `arc_trajectory_sampler/state_adapter.py` now
implements this format for structured trajectory prompts. Training and
evaluation scripts still need to be migrated onto it fully.

## Why This Format

The current system already has:

- a shared typed reasoning IR in `arc_trajectory_sampler/reasoning_ir.py`
- a shared stepwise trajectory object in `arc_trajectory_sampler/stage4_trajectory_dataset.py`
- a modality-neutral `REASONING_STATE_V1` structured prompt path in
  `arc_trajectory_sampler/state_adapter.py`
- decode-aware evaluation centered on short cached continuations beginning at
  `target_action=` in `scripts/model_stack_decode_eval.py`

Recent model results matter here:

- `mla_default` remains the safest default backend
- `mla_sia_prefill_l1` is the better quality-oriented preset on the current
  decode-aware workload
- teacher-forced loss is not a reliable proxy for cached-decode quality

That pushes the interface in a clear direction:

1. maximize reusable prefill context
2. keep decoded continuations short and structured
3. train the model to choose actions and verifier labels, not to regenerate
   whole workspace states
4. let external executors or compilers update the next state

## Design Principles

### 1. One canonical prompt family

All modalities should serialize into one prompt family:

- ARC object-scene trajectories
- synthetic word-problem trajectories
- GSM8K parsed trajectories
- MMLU parsed trajectories

The model should not see separate prompt ontologies for each dataset.

### 2. Prefill-heavy, decode-light

The prompt should contain the full state, trace context, and decode bucket.
The generated target should usually be one short field:

- `target_action=...`
- `target_stop=...`
- `target_answer=...`

This matches the current cached-decode evaluation regime better than asking the
model to emit full next-state payloads.

### 3. Stable lexical atoms for the tokenizer

Given the Epiplex tokenizer decision, the outer format should prefer:

- fixed line prefixes
- stable enum names
- repeated keys in fixed order
- short, regular field values
- canonical minified JSON only for leaf payloads such as action objects

The outer shell should not be arbitrary JSON.

### 4. Externalized state transition

The LM is the policy head, not the executor.

The intended loop is:

1. parse or generate a task into `TrajectoryRecord`
2. serialize a decision state
3. have the LM predict the next action or stop signal
4. apply that action using the executor or teacher trace
5. move to the next decision state

## Canonical Example Types

The canonical unit is a `ReasoningExampleV1`.

It has three record types:

1. `decision_action`
2. `decision_stop`
3. `terminal_answer`

The first should be the primary training unit.

### Shared Metadata Fields

These fields should be present in every record:

- `format_version`
- `record_type`
- `dataset`
- `source_modality`
- `family`
- `difficulty`
- `trajectory_id`
- `step_index`
- `trace_length`
- `trace_step`
- `previous_action`
- `candidate_bucket`
- `answer_format`

### Shared State Fields

These fields describe the current state:

- `state_tokens`
- `state_scalars`
- `verifier_context`

`state_tokens` should be a flat, tokenizer-friendly sequence of symbolic facts.
`state_scalars` should remain a short numeric vector encoded as comma-separated
decimal strings.
`verifier_context` should be compact and should emphasize:

- resolved subgoal count
- unresolved subgoal count
- next subgoal
- current stop eligibility

### Shared Target Fields

Exactly one target family should be primary per example:

- `target_action`
- `target_stop`
- `target_answer`

Auxiliary targets may be stored off-path in metadata, but they should not all be
generated in one long continuation by default.

## Text Serialization

The canonical serialized prompt should look like this:

```text
REASONING_STATE_V1
record_type=decision_action
dataset=gsm8k
source_modality=text_open
family=compose_total
difficulty=3
trajectory_id=gsm8k:train:row_12:step_2
step_index=2
trace_length=5
trace_step=apply
previous_action={"action":{"quantity_ids":["q_first","q_delta_up","q_delta_down"]},"name":"bind"}
candidate_bucket=family:compose_total|step:2|prev:bind
answer_format=open
state_tokens=src.kind=text_open src.qcount=3 prog=ComposeThenReduce subgoal.segment=done subgoal.bind=done subgoal.apply=pending
state_scalars=0.400000,0.600000,0.000000
verifier_context={"next_subgoal":"apply","resolved_subgoal_count":2,"should_stop":false,"unresolved_subgoal_count":3}
target_action=
```

The decoded continuation is only:

```text
{"action":{"derive_quantity_ids":["q_second","q_third"]},"name":"apply"}
```

The key point is that the model decodes only the action payload, not the next
workspace.

## Record-Type Details

### `decision_action`

Primary use:

- behavior cloning
- cached-decode evaluation
- constrained decode over action candidates

Target:

- one canonical action object

Recommended output shape:

```text
target_action={"action":{...},"name":"apply"}
```

### `decision_stop`

Primary use:

- stop / continue calibration
- verifier-oriented supervision

Target:

- `true` or `false`

Recommended output shape:

```text
target_stop=false
```

### `terminal_answer`

Primary use:

- answer correctness
- final-choice generation

Target depends on answer format:

- open-form numeric or short string answer
- multiple-choice letter

Recommended output shape:

```text
target_answer=42
```

or

```text
target_answer=C
```

## Candidate Buckets

The current decode evaluator already benefits from grouping candidates by:

- `family`
- `step_index`
- `previous_action`

That should remain a first-class part of the interface.

Recommended bucket string:

```text
candidate_bucket=family:compose_total|step:2|prev:bind
```

For weaker backoff buckets, the runtime can still degrade to:

- `family + step`
- `family`
- global

But the prompt should always emit the most specific bucket.

## Modality-Specific State Construction

The outer format is shared. Only `state_tokens`, `state_scalars`, and selected
leaf metadata differ by modality.

### ARC / ARC-AGI-2

State tokens should include:

- object summaries
- focus object ids
- scene dimensions
- role bindings
- subgoal status

This is close to the current `state_adapter.py` behavior.

### GSM8K / Synthetic Word Problems

State tokens should include:

- entity ids
- quantity ids
- derivation-rule ids
- reducer type
- question focus tokens
- subgoal status

The executor should apply symbolic derivations outside the LM.

### MMLU

State tokens should include:

- choice ids
- question focus tokens
- prompt structure cue
- supported or eliminated choice ids
- subgoal status

The model should choose among action-like choice-selection decisions rather than
regenerate long textual rationales.

## Action Serialization

Action payloads should stay as canonical minified JSON with sorted keys.

That matches the current training path and keeps candidate-set indexing simple.

Recommended invariant:

- fixed top-level keys: `name`, `action`
- sorted keys
- no pretty-printing
- no extra whitespace

## Current Code Status

### Replace ARC-only prompt naming

Previous prompt header:

```text
ARC_STRUCTURED_STATE
```

Implemented header:

```text
REASONING_STATE_V1
```

### Generalize the adapter

`arc_trajectory_sampler/state_adapter.py` is now a modality-neutral
reasoning-state adapter.

The current adapter now provides:

- no hard-coded assumption that the state contains an ARC `scene`
- modality-specific token extraction behind a shared interface
- shared prompt serializer for all datasets

### Stop using byte-only training text as the long-term default

The current training scripts still use raw UTF-8 byte IDs.

That is fine for tiny ablations, but the intended production path should use the
chosen Epiplex tokenizer on the shared `REASONING_STATE_V1` text format.

### Make `decision_action` the main decode benchmark

Given current cached-decode results, the most stable research target remains:

- prompt: full decision-state prefill
- decode: only `target_action`

Auxiliary tasks such as `target_stop` and `target_answer` should be added as
separate short-target examples, not concatenated into a long mixed target.

## Recommended Training Mix

1. `decision_action` examples as the main supervised corpus
2. `decision_stop` examples as a secondary control objective
3. `terminal_answer` examples as a terminal correctness objective
4. synthetic ARC trajectories for strong local reward and exact execution
5. GSM8K parsed trajectories for text-open symbolic reasoning
6. MMLU parsed trajectories for text-mcq selection behavior

## Non-Goals

This interface does not require:

- free-form chain-of-thought generation
- LM prediction of full next workspace states
- benchmark-specific prompt templates
- on-policy execution inside the prompt itself

Those can be layered later, but they should not define the base format.

## Bottom Line

The right shared interface is:

- modality-neutral
- action-centric
- prefill-heavy
- decode-light
- tokenizer-stable

In practice, that means a single `REASONING_STATE_V1` prompt family whose main
decode target is `target_action=...`, with verifier and terminal-answer
objectives added as separate short continuations.
