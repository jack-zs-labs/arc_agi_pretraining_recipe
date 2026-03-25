# Architecture Alignment

This repo is currently aligned with the benchmark-side and supervision-side parts of the target architecture, not the model-backbone part.

## What Already Exists

- A reference decoder-only transformer package with grouped-query attention, RoPE, KV caching, pluggable attention backends, a local-global hybrid long-context path, reference scale-invariant attention variants, an MLA-inspired latent-KV path, an MLA+SIA hybrid path, selective suffix-layer SIA routing, prefill-only SIA decode mode, a reference sparse-MoE feedforward path, and an explicit inference-budget surface in `models/`.
- A shared benchmark-aware reasoning-budget policy layer in `models/reasoning_budget.py` with `fast` / `balanced` / `deep` effort presets for ARC, GSM8K, MMLU, MMLU-Pro, MMLU-Redux, OlymMATH, CoRe, native `oscar_scope`, structured `oscar_scope_reasoning`, and canonical `oscar_graph_reasoning`, including ARC-specific structured decode hints.
- Named long-context serving presets in `models/`: `mla_default` and `mla_sia_prefill_l1`.
- A shared reasoning IR for ARC-style structured supervision in `arc_trajectory_sampler/reasoning_ir.py`.
- A shared `REASONING_STATE_V1` LM-facing prompt/target format in `docs/unified_reasoning_lm_interface.md`, implemented at the adapter boundary in `arc_trajectory_sampler/state_adapter.py`.
- A critical full-scale training review in `docs/full_scale_reasoning_architecture_review.md`.
- A benchmark-side structured decoding sidecar for ARC `target_action=` generation in `scripts/model_stack_decode_eval.py` via `context_trie` constrained decode over train-split action candidates, a prompt-grounded `schema_json` candidate expansion path, and a `context_then_schema` fallback mode that preserves exact-context buckets before widening coverage.
- ARC synthetic trajectory generation, alternate valid traces, and near-miss negatives in `arc_trajectory_sampler/stage1_latent_sampler.py` through `arc_trajectory_sampler/stage4_trajectory_dataset.py`.
- ARC-specific outer-loop substrate via trajectory compilation, evaluation, and lightweight action-sequence baselines in `experiments/arc_sampler_fqi_benchmark.py` and `experiments/arc_sampler_bc_benchmark.py`.
- Supervision adapters for GSM8K and MMLU in `arc_trajectory_sampler/gsm8k_reasoning_parser.py` and `arc_trajectory_sampler/mmlu_parser.py`, with variable-choice support for MMLU-family benchmarks plus benchmark-safe trainable defaults and explicit audit opt-ins.
- Eval-time loaders for `MMLU-Pro` and `MMLU-Redux` in `arc_trajectory_sampler/mmlu_variants.py`, wired into the shared reasoning-state text interface and the integrated trainer.
- An OlymMATH trajectory adapter in `arc_trajectory_sampler/olympiad_math_parser.py`, `arc_trajectory_sampler/mixed_reasoning_dataset.py`, and `scripts/export_olympiad_math_trace_corpus.py`, wired into the shared reasoning-state text interface, with subject-aware Olympiad trace templates, SymPy-backed canonicalization for scalar/interval/bracketed-scalar answers, and a `lean` concordance path that preserves bilingual informal text plus Lean theorem/proof artifacts.
- A broader `Ptraj` export path in `arc_trajectory_sampler/mixed_reasoning_dataset.py` and `scripts/export_ptraj_corpus.py` that mixes ARC, GSM8K, benchmark-safe MMLU, OlymMATH open/lean, structured CoRe supervision, Oscar doc-grounded reasoning, and canonical Oscar graph reasoning into one JSONL manifest for continued pretraining.
- An Oscar-scope domain path in `arc_trajectory_sampler/oscar_scope_corpus.py`, `arc_trajectory_sampler/oscar_scope_reasoning.py`, `scripts/export_oscar_scope_corpus.py`, `scripts/export_oscar_scope_reasoning_corpus.py`, `scripts/export_oscar_workflow_reasoning_corpus.py`, `scripts/evaluate_oscar_workflow_holdout.py`, and `scripts/train_integrated_reasoning_stack.py` that auto-discovers local process-intelligence specs/meeting notes when available, prefers text sources over PDFs, exposes native domain text as `Pgen`-style `oscar_scope` training data, and treats deterministic reasoning traces as a first-class `oscar_scope_reasoning` benchmark for section anchoring, outline continuation, canonical concept tagging, workflow-heavy business-environment supervision over KPI, bottleneck, and improvement structure, multi-step intervention sequencing with projected reward labels, and cross-case analogy/transfer tasks over shared workflow motifs, with per-trace local graph payloads over document, section, and concept nodes. That path now also includes a canonical KPI-family/intervention-family action surface plus a strict held-out-business-environment evaluator, so workflow abstraction can be checked without leaking held-out intervention labels into the decision-action vocabulary.
- A canonical Oscar graph benchmark in `arc_trajectory_sampler/oscar_graph_reasoning.py`, `scripts/export_oscar_graph_reasoning_corpus.py`, `arc_trajectory_sampler/mixed_reasoning_dataset.py`, and `scripts/train_integrated_reasoning_stack.py` that derives a typed process graph from the local design docs and turns it into first-class `oscar_graph_reasoning` traces for relation classification, neighbor enumeration, path completion, process-to-graph grounding, executor-style motif rollouts, and branching executor-trace `decision_action` records with frontier updates.
- Benchmark-specific Oscar auxiliary heads in `models/oscar_auxiliary.py` and `scripts/train_integrated_reasoning_stack.py` that now supervise structured Oscar reasoning metadata over `oscar_scope_reasoning`, including family ID, section-depth bucket, section-path label, section-parent links, document group, document title, related-document links, multi-label concept tags, workflow KPI IDs, workflow intervention IDs, workflow motif IDs, and workflow reward bucket/score targets.
- A benchmark-native Oscar graph reasoning path that puts the local process graph on the same footing as the other reasoning benchmarks, instead of leaving it only as attached document metadata.
- Candidate-wise Oscar graph auxiliary heads in `models/oscar_graph_auxiliary.py` and `scripts/train_integrated_reasoning_stack.py` that now supervise canonical-relation selection, neighbor-set prediction, path via/target prediction, process-to-graph grounding, rollout motif classification, and stepwise executor-node prediction over `oscar_graph_reasoning`.
- A structured CoRe ingestion path in `arc_trajectory_sampler/core_loader.py`, `arc_trajectory_sampler/core_graph_extractor.py`, `arc_trajectory_sampler/core_reasoning_adapter.py`, and `arc_trajectory_sampler/mixed_reasoning_dataset.py` that now:
  - loads all three dependency kinds: control, data, and infoflow
  - extracts explicit graph views with a parser-backed Python AST path plus heuristic fallback for other languages and parse failures
  - aggregates `list_source` rows into query-level source-set supervision
  - reconstructs most positive traces from the pairwise-positive graph via a lightweight transitive-reduction heuristic
  - serializes code lines, extracted graph nodes/edges, benchmark direct edges, candidate sources, and auxiliary trace/source targets into `REASONING_STATE_V1`
- Benchmark-specific CoRe auxiliary heads in `models/core_auxiliary.py` and `scripts/train_integrated_reasoning_stack.py` that now supervise:
  - query polarity, dependency kind, trace-length bucket, source-count bucket, and infoflow data-edge presence at the example level
  - source-candidate membership and direct-edge-to-target labels at the candidate level
- Benchmark-conditioned structured decision heads in `models/decision_auxiliary.py` and `scripts/train_integrated_reasoning_stack.py` that factor benchmark-local action prediction into name and argument heads, use candidate masking for exact-action reconstruction, and add small benchmark-specific residual adapters on top of the shared decision representation.
- A tokenizer-aligned reference training path in `models/reasoning_tokenizer.py`, `scripts/train_integrated_reasoning_stack.py`, and `scripts/model_stack_training_ablation.py` that fits or loads a shared Epiplex vocabulary over `REASONING_STATE_V1` text instead of hard-coding a byte vocabulary.
- A mixed-benchmark supervised integration path in `scripts/train_integrated_reasoning_stack.py` that trains one decoder across ARC, GSM8K, MMLU, and a structured CoRe slice, while evaluating on MMLU-Pro, MMLU-Redux, and OlymMATH, using the shared reasoning-state text interface, with dense and reference MoE architecture options.
- Small offline RL sandboxes in `experiments/` that are useful for training-loop and reward-shaping iteration, but are not yet the mixed-domain RL stack from the target spec.

## What Is Missing

- No production sparse-MoE stack yet: the repo now has a reference router and expert path, but no large-scale stability recipe, no auxiliary-loss-free balancing, no sparse kernels, and no RL-specific MoE stabilization work.
- No production-grade efficient long-context attention backend such as DeepSeek-style MLA with absorbed projections or lightning attention; the reference stack now has simple hybrid and scale-invariant reference backends plus an MLA-inspired latent-KV backend, but all are intentionally lightweight reference implementations.
- No learned or adaptive reasoning-budget controller yet; the repo now has rule-based benchmark-integrated effort policies, but not a trained policy, not a benchmark-conditioned planner, and not a true test-time compute controller beyond fixed preset selection.
- No multi-token prediction or speculative decoding path.
- No unified mixed-domain RL post-training stack.
- No compiler-grade CoRe graph/IR view yet: the repo now has a stronger explicit extractor with a parser-backed Python path, graph-conditioned CoRe serialization, and learned example-level plus candidate-level auxiliary heads, but not a full multi-language CFG/DFG/IFG pipeline, not token/span-aligned node/edge benchmark heads, and not exact gold trace typing for every infoflow edge.
- No deep Oscar graph supervision stack yet: the repo now has a first-class canonical `oscar_graph_reasoning` benchmark, richer `oscar_scope_reasoning` graph payloads, candidate-wise Oscar graph heads for relations, nodes, and two-hop traversal targets, and small supervised executor-style motif rollouts, but it still lacks span-aligned Oscar node/edge heads, open-ended executable graph traversal/search, and learned abstraction links from local process traces into higher-level canonical process motifs.
- No ARC test-time adaptation sidecar yet: no LoRA refinement loop, no per-task search, and no cost-per-task reporting.
- No split scaling-law reporting across pretraining compute, RL-train compute, and test-time compute.

## Current Interpretation

The strongest current fit to the target spec is:

- Decoder substrate: present, with reference hybrid, scale-invariant, MLA-inspired, and MLA+SIA long-context paths plus suffix-only and prefill-only SIA routing for serving experiments, and now a reference sparse-MoE feedforward option, but still intentionally small and conventional.
- ARC sidecar substrate: partially present.
- MMLU supervision substrate: present for base MMLU, with MMLU-Pro and MMLU-Redux eval adapters now available.
- Olympiad-math supervision substrate: partially present, with an OlymMATH trace adapter, open-answer symbolic canonicalization, subject-shaped reasoning traces for harder math targets, and a `lean` formal-proof concordance variant. The corpus path now labels these separately as `olympiad_math_open` and `olympiad_math_lean`.
- Mixed `Ptraj` corpus substrate: present, with a single exporter that can mix ARC, GSM8K, benchmark-safe MMLU, OlymMATH open/lean, CoRe traces, and Oscar doc-grounded reasoning into one `REASONING_STATE_V1` pretraining manifest.
- Native domain corpus substrate: partially present, with an Oscar-scope exporter and trainer integration that provide both a `Pgen`-like text path for process-intelligence specs and meeting notes and a first-class structured reasoning benchmark plus benchmark-specific auxiliary heads derived from those same documents.
- Oscar process-graph reasoning substrate: partially present, with a first-class canonical `oscar_graph_reasoning` benchmark that trains direct reasoning over the graph structure defined by the local Oscar documentation, process-to-graph grounding tasks that tie surrounding process prose back to that canonical graph, candidate-wise auxiliary heads that make node/relation/traversal supervision explicit in the decoder, supervised motif-rollout tasks, and branching executor-trace `decision_action` records that begin to approximate a graph-executor layer with explicit frontier updates.
- CoRe supervision substrate: partially present, with query-level trace/source supervision, parser-backed Python graph extraction plus heuristic fallback, and both example-level and candidate-level auxiliary targets, but still short of the full benchmark-specific IR/head setup suggested by the target spec.
- Mixed-benchmark supervised training substrate: present, now including distinct `oscar_scope_reasoning` and `oscar_graph_reasoning` training benchmarks when local Oscar docs are available.
- Benchmark-integrated reasoning-budget policy scaffold: present, but rule-based.
- Broad reasoning IR: present.
- Frontier sparse backbone recipe and post-training stack: absent.

In other words, the repo is currently building the outer-loop and dataset plumbing around the future model, not the future model itself.

## Recommended Build Sequence

1. Extend `models/` from the current reference hybrid, MLA-inspired, and reference-MoE paths to a more faithful MLA or lightning-style implementation plus a more realistic sparse expert recipe.
2. Turn the current rule-based reasoning-budget presets into a stronger controller: benchmark-conditioned policies, better decode-time compute scaling, and integration with RL/test-time compute reporting.
3. Extend the current CoRe graph scaffold from the new Python parser-backed path to stronger multi-language extraction: explicit CFG/DFG/IFG edges, typed nodes, and more faithful trace reconstruction.
4. Upgrade the current candidate-level CoRe auxiliary heads into token/span-aligned or graph-node-aligned supervision over full trace nodes, typed edges, and source-set membership.
5. Extend the current supervised Oscar motif rollouts into a stronger graph-executor layer: longer-horizon traversal, branching decisions, abstraction-state updates, and rollout-conditioned decode rather than only single-example supervision.
6. Add RL training code that can mix reasoning, QA, and agent tasks while tracking train-time and test-time compute separately.
7. Add an ARC sidecar loop for per-task adaptation/refinement and report score jointly with cost-per-task.

## Repo Health

The repo now has a fast smoke path in `scripts/smoke_check.sh`. It verifies the current benchmark substrate:

- `models/` compilation and, when a Torch environment is available, the reference decoder smoke run in `scripts/model_stack_smoke.py`
- backend/cache ablations in `scripts/model_stack_ablation.py`
- tiny token-level backend training comparison in `scripts/model_stack_training_ablation.py`
- combined cache/latency/loss reporting in `scripts/model_stack_report.py`
- decode-aware ARC generation evaluation in `scripts/model_stack_decode_eval.py`, including train-split constrained `context_trie` decode, prompt-grounded `schema_json` constrained decode, the `context_then_schema` exact-context fallback hybrid, and policy-driven `policy_default` decode-mode resolution
- integrated mixed-benchmark supervised training smoke coverage in `scripts/train_integrated_reasoning_stack.py`, including benchmark-specific train/eval effort presets plus optional `MMLU-Pro`, `MMLU-Redux`, `OlymMATH`, and structured `CoRe` ingestion
- parser-backed CoRe graph smoke coverage in `scripts/core_graph_probe.py`
- OlymMATH parser and trace-export smoke coverage in `arc_trajectory_sampler/evaluate_olympiad_math_parser.py`, `scripts/export_olympiad_math_trace_corpus.py`, `scripts/export_ptraj_corpus.py`, `scripts/export_oscar_scope_corpus.py`, `scripts/export_oscar_scope_reasoning_corpus.py`, and `scripts/export_oscar_graph_reasoning_corpus.py`, including the `lean` concordance path, the mixed `Ptraj` export path, the native Oscar domain-text export path, the deterministic Oscar doc-grounded reasoning export path, and the canonical Oscar graph reasoning export path
- module compilation
- ARC sampler evaluation
- 3D cart-pole benchmark
- ARC FQI smoke run
- ARC behavior-cloning smoke run
- GSM8K parser smoke run
- MMLU parser smoke run
