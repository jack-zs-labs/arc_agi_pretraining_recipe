#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
model_python_bin="${MODEL_PYTHON_BIN:-}"
output_dir="${SMOKE_OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/arc-agi-smoke.XXXXXX")}"

if [[ -z "$model_python_bin" && -x ".venv_atari/bin/python" ]]; then
  model_python_bin=".venv_atari/bin/python"
fi

if [[ -z "${SMOKE_OUTPUT_DIR:-}" ]]; then
  trap 'rm -rf "$output_dir"' EXIT
fi

echo "[1/8] compile modules"
"$python_bin" -m compileall models arc_trajectory_sampler experiments scripts >/dev/null

echo "[2/8] ARC sampler"
"$python_bin" arc_trajectory_sampler/evaluate_sampler.py \
  --num-samples 5 \
  --output "$output_dir/arc_sampler_summary.json" >/dev/null

echo "[3/8] 3D cart-pole"
"$python_bin" experiments/three_d_cartpole_experiment.py \
  --dataset-sizes 8 \
  --seeds 1 \
  --eval-episodes 8 \
  --output-dir "$output_dir/three_d_cartpole" >/dev/null

echo "[4/8] ARC FQI"
"$python_bin" experiments/arc_sampler_fqi_benchmark.py \
  --dataset-sizes 4 \
  --seeds 1 \
  --eval-episodes 4 \
  --fqi-iterations 1 \
  --optimizer-steps 1 \
  --output-dir "$output_dir/arc_sampler_fqi" \
  --no-progress >/dev/null

echo "[5/8] ARC behavior cloning"
"$python_bin" experiments/arc_sampler_bc_benchmark.py \
  --dataset-sizes 4 \
  --seeds 1 \
  --eval-episodes 4 \
  --epochs 1 \
  --batch-size 16 \
  --output-dir "$output_dir/arc_sampler_bc" \
  --no-progress >/dev/null

echo "[6/8] parser evaluations"
"$python_bin" arc_trajectory_sampler/evaluate_gsm8k_parser.py \
  --data-dir arc_trajectory_sampler/data/gsm8k \
  --max-rows 20 \
  --output "$output_dir/gsm8k_parser_summary.json" >/dev/null
"$python_bin" arc_trajectory_sampler/evaluate_mmlu_parser.py \
  --data-dir arc_trajectory_sampler/data/mmlu \
  --max-rows 20 \
  --output "$output_dir/mmlu_parser_summary.json" >/dev/null
"$python_bin" arc_trajectory_sampler/evaluate_olympiad_math_parser.py \
  --configs en-easy lean \
  --max-rows 12 \
  --output "$output_dir/olympiad_math_parser_summary.json" >/dev/null
"$python_bin" scripts/export_olympiad_math_trace_corpus.py \
  --configs en-easy lean \
  --max-rows 4 \
  --output "$output_dir/olympiad_math_traces.jsonl" \
  --summary-output "$output_dir/olympiad_math_trace_summary.json" >/dev/null
"$python_bin" scripts/export_ptraj_corpus.py \
  --arc-episodes 1 \
  --gsm8k-max-rows 2 \
  --mmlu-max-rows 2 \
  --olympiad-math-configs en-easy lean \
  --olympiad-math-max-rows 2 \
  --core-max-rows 2 \
  --output "$output_dir/ptraj_traces.jsonl" \
  --summary-output "$output_dir/ptraj_trace_summary.json" >/dev/null
"$python_bin" scripts/export_oscar_graph_reasoning_corpus.py \
  --max-examples 12 \
  --output "$output_dir/oscar_graph_reasoning_traces.jsonl" \
  --summary-output "$output_dir/oscar_graph_reasoning_trace_summary.json" >/dev/null
oscar_scope_root="$(cd "$repo_root/../.." && pwd)/oscar_design_docs"
if [[ -d "$oscar_scope_root" ]]; then
  "$python_bin" scripts/export_oscar_scope_corpus.py \
    --max-documents 2 \
    --max-chunks 8 \
    --output "$output_dir/oscar_scope_traces.jsonl" \
    --summary-output "$output_dir/oscar_scope_trace_summary.json" >/dev/null
  "$python_bin" scripts/export_oscar_scope_reasoning_corpus.py \
    --max-documents 2 \
    --max-examples 12 \
    --output "$output_dir/oscar_scope_reasoning_traces.jsonl" \
    --summary-output "$output_dir/oscar_scope_reasoning_trace_summary.json" >/dev/null
  "$python_bin" scripts/export_oscar_workflow_reasoning_corpus.py \
    --max-examples 20 \
    --output "$output_dir/oscar_workflow_reasoning_traces.jsonl" \
    --summary-output "$output_dir/oscar_workflow_reasoning_trace_summary.json" >/dev/null
fi

echo "[7/8] integrated benchmark data path"
"$python_bin" scripts/train_integrated_reasoning_stack.py \
  --data-only \
  --tokenizer byte \
  --no-include-oscar-scope \
  --no-include-oscar-scope-reasoning \
  --arc-train-episodes 2 \
  --arc-val-episodes 1 \
  --gsm8k-max-rows 4 \
  --mmlu-max-rows 4 \
  --olympiad-math-configs en-easy lean \
  --olympiad-math-max-rows 4 \
  --seq-len 64 \
  --output "$output_dir/integrated_reasoning_data_only_summary.json" >/dev/null

echo "[8/8] CoRe graph probe"
"$python_bin" scripts/core_graph_probe.py \
  --languages Python \
  --dependency-kinds infoflow \
  --categories trace \
  --graph-backend auto \
  --max-examples 16 \
  --output "$output_dir/core_graph_probe_summary.json" >/dev/null

if [[ -n "$model_python_bin" ]]; then
  echo "[optional] model stack"
  "$model_python_bin" scripts/model_stack_smoke.py >/dev/null
  "$model_python_bin" scripts/train_integrated_reasoning_stack.py \
    --device cpu \
    --tokenizer byte \
    --no-include-oscar-scope \
    --no-include-oscar-scope-reasoning \
    --architectures dense moe \
    --arc-train-episodes 2 \
    --arc-val-episodes 1 \
    --gsm8k-max-rows 4 \
    --mmlu-max-rows 4 \
    --olympiad-math-configs en-easy lean \
    --olympiad-math-max-rows 4 \
    --core-graph-backend auto \
    --seq-len 64 \
    --batch-size 1 \
    --steps 1 \
    --hidden-size 64 \
    --num-layers 2 \
    --num-heads 4 \
    --num-kv-heads 2 \
    --intermediate-size 128 \
    --latent-kv-dim 16 \
    --output "$output_dir/integrated_reasoning_stack_summary.json" >/dev/null
  if [[ -d "$oscar_scope_root" ]]; then
  "$model_python_bin" scripts/train_integrated_reasoning_stack.py \
      --device cpu \
      --tokenizer byte \
      --architectures dense \
      --no-include-oscar-scope \
      --arc-train-episodes 0 \
      --arc-val-episodes 0 \
      --gsm8k-max-rows 0 \
      --mmlu-max-rows 0 \
      --olympiad-math-max-rows 0 \
      --core-max-rows 0 \
      --oscar-scope-max-documents 2 \
      --oscar-scope-reasoning-max-examples 8 \
      --seq-len 64 \
      --batch-size 2 \
      --steps 1 \
      --hidden-size 64 \
      --num-layers 2 \
      --num-heads 4 \
      --num-kv-heads 2 \
      --intermediate-size 128 \
      --latent-kv-dim 16 \
      --output "$output_dir/integrated_reasoning_stack_oscar_aux_summary.json" >/dev/null
    "$model_python_bin" scripts/train_integrated_reasoning_stack.py \
      --device cpu \
      --tokenizer byte \
      --architectures dense \
      --no-include-oscar-scope \
      --no-include-oscar-scope-reasoning \
      --arc-train-episodes 0 \
      --arc-val-episodes 0 \
      --gsm8k-max-rows 0 \
      --mmlu-max-rows 0 \
      --olympiad-math-max-rows 0 \
      --core-max-rows 0 \
      --oscar-graph-reasoning-max-examples 8 \
      --seq-len 64 \
      --batch-size 2 \
      --steps 1 \
      --hidden-size 64 \
      --num-layers 2 \
      --num-heads 4 \
      --num-kv-heads 2 \
      --intermediate-size 128 \
      --latent-kv-dim 16 \
      --output "$output_dir/integrated_reasoning_stack_oscar_graph_aux_summary.json" >/dev/null
    "$model_python_bin" scripts/evaluate_oscar_workflow_holdout.py \
      --device cpu \
      --tokenizer byte \
      --max-examples 64 \
      --steps 1 \
      --batch-size 2 \
      --decision-batch-size 2 \
      --seq-len 64 \
      --hidden-size 64 \
      --num-layers 2 \
      --num-heads 4 \
      --num-kv-heads 2 \
      --intermediate-size 128 \
      --holdout-environments "GenAI customer support copilot" \
      --output-dir "$output_dir/oscar_workflow_holdout_eval" >/dev/null
  fi
fi

echo "Smoke checks passed."
if [[ -n "${SMOKE_OUTPUT_DIR:-}" ]]; then
  echo "Artifacts: $output_dir"
else
  echo "Artifacts were written to a temporary directory and cleaned up."
fi
