#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_atari/bin/python}"
TOKENIZER_REPO_ROOT="${TOKENIZER_REPO_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/epiplex_tokenizer_trainer}"
TOKENIZER_PYTHON_BIN="${TOKENIZER_PYTHON_BIN:-$PYTHON_BIN}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/epiplex_generic_8h100_burnin_${DATE_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29580}"

MANIFEST_DIR="${RUN_ROOT}/manifest"
TOKENIZER_DIR="${RUN_ROOT}/tokenizer_generic_8k"
PACK_DIR="${RUN_ROOT}/packed_seq2048_epiplex_generic"
BURNIN_DIR="${RUN_ROOT}/burnin_8xh100_step50"
RESUME_DIR="${RUN_ROOT}/burnin_8xh100_resume75"

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing PYTHON_BIN at $PYTHON_BIN" >&2
    exit 1
  fi
  if [[ ! -d "$TOKENIZER_REPO_ROOT" ]]; then
    echo "Missing TOKENIZER_REPO_ROOT at $TOKENIZER_REPO_ROOT" >&2
    exit 1
  fi
fi

mkdir -p "$RUN_ROOT"

echo "[1/5] build manifest"
run_cmd \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/build_pretraining_manifest.py" \
  --output-dir "$MANIFEST_DIR" \
  --validation-fraction "${VALIDATION_FRACTION:-0.005}" \
  --seed "${SEED:-7}" \
  --include-dclm \
  --dclm-max-documents "${DCLM_MAX_DOCUMENTS:-50000}" \
  --dclm-shuffle-buffer-size "${DCLM_SHUFFLE_BUFFER_SIZE:-20000}" \
  --dclm-min-text-chars "${DCLM_MIN_TEXT_CHARS:-256}" \
  --include-oscar-scope \
  --include-oscar-scope-reasoning \
  --include-oscar-graph-reasoning \
  --oscar-scope-auto-discover \
  --oscar-scope-reasoning-max-examples "${OSCAR_SCOPE_REASONING_MAX_EXAMPLES:-1000}" \
  --oscar-graph-reasoning-max-examples "${OSCAR_GRAPH_REASONING_MAX_EXAMPLES:-250}" \
  --oscar-native-repeat "${OSCAR_NATIVE_REPEAT:-4}" \
  --oscar-reasoning-repeat "${OSCAR_REASONING_REPEAT:-8}" \
  --oscar-graph-repeat "${OSCAR_GRAPH_REPEAT:-8}"

echo "[2/5] fit tokenizer"
run_cmd \
  "$TOKENIZER_PYTHON_BIN" "$TOKENIZER_REPO_ROOT/scripts/train_tokenizer.py" \
  --output-dir "$TOKENIZER_DIR" \
  --document-manifest "$MANIFEST_DIR/manifest.json" \
  --document-splits train \
  --sample-max-documents "${TOKENIZER_SAMPLE_MAX_DOCUMENTS:-10000}" \
  --tokenizer epiplex \
  --tokenizer-task "${TOKENIZER_TASK:-generic}" \
  --tokenizer-vocab-size "${TOKENIZER_VOCAB_SIZE:-8192}" \
  --tokenizer-fit-workers "${TOKENIZER_FIT_WORKERS:-8}"

echo "[3/5] pack corpus"
run_cmd \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/pack_pretraining_corpus.py" \
  --document-manifest "$MANIFEST_DIR/manifest.json" \
  --output-dir "$PACK_DIR" \
  --seq-len "${SEQ_LEN:-2048}" \
  --target-shard-sequences "${TARGET_SHARD_SEQUENCES:-16384}" \
  --tokenizer epiplex \
  --tokenizer-load "$TOKENIZER_DIR/reasoning_tokenizer.json"

echo "[4/5] 8xH100 burn-in"
run_cmd \
  env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  PACKED_MANIFEST="$PACK_DIR/packed_manifest.json" \
  OUTPUT_ROOT="$BURNIN_DIR" \
  STEPS="${STEPS:-50}" \
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}" \
  SHARDING_STRATEGY="${SHARDING_STRATEGY:-fsdp_full_shard}" \
  ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}" \
  EVAL_EVERY="${EVAL_EVERY:-25}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}" \
  HIDDEN_SIZE="${HIDDEN_SIZE:-1536}" \
  NUM_LAYERS="${NUM_LAYERS:-24}" \
  NUM_HEADS="${NUM_HEADS:-16}" \
  NUM_KV_HEADS="${NUM_KV_HEADS:-8}" \
  INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-6144}" \
  LATENT_KV_DIM="${LATENT_KV_DIM:-192}" \
  "$REPO_ROOT/scripts/launch_pretraining_lm_8h100.sh"

echo "[5/5] resume check"
run_cmd \
  env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  PACKED_MANIFEST="$PACK_DIR/packed_manifest.json" \
  OUTPUT_ROOT="$RESUME_DIR" \
  RESUME_FROM="$BURNIN_DIR/checkpoints" \
  STEPS="${RESUME_STEPS:-75}" \
  BATCH_SIZE="${BATCH_SIZE:-4}" \
  GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}" \
  SHARDING_STRATEGY="${SHARDING_STRATEGY:-fsdp_full_shard}" \
  ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}" \
  EVAL_EVERY="${RESUME_EVAL_EVERY:-25}" \
  CHECKPOINT_EVERY="${RESUME_CHECKPOINT_EVERY:-25}" \
  HIDDEN_SIZE="${HIDDEN_SIZE:-1536}" \
  NUM_LAYERS="${NUM_LAYERS:-24}" \
  NUM_HEADS="${NUM_HEADS:-16}" \
  NUM_KV_HEADS="${NUM_KV_HEADS:-8}" \
  INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-6144}" \
  LATENT_KV_DIM="${LATENT_KV_DIM:-192}" \
  "$REPO_ROOT/scripts/launch_pretraining_lm_8h100.sh"

echo "Completed burn-in orchestration."
