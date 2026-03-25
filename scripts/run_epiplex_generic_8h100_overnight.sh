#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/load_pretraining_env.sh"
load_pretraining_repo_env "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_atari/bin/python}"
TOKENIZER_REPO_ROOT="${TOKENIZER_REPO_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/epiplex_tokenizer_trainer}"
TOKENIZER_PYTHON_BIN="${TOKENIZER_PYTHON_BIN:-$PYTHON_BIN}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/epiplex_generic_8h100_overnight_${DATE_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29580}"

MANIFEST_DIR="${RUN_ROOT}/manifest"
TOKENIZER_DIR="${RUN_ROOT}/tokenizer_epiplex_generic_8k"
PACK_DIR="${RUN_ROOT}/packed_epiplex_seq2048"
TRAIN_DIR="${RUN_ROOT}/run_8xh100_7b"
HF_UPLOAD_PATH_PREFIX="${HF_UPLOAD_PATH_PREFIX:-runs/$(basename "${RUN_ROOT}")}"
export HF_UPLOAD_PATH_PREFIX

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

echo "[1/6] build final manifest"
if [[ -f "$MANIFEST_DIR/manifest.json" && "${REBUILD_MANIFEST:-0}" != "1" ]]; then
  echo "Using existing manifest at $MANIFEST_DIR/manifest.json"
else
  run_cmd \
    "$PYTHON_BIN" "$REPO_ROOT/scripts/build_pretraining_manifest.py" \
    --output-dir "$MANIFEST_DIR" \
    --validation-fraction "${VALIDATION_FRACTION:-0.005}" \
    --seed "${SEED:-7}" \
    --include-dclm \
    --dclm-max-documents "${DCLM_MAX_DOCUMENTS:-1000000}" \
    --dclm-shuffle-buffer-size "${DCLM_SHUFFLE_BUFFER_SIZE:-20000}" \
    --dclm-min-text-chars "${DCLM_MIN_TEXT_CHARS:-256}" \
    --dclm-min-language-score "${DCLM_MIN_LANGUAGE_SCORE:-0.8}" \
    --dclm-language-allowlist "${DCLM_LANGUAGE_ALLOWLIST:-en}" \
    --include-oscar-scope \
    --include-oscar-scope-reasoning \
    --include-oscar-graph-reasoning \
    --oscar-scope-auto-discover \
    --oscar-scope-max-documents "${OSCAR_SCOPE_MAX_DOCUMENTS:-64}" \
    --oscar-scope-max-chunks "${OSCAR_SCOPE_MAX_CHUNKS:-2048}" \
    --oscar-scope-reasoning-max-examples "${OSCAR_SCOPE_REASONING_MAX_EXAMPLES:-5000}" \
    --oscar-graph-reasoning-max-examples "${OSCAR_GRAPH_REASONING_MAX_EXAMPLES:-5000}" \
    --oscar-native-repeat "${OSCAR_NATIVE_REPEAT:-16}" \
    --oscar-reasoning-repeat "${OSCAR_REASONING_REPEAT:-32}" \
    --oscar-graph-repeat "${OSCAR_GRAPH_REPEAT:-32}"
fi

echo "[2/6] fit final tokenizer"
if [[ -f "$TOKENIZER_DIR/reasoning_tokenizer.json" && "${REFIT_TOKENIZER:-0}" != "1" ]]; then
  echo "Using existing tokenizer at $TOKENIZER_DIR/reasoning_tokenizer.json"
else
  run_cmd \
    "$TOKENIZER_PYTHON_BIN" "$TOKENIZER_REPO_ROOT/scripts/train_tokenizer.py" \
    --output-dir "$TOKENIZER_DIR" \
    --document-manifest "$MANIFEST_DIR/manifest.json" \
    --document-splits train \
    --sample-max-documents "${TOKENIZER_SAMPLE_MAX_DOCUMENTS:-50000}" \
    --tokenizer epiplex \
    --tokenizer-task "${TOKENIZER_TASK:-generic}" \
    --tokenizer-vocab-size "${TOKENIZER_VOCAB_SIZE:-8192}" \
    --tokenizer-fit-workers "${TOKENIZER_FIT_WORKERS:-8}"
fi

echo "[3/6] pack final corpus"
if [[ -f "$PACK_DIR/packed_manifest.json" && "${REPACK_CORPUS:-0}" != "1" ]]; then
  echo "Using existing packed corpus at $PACK_DIR/packed_manifest.json"
else
  run_cmd \
    "$PYTHON_BIN" "$REPO_ROOT/scripts/pack_pretraining_corpus.py" \
    --document-manifest "$MANIFEST_DIR/manifest.json" \
    --output-dir "$PACK_DIR" \
    --seq-len "${SEQ_LEN:-2048}" \
    --target-shard-sequences "${TARGET_SHARD_SEQUENCES:-16384}" \
    --tokenizer epiplex \
    --tokenizer-load "$TOKENIZER_DIR/reasoning_tokenizer.json"
fi

echo "[4/6] validate packed manifest"
run_cmd \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/validate_packed_pretraining_manifest.py" \
  --packed-manifest "$PACK_DIR/packed_manifest.json" \
  --expect-tokenizer-kind "${EXPECT_TOKENIZER_KIND:-epiplex}" \
  --expect-tokenizer-task "${EXPECT_TOKENIZER_TASK:-generic}" \
  --expect-seq-len "${EXPECT_SEQ_LEN:-2048}" \
  --min-train-sequences "${MIN_TRAIN_SEQUENCES:-100000}" \
  --require-corpus dclm \
  --require-corpus-prefix oscar

echo "[5/6] ensure Hugging Face checkpoint repo"
run_cmd \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/ensure_hf_checkpoint_repo.py" \
  --require-enabled

echo "[6/6] launch overnight 7B run"
TRAIN_TIMEOUT_HOURS="${TRAIN_TIMEOUT_HOURS:-36}"
TRAIN_COMMAND=(
  env
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  MASTER_ADDR="$MASTER_ADDR"
  MASTER_PORT="$MASTER_PORT"
  PACKED_MANIFEST="$PACK_DIR/packed_manifest.json"
  OUTPUT_ROOT="$TRAIN_DIR"
  STEPS="${TRAIN_STEPS:-20000}"
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
  EVAL_EVERY="${EVAL_EVERY:-100}"
  MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-8}"
  "$REPO_ROOT/scripts/launch_pretraining_lm_8h100_7b.sh"
)

if [[ "$DRY_RUN" == "1" ]]; then
  if command -v timeout >/dev/null 2>&1 && [[ "$TRAIN_TIMEOUT_HOURS" != "0" ]]; then
    printf '+ '
    printf '%q ' timeout --signal=INT "${TRAIN_TIMEOUT_HOURS}h" "${TRAIN_COMMAND[@]}"
    printf '\n'
  else
    printf '+ '
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
  fi
else
  if command -v timeout >/dev/null 2>&1 && [[ "$TRAIN_TIMEOUT_HOURS" != "0" ]]; then
    timeout --signal=INT "${TRAIN_TIMEOUT_HOURS}h" "${TRAIN_COMMAND[@]}"
  else
    echo "timeout command not available or disabled; launching without wall-clock guard." >&2
    "${TRAIN_COMMAND[@]}"
  fi
fi

echo "Completed overnight orchestration."
