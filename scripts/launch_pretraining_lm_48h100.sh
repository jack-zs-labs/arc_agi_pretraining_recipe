#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/load_pretraining_env.sh"
load_pretraining_repo_env "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv_atari/bin/python}"
NNODES="${NNODES:-6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK to the node rank in [0, ${NNODES}).}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host or IP.}"
MASTER_PORT="${MASTER_PORT:-29500}"
PACKED_MANIFEST="${PACKED_MANIFEST:?Set PACKED_MANIFEST to the packed_manifest.json path.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/pretraining_lm_48h100}"
RESUME_FROM="${RESUME_FROM:-}"
SHARDING_STRATEGY="${SHARDING_STRATEGY:-fsdp_full_shard}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTPUT_ROOT}"

TORCHRUN_ARGS=(
  -m torch.distributed.run
  --nnodes "${NNODES}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/scripts/train_pretraining_lm.py"
  --packed-manifest "${PACKED_MANIFEST}"
  --device cuda
  --precision bf16
  --sharding-strategy "${SHARDING_STRATEGY}"
  --architecture dense
  --steps "${STEPS:-2000}"
  --batch-size "${BATCH_SIZE:-4}"
  --grad-accumulation-steps "${GRAD_ACCUMULATION_STEPS:-8}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}"
  --eval-every "${EVAL_EVERY:-100}"
  --max-eval-batches "${MAX_EVAL_BATCHES:-32}"
  --checkpoint-every "${CHECKPOINT_EVERY:-100}"
  --learning-rate "${LEARNING_RATE:-3e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-2}"
  --lr-warmup-steps "${LR_WARMUP_STEPS:-100}"
  --min-learning-rate-scale "${MIN_LEARNING_RATE_SCALE:-0.1}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
  --prefetch-batches "${PREFETCH_BATCHES:-8}"
  --pin-memory
  --hidden-size "${HIDDEN_SIZE:-1536}"
  --num-layers "${NUM_LAYERS:-24}"
  --num-heads "${NUM_HEADS:-16}"
  --num-kv-heads "${NUM_KV_HEADS:-8}"
  --intermediate-size "${INTERMEDIATE_SIZE:-6144}"
  --latent-kv-dim "${LATENT_KV_DIM:-192}"
  --attention-preset "${ATTENTION_PRESET:-mla_default}"
  --seed "${SEED:-7}"
  --output "${OUTPUT_ROOT}/summary.json"
  --progress-output "${OUTPUT_ROOT}/progress.json"
  --csv-output "${OUTPUT_ROOT}/summary.csv"
  --checkpoint-dir "${OUTPUT_ROOT}/checkpoints"
)

if [[ "${ACTIVATION_CHECKPOINTING}" == "1" || "${ACTIVATION_CHECKPOINTING}" == "true" || "${ACTIVATION_CHECKPOINTING}" == "TRUE" ]]; then
  TORCHRUN_ARGS+=(--activation-checkpointing)
else
  TORCHRUN_ARGS+=(--no-activation-checkpointing)
fi

if [[ -n "${RESUME_FROM}" ]]; then
  TORCHRUN_ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${PYTHON_BIN}" "${TORCHRUN_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec "${PYTHON_BIN}" "${TORCHRUN_ARGS[@]}"
