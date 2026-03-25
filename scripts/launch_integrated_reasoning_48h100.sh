#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv_atari/bin/python}"
NNODES="${NNODES:-6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:?Set NODE_RANK for this node.}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR for the torchrun rendezvous host.}"
MASTER_PORT="${MASTER_PORT:-29500}"
CORPUS_MANIFEST="${CORPUS_MANIFEST:?Set CORPUS_MANIFEST to the exported manifest.json path.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/integrated_reasoning_48h100_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS+=(--resume-from "${RESUME_FROM}")
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes "${NNODES}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  scripts/train_integrated_reasoning_stack.py \
  --architectures dense \
  --corpus-manifest "${CORPUS_MANIFEST}" \
  --device cuda \
  --precision bf16 \
  --disable-core-auxiliary-heads \
  --disable-decision-action-heads \
  --force-full-train-layers \
  --seq-len "${SEQ_LEN:-2048}" \
  --batch-size "${MICRO_BATCH_SIZE:-2}" \
  --decision-batch-size "${DECISION_BATCH_SIZE:-2}" \
  --grad-accumulation-steps "${GRAD_ACCUMULATION_STEPS:-16}" \
  --steps "${STEPS:-2000}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-100}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" \
  --lr-warmup-steps "${LR_WARMUP_STEPS:-100}" \
  --min-learning-rate-scale "${MIN_LR_SCALE:-0.1}" \
  --hidden-size "${HIDDEN_SIZE:-1536}" \
  --num-layers "${NUM_LAYERS:-24}" \
  --num-heads "${NUM_HEADS:-12}" \
  --num-kv-heads "${NUM_KV_HEADS:-4}" \
  --intermediate-size "${INTERMEDIATE_SIZE:-6144}" \
  --latent-kv-dim "${LATENT_KV_DIM:-128}" \
  --output "${OUTPUT_ROOT}/summary.json" \
  --csv-output "${OUTPUT_ROOT}/summary.csv" \
  --checkpoint-dir "${OUTPUT_ROOT}/checkpoints" \
  --no-progress \
  "${RESUME_ARGS[@]}"
