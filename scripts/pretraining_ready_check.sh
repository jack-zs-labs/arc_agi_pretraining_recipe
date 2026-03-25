#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
model_python_bin="${MODEL_PYTHON_BIN:-}"
output_dir="${PRETRAINING_READY_OUTPUT_DIR:-$repo_root/results/pretraining_ready_check}"

if [[ -z "$model_python_bin" && -x ".venv_atari/bin/python" ]]; then
  model_python_bin=".venv_atari/bin/python"
fi

if [[ -z "$model_python_bin" ]]; then
  echo "MODEL_PYTHON_BIN is not set and .venv_atari/bin/python was not found." >&2
  echo "A Torch-enabled Python is required for the packed-LM training smoke." >&2
  exit 1
fi

rm -rf "$output_dir"
mkdir -p "$output_dir"

echo "[1/4] compile modules"
"$python_bin" -m compileall models arc_trajectory_sampler training scripts >/dev/null

echo "[2/4] build compact large-pretraining profile"
"$python_bin" scripts/build_large_pretraining_profile.py \
  --output-dir "$output_dir/profile" \
  --tokenizer byte \
  --seq-len 128 \
  --target-shard-sequences 64 \
  --pgen-gsm8k-max-rows 8 \
  --pgen-mmlu-max-rows 8 \
  --benchmark-train-repeat 4 \
  --oscar-native-repeat 4 \
  --ptraj-gsm8k-max-rows 4 \
  --ptraj-mmlu-max-rows 4 \
  --ptraj-core-max-rows 4 \
  --arc-episodes 2 \
  --gsm8k-bench-max-rows 8 \
  --mmlu-bench-max-rows 8 \
  --oscar-scope-max-documents 4 \
  --oscar-scope-max-chunks 32 \
  --oscar-scope-reasoning-max-examples 24 \
  --oscar-graph-reasoning-max-examples 12 \
  >/dev/null

manifest_path="$output_dir/profile/manifest/manifest.json"
packed_manifest_path="$output_dir/profile/packed/packed_manifest.json"
summary_path="$output_dir/profile/summary.json"

for path in "$manifest_path" "$packed_manifest_path" "$summary_path"; do
  if [[ ! -f "$path" ]]; then
    echo "Expected artifact is missing: $path" >&2
    exit 1
  fi
done

echo "[3/4] validate preflight summaries"
"$python_bin" - <<'PY' "$summary_path"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))

manifest_preflight = dict(summary.get("manifest_preflight") or {})
packed_preflight = dict(summary.get("packed_preflight") or {})

errors = []
if manifest_preflight.get("status") != "pass":
    errors.append("document-manifest preflight did not pass")
if packed_preflight.get("status") != "pass":
    errors.append("packed-manifest preflight did not pass")
if not str(summary.get("packed_manifest_path", "")).strip():
    errors.append("summary is missing packed_manifest_path")
if errors:
    raise SystemExit("; ".join(errors))
PY

echo "[4/4] packed-LM training smoke"
"$model_python_bin" scripts/train_pretraining_lm.py \
  --packed-manifest "$packed_manifest_path" \
  --steps 1 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --eval-every 1 \
  --max-eval-batches 1 \
  --checkpoint-every 1 \
  --grad-accumulation-steps 1 \
  --learning-rate 0.0002 \
  --architecture moe \
  --hidden-size 64 \
  --num-layers 2 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --intermediate-size 128 \
  --latent-kv-dim 16 \
  --attention-preset mla_default \
  --precision fp32 \
  --device cpu \
  --checkpoint-dir "$output_dir/train_smoke/checkpoints" \
  --output "$output_dir/train_smoke/train_summary.json" \
  --progress-output "$output_dir/train_smoke/train_progress.json" \
  --csv-output "$output_dir/train_smoke/train_metrics.csv" \
  >/dev/null

latest_checkpoint="$output_dir/train_smoke/checkpoints/latest.txt"
train_summary="$output_dir/train_smoke/train_summary.json"

for path in "$latest_checkpoint" "$train_summary"; do
  if [[ ! -f "$path" ]]; then
    echo "Expected training smoke artifact is missing: $path" >&2
    exit 1
  fi
done

echo "Pretraining readiness check passed."
echo "Artifacts: $output_dir"
