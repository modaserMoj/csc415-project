#!/usr/bin/env bash
# Evaluate the phase1 model on the configured datasets.
# Writes results/*.json files for the phase1 model.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p results logs

# Prefer project virtualenv Python (works on Windows + Git Bash).
if [ -x "$PROJECT_ROOT/env/Scripts/python.exe" ]; then
  PYTHON_BIN="$PROJECT_ROOT/env/Scripts/python.exe"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
PYTHON_UNBUFFERED="${PYTHON_UNBUFFERED:-1}"

# Datasets
DATASETS=(
  "data/gsm8k/test.parquet"
  "data/math/test.parquet"
  "data/svamp/test.parquet"
  "data/multiarith/test.parquet"
)
DATASET_NAMES=(gsm8k math svamp multiarith)

# Phase1 checkpoint: matches phase1/scripts/train_phase1.sh (--output_dir checkpoints/phase1).
# Override if you used a different run, e.g. README's checkpoints/phase1_0.5b_fast:
#   MODEL=checkpoints/phase1_0.5b_fast MODEL_NAME=phase1_0.5b_fast bash eval/eval_phase1.sh
MODEL="${MODEL:-checkpoints/phase1}"
MODEL_NAME="${MODEL_NAME:-phase1}"

# Default to a quick smoke test unless overridden.
# For full runs, set MAX_SAMPLES=1000 (or your desired value).
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  DATASET_NAME="${DATASET_NAMES[$i]}"

  echo "=========================================="
  echo "Evaluating: ${MODEL_NAME} on ${DATASET_NAME}"
  echo "=========================================="

  if [ -f "$DATASET" ]; then
    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    "$PYTHON_BIN" -u eval/evaluate.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --output_file "results/${MODEL_NAME}_${DATASET_NAME}.json" \
      --max_samples "$MAX_SAMPLES" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --batch_size "$BATCH_SIZE" \
      2>&1 | tee "logs/eval_${MODEL_NAME}_${DATASET_NAME}.log"
  else
    echo "Skipping ${DATASET_NAME}: ${DATASET} not found"
  fi
done

echo ""
echo "=== Phase1 evaluations complete ==="
echo ""

# Print summary for phase1
echo "Dataset   | Accuracy"
echo "----------|---------"
for d in "${DATASET_NAMES[@]}"; do
  result_file="results/${MODEL_NAME}_${d}.json"
  if [ -f "$result_file" ]; then
    acc=$("$PYTHON_BIN" -c "import json; x=json.load(open('${result_file}')); print(f\"{x['accuracy']*100:.1f}%\")")
    echo "$(printf '%-10s' "$d")| $acc"
  else
    echo "$(printf '%-10s' "$d")| -"
  fi
done
