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

# Phase1 model
MODEL="modaserMoj/csc415-phase1-0.5b-fast"
MODEL_NAME="phase1_0.5b"

# Default to a quick smoke test unless overridden.
# For full runs, set MAX_SAMPLES=1000 (or your desired value).
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "=========================================="
echo "Evaluating: phase1_0.5b on math"
echo "=========================================="

if [ -f "data/math/test.parquet" ]; then
  PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
  "$PYTHON_BIN" -u eval/evaluate.py \
    --model "$MODEL" \
    --dataset "data/math/test.parquet" \
    --output_file "results/${MODEL_NAME}_math.json" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    # --progress_interval 5 \
    2>&1 | tee "logs/eval_${MODEL_NAME}_math.log"
else
  echo "Skipping math: data/math/test.parquet not found"
fi

echo ""
echo "=== Phase1 evaluations complete ==="
echo ""

# Print summary for phase1
echo "Dataset   | Accuracy"
echo "----------|---------"
result_file="results/${MODEL_NAME}_math.json"
if [ -f "$result_file" ]; then
  acc=$("$PYTHON_BIN" -c "import json; d=json.load(open('$result_file')); print(f\"{d['accuracy']*100:.1f}%\")")
  echo "$(printf '%-10s' "math")| $acc"
else
  echo "$(printf '%-10s' "math")| -"
fi