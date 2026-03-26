#!/usr/bin/env bash
# Evaluate the phase1 model on the configured datasets.
# Writes results/*.json files for the phase1 model.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p results logs

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

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

echo "=========================================="
echo "Evaluating: phase1_0.5b on gsm8k"
echo "=========================================="

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python eval/evaluate.py \
  --model "$MODEL" \
  --dataset "data/gsm8k/test.parquet" \
  --output_file "results/${MODEL_NAME}_gsm8k.json" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size 8 \
  > "logs/eval_${MODEL_NAME}_gsm8k.log" 2>&1

echo "=========================================="
echo "Evaluating: phase1_0.5b on math"
echo "=========================================="

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python eval/evaluate.py \
  --model "$MODEL" \
  --dataset "data/math/test.parquet" \
  --output_file "results/${MODEL_NAME}_math.json" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size 8 \
  > "logs/eval_${MODEL_NAME}_math.log" 2>&1

echo "=========================================="
echo "Evaluating: phase1_0.5b on svamp"
echo "=========================================="

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python eval/evaluate.py \
  --model "$MODEL" \
  --dataset "data/svamp/test.parquet" \
  --output_file "results/${MODEL_NAME}_svamp.json" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size 8 \
  > "logs/eval_${MODEL_NAME}_svamp.log" 2>&1

echo "=========================================="
echo "Evaluating: phase1_0.5b on multiarith"
echo "=========================================="

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python eval/evaluate.py \
  --model "$MODEL" \
  --dataset "data/multiarith/test.parquet" \
  --output_file "results/${MODEL_NAME}_multiarith.json" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --batch_size 8 \
  > "logs/eval_${MODEL_NAME}_multiarith.log" 2>&1

echo ""
echo "=== Phase1 evaluations complete ==="
echo ""

# Print summary for phase1
echo "Dataset   | Accuracy"
echo "----------|---------"
for d_idx in "${!DATASET_NAMES[@]}"; do
  dataset_name="${DATASET_NAMES[$d_idx]}"
  result_file="results/${MODEL_NAME}_${dataset_name}.json"
  if [ -f "$result_file" ]; then
    acc=$(python -c "import json; d=json.load(open('$result_file')); print(f\"{d['accuracy']*100:.1f}%\")")
    echo "$(printf '%-10s' "$dataset_name")| $acc"
  else
    echo "$(printf '%-10s' "$dataset_name")| -"
  fi
done