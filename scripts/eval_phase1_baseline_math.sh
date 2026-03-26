#!/usr/bin/env bash
# Evaluate phase1_baseline_math model on all datasets
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p results logs

DATASETS=(
  "data/gsm8k/test.parquet"
  "data/math/test.parquet"
  "data/svamp/test.parquet"
  "data/multiarith/test.parquet"
)
DATASET_NAMES=(gsm8k math svamp multiarith)

MODEL="checkpoints/phase1_baseline_math_0.5b"
MODEL_NAME="phase1_baseline_math_0.5b"

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  DATASET_NAME="${DATASET_NAMES[$i]}"
  
  echo "=========================================="
  echo "Evaluating: phase1_baseline_math on $DATASET_NAME"
  echo "=========================================="
  
  PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
  python eval/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output_file "results/${MODEL_NAME}_${DATASET_NAME}.json" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size 8 \
    > "logs/eval_${MODEL_NAME}_${DATASET_NAME}.log" 2>&1
done

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to results/"
echo "=========================================="
