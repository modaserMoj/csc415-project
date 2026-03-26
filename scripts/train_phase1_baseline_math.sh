#!/usr/bin/env bash
# Phase 1 training on baseline_math model
# Applies RND exploration to the MATH-finetuned model
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p checkpoints logs

MODEL="modaserMoj/csc415-baseline-math-0.5b"
OUTPUT_DIR="checkpoints/phase1_baseline_math_0.5b"

echo "=========================================="
echo "Phase 1 Training: baseline_math + RND"
echo "=========================================="

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python3 phase1/main_phase1.py \
  --model "$MODEL" \
  --train_file data/phase1_mix/train.parquet \
  --eval_file data/phase1_mix/test.parquet \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs 3 \
  --batch_size 20 \
  --grad_accum 4 \
  --num_generations 5 \
  --max_completion_length 512 \
  --beta 0.04 \
  --lr 5e-7 \
  --save_steps 200 \
  --logging_steps 1 \
  --rnd_device cpu \
  > "logs/train_phase1_baseline_math.log" 2>&1

echo "Phase 1 training completed!"
echo "Checkpoint saved: $OUTPUT_DIR"
