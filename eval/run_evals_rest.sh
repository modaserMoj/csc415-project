#!/usr/bin/env bash
# Run all 5 models on SVAMP then Countdown-4, 512 max new tokens each.
# Prep: see comments at bottom. Run: bash eval/run_evals_svamp.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs

echo "=== Cross-evals GSM8K <-> MATH (no extra training) ==="

# MATH-tuned models on GSM8K (baseline and Phase 2)
PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/baseline_math_0.5b \
  --dataset data/gsm8k/test.parquet \
  --output_file results/baseline_math_0.5b_gsm8k.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_math_0.5b_gsm8k.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/phase2_math_0.5b \
  --dataset data/gsm8k/test.parquet \
  --output_file results/phase2_math_0.5b_gsm8k.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_math_0.5b_gsm8k.log 2>&1

# GSM8K-tuned models on MATH (baseline and Phase 2)
PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/baseline_gsm8k_0.5b \
  --dataset data/math/test.parquet \
  --output_file results/baseline_gsm8k_0.5b_math.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_gsm8k_0.5b_math.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/phase2_gsm8k_0.5b \
  --dataset data/math/test.parquet \
  --output_file results/phase2_gsm8k_0.5b_math.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_gsm8k_0.5b_math.log 2>&1