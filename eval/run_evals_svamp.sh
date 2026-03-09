#!/usr/bin/env bash
# Run all 5 models on SVAMP then Countdown-4, 512 max new tokens each.
# Prep: see comments at bottom. Run: bash eval/run_evals_svamp.sh
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset data/svamp/test.parquet \
  --output_file results/base_0.5b_svamp.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_base_0.5b_svamp.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/baseline_gsm8k_0.5b \
  --dataset data/svamp/test.parquet \
  --output_file results/baseline_gsm8k_0.5b_svamp.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_gsm8k_0.5b_svamp.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/baseline_math_0.5b \
  --dataset data/svamp/test.parquet \
  --output_file results/baseline_math_0.5b_svamp.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_math_0.5b_svamp.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/phase2_gsm8k_0.5b \
  --dataset data/svamp/test.parquet \
  --output_file results/phase2_gsm8k_0.5b_svamp.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_gsm8k_0.5b_svamp.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/phase2_math_0.5b \
  --dataset data/svamp/test.parquet \
  --output_file results/phase2_math_0.5b_svamp.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_math_0.5b_svamp.log 2>&1

echo "=== SVAMP done; starting Countdown-4 ==="

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset data/countdown/test.parquet \
  --output_file results/base_0.5b_countdown4.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_base_0.5b_countdown4.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/baseline_gsm8k_0.5b \
  --dataset data/countdown/test.parquet \
  --output_file results/baseline_gsm8k_0.5b_countdown4.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_gsm8k_0.5b_countdown4.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/baseline_math_0.5b \
  --dataset data/countdown/test.parquet \
  --output_file results/baseline_math_0.5b_countdown4.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_math_0.5b_countdown4.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/phase2_gsm8k_0.5b \
  --dataset data/countdown/test.parquet \
  --output_file results/phase2_gsm8k_0.5b_countdown4.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_gsm8k_0.5b_countdown4.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model checkpoints/phase2_math_0.5b \
  --dataset data/countdown/test.parquet \
  --output_file results/phase2_math_0.5b_countdown4.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_math_0.5b_countdown4.log 2>&1

# --- PREP (run once from project root) ---
#   SVAMP:    python data/prepare_data.py --dataset svamp    --local_dir data/svamp
#   Countdown: python data/prepare_data.py --dataset countdown --local_dir data/countdown
# --- RUN ---
#   bash eval/run_evals_svamp.sh
#   (or: cd /workspace/csc415-project && bash eval/run_evals_svamp.sh)
