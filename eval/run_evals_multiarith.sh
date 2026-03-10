#!/usr/bin/env bash
# Run all 5 models on MultiArith (600 examples), 512 max new tokens.
# Prep: python data/prepare_data.py --dataset multiarith --local_dir data/multiarith
# Run:  bash eval/run_evals_multiarith.sh
set -eu

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p logs

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model Qwen/Qwen2.5-0.5B \
  --dataset data/multiarith/test.parquet \
  --output_file results/base_0.5b_multiarith.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_base_0.5b_multiarith.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model modaserMoj/csc415-baseline-gsm8k-0.5b \
  --dataset data/multiarith/test.parquet \
  --output_file results/baseline_gsm8k_0.5b_multiarith.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_gsm8k_0.5b_multiarith.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model modaserMoj/csc415-baseline-math-0.5b \
  --dataset data/multiarith/test.parquet \
  --output_file results/baseline_math_0.5b_multiarith.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_baseline_math_0.5b_multiarith.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model modaserMoj/csc415-phase2-gsm8k-0.5b \
  --dataset data/multiarith/test.parquet \
  --output_file results/phase2_gsm8k_0.5b_multiarith.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_gsm8k_0.5b_multiarith.log 2>&1

PYTHONPATH="${PYTHONPATH:-}:$(pwd)" python eval/evaluate.py \
  --model modaserMoj/csc415-phase2-math-0.5b \
  --dataset data/multiarith/test.parquet \
  --output_file results/phase2_math_0.5b_multiarith.json \
  --max_samples 1000 \
  --max_new_tokens 512 \
  --batch_size 8 \
  > logs/eval_phase2_math_0.5b_multiarith.log 2>&1
