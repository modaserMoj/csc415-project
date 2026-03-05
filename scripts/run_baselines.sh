#!/usr/bin/env bash
# Run 3 baseline GRPO fine-tuning runs directly from base Qwen (no Phase 1).
# These serve as controls to measure the effect of Phase 1 exploration.
set -euxo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Baseline-A: GRPO on GSM8K ==="
BASE_MODEL=Qwen/Qwen2.5-1.5B \
DATA_DIR=data/gsm8k \
OUTPUT_DIR=checkpoints/baseline_gsm8k \
LOG_NAME=baseline_gsm8k \
    bash phase2/scripts/train_phase2.sh

echo "=== Baseline-B: GRPO on MATH ==="
BASE_MODEL=Qwen/Qwen2.5-1.5B \
DATA_DIR=data/math \
OUTPUT_DIR=checkpoints/baseline_math \
LOG_NAME=baseline_math \
    bash phase2/scripts/train_phase2.sh

echo "=== Baseline-C: GRPO on Countdown-4 ==="
BASE_MODEL=Qwen/Qwen2.5-1.5B \
DATA_DIR=data/countdown \
OUTPUT_DIR=checkpoints/baseline_countdown \
LOG_NAME=baseline_countdown \
    bash phase2/scripts/train_phase2.sh

echo "=== All baselines complete ==="
