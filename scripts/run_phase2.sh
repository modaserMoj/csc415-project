#!/usr/bin/env bash
# Run all 3 Phase 2 fine-tuning runs sequentially.
set -euxo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PHASE1_CKPT="${PHASE1_CKPT:=checkpoints/phase1}"

echo "=== Phase 2-A: Fine-tune on GSM8K ==="
BASE_MODEL="$PHASE1_CKPT" bash phase2/scripts/train_phase2_gsm8k.sh

echo "=== Phase 2-B: Fine-tune on MATH ==="
BASE_MODEL="$PHASE1_CKPT" bash phase2/scripts/train_phase2_math.sh

echo "=== Phase 2-C: Fine-tune on Countdown-4 ==="
BASE_MODEL="$PHASE1_CKPT" bash phase2/scripts/train_phase2_countdown.sh

echo "=== All Phase 2 runs complete ==="
