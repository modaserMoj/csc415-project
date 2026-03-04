#!/usr/bin/env bash
# Phase 2: Task Fine-tuning with GRPO + correctness reward (TRL)
# Tuned for: 1x RTX 4080 (16 GB VRAM), Qwen2.5-0.5B

set -euxo pipefail

: "${BASE_MODEL:=checkpoints/phase1}"
: "${DATA_DIR:=data/countdown}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase2/main_phase2.py \
    --model "$BASE_MODEL" \
    --train_file "$DATA_DIR/train.parquet" \
    --eval_file "$DATA_DIR/test.parquet" \
    --output_dir checkpoints/phase2 \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --num_generations 5 \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --beta 0.001 \
    --lr 1e-6 \
    --save_steps 50 \
    --logging_steps 1 \
    2>&1 | tee logs/phase2_train.log
