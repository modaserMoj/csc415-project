#!/usr/bin/env bash
# Phase 1: Cognitive Pretraining with GRPO + RND (TRL)
# Tuned for: 1x RTX 4080 (16 GB VRAM), Qwen2.5-0.5B

set -euxo pipefail

: "${BASE_MODEL:=Qwen/Qwen2.5-0.5B}"
: "${DATA_DIR:=data/phase1_mix}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase1/main_phase1.py \
    --model "$BASE_MODEL" \
    --train_file "$DATA_DIR/train.parquet" \
    --eval_file "$DATA_DIR/test.parquet" \
    --output_dir checkpoints/phase1 \
    --num_epochs 3 \
    --batch_size 5 \
    --grad_accum 4 \
    --num_generations 5 \
    --max_prompt_length 512 \
    --max_completion_length 256 \
    --beta 0.001 \
    --lr 1e-6 \
    --save_steps 50 \
    --logging_steps 1 \
    --rnd_device cpu \
    --rnd_reward_scale 1.0 \
    --rnd_reward_norm batch \
    2>&1 | tee logs/phase1_train.log
