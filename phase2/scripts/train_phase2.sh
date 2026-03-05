#!/usr/bin/env bash

set -euxo pipefail

: "${BASE_MODEL:=checkpoints/phase1}"
: "${DATA_DIR:=data/countdown}"
: "${OUTPUT_DIR:=checkpoints/phase2}"
: "${LOG_NAME:=phase2_train}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase2/main_phase2.py \
    --model "$BASE_MODEL" \
    --train_file "$DATA_DIR/train.parquet" \
    --eval_file "$DATA_DIR/test.parquet" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 20 \
    --grad_accum 4 \
    --num_generations 5 \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --beta 0.04 \
    --lr 5e-7 \
    --save_steps 200 \
    --logging_steps 1 \
    2>&1 | tee "logs/${LOG_NAME}.log"
