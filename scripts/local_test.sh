#!/usr/bin/env bash
# Quick end-to-end test: 2 training steps each for Phase 1, Phase 2, and baseline.
# Uses tiny batch sizes and short sequences to fit on limited GPU memory.
# Expected runtime: ~5-10 minutes on a 16GB GPU.
set -euxo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"

# ---------- 1. Prepare tiny test dataset ----------
echo "=== Preparing tiny test datasets ==="
python data/prepare_data.py --dataset countdown --local_dir data/test_countdown --size 50 --num_operands 4

# ---------- 2. Phase 1: RND exploration (2 steps) ----------
echo "=== Phase 1 local test (2 steps) ==="
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase1/main_phase1.py \
    --model "$MODEL" \
    --train_file data/test_countdown/train.parquet \
    --eval_file data/test_countdown/test.parquet \
    --output_dir checkpoints/test_phase1 \
    --max_steps 2 \
    --batch_size 2 \
    --grad_accum 1 \
    --num_generations 2 \
    --max_prompt_length 128 \
    --max_completion_length 64 \
    --beta 0.04 \
    --lr 5e-7 \
    --save_steps 999 \
    --logging_steps 1 \
    --rnd_device cuda \
    2>&1 | tee logs/test_phase1.log

echo "=== Phase 1 test PASSED ==="

# ---------- 3. Phase 2: correctness fine-tune (2 steps) ----------
echo "=== Phase 2 local test (2 steps) ==="
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase2/main_phase2.py \
    --model checkpoints/test_phase1 \
    --train_file data/test_countdown/train.parquet \
    --eval_file data/test_countdown/test.parquet \
    --output_dir checkpoints/test_phase2 \
    --max_steps 2 \
    --batch_size 2 \
    --grad_accum 1 \
    --num_generations 2 \
    --max_prompt_length 128 \
    --max_completion_length 64 \
    --beta 0.04 \
    --lr 5e-7 \
    --save_steps 999 \
    --logging_steps 1 \
    2>&1 | tee logs/test_phase2.log

echo "=== Phase 2 test PASSED ==="

# ---------- 4. Baseline: correctness fine-tune from base (2 steps) ----------
echo "=== Baseline local test (2 steps) ==="
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase2/main_phase2.py \
    --model "$MODEL" \
    --train_file data/test_countdown/train.parquet \
    --eval_file data/test_countdown/test.parquet \
    --output_dir checkpoints/test_baseline \
    --max_steps 2 \
    --batch_size 2 \
    --grad_accum 1 \
    --num_generations 2 \
    --max_prompt_length 128 \
    --max_completion_length 64 \
    --beta 0.04 \
    --lr 5e-7 \
    --save_steps 999 \
    --logging_steps 1 \
    2>&1 | tee logs/test_baseline.log

echo "=== Baseline test PASSED ==="

# ---------- 5. Eval on Phase 2 checkpoint ----------
echo "=== Eval local test ==="
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python eval/evaluate.py \
    --model checkpoints/test_phase2 \
    --dataset data/test_countdown/test.parquet \
    --max_samples 10 \
    --max_new_tokens 64 \
    2>&1 | tee logs/test_eval.log

echo "=== Eval test PASSED ==="

echo ""
echo "============================================"
echo "  ALL LOCAL TESTS PASSED"
echo "  Pipeline is ready for Vast.ai"
echo "============================================"

# Clean up test checkpoints
rm -rf checkpoints/test_phase1 checkpoints/test_phase2 checkpoints/test_baseline data/test_countdown
echo "Cleaned up test artifacts."
