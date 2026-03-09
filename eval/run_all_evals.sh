#!/usr/bin/env bash
# Evaluate all 5 models (base, 2 baselines, 2 phase2) on all 5 datasets.
# Produces a results/ directory with JSON files for each (model, dataset) pair.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p results logs

DATASETS=(
    "data/gsm8k/test.parquet"
    "data/math/test.parquet"
    "data/countdown/test.parquet"
    "data/svamp/test.parquet"
    "data/countdown3/test.parquet"
)

DATASET_NAMES=(gsm8k math countdown4 svamp countdown3)

# 0.5B setup: base, 2 baselines (gsm8k, math), 2 phase2 (gsm8k, math) — no countdown fine-tuning
MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "checkpoints/baseline_gsm8k_0.5b"
    "checkpoints/baseline_math_0.5b"
    "checkpoints/phase2_gsm8k_0.5b"
    "checkpoints/phase2_math_0.5b"
)

MODEL_NAMES=(base_0.5b baseline_gsm8k_0.5b baseline_math_0.5b phase2_gsm8k_0.5b phase2_math_0.5b)

MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

for m_idx in "${!MODELS[@]}"; do
    model="${MODELS[$m_idx]}"
    model_name="${MODEL_NAMES[$m_idx]}"

    if [[ "$model" == checkpoints/* ]] && [ ! -d "$model" ]; then
        echo "SKIP: $model not found"
        continue
    fi

    for d_idx in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$d_idx]}"
        dataset_name="${DATASET_NAMES[$d_idx]}"

        if [ ! -f "$dataset" ]; then
            echo "SKIP: $dataset not found"
            continue
        fi

        output_file="results/${model_name}_${dataset_name}.json"

        if [ -f "$output_file" ]; then
            echo "SKIP: $output_file already exists"
            continue
        fi

        echo "=========================================="
        echo "Evaluating: ${model_name} on ${dataset_name}"
        echo "=========================================="

        PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
        python eval/evaluate.py \
            --model "$model" \
            --dataset "$dataset" \
            --output_file "$output_file" \
            --max_samples "$MAX_SAMPLES" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --batch_size 8
    done
done

echo ""
echo "=== All evaluations complete ==="
echo ""

# Print summary table
echo "Model                  | GSM8K  | MATH   | Count4 | SVAMP  | Count3"
echo "-----------------------|--------|--------|--------|--------|-------"
for m_idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$m_idx]}"
    row=$(printf "%-23s" "$model_name")
    for d_idx in "${!DATASET_NAMES[@]}"; do
        dataset_name="${DATASET_NAMES[$d_idx]}"
        result_file="results/${model_name}_${dataset_name}.json"
        if [ -f "$result_file" ]; then
            acc=$(python -c "import json; d=json.load(open('$result_file')); print(f\"{d['accuracy']*100:.1f}%\")")
            row="$row| $(printf '%7s' "$acc")"
        else
            row="$row|    -  "
        fi
    done
    echo "$row"
done
