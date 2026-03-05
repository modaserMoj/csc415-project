#!/usr/bin/env bash
# Evaluate all 7 models on all 5 datasets.
# Produces a results/ directory with JSON files for each (model, dataset) pair.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p results

DATASETS=(
    "data/gsm8k/test.parquet"
    "data/math/test.parquet"
    "data/countdown/test.parquet"
    "data/svamp/test.parquet"
    "data/countdown3/test.parquet"
)

DATASET_NAMES=(gsm8k math countdown4 svamp countdown3)

MODELS=(
    "Qwen/Qwen2.5-1.5B"
    "checkpoints/baseline_gsm8k"
    "checkpoints/baseline_math"
    "checkpoints/baseline_countdown"
    "checkpoints/phase2_gsm8k"
    "checkpoints/phase2_math"
    "checkpoints/phase2_countdown"
)

MODEL_NAMES=(base baseline_gsm8k baseline_math baseline_countdown phase2_gsm8k phase2_math phase2_countdown)

MAX_SAMPLES="${MAX_SAMPLES:-500}"

for m_idx in "${!MODELS[@]}"; do
    model="${MODELS[$m_idx]}"
    model_name="${MODEL_NAMES[$m_idx]}"

    if [ ! -d "$model" ] && [[ "$model" != *"/"* ]]; then
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
