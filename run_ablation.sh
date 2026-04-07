#!/usr/bin/env bash

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B}"
DATA_DIR="${DATA_DIR:-data/phase1_mix}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-data}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-checkpoints/ablation}"
LOGS_ROOT="${LOGS_ROOT:-logs/ablation}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation}"

MAX_STEPS="${MAX_STEPS:-500}"

MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-500}"

RUN_ONLY="${RUN_ONLY:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINTS_ROOT" "$LOGS_ROOT" "$RESULTS_ROOT"

BASE_EPOCHS=3
BASE_BATCH=20
BASE_GRAD_ACCUM=4
BASE_NUM_GEN=5
BASE_MAX_COMP=512
BASE_BETA=0.04
BASE_LR=5e-7
BASE_REWARD_SCALE=1.0
BASE_REWARD_NORM=batch
BASE_SAVE_STEPS=200

declare -a EXPERIMENTS=(
    "baseline|                          |      |Baseline (all defaults)"
    "A1      |--rnd_reward_norm         |running|Reward norm: running stats"
    "A3      |--beta                    |0.01   |KL penalty: lower (more exploration)"
    "A4      |--beta                    |0.1    |KL penalty: higher (stay close to base)"
)

# helper function to run only one at a time
train_run() {
    local name="$1"
    local extra_flag="$2"
    local extra_val="$3"

    local out_dir="$CHECKPOINTS_ROOT/$name"
    local log_file="$LOGS_ROOT/train_${name}.log"
    mkdir -p "$out_dir"

    echo ""
    echo "  TRAINING: $name"
    [[ -n "$extra_flag" ]] && echo "  Changed: $extra_flag $extra_val"
    echo "  Output:   $out_dir"
    echo "  Log:      $log_file"

    # Build the extra-param fragment (empty string for baseline)
    local extra_args=""
    if [[ -n "${extra_flag// /}" ]]; then
        extra_args="$extra_flag $extra_val"
    fi

    # shellcheck disable=SC2086
    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    python phase1/main_phase1.py \
        --model         "$BASE_MODEL" \
        --train_file    "$DATA_DIR/train.parquet" \
        --eval_file     "$DATA_DIR/test.parquet" \
        --output_dir    "$out_dir" \
        --num_epochs    "$BASE_EPOCHS" \
        --batch_size    "$BASE_BATCH" \
        --grad_accum    "$BASE_GRAD_ACCUM" \
        --num_generations "$BASE_NUM_GEN" \
        --max_completion_length "$BASE_MAX_COMP" \
        --beta          "$BASE_BETA" \
        --lr            "$BASE_LR" \
        --max_steps     "$MAX_STEPS" \
        --save_steps    "$BASE_SAVE_STEPS" \
        --logging_steps 1 \
        --rnd_device    cuda \
        --rnd_reward_scale "$BASE_REWARD_SCALE" \
        --rnd_reward_norm  "$BASE_REWARD_NORM" \
        $extra_args \
        2>&1 | tee "$log_file"

    echo "  ✓ Training complete: $name"
}

eval_run() {
    local name="$1"
    local ckpt_dir="$CHECKPOINTS_ROOT/$name"

    if [[ ! -d "$ckpt_dir" ]]; then
        echo "Checkpoint not found for $name at $ckpt_dir , skipping eval"
        return
    fi

    local datasets=(gsm8k math svamp multiarith)

    echo ""
    echo "  EVALUATING: $name"

    for ds in "${datasets[@]}"; do
        local ds_file="$EVAL_DATA_ROOT/$ds/test.parquet"
        local out_file="$RESULTS_ROOT/${name}_${ds}.json"
        local log_file="$LOGS_ROOT/eval_${name}_${ds}.log"

        if [[ ! -f "$ds_file" ]]; then
            echo "Dataset not found: $ds_file , skipping"
            continue
        fi

        echo "  → $ds"
        PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
        python eval/evaluate.py \
            --model       "$ckpt_dir" \
            --dataset     "$ds_file" \
            --output_file "$out_file" \
            --max_samples "$MAX_EVAL_SAMPLES" \
            --max_new_tokens 512 \
            --batch_size  8 \
            2>&1 | tee "$log_file"
    done
}

print_summary() {
    local datasets=(gsm8k math svamp multiarith)
    local names=()
    for row in "${EXPERIMENTS[@]}"; do
        local name
        name="$(echo "$row" | awk -F'|' '{print $1}' | xargs)"
        names+=("$name")
    done

    echo ""
    echo "  ABLATION RESULTS SUMMARY"
    printf "%-12s" "Run"
    for ds in "${datasets[@]}"; do printf "  %-10s" "$ds"; done
    echo ""
    printf "%-12s" "───────────"
    for ds in "${datasets[@]}"; do printf "  %-10s" "──────────"; done
    echo ""

    for name in "${names[@]}"; do
        printf "%-12s" "$name"
        for ds in "${datasets[@]}"; do
            local out_file="$RESULTS_ROOT/${name}_${ds}.json"
            if [[ -f "$out_file" ]]; then
                local acc
                acc=$(python3 -c "import json; x=json.load(open('$out_file')); print(f\"{x['accuracy']*100:.1f}%\")" 2>/dev/null || echo "err")
                printf "  %-10s" "$acc"
            else
                printf "  %-10s" "-"
            fi
        done
        echo ""
    done

    echo ""
    echo "Full results in: $RESULTS_ROOT/"
    echo "Training logs:   $LOGS_ROOT/"
}

echo ""

for row in "${EXPERIMENTS[@]}"; do
    name="$(echo "$row"   | awk -F'|' '{print $1}' | xargs)"
    flag="$(echo "$row"   | awk -F'|' '{print $2}' | xargs)"
    val="$(echo "$row"    | awk -F'|' '{print $3}' | xargs)"
    desc="$(echo "$row"   | awk -F'|' '{print $4}' | xargs)"

    if [[ -n "$RUN_ONLY" && "$name" != "$RUN_ONLY" ]]; then
        continue
    fi

    echo ""
    echo "$name: $desc"

    [[ "$EVAL_ONLY" == "0" ]] && train_run "$name" "$flag" "$val"
    [[ "$TRAIN_ONLY" == "0" ]] && eval_run "$name"
done

if [[ -z "$RUN_ONLY" && "$TRAIN_ONLY" == "0" ]]; then
    print_summary
fi

echo ""
echo "Done."
