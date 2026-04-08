
#!/usr/bin/env bash

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
DATA_DIR="${DATA_DIR:-data/phase1_mix}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-data}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-checkpoints/seeds}"
LOGS_ROOT="${LOGS_ROOT:-logs/seeds}"
RESULTS_ROOT="${RESULTS_ROOT:-results/seeds}"

SEEDS="${SEEDS:-42 1 123 456 67}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_STEPS_P2="${MAX_STEPS_P2:-500}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-500}"
EVAL_ONLY="${EVAL_ONLY:-0}"

BASE_EPOCHS=3
BASE_BATCH=2
BASE_NUM_GEN=4
BASE_GRAD_ACCUM=4
BASE_MAX_COMP=512
BASE_BETA=0.04
BASE_LR=5e-7
BASE_REWARD_SCALE=1.0
BASE_REWARD_NORM=batch
BASE_SAVE_STEPS=500

P2_BATCH=2
P2_NUM_GEN=4
P2_GRAD_ACCUM=4
P2_MAX_COMP=512
P2_BETA=0.04
P2_LR=5e-7
P2_SAVE_STEPS=500

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINTS_ROOT" "$LOGS_ROOT" "$RESULTS_ROOT"

train_seed() {
    local seed="$1"
    local name="seed_${seed}"
    local out_dir="$CHECKPOINTS_ROOT/$name"
    local log_file="$LOGS_ROOT/train_${name}.log"
    mkdir -p "$out_dir"

    echo ""
    echo "  TRAINING PHASE 1: $name"
    echo "  Seed: $seed"

    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
        --seed          "$seed" \
        2>&1 | tee "$log_file"

    echo "Phase 1 training complete: $name"
}

train_phase2_seed() {
    local seed="$1"
    local name="seed_${seed}"
    local p1_ckpt="$CHECKPOINTS_ROOT/$name"
    local out_dir="$CHECKPOINTS_ROOT/${name}_phase2_gsm8k"
    local log_file="$LOGS_ROOT/train_${name}_phase2_gsm8k.log"
    mkdir -p "$out_dir"

    echo ""
    echo "  TRAINING PHASE 2 (GSM8K): $name"
    echo "  Phase 1 checkpoint: $p1_ckpt"

    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python phase2/main_phase2.py \
        --model         "$p1_ckpt" \
        --train_file    "$EVAL_DATA_ROOT/gsm8k/train.parquet" \
        --eval_file     "$EVAL_DATA_ROOT/gsm8k/test.parquet" \
        --output_dir    "$out_dir" \
        --num_epochs    "$BASE_EPOCHS" \
        --batch_size    "$P2_BATCH" \
        --grad_accum    "$P2_GRAD_ACCUM" \
        --num_generations "$P2_NUM_GEN" \
        --max_completion_length "$P2_MAX_COMP" \
        --beta          "$P2_BETA" \
        --lr            "$P2_LR" \
        --max_steps     "$MAX_STEPS_P2" \
        --save_steps    "$P2_SAVE_STEPS" \
        --logging_steps 1 \
        2>&1 | tee "$log_file"

    echo "Phase 2 training complete: $name"
}

eval_seed() {
    local seed="$1"
    local name="seed_${seed}"
    local ckpt_dir="$CHECKPOINTS_ROOT/$name"

    if [[ ! -d "$ckpt_dir" ]]; then
        echo "Checkpoint not found for $name — skipping phase 1 eval"
        return
    fi

    local datasets=(gsm8k math svamp multiarith)

    echo ""
    echo "  EVALUATING PHASE 1: $name"

    for ds in "${datasets[@]}"; do
        local ds_file="$EVAL_DATA_ROOT/$ds/test.parquet"
        local out_file="$RESULTS_ROOT/${name}_${ds}.json"
        local log_file="$LOGS_ROOT/eval_${name}_${ds}.log"

        if [[ ! -f "$ds_file" ]]; then
            echo "Dataset not found: $ds_file — skipping"
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

eval_phase2_seed() {
    local seed="$1"
    local name="seed_${seed}"
    local ckpt_dir="$CHECKPOINTS_ROOT/${name}_phase2_gsm8k"

    if [[ ! -d "$ckpt_dir" ]]; then
        echo "Phase 2 checkpoint not found for $name — skipping phase 2 eval"
        return
    fi

    local out_file="$RESULTS_ROOT/${name}_phase2_gsm8k.json"
    local log_file="$LOGS_ROOT/eval_${name}_phase2_gsm8k.log"
    local ds_file="$EVAL_DATA_ROOT/gsm8k/test.parquet"

    echo ""
    echo "  EVALUATING PHASE 2 (GSM8K): $name"

    PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
    python eval/evaluate.py \
        --model       "$ckpt_dir" \
        --dataset     "$ds_file" \
        --output_file "$out_file" \
        --max_samples "$MAX_EVAL_SAMPLES" \
        --max_new_tokens 512 \
        --batch_size  8 \
        2>&1 | tee "$log_file"
}


print_summary() {
    local datasets=(gsm8k math svamp multiarith)

    echo ""
    echo "  PHASE 1 RESULTS"
    printf "%-12s" "Run"
    for ds in "${datasets[@]}"; do printf "  %-10s" "$ds"; done
    echo ""
    printf "%-12s" "───────────"
    for ds in "${datasets[@]}"; do printf "  %-10s" "──────────"; done
    echo ""

    for seed in $SEEDS; do
        local name="seed_${seed}"
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
    printf "%-12s" "mean±std"
    for ds in "${datasets[@]}"; do
        python3 - <<EOF
import json, os, statistics
accs = []
for seed in "${SEEDS}".split():
    f = "$RESULTS_ROOT/seed_{}_${ds}.json".format(seed)
    if os.path.exists(f):
        accs.append(json.load(open(f))["accuracy"] * 100)
if len(accs) > 1:
    print(f"  {statistics.mean(accs):.1f}±{statistics.stdev(accs):.1f}%", end="")
elif len(accs) == 1:
    print(f"  {accs[0]:.1f}%     ", end="")
else:
    print("  -         ", end="")
EOF
    done
    echo ""

    echo ""
    echo "  PHASE 2 RESULTS (GSM8K)"
    printf "%-12s  %-10s\n" "Run" "gsm8k"
    printf "%-12s  %-10s\n" "───────────" "──────────"
    for seed in $SEEDS; do
        local name="seed_${seed}"
        local out_file="$RESULTS_ROOT/${name}_phase2_gsm8k.json"
        if [[ -f "$out_file" ]]; then
            local acc
            acc=$(python3 -c "import json; x=json.load(open('$out_file')); print(f\"{x['accuracy']*100:.1f}%\")" 2>/dev/null || echo "err")
            printf "%-12s  %-10s\n" "$name" "$acc"
        else
            printf "%-12s  %-10s\n" "$name" "-"
        fi
    done

    echo ""
    echo "Full results in: $RESULTS_ROOT/"
    echo "Training logs:   $LOGS_ROOT/"
}

echo ""

for seed in $SEEDS; do
    ckpt="$CHECKPOINTS_ROOT/seed_${seed}"
    ckpt_p2="$CHECKPOINTS_ROOT/seed_${seed}_phase2_gsm8k"

    if [[ "$EVAL_ONLY" == "0" ]]; then
        # Phase 1 — skip if checkpoint exists
        if [[ -d "$ckpt" && -f "$ckpt/model.safetensors" ]]; then
            echo ""
            echo "Phase 1 checkpoint exists for seed_${seed} — skipping training"
        else
            train_seed "$seed"
        fi

        # Phase 2 — skip if checkpoint exists
        if [[ -d "$ckpt_p2" && -f "$ckpt_p2/model.safetensors" ]]; then
            echo ""
            echo "Phase 2 checkpoint exists for seed_${seed} — skipping training"
        else
            train_phase2_seed "$seed"
        fi
    fi

    eval_seed "$seed"
    eval_phase2_seed "$seed"
done

print_summary

echo ""
echo "Done."
