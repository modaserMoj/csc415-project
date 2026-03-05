#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 1. Install project dependencies ----------
pip install --upgrade pip
pip install -r requirements.txt

# ---------- 2. Prepare datasets ----------
python data/prepare_data.py --dataset countdown  --local_dir data/countdown
python data/prepare_data.py --dataset countdown3 --local_dir data/countdown3
python data/prepare_data.py --dataset gsm8k      --local_dir data/gsm8k
python data/prepare_data.py --dataset math       --local_dir data/math
python data/prepare_data.py --dataset svamp      --local_dir data/svamp
python data/prepare_data.py --dataset phase1_mix --local_dir data/phase1_mix

# ---------- 3. Create directories ----------
mkdir -p logs checkpoints

echo ""
echo "=== Setup complete ==="
echo "Run Phase 1:  nohup bash phase1/scripts/train_phase1.sh > logs/phase1_train.log 2>&1 &"
echo "Run Phase 2:  nohup bash phase2/scripts/train_phase2.sh > logs/phase2_train.log 2>&1 &"
