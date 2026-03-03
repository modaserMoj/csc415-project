#!/usr/bin/env bash
# One-time setup: clone TinyZero, create venv, install everything.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 1. Python virtual environment ----------
if [ ! -d "env" ]; then
    python3 -m venv env
fi
source env/bin/activate

# ---------- 2. Clone TinyZero ----------
if [ ! -d "TinyZero" ]; then
    git clone https://github.com/Jiayi-Pan/TinyZero.git
fi

# ---------- 3. Install TinyZero (verl) ----------
pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 --no-build-isolation
pip install ray
cd TinyZero && pip install -e . && cd ..
pip install flash-attn --no-build-isolation

# ---------- 4. Project-specific deps ----------
pip install -r requirements.txt

# ---------- 5. Create log directory ----------
mkdir -p logs

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source env/bin/activate"
echo "Prepare data:   python data/prepare_data.py --dataset countdown --local_dir data/countdown"
echo "Run Phase 1:    export BASE_MODEL=Qwen/Qwen2.5-1.5B DATA_DIR=data/countdown && bash phase1/scripts/train_phase1.sh"
