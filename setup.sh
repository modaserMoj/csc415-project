#!/usr/bin/env bash
# One-time setup: clone TinyZero, create venv, install everything.
# Venv goes on /virtual (local disk with 131GB) to avoid NFS home quota limits.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="/virtual/$(whoami)/csc415/venv"

# ---------- 1. Python virtual environment (on local disk) ----------
mkdir -p "$(dirname "$VENV_DIR")"
if [ ! -d "$VENV_DIR" ]; then
    pip install --user virtualenv
    python3 -m virtualenv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Symlink so other scripts can use .venv as a shortcut
ln -sfn "$VENV_DIR" .venv

# Point pip cache to local disk too
export PIP_CACHE_DIR="/virtual/$(whoami)/csc415/pip-cache"
mkdir -p "$PIP_CACHE_DIR"

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
echo "Venv location: $VENV_DIR"
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "  (or:          source .venv/bin/activate)"
echo "Prepare data:   python data/prepare_data.py --dataset phase1_mix --local_dir data/phase1_mix"
echo "Run Phase 1:    export BASE_MODEL=Qwen/Qwen2.5-1.5B DATA_DIR=data/phase1_mix && bash phase1/scripts/train_phase1.sh"
