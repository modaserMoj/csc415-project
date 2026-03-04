#!/usr/bin/env bash
# One-time setup: create venv and install all dependencies.
# Venv goes on /virtual (local disk) to avoid NFS home quota limits.
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

ln -sfn "$VENV_DIR" .venv

export PIP_CACHE_DIR="/virtual/$(whoami)/csc415/pip-cache"
mkdir -p "$PIP_CACHE_DIR"

# ---------- 2. Install PyTorch (CUDA 12.1) ----------
pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# ---------- 3. Install project dependencies ----------
pip install -r requirements.txt

# ---------- 4. Optional: flash-attn for faster training ----------
TMPDIR="/virtual/$(whoami)/csc415/tmp" pip install flash-attn --no-build-isolation --no-cache-dir || \
    echo "flash-attn install failed (optional — training will still work without it)"

# ---------- 5. Create log directory ----------
mkdir -p logs

echo ""
echo "=== Setup complete ==="
echo "Venv location: $VENV_DIR"
echo "Activate with:  source .venv/bin/activate"
echo "Prepare data:   python data/prepare_data.py --dataset phase1_mix --local_dir data/phase1_mix"
echo "Run Phase 1:    bash phase1/scripts/train_phase1.sh"
