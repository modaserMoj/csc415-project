#!/usr/bin/env bash
# Quick smoke test (~5 min) to verify everything works before a full run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# ---------- 1. venv ----------
echo "=== 1/6  Checking venv ==="
if [ ! -d ".venv" ]; then fail "No .venv found — run setup.sh first"; fi
source .venv/bin/activate
pass "venv activated"

# ---------- 2. Python imports ----------
echo "=== 2/6  Checking imports ==="
python -c "
import torch, trl, accelerate, pandas
from sentence_transformers import SentenceTransformer
from trl import GRPOTrainer, GRPOConfig
print(f'torch {torch.__version__}  cuda {torch.cuda.is_available()}  trl {trl.__version__}')
" || fail "Import check failed"
pass "all imports OK"

# ---------- 3. GPU ----------
echo "=== 3/6  Checking GPU ==="
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU detected'
name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'{name}  {mem:.1f} GB')
" || fail "GPU check failed"
pass "GPU accessible"

# ---------- 4. RND module ----------
echo "=== 4/6  Testing RND module ==="
PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python -c "
from phase1.rnd_module import RNDModule
rnd = RNDModule(device='cuda')
rewards, metrics = rnd.compute_rewards(['This is a test response about math.', 'Another reasoning chain with numbers 2+3=5.'])
print(f'rewards shape: {rewards.shape}  metrics: {metrics}')
assert rewards.shape[0] == 2
" || fail "RND module test failed"
pass "RND module works on GPU"

# ---------- 5. Tiny dataset ----------
echo "=== 5/6  Preparing tiny test dataset ==="
python data/prepare_data.py --dataset countdown --local_dir data/smoke_test --size 100 --num_operands 4
pass "test dataset created (100 samples)"

# ---------- 6. Model download test ----------
echo "=== 6/6  Checking model access ==="
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
print(f'tokenizer vocab size: {tok.vocab_size}')
" || fail "Cannot download Qwen2.5-0.5B — check internet / HF access"
pass "Qwen2.5-0.5B accessible"

echo ""
echo -e "${GREEN}=== All checks passed! Ready for training. ===${NC}"
echo ""
echo "Next steps:"
echo "  tmux new -s phase1"
echo "  source .venv/bin/activate"
echo "  export BASE_MODEL=Qwen/Qwen2.5-0.5B"
echo "  export DATA_DIR=data/phase1_mix"
echo "  bash phase1/scripts/train_phase1.sh"
