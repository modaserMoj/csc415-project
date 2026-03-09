#!/usr/bin/env bash
# Push local checkpoints to Hugging Face (modaserMoj).
# Run from project root. Requires: pip install transformers huggingface_hub
# Login once: huggingface-cli login

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

HF_USER="${HF_USER:-modaserMoj}"

push_one() {
    local local_dir="$1"
    local repo_id="$2"
    if [ ! -d "$local_dir" ]; then
        echo "SKIP: $local_dir not found"
        return 0
    fi
    echo "Pushing $local_dir -> $HF_USER/$repo_id"
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
p = '$local_dir'
repo = '$HF_USER/$repo_id'
m = AutoModelForCausalLM.from_pretrained(p)
t = AutoTokenizer.from_pretrained(p)
m.push_to_hub(repo)
t.push_to_hub(repo)
print('Done:', repo)
"
}

# Phase 1 already backed up as: modaserMoj/csc415-phase1-0.5b-fast

push_one "checkpoints/baseline_gsm8k_0.5b" "csc415-baseline-gsm8k-0.5b"
push_one "checkpoints/baseline_math_0.5b"  "csc415-baseline-math-0.5b"
push_one "checkpoints/phase2_gsm8k_0.5b"   "csc415-phase2-gsm8k-0.5b"
push_one "checkpoints/phase2_math_0.5b"   "csc415-phase2-math-0.5b"

echo "All done."
