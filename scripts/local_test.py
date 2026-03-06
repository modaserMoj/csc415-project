"""
Quick end-to-end pipeline test. Works on any OS (Windows, Linux, macOS).
Auto-detects GPU; falls back to CPU with a smaller model if none found.
Runs 2 training steps for Phase 1, Phase 2, and baseline, then eval.
"""

import subprocess
import sys
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
os.makedirs("logs", exist_ok=True)

PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

if HAS_CUDA:
    MODEL = "Qwen/Qwen2.5-1.5B"
    RND_DEVICE = "cuda"
    print("=== GPU detected ===")
else:
    MODEL = "Qwen/Qwen2.5-0.5B"
    RND_DEVICE = "cpu"
    print("=== No GPU — running on CPU (smaller model) ===")


def run(description, cmd):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=ENV)
    if result.returncode != 0:
        print(f"\n*** FAILED: {description} (exit code {result.returncode}) ***")
        sys.exit(1)
    print(f"  PASSED: {description}")


# 1. Prepare tiny dataset
run("Prepare tiny test dataset (50 samples)", [
    PYTHON, "data/prepare_data.py",
    "--dataset", "countdown",
    "--local_dir", "data/test_countdown",
    "--size", "50",
    "--num_operands", "4",
])

# 2. Phase 1: RND exploration (2 steps)
run("Phase 1 — RND exploration (2 steps)", [
    PYTHON, "phase1/main_phase1.py",
    "--model", MODEL,
    "--train_file", "data/test_countdown/train.parquet",
    "--eval_file", "data/test_countdown/test.parquet",
    "--output_dir", "checkpoints/test_phase1",
    "--max_steps", "2",
    "--batch_size", "2",
    "--grad_accum", "1",
    "--num_generations", "2",
    "--max_completion_length", "64",
    "--beta", "0.04",
    "--lr", "5e-7",
    "--save_steps", "999",
    "--logging_steps", "1",
    "--rnd_device", RND_DEVICE,
])

# 3. Phase 2: correctness fine-tune from Phase 1 checkpoint (2 steps)
run("Phase 2 — correctness fine-tune from Phase 1 (2 steps)", [
    PYTHON, "phase2/main_phase2.py",
    "--model", "checkpoints/test_phase1",
    "--train_file", "data/test_countdown/train.parquet",
    "--eval_file", "data/test_countdown/test.parquet",
    "--output_dir", "checkpoints/test_phase2",
    "--max_steps", "2",
    "--batch_size", "2",
    "--grad_accum", "1",
    "--num_generations", "2",
    "--max_completion_length", "64",
    "--beta", "0.04",
    "--lr", "5e-7",
    "--save_steps", "999",
    "--logging_steps", "1",
])

# 4. Baseline: correctness fine-tune from base model (2 steps)
run("Baseline — correctness fine-tune from base model (2 steps)", [
    PYTHON, "phase2/main_phase2.py",
    "--model", MODEL,
    "--train_file", "data/test_countdown/train.parquet",
    "--eval_file", "data/test_countdown/test.parquet",
    "--output_dir", "checkpoints/test_baseline",
    "--max_steps", "2",
    "--batch_size", "2",
    "--grad_accum", "1",
    "--num_generations", "2",
    "--max_completion_length", "64",
    "--beta", "0.04",
    "--lr", "5e-7",
    "--save_steps", "999",
    "--logging_steps", "1",
])

# 5. Eval on Phase 2 checkpoint
run("Evaluation — score Phase 2 checkpoint", [
    PYTHON, "eval/evaluate.py",
    "--model", "checkpoints/test_phase2",
    "--dataset", "data/test_countdown/test.parquet",
    "--max_samples", "10",
    "--max_new_tokens", "64",
])

# Clean up
print("\nCleaning up test artifacts...")
for d in ["checkpoints/test_phase1", "checkpoints/test_phase2", "checkpoints/test_baseline", "data/test_countdown"]:
    shutil.rmtree(d, ignore_errors=True)

print(f"""
{'='*60}
  ALL LOCAL TESTS PASSED
  Pipeline is ready for Vast.ai
{'='*60}
""")
