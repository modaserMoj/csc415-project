"""
Run all 5 models on MultiArith. Use this on Windows instead of the .sh script.
  Prep: python data/prepare_data.py --dataset multiarith --local_dir data/multiarith
  Run:  python eval/run_evals_multiarith.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT)
os.makedirs("logs", exist_ok=True)

DATASET = "data/multiarith/test.parquet"
MODELS = [
    ("Qwen/Qwen2.5-0.5B", "base_0.5b"),
    # ("modaserMoj/csc415-baseline-gsm8k-0.5b", "baseline_gsm8k_0.5b"),
    # ("modaserMoj/csc415-baseline-math-0.5b", "baseline_math_0.5b"),
    ("modaserMoj/csc415-phase2-gsm8k-0.5b", "phase2_gsm8k_0.5b"),
    # ("modaserMoj/csc415-phase2-math-0.5b", "phase2_math_0.5b"),
]

env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([ROOT, env.get("PYTHONPATH", "")])

for model_path, name in MODELS:
    out = f"results/{name}_multiarith.json"
    log = f"logs/eval_{name}_multiarith.log"
    cmd = [
        sys.executable, "eval/evaluate.py",
        "--model", model_path,
        "--dataset", DATASET,
        "--output_file", out,
        "--max_samples", "1000",
        "--max_new_tokens", "512",
        "--batch_size", "8",
    ]
    print(f"Running: {name} -> {out}")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if rc.returncode != 0:
        print(f"  FAILED (see {log})")
        sys.exit(rc.returncode)
print("Done.")
