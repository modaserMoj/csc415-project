"""
Monitor training progress. Run anytime during or after training.

Usage:
    python scripts/check_training.py logs/phase1_train.log
    python scripts/check_training.py logs/baseline_gsm8k.log
    python scripts/check_training.py --eval checkpoints/phase1  # quick eval on 50 samples
"""

import argparse
import json
import os
import re
import sys


def parse_log(log_path):
    """Extract training metrics from a log file."""
    steps = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                line = line.replace("'", '"')
                data = json.loads(line)
                steps.append(data)
            except (json.JSONDecodeError, ValueError):
                continue
    return steps


def show_log_summary(log_path):
    steps = parse_log(log_path)
    if not steps:
        print(f"No training metrics found in {log_path}")
        return

    print(f"\nLog: {log_path}")
    print(f"Steps logged: {len(steps)}")
    print(f"Last step: {steps[-1].get('epoch', '?'):.2f} epochs")

    is_phase1 = any("rnd_reward" in str(s) for s in steps)

    if is_phase1:
        print("\n--- Phase 1 (RND) Health Check ---")
        first5 = steps[:5]
        last5 = steps[-5:]

        early_rnd = [float(s.get("rewards/rnd_reward/mean", s.get("reward", 0))) for s in first5]
        late_rnd = [float(s.get("rewards/rnd_reward/mean", s.get("reward", 0))) for s in last5]
        early_kl = [float(s.get("kl", 0)) for s in first5]
        late_kl = [float(s.get("kl", 0)) for s in last5]
        early_entropy = [float(s.get("entropy", 0)) for s in first5]
        late_entropy = [float(s.get("entropy", 0)) for s in last5]

        avg = lambda x: sum(x) / len(x) if x else 0

        print(f"  RND reward mean:  early={avg(early_rnd):.4f}  latest={avg(late_rnd):.4f}")
        print(f"  KL divergence:    early={avg(early_kl):.4f}  latest={avg(late_kl):.4f}")
        print(f"  Entropy:          early={avg(early_entropy):.4f}  latest={avg(late_entropy):.4f}")

        last = steps[-1]
        kl = float(last.get("kl", 0))
        entropy = float(last.get("entropy", 0))
        reward_std = float(last.get("reward_std", 0))

        print("\n  Diagnostics:")
        if kl > 5:
            print("  WARNING: KL is very high — model may be diverging too far from base")
        elif kl < 0.0001 and len(steps) > 10:
            print("  WARNING: KL near zero — model may not be learning")
        else:
            print("  OK: KL is in healthy range")

        if entropy < 0.1 and len(steps) > 10:
            print("  WARNING: Entropy very low — possible mode collapse")
        else:
            print("  OK: Entropy looks healthy")

        if reward_std < 0.01 and len(steps) > 10:
            print("  WARNING: reward_std near zero — completions may be identical")
        else:
            print("  OK: Completion diversity looks healthy")
    else:
        print("\n--- Phase 2 / Baseline Health Check ---")
        first5 = steps[:5]
        last5 = steps[-5:]

        early_reward = [float(s.get("reward", 0)) for s in first5]
        late_reward = [float(s.get("reward", 0)) for s in last5]
        early_kl = [float(s.get("kl", 0)) for s in first5]
        late_kl = [float(s.get("kl", 0)) for s in last5]

        avg = lambda x: sum(x) / len(x) if x else 0

        print(f"  Reward mean:      early={avg(early_reward):.4f}  latest={avg(late_reward):.4f}")
        print(f"  KL divergence:    early={avg(early_kl):.4f}  latest={avg(late_kl):.4f}")

        if avg(late_reward) > avg(early_reward):
            print("  OK: Reward is increasing — model is learning")
        elif len(steps) > 20:
            print("  WARNING: Reward not increasing after 20+ steps")

    print(f"\n  Last 3 steps:")
    for s in steps[-3:]:
        step_time = s.get("step_time", "?")
        epoch = s.get("epoch", "?")
        reward = s.get("reward", "?")
        kl = s.get("kl", "?")
        print(f"    epoch={epoch}  reward={reward}  kl={kl}  step_time={step_time}s")


def quick_eval(checkpoint_path, dataset_path=None, n=50):
    """Run a fast eval on a checkpoint to gauge accuracy."""
    if dataset_path is None:
        for candidate in ["data/gsm8k/test.parquet", "data/countdown/test.parquet", "data/math/test.parquet"]:
            if os.path.exists(candidate):
                dataset_path = candidate
                break
    if dataset_path is None:
        print("No test dataset found. Run setup.sh first.")
        return

    print(f"\nQuick eval: {checkpoint_path} on {dataset_path} ({n} samples)")
    import subprocess
    cmd = [
        sys.executable, "eval/evaluate.py",
        "--model", checkpoint_path,
        "--dataset", dataset_path,
        "--max_samples", str(n),
        "--max_new_tokens", "256",
        "--batch_size", "4",
    ]
    subprocess.run(cmd, env={**os.environ, "PYTHONPATH": os.getcwd()})


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("log_or_checkpoint", help="Path to log file or checkpoint dir")
    parser.add_argument("--eval", action="store_true", help="Run quick eval on a checkpoint")
    parser.add_argument("--dataset", default=None, help="Dataset for eval")
    parser.add_argument("--n", type=int, default=50, help="Number of eval samples")
    args = parser.parse_args()

    if args.eval or os.path.isdir(args.log_or_checkpoint):
        quick_eval(args.log_or_checkpoint, args.dataset, args.n)
    else:
        show_log_summary(args.log_or_checkpoint)


if __name__ == "__main__":
    main()
