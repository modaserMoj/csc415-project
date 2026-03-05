"""
Unified data preparation for all datasets used in Phase 1 and Phase 2.

Converts HuggingFace datasets into the parquet format expected by TinyZero/veRL:
    - prompt          : the formatted prompt string
    - data_source     : dataset identifier (used to dispatch the scoring function)
    - reward_model    : dict with ground_truth info

Supported datasets:
    - countdown   (Phase 1 + 2)
    - gsm8k       (Phase 1 + 2)
    - math        (Phase 1 + 2)
    - svamp       (held-out generalization eval)
    - phase1_mix  (all three combined + shuffled for Phase 1)

Usage:
    python data/prepare_data.py --dataset countdown  --local_dir ./data/countdown
    python data/prepare_data.py --dataset gsm8k      --local_dir ./data/gsm8k
    python data/prepare_data.py --dataset math       --local_dir ./data/math
    python data/prepare_data.py --dataset svamp      --local_dir ./data/svamp
    python data/prepare_data.py --dataset phase1_mix --local_dir ./data/phase1_mix
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def prepare_countdown(local_dir: str, num_operands: int = 4, size: int = 490_000, test_size: int = 10_000):
    """Generate Countdown game data: find an equation using given numbers to reach a target."""

    os.makedirs(local_dir, exist_ok=True)
    ops = ["+", "-", "*"]

    def make_sample():
        numbers = [random.randint(1, 100) for _ in range(num_operands)]
        chosen_ops = [random.choice(ops) for _ in range(num_operands - 1)]
        expr = str(numbers[0])
        for op, n in zip(chosen_ops, numbers[1:]):
            expr += f" {op} {n}"
        target = eval(expr)
        return numbers, target

    def build_row(numbers, target):
        prompt = (
            f"Using the numbers {numbers}, create an equation that equals {target}. "
            f"You can use basic arithmetic operations (+, -, *, /). "
            f"Each number must be used exactly once. "
            f"Show your work in <think> </think> tags. "
            f"And return the final equation in <answer> </answer> tags, "
            f"for example <answer> 1 + 2 </answer>."
        )
        return {
            "data_source": "countdown",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": {"target": target, "numbers": numbers}},
            "extra_info": {"numbers": numbers, "target": target},
        }

    train_rows, test_rows = [], []
    for _ in range(size):
        numbers, target = make_sample()
        train_rows.append(build_row(numbers, target))
    for _ in range(test_size):
        numbers, target = make_sample()
        test_rows.append(build_row(numbers, target))

    pd.DataFrame(train_rows).to_parquet(os.path.join(local_dir, "train.parquet"))
    pd.DataFrame(test_rows).to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"Countdown: {len(train_rows)} train, {len(test_rows)} test → {local_dir}")


def prepare_gsm8k(local_dir: str):
    """Download and format GSM8K from HuggingFace."""
    from datasets import load_dataset

    os.makedirs(local_dir, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")

    def format_split(split):
        rows = []
        for ex in split:
            prompt = (
                f"Solve this math problem step by step.\n\n{ex['question']}\n\n"
                f"Show your work in <think> </think> tags. "
                f"Give your final numerical answer in <answer> </answer> tags."
            )
            answer = ex["answer"].split("####")[-1].strip()
            rows.append({
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"question": ex["question"], "answer": answer},
            })
        return rows

    train_rows = format_split(ds["train"])
    test_rows = format_split(ds["test"])

    pd.DataFrame(train_rows).to_parquet(os.path.join(local_dir, "train.parquet"))
    pd.DataFrame(test_rows).to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"GSM8K: {len(train_rows)} train, {len(test_rows)} test → {local_dir}")


def prepare_math(local_dir: str):
    """Download and format the MATH dataset from HuggingFace."""
    from datasets import load_dataset

    os.makedirs(local_dir, exist_ok=True)
    ds = load_dataset("nlile/hendrycks-MATH-benchmark")

    def format_split(split):
        rows = []
        for ex in split:
            prompt = (
                f"Solve this problem step by step.\n\n{ex['problem']}\n\n"
                f"Show your work in <think> </think> tags. "
                f"Give your final answer in <answer> </answer> tags."
            )
            rows.append({
                "data_source": "lighteval/MATH",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ex["solution"]},
                "extra_info": {"problem": ex["problem"], "level": ex.get("level", ""), "type": ex.get("type", "")},
            })
        return rows

    train_rows = format_split(ds["train"])
    test_rows = format_split(ds["test"])

    pd.DataFrame(train_rows).to_parquet(os.path.join(local_dir, "train.parquet"))
    pd.DataFrame(test_rows).to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"MATH: {len(train_rows)} train, {len(test_rows)} test → {local_dir}")


def prepare_svamp(local_dir: str):
    """Download and format SVAMP from HuggingFace (held-out generalization eval)."""
    from datasets import load_dataset

    os.makedirs(local_dir, exist_ok=True)
    ds = load_dataset("ChilleD/SVAMP")

    def format_split(split):
        rows = []
        for ex in split:
            body = ex.get("Body", ex.get("body", ""))
            question = ex.get("Question", ex.get("question", ""))
            answer = str(ex.get("Answer", ex.get("answer", "")))
            prompt = (
                f"Solve this math problem step by step.\n\n{body} {question}\n\n"
                f"Show your work in <think> </think> tags. "
                f"Give your final numerical answer in <answer> </answer> tags."
            )
            rows.append({
                "data_source": "svamp",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"body": body, "question": question, "answer": answer},
            })
        return rows

    if "test" in ds:
        test_rows = format_split(ds["test"])
        train_rows = format_split(ds["train"]) if "train" in ds else test_rows
    else:
        all_rows = format_split(ds["train"])
        split_idx = int(len(all_rows) * 0.8)
        random.shuffle(all_rows)
        train_rows, test_rows = all_rows[:split_idx], all_rows[split_idx:]

    pd.DataFrame(train_rows).to_parquet(os.path.join(local_dir, "train.parquet"))
    pd.DataFrame(test_rows).to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"SVAMP: {len(train_rows)} train, {len(test_rows)} test → {local_dir}")


def _collect_rows(prepare_fn, **kwargs):
    """Run a prepare function but capture the rows instead of writing to disk."""
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        prepare_fn(tmp, **kwargs)
        train_df = pd.read_parquet(os.path.join(tmp, "train.parquet"))
        test_df = pd.read_parquet(os.path.join(tmp, "test.parquet"))
        return train_df, test_df
    finally:
        shutil.rmtree(tmp)


def prepare_phase1_mix(local_dir: str, countdown_size: int = 10_000, num_operands: int = 4):
    """Build a combined, shuffled dataset from Countdown + GSM8K + MATH.

    Since GSM8K (~7.5k train) and MATH (~7.5k train) are much smaller than the
    full 490k Countdown set, we downsample Countdown to keep the mix roughly
    balanced.  The default countdown_size=10_000 gives an approximate 1:1:1 ratio.
    Adjust via --countdown_mix_size if needed.

    All rows keep their original data_source field so the reward manager can
    dispatch to the right scoring function (though Phase 1 ignores it — RND
    doesn't need ground truth).
    """
    os.makedirs(local_dir, exist_ok=True)

    print("Preparing Countdown …")
    cd_train, cd_test = _collect_rows(
        prepare_countdown, num_operands=num_operands, size=countdown_size, test_size=2_000,
    )
    print("Preparing GSM8K …")
    gsm_train, gsm_test = _collect_rows(prepare_gsm8k)
    print("Preparing MATH …")
    math_train, math_test = _collect_rows(prepare_math)

    # JSON-serialize reward_model so all datasets have the same column type.
    # (Countdown ground_truth is a dict, GSM8K/MATH is a string — can't mix in parquet.)
    # Phase 1 (RND) doesn't read this column; Phase 2 uses single-dataset parquets.
    for df in [cd_train, cd_test, gsm_train, gsm_test, math_train, math_test]:
        df["reward_model"] = df["reward_model"].apply(lambda x: json.dumps(x, cls=_NumpyEncoder))

    train_df = pd.concat([cd_train, gsm_train, math_train], ignore_index=True).sample(frac=1, random_state=42)
    test_df = pd.concat([cd_test, gsm_test, math_test], ignore_index=True).sample(frac=1, random_state=42)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    for src in train_df["data_source"].value_counts().items():
        print(f"  train  {src[0]:25s}  {src[1]:>6d} rows")
    for src in test_df["data_source"].value_counts().items():
        print(f"  test   {src[0]:25s}  {src[1]:>6d} rows")
    print(f"Phase 1 mix: {len(train_df)} train, {len(test_df)} test → {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--dataset", required=True,
                        choices=["countdown", "countdown3", "gsm8k", "math", "svamp", "phase1_mix"])
    parser.add_argument("--local_dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--num_operands", type=int, default=4, help="Countdown: number of operands")
    parser.add_argument("--size", type=int, default=490_000, help="Countdown: training set size")
    parser.add_argument("--countdown_mix_size", type=int, default=10_000,
                        help="Countdown rows to include in phase1_mix (balances against GSM8K/MATH)")
    args = parser.parse_args()

    if args.dataset == "countdown":
        prepare_countdown(args.local_dir, num_operands=args.num_operands, size=args.size)
    elif args.dataset == "countdown3":
        prepare_countdown(args.local_dir, num_operands=3, size=args.size, test_size=10_000)
    elif args.dataset == "gsm8k":
        prepare_gsm8k(args.local_dir)
    elif args.dataset == "math":
        prepare_math(args.local_dir)
    elif args.dataset == "svamp":
        prepare_svamp(args.local_dir)
    elif args.dataset == "phase1_mix":
        prepare_phase1_mix(args.local_dir, countdown_size=args.countdown_mix_size, num_operands=args.num_operands)
