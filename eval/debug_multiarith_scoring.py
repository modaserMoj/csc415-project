"""
Quick sanity check for MultiArith prompts and scoring.

Runs on a few examples from data/multiarith/test.parquet and shows:
- question text
- ground-truth answer
- scores for some synthetic completions

Run:
    python eval/debug_multiarith_scoring.py
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase2.reward_manager import score_gsm8k


DATA_PATH = os.path.join(ROOT, "data", "multiarith", "test.parquet")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Missing {DATA_PATH}. Run:")
        print("  python data/prepare_data.py --dataset multiarith --local_dir data/multiarith")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded MultiArith test set with {len(df)} rows.")
    print("Columns:", list(df.columns))

    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        rm = row["reward_model"]
        if isinstance(rm, str):
            rm = json.loads(rm)
        gt = str(rm["ground_truth"]).strip()

        # Prompt content (user text)
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, list) and prompt_msgs and isinstance(prompt_msgs[0], dict):
            question = prompt_msgs[0].get("content", "")
        else:
            question = str(prompt_msgs)

        print("\n=== Example", idx, "===")
        print("data_source:", row.get("data_source"))
        print("prompt (truncated):", question[:200].replace("\n", " "))
        print("ground_truth:", gt)

        # Synthetic completions to test the scorer
        c1 = f"<think> ... </think> <answer> {gt} </answer>"
        c2 = f"The answer is {gt}."
        wrong = "<answer> 999999 </answer>"

        print("score_gsm8k(correct in <answer>):", score_gsm8k(c1, gt))
        print("score_gsm8k(correct as last number):", score_gsm8k(c2, gt))
        print("score_gsm8k(wrong):", score_gsm8k(wrong, gt))


if __name__ == "__main__":
    main()

