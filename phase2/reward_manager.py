"""
Phase 2 scoring functions — standalone correctness checking.

Each function takes the model's completion text and a ground-truth value,
returning 1.0 for correct and 0.0 for incorrect.  These replace the veRL
scoring utilities so we have no TinyZero dependency.
"""

import json
import re


def score_countdown(completion: str, ground_truth: dict) -> float:
    """Check if the completion contains a valid equation reaching the target."""
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    # Look for an equation pattern like "23 + 45 - 12 + 8 = 64"
    equation_match = re.search(r"([\d\s\+\-\*\/\(\)]+)=\s*(\d+)", completion)
    if not equation_match:
        return 0.0

    expr = equation_match.group(1).strip()
    try:
        result = eval(expr, {"__builtins__": {}})
    except Exception:
        return 0.0

    if abs(result - target) > 1e-6:
        return 0.0

    # Verify only allowed numbers are used
    used_numbers = sorted([int(n) for n in re.findall(r"\d+", expr)])
    allowed_numbers = sorted(numbers)
    if used_numbers != allowed_numbers:
        return 0.0

    return 1.0


def score_gsm8k(completion: str, ground_truth: str) -> float:
    """Check if the completion contains the correct final numeric answer.

    GSM8K convention: the answer appears after '####' or as the last number.
    """
    gt = ground_truth.strip().replace(",", "")

    # Look for #### marker first
    marker = completion.rfind("####")
    if marker != -1:
        answer_text = completion[marker + 4:].strip()
        answer_text = answer_text.replace(",", "").split()[0] if answer_text else ""
        if answer_text == gt:
            return 1.0

    # Fallback: match the last number in the completion
    all_nums = re.findall(r"-?\d[\d,]*\.?\d*", completion.replace(",", ""))
    if all_nums and all_nums[-1] == gt:
        return 1.0

    return 0.0


def score_math(completion: str, ground_truth: str) -> float:
    r"""Check if the completion contains the correct answer in \boxed{}.

    Simple string matching after normalising whitespace.  This won't catch
    all equivalent LaTeX forms but works for the majority of MATH problems.
    """
    gt = ground_truth.strip()

    # Extract all \boxed{...} contents
    boxed = re.findall(r"\\boxed\{([^}]+)\}", completion)
    if not boxed:
        # Fallback: check if gt appears literally in the last line
        last_line = completion.strip().split("\n")[-1]
        return 1.0 if gt in last_line else 0.0

    predicted = boxed[-1].strip()
    return 1.0 if predicted == gt else 0.0


SCORE_FNS = {
    "countdown": score_countdown,
    "openai/gsm8k": score_gsm8k,
    "svamp": score_gsm8k,
    "lighteval/MATH": score_math,
    "nlile/hendrycks-MATH-benchmark": score_math,
}


def correctness_reward(prompts, completions, data_source, reward_model, **kwargs):
    """TRL-compatible reward function for Phase 2 correctness scoring."""
    rewards = []
    for prompt, completion, ds, rm_raw in zip(prompts, completions, data_source, reward_model):
        rm = json.loads(rm_raw) if isinstance(rm_raw, str) else rm_raw
        gt = rm["ground_truth"]

        score_fn = None
        for key, fn in SCORE_FNS.items():
            if key in ds:
                score_fn = fn
                break

        if score_fn is None:
            rewards.append(0.0)
            continue

        full_text = prompt + completion
        rewards.append(score_fn(full_text, gt))

    return rewards
