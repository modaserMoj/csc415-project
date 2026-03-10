"""
Phase 2 scoring functions — standalone correctness checking.

Each function takes the model's completion text and a ground-truth value,
returning 1.0 for correct and 0.0 for incorrect.  These replace the veRL
scoring utilities so we have no TinyZero dependency.
"""

import json
import re
from typing import Any


def _to_text(x: Any) -> str:
    """Convert TRL-style prompt/completion objects into plain text.

    TRL can pass:
      - a plain string
      - a dict with a 'content' field
      - a list of chat messages [{"role": ..., "content": ...}, ...]
    """
    if isinstance(x, list):
        if not x:
            return ""
        last = x[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(x, dict):
        return str(x.get("content", ""))
    return str(x)


def _to_scalar(x: Any):
    """Best-effort conversion of pyarrow / numpy wrappers into a Python scalar."""
    if hasattr(x, "as_py"):
        x = x.as_py()
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _to_scalar(x[0])
    return x


def score_countdown(completion: str, ground_truth: dict) -> float:
    """Check if the completion contains a valid equation reaching the target."""
    raw_target = ground_truth["target"]
    target = int(_to_scalar(raw_target))

    raw_numbers = ground_truth["numbers"]
    if hasattr(raw_numbers, "tolist"):
        raw_numbers = raw_numbers.tolist()
    elif not isinstance(raw_numbers, (list, tuple)):
        raw_numbers = [raw_numbers]
    numbers = [_to_scalar(n) for n in raw_numbers]
    allowed_numbers = sorted(int(n) for n in numbers)

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


def _normalize_math_answer(s: str) -> str:
    """Strip whitespace and optional outer $ for comparison."""
    s = s.strip().strip("$").strip()
    return s


def _extract_canonical_answer(ground_truth: str) -> str:
    """If ground_truth is a full solution with \\boxed{}, use the boxed content."""
    gt = ground_truth.strip()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", gt)
    if boxed:
        return _normalize_math_answer(boxed[-1])
    return _normalize_math_answer(gt)


def score_math(completion: str, ground_truth: str) -> float:
    r"""Check if the completion contains the correct answer.

    Accepts either \boxed{...} or <answer>...</answer> (we instruct the latter).
    Ground truth may be a full solution string with \boxed{}; we use the boxed part.
    """
    gt_canonical = _extract_canonical_answer(ground_truth)
    if not gt_canonical:
        return 0.0

    # 1) Prefer <answer>...</answer> (what we ask for in the prompt)
    answer_tag = re.findall(r"<answer>\s*([\s\S]*?)\s*</answer>", completion, re.IGNORECASE)
    if answer_tag:
        predicted = _normalize_math_answer(answer_tag[-1])
        if predicted == gt_canonical:
            return 1.0
        # Relaxed: gt might be LaTeX, predicted might be simplified
        if gt_canonical in predicted or predicted in gt_canonical:
            return 1.0

    # 2) Standard MATH format \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]*)\}", completion)
    if boxed:
        predicted = _normalize_math_answer(boxed[-1])
        if predicted == gt_canonical:
            return 1.0
        if gt_canonical in predicted or predicted in gt_canonical:
            return 1.0

    # 3) Fallback: gt appears in completion (e.g. last line)
    if gt_canonical in completion:
        return 1.0
    last_line = completion.strip().split("\n")[-1] if completion.strip() else ""
    if gt_canonical in last_line:
        return 1.0

    return 0.0


SCORE_FNS = {
    "countdown": score_countdown,
    "openai/gsm8k": score_gsm8k,
    "svamp": score_gsm8k,
    "multiarith": score_gsm8k,
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

        prompt_text = _to_text(prompt)
        completion_text = _to_text(completion)
        full_text = prompt_text + completion_text

        rewards.append(score_fn(full_text, gt))

    return rewards
