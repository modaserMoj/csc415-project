#!/usr/bin/env python3
"""
Evaluate a model on a math dataset.

Loads a model, generates completions for prompts in a parquet dataset,
scores them using the appropriate reward function, and saves accuracy to JSON.
"""

import argparse
import json
import re
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _to_text(x: Any) -> str:
    """Convert TRL-style prompt/completion objects into plain text."""
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


def score_gsm8k(completion: str, ground_truth: str) -> float:
    """Check if the completion contains the correct final numeric answer."""
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
    """Check if the completion contains the correct final answer for MATH dataset."""
    gt = ground_truth.strip().replace(",", "")

    # Look for \boxed{answer}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", completion)
    if boxed_match:
        answer_text = boxed_match.group(1).strip().replace(",", "")
        if answer_text == gt:
            return 1.0

    # Fallback: last number
    all_nums = re.findall(r"-?\d[\d,]*\.?\d*", completion.replace(",", ""))
    if all_nums and all_nums[-1] == gt:
        return 1.0

    return 0.0


def score_svamp(completion: str, ground_truth: str) -> float:
    """Check if the completion contains the correct answer for SVAMP."""
    gt = str(ground_truth).strip().replace(",", "")

    # Look for final answer in <answer> tags or last number
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip().replace(",", "")
        if answer_text == gt:
            return 1.0

    all_nums = re.findall(r"-?\d[\d,]*\.?\d*", completion.replace(",", ""))
    if all_nums and all_nums[-1] == gt:
        return 1.0

    return 0.0


def score_multiarith(completion: str, ground_truth: str) -> float:
    """Check if the completion contains the correct answer for MultiArith."""
    gt = str(ground_truth).strip().replace(",", "")

    # Similar to SVAMP
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip().replace(",", "")
        if answer_text == gt:
            return 1.0

    all_nums = re.findall(r"-?\d[\d,]*\.?\d*", completion.replace(",", ""))
    if all_nums and all_nums[-1] == gt:
        return 1.0

    return 0.0


SCORE_FNS = {
    "openai/gsm8k": score_gsm8k,
    "hendrycks/math": score_math,
    "ChilleD/SVAMP": score_svamp,
    "ChilleD/MultiArith": score_multiarith,
    # Local shorthand names used in this repo's parquet files.
    "svamp": score_svamp,
    "multiarith": score_multiarith,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on math dataset")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Path to parquet dataset")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_parquet(args.dataset)
    if len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )

    correct = 0
    total = len(df)
    total_batches = (total + args.batch_size - 1) // args.batch_size if total > 0 else 0
    progress_interval = 25
    print(f"Starting evaluation on {total} samples...")

    for i in range(0, total, args.batch_size):
        batch_idx = i // args.batch_size + 1
        batch = df.iloc[i:i+args.batch_size]
        prompts = batch["prompt"].tolist()
        
        ground_truths = []
        for x in batch["reward_model"]:
            if isinstance(x, str):
                gt = json.loads(x)["ground_truth"]
            elif isinstance(x, dict):
                gt = x["ground_truth"]
            else:
                gt = x
            ground_truths.append(gt)
        data_sources = batch["data_source"].tolist()

        # Prepare inputs
        inputs = []
        for prompt in prompts:
            # Parse JSON string back to list of messages
            if isinstance(prompt, str):
                messages = json.loads(prompt)
            else:
                messages = prompt

            # Some parquet rows store prompt as numpy.ndarray of dicts.
            if hasattr(messages, "tolist"):
                messages = messages.tolist()

            if isinstance(messages, list):
                text_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "user")).strip().lower()
                        content = str(msg.get("content", "")).strip()
                        if not content:
                            continue
                        if role == "assistant":
                            text_parts.append(f"Assistant: {content}")
                        else:
                            text_parts.append(f"Human: {content}")
                    else:
                        text_parts.append(str(msg))

                # Ensure model sees a turn to complete.
                text = "\n".join(text_parts).strip()
                if "Assistant:" not in text_parts[-1] if text_parts else True:
                    text = f"{text}\nAssistant:"
                inputs.append(text)
            else:
                inputs.append(str(messages))

        tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # Greedy for evaluation
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        completions = []
        for output in outputs:
            decoded = tokenizer.decode(output[tokenized["input_ids"].shape[1]:], skip_special_tokens=True)
            completions.append(decoded)

        # Score
        batch_correct = 0
        for completion, gt, ds in zip(completions, ground_truths, data_sources):
            score_fn = SCORE_FNS.get(ds, lambda c, g: 0.0)
            if score_fn(completion, gt) == 1.0:
                batch_correct += 1
                correct += 1
        if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches:
            processed = i + len(batch)
            running_acc = correct / processed if processed > 0 else 0.0
            print(
                f"Progress: batch {batch_idx}/{total_batches} "
                f"| processed {processed}/{total} "
                f"| running_acc={running_acc:.4f}"
            )

    accuracy = correct / total if total > 0 else 0.0

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()