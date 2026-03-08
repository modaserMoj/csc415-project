"""
Evaluate a model checkpoint on a dataset.

Loads the model, generates completions for each test prompt, then scores
them using the appropriate scoring function. Reports accuracy and saves
per-example results.

Usage:
    python eval/evaluate.py \
        --model Qwen/Qwen2.5-1.5B \
        --dataset data/gsm8k/test.parquet \
        --output_file results/base_gsm8k.json \
        --max_samples 500
"""

import argparse
import ast
import json
import os
import sys
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase2.reward_manager import SCORE_FNS


def extract_answer_tag(text: str) -> str:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def score_example(completion: str, data_source: str, reward_model: dict) -> float:
    gt = reward_model["ground_truth"]
    for key, fn in SCORE_FNS.items():
        if key in data_source:
            return fn(completion, gt)
    return 0.0


def _to_python(obj):
    """Force pyarrow / numpy objects into plain Python types."""
    if hasattr(obj, "as_py"):
        return obj.as_py()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def render_prompt(prompt_obj, tokenizer) -> str:
    """Render dataset prompt into a model-ready string.

    Prompts may arrive as:
      - list/dict chat messages (native or pyarrow)
      - plain strings
      - stringified Python lists/dicts from parquet round-trips
    """
    prompt_obj = _to_python(prompt_obj)

    if isinstance(prompt_obj, list):
        msgs = [_to_python(m) for m in prompt_obj]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt_obj, dict):
        return tokenizer.apply_chat_template([prompt_obj], tokenize=False, add_generation_prompt=True)
    if isinstance(prompt_obj, str):
        stripped = prompt_obj.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = ast.literal_eval(stripped)
                return render_prompt(parsed, tokenizer)
            except Exception:
                pass
        return prompt_obj
    return str(prompt_obj)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--dataset", required=True, help="Path to test.parquet file")
    parser.add_argument("--output_file", default=None, help="JSON file to save per-example results")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval to N samples")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_parquet(args.dataset)
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    prompts = []
    for _, row in df.iterrows():
        prompts.append(str(render_prompt(row["prompt"], tokenizer)))

    data_sources = df["data_source"].tolist()
    reward_models = []
    for rm in df["reward_model"].tolist():
        rm = _to_python(rm) if not isinstance(rm, (str, dict)) else rm
        if isinstance(rm, str):
            reward_models.append(json.loads(rm))
        else:
            reward_models.append(rm)

    results = []
    correct = 0
    total = len(prompts)

    print(f"Evaluating {total} examples (batch_size={args.batch_size})...")
    for i in range(0, total, args.batch_size):
        # Coerce any residual odd types (e.g. lists/dicts) to strings
        batch_prompts = [str(p) for p in prompts[i : i + args.batch_size]]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
            outputs = model.generate(**inputs, **gen_kwargs)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        for j, output in enumerate(outputs):
            idx = i + j

            # With left padding, each row's prompt length is the sum of its attention mask.
            prompt_len = int(attn_mask[j].sum().item())
            new_tokens = output[prompt_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Score only the generated completion to avoid accidental prompt leakage.
            score = score_example(completion, data_sources[idx], reward_models[idx])
            correct += score

            results.append({
                "idx": idx,
                "data_source": data_sources[idx],
                "score": score,
                "completion": completion[:500],
            })

        done = min(i + args.batch_size, total)
        acc_so_far = correct / done if done > 0 else 0
        print(f"  [{done}/{total}]  accuracy so far: {acc_so_far:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Samples:  {total}")
    print(f"Correct:  {int(correct)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "total": total,
            "correct": int(correct),
            "accuracy": accuracy,
            "examples": results,
        }
        with open(args.output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
