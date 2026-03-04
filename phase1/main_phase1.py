"""
Phase 1 entry point: Cognitive Pretraining with GRPO + RND.

Trains a base LLM using TRL's GRPOTrainer where the *only* reward signal
is RND prediction error (curiosity / novelty).  A KL penalty against
the base model keeps generations coherent.

Usage:
    python phase1/main_phase1.py \
        --model Qwen/Qwen2.5-0.5B \
        --train_file data/phase1_mix/train.parquet \
        --eval_file data/phase1_mix/test.parquet
"""

import argparse

import pandas as pd
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

from phase1.rnd_module import RNDModule


def make_rnd_reward_fn(rnd_config: dict):
    """Create a TRL-compatible reward function backed by an RND module."""
    rnd = RNDModule(**rnd_config)
    _step = {"n": 0}

    def rnd_reward(completions, **kwargs):
        texts = []
        for c in completions:
            if isinstance(c, list):
                texts.append(c[-1]["content"] if c else "")
            elif isinstance(c, dict):
                texts.append(c.get("content", str(c)))
            else:
                texts.append(str(c))
        rewards, metrics = rnd.compute_rewards(texts)
        _step["n"] += 1
        if _step["n"] % 10 == 0:
            print(f"[RND step {_step['n']}] {metrics}")
        return rewards.tolist()

    rnd_reward._rnd_module = rnd
    return rnd_reward


def main():
    parser = argparse.ArgumentParser(description="Phase 1: GRPO + RND pretraining")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", default="checkpoints/phase1")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.001,
                        help="KL penalty coefficient against reference model")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)

    parser.add_argument("--rnd_device", default="cpu")
    parser.add_argument("--rnd_reward_scale", type=float, default=1.0)
    parser.add_argument("--rnd_reward_norm", default="batch")
    parser.add_argument("--rnd_embedding_model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_file)
    eval_df = pd.read_parquet(args.eval_file)
    train_dataset = Dataset.from_pandas(train_df[["prompt"]])
    eval_dataset = Dataset.from_pandas(eval_df[["prompt"]])

    rnd_config = {
        "embedding_model": args.rnd_embedding_model,
        "device": args.rnd_device,
        "reward_scale": args.rnd_reward_scale,
        "reward_norm": args.rnd_reward_norm,
    }
    reward_fn = make_rnd_reward_fn(rnd_config)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.lr,
        gradient_checkpointing=True,
        optim="adafactor",
        bf16=True,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="none",
        save_total_limit=3,
        temperature=1.0,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model()

    rnd_module = reward_fn._rnd_module
    rnd_module.save(f"{args.output_dir}/rnd_checkpoint.pt")
    print(f"RND checkpoint saved to {args.output_dir}/rnd_checkpoint.pt")


if __name__ == "__main__":
    main()
