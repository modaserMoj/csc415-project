"""
Phase 2 entry point: Task Fine-tuning with GRPO + correctness reward.

Loads the Phase 1 checkpoint (diverse-reasoning model) and fine-tunes it
on specific tasks where reward = 1 for correct answers, 0 otherwise.

Usage:
    python phase2/main_phase2.py \
        --model checkpoints/phase1 \
        --train_file data/countdown/train.parquet \
        --eval_file data/countdown/test.parquet
"""

import argparse

import pandas as pd
import torch
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

from phase2.reward_manager import correctness_reward


def main():
    parser = argparse.ArgumentParser(description="Phase 2: GRPO + correctness fine-tuning")
    parser.add_argument("--model", required=True,
                        help="Phase 1 checkpoint path or HF model id (for baseline)")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_dir", default="checkpoints/phase2")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Stop after N optimizer steps (-1 = full epochs)")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_file)
    eval_df = pd.read_parquet(args.eval_file)

    keep_cols = [c for c in ["prompt", "data_source", "reward_model"] if c in train_df.columns]
    train_dataset = Dataset.from_pandas(train_df[keep_cols])
    eval_dataset = Dataset.from_pandas(eval_df[keep_cols])

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.lr,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="none",
        save_total_limit=3,
        temperature=1.0,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=correctness_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
