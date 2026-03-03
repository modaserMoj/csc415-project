"""
Phase 1 Reward Manager — replaces correctness rewards with RND novelty rewards.

Plugs into TinyZero / veRL's training loop as a drop-in replacement for the
standard RewardManager. Instead of checking answers against ground truth,
it rewards responses whose reasoning patterns are *novel* to the predictor
network (high RND prediction error).
"""

import torch
from verl import DataProto

from phase1.rnd_module import RNDModule


class RNDRewardManager:
    """Compute per-token reward tensors using RND novelty scores."""

    def __init__(self, tokenizer, rnd_config: dict | None = None, num_examine: int = 0):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.rnd = RNDModule(**(rnd_config or {}))
        self._step = 0

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        response_texts = []
        valid_lengths = []

        for i in range(len(data)):
            item = data[i]

            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = item.batch["attention_mask"][:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_resp_len = item.batch["attention_mask"][prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            text = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)
            response_texts.append(text)
            valid_lengths.append(valid_resp_len.item())

            if i < self.num_examine:
                full = self.tokenizer.decode(torch.cat((valid_prompt_ids, valid_resp_ids)))
                print(f"[Phase1 Sample {i}] {full[:500]}...")

        rnd_rewards, metrics = self.rnd.compute_rewards(response_texts)

        # Place the scalar reward at the last valid token of each response
        for i in range(len(data)):
            reward_tensor[i, int(valid_lengths[i]) - 1] = rnd_rewards[i]

        self._step += 1
        if self._step % 10 == 0:
            print(f"[RND step {self._step}] {metrics}")

        return reward_tensor

    # Convenience wrappers for checkpoint save/load
    def save_rnd(self, path: str):
        self.rnd.save(path)

    def load_rnd(self, path: str):
        self.rnd.load(path)
