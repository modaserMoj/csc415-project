"""
Phase 2 Reward Manager — standard correctness-based rewards.

Takes the Phase 1 checkpoint (a model trained for diverse reasoning via RND)
and fine-tunes it with outcome-based rewards:
    correct answer → reward = 1
    wrong answer   → reward = 0

This is essentially the same RewardManager that TinyZero ships with, but
centralised here so the two phases share a consistent project layout.
"""

import torch
from verl import DataProto
from verl.utils.reward_score import gsm8k, math, countdown


def _get_score_fn(data_source: str):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score
    if data_source == "lighteval/MATH":
        return math.compute_score
    if "countdown" in data_source:
        return countdown.compute_score
    raise NotImplementedError(f"No scoring function for data_source={data_source}")


class CorrectnessRewardManager:
    """Reward = 1 if the model's final answer matches ground truth, else 0."""

    def __init__(self, tokenizer, num_examine: int = 0):
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        printed: dict[str, int] = {}

        for i in range(len(data)):
            item = data[i]

            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = item.batch["attention_mask"][:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_resp_len = item.batch["attention_mask"][prompt_len:].sum()
            valid_resp_ids = response_ids[:valid_resp_len]

            full_text = self.tokenizer.decode(torch.cat((valid_prompt_ids, valid_resp_ids)))
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch["data_source"]

            score = _get_score_fn(data_source)(solution_str=full_text, ground_truth=ground_truth)
            reward_tensor[i, int(valid_resp_len) - 1] = score

            if data_source not in printed:
                printed[data_source] = 0
            if printed[data_source] < self.num_examine:
                printed[data_source] += 1
                print(full_text)

        return reward_tensor
