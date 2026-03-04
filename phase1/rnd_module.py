"""
Random Network Distillation (RND) module for curiosity-driven exploration.

The core idea: a frozen random target network produces embeddings that a trainable
predictor network tries to match. On novel inputs the predictor hasn't seen before,
prediction error is high — giving us a novelty/curiosity reward signal.

Reference: Burda et al., "Exploration by Random Network Distillation" (2018)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer


class RNDNetwork(nn.Module):
    """MLP used for both the frozen target and trainable predictor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule:
    """Manages the RND target/predictor pair and computes novelty rewards.

    Workflow per batch:
        1. Encode response texts → fixed-dim embeddings (sentence-transformer)
        2. Pass embeddings through frozen target network → target features
        3. Pass embeddings through trainable predictor   → predicted features
        4. RND reward = MSE(predicted, target) per sample
        5. Update predictor so it learns patterns seen so far
           (future encounters with similar reasoning → lower reward)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        rnd_output_dim: int = 128,
        predictor_lr: float = 1e-4,
        device: str = "cuda",
        reward_scale: float = 1.0,
        reward_norm: str = "batch",
    ):
        self.device = device
        self.reward_scale = reward_scale
        self.reward_norm = reward_norm

        # Sentence encoder: converts response text → dense vector
        self.encoder = SentenceTransformer(embedding_model, device=device)

        # Target network — frozen random weights
        self.target = RNDNetwork(embedding_dim, hidden_dim, rnd_output_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        # Predictor network — trained to match the target
        self.predictor = RNDNetwork(embedding_dim, hidden_dim, rnd_output_dim).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=predictor_lr)

        # Running statistics for reward normalisation
        self._running_mean = 0.0
        self._running_var = 1.0
        self._running_n = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_rewards(self, response_texts: list[str]) -> tuple[torch.Tensor, dict]:
        """Full pipeline: encode → reward → update predictor.

        Returns the rewards computed *before* the predictor update so the
        novelty signal reflects the predictor's state prior to this batch.
        """
        embeddings = self._encode(response_texts)

        # Compute rewards (no grad for the reward values themselves)
        with torch.no_grad():
            target_feat = self.target(embeddings)
            pred_feat = self.predictor(embeddings)
            raw_rewards = ((pred_feat - target_feat) ** 2).mean(dim=-1)

        normed_rewards = self._normalise(raw_rewards) * self.reward_scale

        # Now update the predictor on this batch
        pred_loss = self._update_predictor(embeddings)

        metrics = {
            "rnd/raw_reward_mean": raw_rewards.mean().item(),
            "rnd/raw_reward_std": raw_rewards.std().item(),
            "rnd/normed_reward_mean": normed_rewards.mean().item(),
            "rnd/predictor_loss": pred_loss,
        }
        return normed_rewards, metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder.encode(
                texts, convert_to_tensor=True, device=self.device, show_progress_bar=False,
            )

    def _normalise(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_norm == "batch":
            return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        if self.reward_norm == "running":
            b_mean = rewards.mean().item()
            b_var = rewards.var().item()
            b_n = rewards.shape[0]

            delta = b_mean - self._running_mean
            total = self._running_n + b_n
            self._running_mean += delta * b_n / max(total, 1)
            m_a = self._running_var * self._running_n
            m_b = b_var * b_n
            self._running_var = (m_a + m_b + delta ** 2 * self._running_n * b_n / max(total, 1)) / max(total, 1)
            self._running_n = total
            return (rewards - self._running_mean) / (self._running_var ** 0.5 + 1e-8)

        return rewards  # "none"

    def _update_predictor(self, embeddings: torch.Tensor) -> float:
        self.predictor.train()
        embeddings = embeddings.detach().clone()
        with torch.no_grad():
            target_feat = self.target(embeddings)
        pred_feat = self.predictor(embeddings)
        loss = ((pred_feat - target_feat) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "target": self.target.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "running_mean": self._running_mean,
                "running_var": self._running_var,
                "running_n": self._running_n,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.target.load_state_dict(ckpt["target"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._running_mean = ckpt["running_mean"]
        self._running_var = ckpt["running_var"]
        self._running_n = ckpt["running_n"]
