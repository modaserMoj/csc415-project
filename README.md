# CSC 415 — Curiosity-Driven Reasoning via RND + GRPO

Two-phase RL training pipeline that teaches an LLM to reason diversely before
fine-tuning for task accuracy. Built on [TinyZero](https://github.com/Jiayi-Pan/TinyZero) (veRL).

## Architecture

```
Phase 1 (Cognitive Pretraining)        Phase 2 (Task Fine-tuning)
─────────────────────────────────      ─────────────────────────────
Reward = RND prediction error          Reward = correctness (0 / 1)
  → encourages novel reasoning           → steers toward right answers
KL penalty keeps outputs coherent      Starts from Phase 1 checkpoint
No ground-truth labels needed          Uses ground-truth labels
```

## Project layout

```
├── phase1/                    # Cognitive Pretraining (RND + GRPO)
│   ├── rnd_module.py          # Target & predictor networks, RND reward
│   ├── rnd_reward_manager.py  # veRL-compatible RewardManager
│   ├── main_phase1.py         # Training entry point
│   ├── config/                # Hydra config (GRPO defaults + RND params)
│   └── scripts/               # Shell script to launch training
├── phase2/                    # Task Fine-tuning (correctness + GRPO)
│   ├── reward_manager.py      # Standard correctness reward
│   ├── main_phase2.py         # Training entry point
│   ├── config/                # Hydra config
│   └── scripts/               # Shell script to launch training
├── data/
│   └── prepare_data.py        # Dataset prep (Countdown, GSM8K, MATH)
├── setup.sh                   # One-time env + TinyZero setup
└── requirements.txt
```

## Quick start

```bash
# 1. Setup (clones TinyZero, installs deps)
bash setup.sh
source .venv/bin/activate

# 2. Prepare data — combined Countdown + GSM8K + MATH for Phase 1
python data/prepare_data.py --dataset phase1_mix --local_dir data/phase1_mix

# 3. Phase 1 — curiosity-driven pretraining on all three datasets
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export DATA_DIR=data/phase1_mix
bash phase1/scripts/train_phase1.sh

# 4. Phase 2 — fine-tune the Phase 1 checkpoint for correctness
export BASE_MODEL=checkpoints/csc415_phase1/phase1_rnd_grpo  # Phase 1 output
export DATA_DIR=data/countdown   # or data/gsm8k for GSM8K evaluation
bash phase2/scripts/train_phase2.sh
```

## How it works

### Phase 1 — RND Exploration

1. Prompts are sampled from the training set (Countdown, GSM8K, MATH, etc.)
2. The LLM generates a **group** of responses per prompt (GRPO, `n=5`)
3. Each response is encoded via a sentence-transformer and passed through two MLPs:
   - **Target network** (frozen random weights) → target embedding
   - **Predictor network** (trainable) → predicted embedding
4. `RND reward = ||predicted − target||²` — high for novel reasoning patterns
5. GRPO uses within-group reward comparisons to update the LLM policy
6. The predictor is updated to recognise patterns seen so far (drives novelty down over time)
7. KL penalty against the base model keeps outputs coherent

### Phase 2 — Correctness Fine-tuning

1. Load the Phase 1 checkpoint (model that reasons diversely)
2. Generate grouped responses, score each: correct=1, wrong=0
3. GRPO updates the policy to make correct reasoning more likely

## Key hyperparameters

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Reward | RND prediction error | Correctness (0/1) |
| `algorithm.adv_estimator` | `grpo` | `grpo` |
| `rollout.n` (group size) | 5 | 5 |
| `kl_loss_coef` | 0.001 | 0.001 |
| `rnd.hidden_dim` | 256 | — |
| `rnd.reward_norm` | batch | — |

## GPU requirements

- **1 GPU**: works for models ≤ 1.5B (Qwen2.5-0.5B, Qwen2.5-1.5B)
- **2+ GPUs**: needed for 3B+ models (set `N_GPUS` and `ROLLOUT_TP_SIZE`)
