# CSC 415 Project — Two-Phase LLM Training with Intrinsic Motivation

Train an LLM using a two-phase pipeline:
1. **Phase 1 (Exploration):** GRPO + RND curiosity reward — the model learns diverse reasoning patterns without task-specific correctness.
2. **Phase 2 (Exploitation):** GRPO + correctness reward — fine-tune the Phase 1 checkpoint on specific tasks (GSM8K, MATH, Countdown).

## Architecture

- **Model:** Qwen2.5-1.5B
- **RL Algorithm:** GRPO (Group Relative Policy Optimization) via HuggingFace TRL
- **Intrinsic Reward:** Random Network Distillation (RND) for Phase 1 novelty signal
- **Datasets:** GSM8K, MATH, Countdown (mixed for Phase 1, per-task for Phase 2)

## Quick Start (Vast.ai)

1. Rent an A100 80GB instance with a PyTorch Docker template
2. SSH in and run:

```bash
git clone https://github.com/modaserMoj/csc415-project.git
cd csc415-project
bash setup.sh
bash smoke_test.sh
nohup bash phase1/scripts/train_phase1.sh > logs/phase1_train.log 2>&1 &
```

3. Monitor training:
```bash
tail -f logs/phase1_train.log
```

4. After Phase 1 completes, run Phase 2:
```bash
nohup bash phase2/scripts/train_phase2.sh > logs/phase2_train.log 2>&1 &
```

## Training Settings (A100 80GB)

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Model | Qwen2.5-1.5B | Phase 1 checkpoint |
| Batch size | 20 | 20 |
| Grad accumulation | 4 | 4 |
| Num generations | 5 | 5 |
| Max prompt length | 512 | 512 |
| Max completion length | 512 | 512 |
| KL penalty (beta) | 0.04 | 0.04 |
| Learning rate | 5e-7 | 5e-7 |
| Epochs | 3 | 3 |
| Reward | RND novelty | Correctness (0/1) |

### Estimated Training Time
- **Phase 1:** ~6-10 hours on A100 80GB
- **Phase 2:** ~2-4 hours on A100 80GB (smaller per-task datasets)

## Baselines (Qwen2.5-1.5B)

| Benchmark | Base Model | Expected After Phase 2 |
|---|---|---|
| GSM8K | ~40-50% | 55-65% |
| MATH | ~20-30% | 30-40% |
| Countdown | ~0-5% | 30-50% |

## Project Structure

```
├── data/
│   ├── prepare_data.py          # Dataset preparation
│   ├── countdown/               # Countdown dataset (generated)
│   ├── gsm8k/                   # GSM8K dataset
│   ├── math/                    # MATH dataset
│   └── phase1_mix/              # Mixed dataset for Phase 1
├── phase1/
│   ├── main_phase1.py           # Phase 1 training entry point
│   ├── rnd_module.py            # RND novelty reward module
│   └── scripts/train_phase1.sh  # Phase 1 launch script
├── phase2/
│   ├── main_phase2.py           # Phase 2 training entry point
│   ├── reward_manager.py        # Correctness scoring functions
│   └── scripts/train_phase2.sh  # Phase 2 launch script
├── setup.sh                     # One-time environment setup
├── smoke_test.sh                # Quick verification
└── requirements.txt             # Python dependencies
```
