# CSC 415 Project — Two-Phase RL Training (GRPO + RND)

This repo trains a small LLM with a **two-phase RL pipeline**:

1. **Phase 1 (Exploration):** GRPO + RND intrinsic reward to encourage diverse reasoning (not task-specific correctness).
2. **Phase 2 (Exploitation):** GRPO + correctness reward to fine-tune on specific tasks.

## Architecture

- **Base model**: `Qwen/Qwen2.5-0.5B`
- **RL algorithm**: GRPO (HuggingFace TRL)
- **Phase 1 intrinsic reward**: Random Network Distillation (RND)
- **Phase 2 reward**: outcome correctness (0/1) via `phase2/reward_manager.py`

## Datasets

- **Training tasks**:
  - `openai/gsm8k` (GSM8K)
  - `nlile/hendrycks-MATH-benchmark` (MATH; stored as `data_source="lighteval/MATH"` in prepared parquet)
- **Generalization eval tasks**:
  - `ChilleD/SVAMP` (SVAMP)
  - `ChilleD/MultiArith` (MultiArith)
- **Phase 1 mix**:
  - Combined dataset built by `data/prepare_data.py --dataset phase1_mix` (includes Countdown + GSM8K + MATH).

## Quick start (Linux / remote)

```bash
git clone https://github.com/modaserMoj/csc415-project.git
cd csc415-project
bash setup.sh
```

### Phase 1 (exploration)

```bash
PYTHONPATH="$(pwd)" nohup python phase1/main_phase1.py \
  --model Qwen/Qwen2.5-0.5B \
  --train_file data/phase1_mix/train.parquet \
  --eval_file data/phase1_mix/test.parquet \
  --output_dir checkpoints/phase1_0.5b_fast \
  > logs/phase1_0.5b_fast.log 2>&1 &
```

### Phase 2 (fine-tune) and baselines

- **Phase 2 (from Phase 1 checkpoint)**:

```bash
bash scripts/run_phase2.sh
```

- **Baselines (from base model, no Phase 1)**:

```bash
bash scripts/run_baselines.sh
```

## Evaluation

Run all 5 models on the configured datasets:

```bash
MAX_SAMPLES=1000 MAX_NEW_TOKENS=256 bash eval/run_all_evals.sh
```

Summarize everything in `results/*.json` into a single table:

```bash
python eval/summarize_results.py
```

This writes `results/summary.txt`.

## Project structure

```
├── data/
│   └── prepare_data.py          # Dataset preparation (parquet)
├── phase1/
│   ├── main_phase1.py           # Phase 1 training entry point
│   └── rnd_module.py            # RND intrinsic reward module
├── phase2/
│   ├── main_phase2.py           # Phase 2 training entry point
│   ├── reward_manager.py        # Scoring functions (GSM8K / MATH / SVAMP / MultiArith / Countdown)
│   └── scripts/train_phase2.sh  # Generic Phase 2 launcher used by wrappers
├── eval/
│   ├── evaluate.py              # Generate + score to results/*.json
│   ├── run_all_evals.sh         # Run full eval suite
│   └── summarize_results.py     # Build results/summary.txt
├── setup.sh                     # One-time environment setup
└── requirements.txt             # Python dependencies
```
