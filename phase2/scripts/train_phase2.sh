#!/usr/bin/env bash
# Phase 2: Task Fine-tuning with GRPO + correctness reward
# Tuned for: 1x RTX 4080 (16 GB VRAM), Qwen2.5-0.5B
#
# Required env vars:
#   BASE_MODEL   — path to Phase 1 checkpoint (or HF model for baseline comparison)
#   DATA_DIR     — directory containing train.parquet & test.parquet
#
# Optional env vars:
#   N_GPUS            (default 1)
#   ROLLOUT_TP_SIZE   (default 1)
#   EXPERIMENT_NAME   (default phase2_correctness_grpo)

set -euxo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${N_GPUS:=1}"
: "${ROLLOUT_TP_SIZE:=1}"
: "${EXPERIMENT_NAME:=phase2_correctness_grpo}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase2/main_phase2.py \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger="['console']" \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=csc415_phase2 \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=15 \
    2>&1 | tee logs/phase2_train.log
