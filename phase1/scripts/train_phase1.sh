#!/usr/bin/env bash
# Phase 1: Cognitive Pretraining with GRPO + RND
# Tuned for: 1x RTX 4080 (16 GB VRAM), Qwen2.5-1.5B
#
# Required env vars:
#   BASE_MODEL   — HF model id or local path  (e.g. Qwen/Qwen2.5-1.5B)
#   DATA_DIR     — directory containing train.parquet & test.parquet
#
# Optional env vars:
#   N_GPUS            (default 1)
#   ROLLOUT_TP_SIZE   (default 1)
#   EXPERIMENT_NAME   (default phase1_rnd_grpo)

set -euxo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS

: "${N_GPUS:=1}"
: "${ROLLOUT_TP_SIZE:=1}"
: "${EXPERIMENT_NAME:=phase1_rnd_grpo}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}" \
python phase1/main_phase1.py \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.optim.lr=1e-5 \
    critic.model.path="$BASE_MODEL" \
    critic.ppo_micro_batch_size=4 \
    critic.model.enable_gradient_checkpointing=True \
    critic.fsdp_config.param_offload=True \
    critic.fsdp_config.grad_offload=True \
    critic.fsdp_config.optimizer_offload=True \
    trainer.logger="['console']" \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=csc415_phase1 \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=15 \
    rnd.embedding_model=all-MiniLM-L6-v2 \
    rnd.embedding_dim=384 \
    rnd.hidden_dim=256 \
    rnd.rnd_output_dim=128 \
    rnd.predictor_lr=1e-4 \
    rnd.reward_scale=1.0 \
    rnd.device=cpu \
    rnd.reward_norm=batch \
    2>&1 | tee logs/phase1_train.log
