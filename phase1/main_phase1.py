"""
Phase 1 entry point: Cognitive Pretraining with GRPO + RND.

Trains the base LLM using group-relative policy optimisation where the *only*
reward signal is RND prediction error (curiosity / novelty).  A KL penalty
against the base model keeps generations coherent.

Usage:
    python phase1/main_phase1.py           \
        actor_rollout_ref.model.path=...   \
        data.train_files=...               \
        data.val_files=...                 \
        +rnd.device=cuda                   \
        ...                                \
        (any veRL/TinyZero override)

The Hydra config is loaded from TinyZero's default ppo_trainer.yaml;
all Phase-1-specific defaults are overridden in scripts/train_phase1.sh.
"""

import ray
import hydra

from phase1.rnd_reward_manager import RNDRewardManager


@hydra.main(config_path="config", config_name="phase1_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                }
            }
        )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from pprint import pprint

    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    # ---- worker setup (mirrors TinyZero's main_ppo.py) ----
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # ---- RND reward function (the Phase 1 novelty) ----
    rnd_cfg = OmegaConf.to_container(config.get("rnd", {}), resolve=True) or {}
    reward_fn = RNDRewardManager(tokenizer=tokenizer, rnd_config=rnd_cfg, num_examine=0)
    val_reward_fn = RNDRewardManager(tokenizer=tokenizer, rnd_config=rnd_cfg, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping,
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
