"""Entry point for TTT-Discover training via the verl framework.

This module mirrors ``verl.trainer.main_ppo`` but:
1. Uses ``TTTDiscoverDataset`` instead of ``RLHFDataset``.
2. Uses ``PUCTSampler`` for curriculum learning.
3. Imports ``ttt_advantages`` to register the ``"entropic"`` advantage estimator.
4. Uses a dummy retain dataset (TTT-Discover doesn't need retain loss).
5. Monkey-patches ``RayPPOTrainer.__init__`` so that custom advantage
   estimators (like ``"entropic"``) set ``use_critic = False`` instead of
   raising ``NotImplementedError``.  This avoids modifying any file in
   ``unlearn/``.

Usage::

    python -m ttt_discover.verl_integration.main_ttt \\
        algorithm.adv_estimator=entropic \\
        ...
"""

from __future__ import annotations

import os
import socket
import sys

import hydra
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

# --- CRITICAL: register the "entropic" advantage estimator ---
# This import triggers the @register_adv_est("entropic") decorator.
import ttt_discover.verl_integration.ttt_advantages  # noqa: F401

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available


# ---------------------------------------------------------------------------
# Monkey-patch: allow custom advantage estimators in RayPPOTrainer.__init__
# ---------------------------------------------------------------------------
# The original code raises NotImplementedError for unknown adv_estimators.
# We wrap the original __init__ to catch that and set use_critic = False.

_original_init = RayPPOTrainer.__init__


def _patched_init(self, *args, **kwargs):
    """Wrapped __init__ that gracefully handles custom adv estimators."""
    try:
        _original_init(self, *args, **kwargs)
    except NotImplementedError:
        # The original __init__ raised because the adv_estimator was not in
        # its known list.  We set use_critic = False (custom estimators like
        # "entropic" don't need a critic) and re-run the rest of __init__
        # that comes after the adv_estimator check.
        #
        # At this point, everything before the raise has already executed.
        # We just need to set use_critic and call _validate_config + _create_dataloader.
        self.use_critic = False
        self._validate_config()

        # Reconstruct the arguments that _create_dataloader needs.
        # They were passed as positional/keyword args to __init__.
        import inspect
        sig = inspect.signature(_original_init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments

        self._create_dataloader(
            a.get("train_dataset"),
            a.get("val_dataset"),
            a.get("retain_dataset"),
            a.get("collate_fn"),
            a.get("train_sampler"),
            a.get("retain_sampler"),
        )


RayPPOTrainer.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for TTT-Discover training with Hydra configuration."""
    run_ttt(config)


def run_ttt(config) -> None:
    """Initialise Ray cluster and run TTT-Discover training."""
    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing TTT-Discover training."""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local
        import warnings

        # --- Re-register the advantage estimator inside the Ray worker ---
        import ttt_discover.verl_integration.ttt_advantages  # noqa: F401

        # --- Re-apply the monkey-patch inside the Ray worker process ---
        _apply_monkey_patch()

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download model checkpoint
        warnings.warn("=== START `copy_local_path_from_hdfs` ===", UserWarning)
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Tokenizer / processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Worker classes (same as main_ppo)
        warnings.warn("=== START `define worker classes` ===", UserWarning)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Role → worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Resource pools
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Reward model (optional)
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Reference policy (optional)
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Reward function
        warnings.warn("=== START `RewardManager init` ===", UserWarning)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )

        from verl.utils.dataset.rl_dataset import collate_fn

        # ---- TTT-Discover custom dataset ----
        from ttt_discover.verl_integration.ttt_dataset import TTTDiscoverDataset

        warnings.warn("=== Loading TTTDiscoverDataset ===", UserWarning)
        train_dataset = TTTDiscoverDataset(
            data_files=config.data.get("train_files", []),
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )

        # Validation dataset — use the same TTT dataset (or a regular one if val_files exist)
        val_dataset = _create_val_dataset(config, tokenizer, processor)

        # PUCT Sampler
        from ttt_discover.verl_integration.puct_sampler import PUCTSampler

        train_sampler = PUCTSampler(
            data_source=train_dataset,
            data_config=config.data,
        )

        # Retain dataset — TTT-Discover uses RETAIN_COEF=0, but the trainer
        # still iterates the retain dataloader.  We create a dummy that yields
        # empty batches compatible with the trainer's expectations.
        retain_dataset = _create_retain_dataset(config, tokenizer, processor)
        retain_sampler = _create_retain_sampler(config, retain_dataset)

        # ---- Trainer ----
        warnings.warn("=== INIT `RayPPOTrainer` ===", UserWarning)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            retain_dataset=retain_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            retain_sampler=retain_sampler,
        )
        warnings.warn("=== START `trainer.init_workers` ===", UserWarning)
        trainer.init_workers()
        warnings.warn("=== START `trainer.fit` ===", UserWarning)
        trainer.fit()


# ---------------------------------------------------------------------------
# Helper: re-apply the monkey-patch (for use inside Ray worker processes)
# ---------------------------------------------------------------------------

def _apply_monkey_patch():
    """Apply the RayPPOTrainer monkey-patch in the current process.

    This must be called inside each Ray worker because they run in separate
    Python processes that don't inherit module-level patches from the driver.
    """
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _Trainer
    import inspect

    # Guard: don't double-patch
    if getattr(_Trainer.__init__, "_ttt_patched", False):
        return

    _orig = _Trainer.__init__

    def _patched(self, *args, **kwargs):
        try:
            _orig(self, *args, **kwargs)
        except NotImplementedError:
            self.use_critic = False
            self._validate_config()
            sig = inspect.signature(_orig)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            self._create_dataloader(
                a.get("train_dataset"),
                a.get("val_dataset"),
                a.get("retain_dataset"),
                a.get("collate_fn"),
                a.get("train_sampler"),
                a.get("retain_sampler"),
            )

    _patched._ttt_patched = True
    _Trainer.__init__ = _patched


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _create_val_dataset(config, tokenizer, processor):
    """Create validation dataset — falls back to TTTDiscoverDataset if no val_files."""
    val_files = config.data.get("val_files", None)
    if val_files and len(val_files) > 0:
        from verl.utils.dataset.rl_dataset import RLHFDataset

        try:
            return RLHFDataset(
                data_files=val_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to load val dataset from {val_files}: {exc}", UserWarning)

    # Fallback: use the train dataset for validation too
    from ttt_discover.verl_integration.ttt_dataset import TTTDiscoverDataset

    return TTTDiscoverDataset(
        data_files=[],
        tokenizer=tokenizer,
        processor=processor,
        config=config.data,
    )


class _DummyRetainDataset(Dataset):
    """Minimal retain dataset producing zero tensors.

    The ray_trainer copies retain_input_ids, retain_response_ids,
    retain_attention_mask, retain_position_ids, and retain_loss_mask
    from the retain batch into the main batch.  With RETAIN_COEF=0 the
    retain loss is zero-weighted, so these tensors just need the right
    shape — the values don't matter.
    """

    def __init__(self, data_config, tokenizer):
        self.max_prompt_length = int(data_config.get("max_prompt_length", 1024))
        self.max_response_length = int(data_config.get("max_response_length", 2048))
        self.total_length = self.max_prompt_length + self.max_response_length
        self.pad_token_id = tokenizer.pad_token_id or 0
        # We need enough items so the DataLoader can draw retain_batch_size items
        self._size = 1000

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        input_ids = torch.full(
            (self.total_length,), self.pad_token_id, dtype=torch.long,
        )
        attention_mask = torch.zeros(self.total_length, dtype=torch.long)
        position_ids = torch.zeros(self.total_length, dtype=torch.long)
        response_ids = torch.full(
            (self.max_response_length,), self.pad_token_id, dtype=torch.long,
        )
        loss_mask = torch.zeros(self.max_response_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "response_ids": response_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


def _create_retain_dataset(config, tokenizer, processor):
    """Create retain dataset — uses actual files if provided, else a dummy.

    The dummy retain dataset produces tensors with the correct fields
    (input_ids, response_ids, attention_mask, position_ids, loss_mask)
    expected by ray_trainer.py.  Since RETAIN_COEF=0, the loss from these
    dummy batches is zero-weighted.
    """
    retain_files = config.data.get("retain_files", None)

    # Try loading real retain files if provided (non-empty list)
    if retain_files and len(retain_files) > 0:
        try:
            from verl.utils.dataset.unlearn_dataset import RetainDataset

            return RetainDataset(
                data_files=retain_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to load retain dataset: {exc}. Using dummy.", UserWarning)

    # Fallback: dummy dataset producing zero tensors with correct shapes
    return _DummyRetainDataset(config.data, tokenizer)


def _create_retain_sampler(config, retain_dataset):
    """Create a sampler for the retain dataset."""
    import torch
    from torch.utils.data import RandomSampler

    gen = torch.Generator()
    gen.manual_seed(config.data.get("seed", 1))
    return RandomSampler(data_source=retain_dataset, generator=gen)


if __name__ == "__main__":
    main()
