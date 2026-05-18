"""Base class for PPO trainers.

Extracts the shared training infrastructure (model creation, dataloaders,
checkpointing, evaluation, stats logging) so that concrete trainers only need
to implement rollout-specific methods.
"""

from __future__ import annotations

import abc
import os
from typing import Any

import torch.distributed as dist
from datasets import Dataset
from astraflow.train_worker.api.alloc_mode import AllocationMode
from astraflow.train_worker.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
)
from astraflow.train_worker.api.engine_api import InferenceEngine
from astraflow.train_worker.api.io_struct import FinetuneSpec, StepInfo
from astraflow.train_worker.engine.ppo.actor import FSDPPPOActor, MegatronPPOActor
from astraflow.train_worker.engine.ppo.critic import FSDPPPOCritic, MegatronPPOCritic
from astraflow.train_worker.platforms import current_platform
from astraflow.train_worker.utils import logging, perf_tracer, seeding
from astraflow.train_worker.utils.hf_utils import load_hf_processor_and_tokenizer
from astraflow.train_worker.utils.recover import RecoverHandler
from astraflow.train_worker.utils.saver import Saver
from astraflow.train_worker.utils.stats_logger import StatsLogger

logger = logging.getLogger(__name__)


class PPOTrainerBase(abc.ABC):
    """Base class providing shared PPO training infrastructure.

    Subclasses must implement:
    - ``_init_rollout`` — create inference engine(s)
    - ``prepare_batch_from_buffer`` — fetch a training batch
    - ``train`` — the main training loop
    """

    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | dict[str, tuple[Dataset, int]] | None = None,
    ):
        """Initialize PPOTrainerBase.

        Parameters
        ----------
        config : PPOConfig
            PPO training configuration
        train_dataset : Dataset
            Training dataset
        valid_dataset : Dataset | dict[str, tuple[Dataset, int]] | None
            Validation dataset(s). Can be:
            - A single Dataset for simple evaluation
            - A dict mapping dataset names to (dataset, repeat) tuples where
              repeat is the number of evaluation runs for computing eval metrics
            - None to skip validation
        """
        rank = int(os.getenv("RANK", "0"))

        # Configure performance tracer
        if config.perf_tracer is not None:
            perf_tracer.configure(config.perf_tracer, rank=rank)

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse allocation mode.
        self.allocation_mode = AllocationMode.resolve(config.allocation_mode)

        # Create models: actor, critic, etc.
        self.actor = self._create_actor(config.actor)
        self.critic = None
        if config.critic is not None:
            self.critic = self._create_critic(config.critic)
        self.ref = None
        if (
            (config.actor.kl_ctl > 0 or config.actor.kl_penalty_coef > 0)
            and config.ref is not None
        ):
            self.ref = self._create_actor(config.ref)

        self.train_dataset = train_dataset

        # Initialize inference
        self.rollout = self._init_rollout(config.rollout, is_eval=False)
        self.eval_rollout = self._init_rollout(config.rollout, is_eval=True)

        # Use top-level train_batch_size; fall back to train_dataset config.
        train_bs = config.train_batch_size
        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(train_dataset),
            train_batch_size=train_bs,
            _total_train_steps_override=config.total_train_steps,
        )

        self.ft_spec = ft_spec

        # Initialize models
        self.parallel_strategy = self.allocation_mode.train
        assert self.parallel_strategy is not None
        engine_init_kwargs = {"addr": None, "ft_spec": ft_spec}
        self.actor.initialize(**engine_init_kwargs)
        if self.critic is not None:
            self.critic.initialize(**engine_init_kwargs)
        if self.ref is not None:
            self.ref.initialize(**engine_init_kwargs)

        # Connect to inference engine for weight transfer.
        # TCP mode (rollout=None): weight transfer is handled externally by
        # WeightManager / sender_agent — no WeightUpdateMeta needed.
        self.weight_update_meta = None
        if self.rollout is not None:
            self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)

        # Set up statistics logging (wandb, tensoboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.stats_logger,
            None,  # dataloader (unused in v2)
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _init_rollout(
        self, rollout_config: InferenceEngineConfig, is_eval: bool = False
    ) -> InferenceEngine:
        """Create an inference engine for rollout or evaluation."""
        ...

    @abc.abstractmethod
    def prepare_batch_from_buffer(self, *args, **kwargs) -> dict[str, Any]:
        """Fetch a training batch from the rollout buffer."""
        ...

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Run the training loop."""
        ...

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_actor(self, actor_config: PPOActorConfig):
        if self.allocation_mode.train_backend == "fsdp":
            actor = FSDPPPOActor(config=actor_config)
        elif self.allocation_mode.train_backend == "megatron":
            actor = MegatronPPOActor(config=actor_config)
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.train_backend}, expected fsdp or megatron"
            )
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _create_critic(self, critic_config: PPOCriticConfig):
        if self.allocation_mode.train_backend == "fsdp":
            critic = FSDPPPOCritic(config=critic_config)
        elif self.allocation_mode.train_backend == "megatron":
            critic = MegatronPPOCritic(config=critic_config)
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.train_backend}, expected fsdp or megatron"
            )
        critic.create_process_group(parallel_strategy=self.allocation_mode.train)
        return critic

    # ------------------------------------------------------------------
    # Checkpointing / saving
    # ------------------------------------------------------------------

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        import time as _time
        _rank = dist.get_rank() if dist.is_initialized() else 0
        # Save as HF models for evaluation
        _t0 = _time.monotonic()
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        print(f"[save_hf] rank={_rank} step={global_step} saver.save(actor) {_time.monotonic()-_t0:.1f}s", flush=True)
        if self.critic is not None:
            _t1 = _time.monotonic()
            self.saver.save(
                self.critic,
                epoch,
                epoch_step,
                global_step,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name="critic",
            )
            print(f"[save_hf] rank={_rank} step={global_step} saver.save(critic) {_time.monotonic()-_t1:.1f}s", flush=True)
        _t2 = _time.monotonic()
        dist.barrier(group=self.actor.cpu_group)
        print(f"[save_hf] rank={_rank} step={global_step} dist.barrier {_time.monotonic()-_t2:.1f}s", flush=True)
        current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        import time as _time
        _rank = dist.get_rank() if dist.is_initialized() else 0
        # Save recoverable checkpoints
        to_save = dict(default=self.actor)
        if self.critic is not None:
            to_save["critic"] = self.critic
        buffer = getattr(self, "rollout_buffer", None)
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=self.ft_spec.steps_per_epoch,
        )
        _t0 = _time.monotonic()
        self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.stats_logger,
            None,  # dataloader (unused in v2)
            buffer=buffer,
            tokenizer=self.tokenizer,
            processor=self.processor,
            rollout_dataloader=getattr(self, "rollout_dataloader", None),
        )
        print(f"[save_recover] rank={_rank} step={global_step} recover_handler.dump {_time.monotonic()-_t0:.1f}s", flush=True)

        _t1 = _time.monotonic()
        dist.barrier(group=self.actor.cpu_group)
        print(f"[save_recover] rank={_rank} step={global_step} dist.barrier {_time.monotonic()-_t1:.1f}s", flush=True)
        current_platform.synchronize()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # Upload statistics to the logger (e.g., wandb)
        stats = self.actor.export_stats()
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    # ------------------------------------------------------------------
    # Context manager & cleanup
    # ------------------------------------------------------------------

    def close(self):
        self.stats_logger.close()
        if self.eval_rollout is not None:
            self.eval_rollout.destroy()
        if self.rollout is not None:
            self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        if self.critic is not None:
            self.critic.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
