"""Simplified PPO trainer using AstraFlow HTTP service.

This trainer has a clean training loop:
  get_batch → distribute → train_step → wm.offload →
  save checkpoint → notify_version → next iteration

All orchestration logic (data acquisition, buffering, eval, version
management, pause/resume) is handled by the AstraFlow HTTP service.
The trainer only needs a few HTTP calls per iteration.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset

from astraflow.train_worker.api.cli_args import InferenceEngineConfig, PPOConfig
from astraflow.train_worker.api.engine_api import InferenceEngine
from astraflow.train_worker.utils.dist_rollout import _slice_tensor_dict, redistribute
from astraflow.train_worker.platforms import current_platform
from astraflow.train_worker.utils import logging, perf_tracer, stats_tracker
from astraflow.train_worker.utils.data import (
    broadcast_tensor_container,
    get_batch_size,
    tensor_container_to,
)
from astraflow.train_worker.utils.device import clear_memory, log_gpu_stats
from astraflow.train_worker.utils.perf_tracer import Category

from .astraflow_client import AstraFlowClient
from .ppo_base import PPOTrainerBase

logger = logging.getLogger(__name__)


class AstraFlowPPOTrainer(PPOTrainerBase):
    """PPO trainer backed by AstraFlow HTTP service.

    Delegates all orchestration to the AstraFlow HTTP service and only
    handles:
    - Batch distribution across DP ranks
    - PPO compute (forward/backward, all DP ranks)
    - Weight transfer via TCP (WeightManager → sender agent → RaaS)
    - Checkpoint saving
    """

    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | dict[str, tuple[Dataset, int]] | None = None,
    ):
        super().__init__(config, train_dataset, valid_dataset)

        # AstraFlow HTTP client
        service_url = os.environ.get("ASTRAFLOW_URL", "http://localhost:8000")
        model_id = getattr(config, "model_id", None)
        self.astraflow = AstraFlowClient(
            service_url=service_url,
            model_id=model_id,
        )

        self._is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

        # Initialize connection to AstraFlow service
        self.astraflow.initialize(verbose=self._is_rank0)

        # WeightManager for TCP weight transfer (initialized after actor is ready).
        self.weight_manager = None
        self._tcp_raas_urls: list[str] = []
        raas_url = os.environ.get("ASTRAFLOW_RAAS_URL", "").strip()
        if not raas_url and model_id is None:
            # Single-model mode requires a direct RaaS URL for the
            # sender agent to call /get_receive_instances and
            # /update_weights on RaaS.
            raise RuntimeError(
                "ASTRAFLOW_RAAS_URL must be set when model_id is not configured. "
                "In multi-model mode (model_id set), AstraFlow coordinates "
                "weight transfer via the RaaS pool, so ASTRAFLOW_RAAS_URL "
                "is not needed."
            )
        if raas_url:
            self._tcp_raas_urls = [raas_url]
        self._init_weight_manager()

        self._closed = False
        # Set True only when train() returns from its loop without
        # exception. Gates cluster-wide shutdown in _shutdown_and_exit so
        # an unrecoverable trainer error doesn't take down AstraFlow + RaaS.
        self._completed_normally = False

        # After recovery, push recovered weights to RaaS so it
        # generates rollouts with the correct model instead of the pretrained
        # base model.  Must happen after _init_weight_manager() which creates
        # the shared-memory buffer and sender agent.
        self._recovered_version: int | None = None
        if self.recover_info is not None and self.weight_manager is not None:
            version = self.recover_info.last_step_info.global_step + 1
            self._recovered_version = version
            # Set actor (and critic) version — skipped by recover.py
            # because inference_engine is None.
            self.actor.set_version(version)
            if self.critic is not None:
                self.critic.set_version(version)
            if self._is_rank0:
                logger.info(
                    "Recovery: pushing recovered weights to RaaS "
                    "(version=%d)",
                    version,
                )
            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if self.config.actor.use_lora:
                from peft import get_peft_model_state_dict
                from torch.distributed.checkpoint.state_dict import (
                    get_model_state_dict,
                    StateDictOptions,
                )

                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                state_dict = get_model_state_dict(self.actor.model, options=options)
                state_dict = get_peft_model_state_dict(
                    self.actor.model, state_dict=state_dict
                )
                self.weight_manager.offload(
                    state_dict.items(),
                    version=version,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                self.weight_manager.offload(
                    self._get_named_params_for_offload(),
                    version=version,
                    rank=rank,
                    world_size=world_size,
                )
            # Recovery weight loading is handled by AstraFlow: signal_ready
            # sets recovered_version, and the version barrier in
            # notify_version triggers notify_all_versions to pull weights.
            # Drain the delta/no_delta message left by the initial offload
            # so it doesn't shift the queue for subsequent steps.
            # Use 180s timeout: first delta after recovery compares against
            # uninitialized buffer and can take ~85s for a 4B model.
            self.weight_manager.wait_delta_ready(timeout=180.0)

        # Signal readiness — AstraFlow starts data acquisition only
        # after both RaaS and trainer are ready.
        if self._is_rank0:
            sender_endpoint = self.weight_manager.get_sender_endpoint()
            self.astraflow.signal_ready(
                train_batch_size=self.config.train_batch_size,
                sender_endpoint=sender_endpoint,
                recovered_version=self._recovered_version,
            )

    def _init_rollout(
        self, rollout_config: InferenceEngineConfig, is_eval: bool = False
    ) -> InferenceEngine | None:
        """Return None — weight transfer is handled by WeightManager / sender agent."""
        return None

    @property
    def _is_megatron(self) -> bool:
        try:
            from astraflow.train_worker.engine.megatron_engine import MegatronEngine
        except ImportError:
            return False
        return isinstance(self.actor, MegatronEngine)

    def _get_named_params_for_offload(self):
        """Return the (name, tensor) stream for WeightManager.offload().

        - Megatron: a fresh ``export_hf_named_params`` generator that yields
          gathered HF-layout tensors (handles TP/PP/EP/VPP). WeightManager
          streams it into the HF buffer on the writer rank.
        - FSDP: raw ``model.named_parameters()`` (DTensor shards handled by
          WeightManager's shard-copy / all-gather paths).
        """
        if self._is_megatron:
            return self.actor.export_hf_named_params()
        try:
            return self.actor.model.named_parameters(remove_duplicate=False)
        except TypeError:
            return self.actor.model.named_parameters()

    def _init_weight_manager(self) -> None:
        """Initialize WeightManager for TCP-based weight transfer.

        Must be called after ``super().__init__()`` so that ``self.actor.model``
        is available (actor is initialized inside ``PPOTrainerBase.__init__``).
        """
        import socket

        from astraflow.core.weight_manager import WeightManager, WeightManagerConfig
        from astraflow.core.weight_manager.transfer.config import (
            SenderAgentConfig,
            TransferEngineConfig,
        )

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        global_rank = dist.get_rank() if dist.is_initialized() else 0

        handshake_port = int(os.getenv("WEIGHT_TRANSFER_HANDSHAKE_PORT", "21000"))
        http_port = int(os.getenv("WEIGHT_TRANSFER_HTTP_PORT", os.getenv("WEIGHT_TRANSFER_RPYC_PORT", "18861")))

        # The sender subprocess (spawn-child of local_rank 0) redirects its
        # stdout/stderr onto this path so fatal-signal tracebacks survive
        # independent of the torchrun/tee pipeline. `fileroot` lives on a
        # few different sub-configs depending on which code path populates
        # it (recover/saver/stats_logger); fall back across them so any one
        # being unset still gives us a usable path.
        fileroot = (
            getattr(getattr(self.config, "recover", None), "fileroot", None)
            or getattr(getattr(self.config, "saver", None), "fileroot", None)
            or "/tmp/astraflow"
        )
        sender_log_file = os.path.join(fileroot, "logs", "sender.log")

        sender_config = SenderAgentConfig(
            trainer_global_rank=global_rank,
            trainer_world_size=dist.get_world_size() if dist.is_initialized() else 1,
            engine_configs=[
                TransferEngineConfig(
                    local_hostname=socket.gethostname(),
                    handshake_port=handshake_port,
                )
            ],
            http_bind_port=http_port,
            log_file=sender_log_file,
        )

        # Transfer strategies: config field, env var override for compat
        strategies_env = os.getenv("WEIGHT_TRANSFER_STRATEGIES")
        if strategies_env is not None:
            strategies = [s.strip() for s in strategies_env.split(",")]
        elif self.config.weight_transfer_strategies == "delta":
            strategies = ["full", "delta"]
        else:
            strategies = ["full"]

        wm_config = WeightManagerConfig(
            sender_config=sender_config,
            strategies=strategies,
        )
        self.weight_manager = WeightManager(wm_config)

        # Build LoRA metadata for the sender agent / receiver so the RaaS
        # side can save weights in PEFT adapter format.
        lora_config = None
        if self.config.actor.use_lora:
            peft_cfg = self.actor.model.peft_config["default"]
            lora_config = {
                "peft_type": "LORA",
                "r": peft_cfg.r,
                "lora_alpha": peft_cfg.lora_alpha,
                "target_modules": list(peft_cfg.target_modules),
            }

        # Megatron HF-export mode: the buffer is sized from the full HF
        # weight layout, and offload streams gathered HF tensors into it.
        # This keeps the sender/RaaS path identical to FSDP (delta in HF
        # space) and works under any TP/PP/EP/VPP combination.
        megatron_hf_meta = None
        if self._is_megatron:
            megatron_hf_meta = self.actor.get_hf_weight_metadata()

        # Determine HSDP replica rank (0 = primary, >0 = secondary).
        dp_replicate_rank = 0
        if hasattr(self.actor, 'world_mesh') and "dp_replicate" in self.actor.world_mesh.mesh_dim_names:
            dp_replicate_rank = self.actor.world_mesh["dp_replicate"].get_local_rank()

        if self.config.actor.use_lora:
            # Use get_peft_model_state_dict so the buffer layout (key names +
            # sizes) exactly matches what offload writes for LoRA.
            from peft import get_peft_model_state_dict

            raw_state = dict(self.actor.model.named_parameters())
            lora_params = get_peft_model_state_dict(self.actor.model, state_dict=raw_state)
            logger.info("[DEBUG-INIT] raw_state keys (%d): %s", len(raw_state), list(raw_state.keys())[:10])
            logger.info("[DEBUG-INIT] lora_params keys (%d): %s", len(lora_params), list(lora_params.keys())[:10])
            self.weight_manager.initialize(
                lora_params.items(), local_rank, global_rank,
                lora_config=lora_config,
                dp_replicate_rank=dp_replicate_rank,
            )
        else:
            # In Megatron HF-export mode the layout comes from
            # megatron_hf_meta, so named_params is unused at init time.
            named_params = (
                iter(()) if self._is_megatron
                else self._get_named_params_for_offload()
            )
            self.weight_manager.initialize(
                named_params, local_rank, global_rank,
                megatron_hf_meta=megatron_hf_meta,
                dp_replicate_rank=dp_replicate_rank,
            )
        logger.info(
            "WeightManager initialised for TCP weight transfer "
            "(http_port=%d handshake_port=%d strategies=%s)",
            http_port,
            handshake_port,
            strategies,
        )

    def _filter_zero_adv_in_batch(
        self,
        batch: dict[str, Any],
        buffer_stats: dict[str, float],
    ) -> dict[str, Any] | None:
        """Drop sequences whose stamped group_reward_std == 0.

        Runs on rank 0 only, before the cross-rank broadcast. Survivor
        count is rounded down to a multiple of ``dp_world_size *
        group_size`` so the downstream DP-slice and ``redistribute``
        (which asserts ``per_rank_bs % granularity == 0``) both stay
        clean. In multi-agent workflows per-model groups can be smaller
        than ``group_size``, so filter survivors are not guaranteed to
        be multiples of ``group_size`` on their own — round-down here
        absorbs the slack.

        Returns None if nothing survives — caller should skip the step.
        """
        g_std = batch.get("group_reward_std")
        if g_std is None or not torch.is_tensor(g_std):
            # Workflow did not stamp group stats; fail-open (keep all).
            return batch

        g_std_1d = g_std[:, 0] if g_std.ndim >= 2 else g_std
        keep = g_std_1d > 0
        n_total = int(keep.numel())
        n_keep = int(keep.sum().item())
        n_dropped = n_total - n_keep

        buffer_stats["rollout/zero_adv_dropped"] = float(n_dropped)
        buffer_stats["rollout/zero_adv_frac"] = n_dropped / max(n_total, 1)

        if n_keep == 0:
            buffer_stats["rollout/effective_batch_size"] = 0.0
            buffer_stats["rollout/round_down_dropped"] = 0.0
            return None

        dp_ws = self.actor.data_parallel_world_size
        group_size = self.config.actor.group_size or 1
        unit = dp_ws * group_size
        n_target = (n_keep // unit) * unit
        if n_target == 0:
            buffer_stats["rollout/effective_batch_size"] = 0.0
            buffer_stats["rollout/round_down_dropped"] = float(n_keep)
            return None

        # If rounding loses samples, drop the trailing keepers (deterministic).
        if n_target < n_keep:
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
            keep[keep_idx[n_target:]] = False

        buffer_stats["rollout/effective_batch_size"] = float(n_target)
        buffer_stats["rollout/round_down_dropped"] = float(n_keep - n_target)

        return {
            k: (v[keep] if torch.is_tensor(v) and v.shape[0] == n_total else v)
            for k, v in batch.items()
        }

    def prepare_batch_from_buffer(
        self,
        timeout: float | None = None,
        granularity: int | None = None,
        version: int | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, float]]:
        """Get batch from AstraFlow and distribute across DP ranks.

        Only rank 0 fetches the batch from the AstraFlow HTTP service.
        The batch is then broadcast and redistributed across all DP ranks.

        When ``actor.filter_zero_adv_in_batch`` is True and the entire
        fetched batch is zero-advantage (or rounding to dp_world_size
        leaves 0 survivors), returns ``(None, buffer_stats)`` and the
        caller should skip the training step.

        Returns ``(batch, buffer_stats)`` where buffer_stats are for wandb.
        """
        batch = None
        buffer_stats: dict[str, float] = {}
        if self._is_rank0:
            batch, buffer_stats = self.astraflow.get_batch(
                timeout=timeout, version=version,
            )
            # Compute batch reward mean before any transformation.
            batch_rewards = batch.get("rewards")
            if batch_rewards is not None and torch.is_tensor(batch_rewards):
                r = batch_rewards.detach().float().flatten()
                if r.numel() > 0:
                    buffer_stats["rollout/batch_reward_mean"] = float(r.mean().item())

            if self.config.actor.filter_zero_adv_in_batch:
                batch = self._filter_zero_adv_in_batch(batch, buffer_stats)

            if batch is not None:
                batch = tensor_container_to(batch, current_platform.current_device())

        # Broadcast skip-flag so all ranks agree on whether to skip the step.
        skip_flag = torch.zeros(
            1, dtype=torch.int32, device=current_platform.current_device(),
        )
        if self._is_rank0 and batch is None:
            skip_flag.fill_(1)
        dist.broadcast(skip_flag, src=0)
        if int(skip_flag.item()) == 1:
            return None, buffer_stats

        # Broadcast raw batch from rank 0 to all ranks
        batch = broadcast_tensor_container(
            batch,
            src_rank=0,
            group=None,
        )

        current_platform.synchronize()
        dist.barrier(group=self.actor.cpu_group)

        # DP-slice per rank
        if batch is not None:
            dp_rank = self.actor.data_parallel_rank
            dp_world_size = self.actor.data_parallel_world_size
            batch_size = get_batch_size(batch)
            samples_per_rank = batch_size // dp_world_size
            start_idx = dp_rank * samples_per_rank
            end_idx = start_idx + samples_per_rank
            batch = _slice_tensor_dict(batch, start_idx, end_idx)

        # Redistribute (FFD load-balance by sequence length)
        redist = redistribute(
            batch,
            granularity=granularity or self.config.actor.group_size,
            group=self.actor.data_parallel_group,
        )
        batch = redist.data

        current_platform.synchronize()
        dist.barrier(group=self.actor.cpu_group)

        # Broadcast to model-parallel group
        batch = broadcast_tensor_container(
            batch,
            src_rank=self.actor.current_data_parallel_head(),
            group=self.actor.context_and_model_parallel_group,
        )

        current_platform.synchronize()
        dist.barrier(group=self.actor.cpu_group)

        return batch, buffer_stats

    def _log_pre_train_eval_to_wandb(self, eval_results: dict) -> None:
        """Log pre-train eval results directly to wandb at step=0.

        Mirrors the scoping used by the in-loop eval log
        (``eval-avg/<ds>/avg@k``, ``eval-pass/<ds>/pass@k``,
        ``eval-avg/overall_avg``, ``eval-pass/overall_pass@k``) but bypasses
        ``stats_logger.commit`` so the monotonic log-step guard doesn't push
        the first training step's commit to wandb step 1.
        """
        try:
            import wandb
        except ImportError:
            return
        if getattr(wandb, "run", None) is None:
            return

        datasets = eval_results.get("datasets", {})
        all_ks = {
            int(ds["pass_k"])
            for ds in datasets.values()
            if ds.get("pass_k") is not None
        }
        has_pass_at_k = any(k > 1 for k in all_ks)

        payload: dict[str, float] = {}
        for name, ds in datasets.items():
            avg_k = ds.get("pass_k")
            avg_metric_name = f"avg@{int(avg_k)}" if avg_k is not None else "avg@k"
            payload[f"eval-avg/{name}/{avg_metric_name}"] = ds.get("avg@k", 0.0)
            pass_at_k = ds.get("pass@k")
            pass_k = ds.get("pass_k")
            if pass_at_k is not None and pass_k is not None and int(pass_k) > 1:
                payload[f"eval-pass/{name}/pass@{int(pass_k)}"] = pass_at_k

        if "overall_avg@k" in eval_results:
            payload["eval-avg/overall_avg"] = eval_results["overall_avg@k"]
            if "overall_pass@k" in eval_results and has_pass_at_k:
                metric_name = (
                    f"overall_pass@{next(iter(all_ks))}"
                    if len(all_ks) == 1
                    else "overall_pass@k"
                )
                payload[f"eval-pass/{metric_name}"] = eval_results["overall_pass@k"]

        if payload:
            wandb.log(payload, step=0)

    def train(
        self,
        total_epochs: int | None = None,
        granularity: int | None = None,
    ):
        """Run the training loop.

        Workflow specs, data acquisition, and eval are all managed by
        the AstraFlow HTTP service.

        Parameters
        ----------
        total_epochs : int | None
            Override total epochs from config.
        granularity : int | None
            Granularity for batch redistribution.
        """
        try:
            config = self.config
            start_step = (
                self.recover_info.last_step_info.next().global_step
                if self.recover_info is not None
                else 0
            )

            if total_epochs is None:
                total_epochs = config.total_train_epochs
            if total_epochs <= 0:
                raise ValueError(f"Total epochs must be positive: {total_epochs}")
            steps_per_epoch = self.ft_spec.steps_per_epoch
            max_steps = total_epochs * steps_per_epoch

            # Pre-training eval: evaluate the initial checkpoint at version=0
            # before any training step. Skipped on recovery runs (start_step > 0)
            # and when disabled via config.
            if (
                start_step == 0
                and getattr(config.evaluator, "eval_at_start", False)
                and getattr(config.evaluator, "freq_steps", None) is not None
            ):
                if self._is_rank0:
                    print(
                        "[Trainer] [pre-train] Running eval at version=0 ...",
                        flush=True,
                    )
                    with stats_tracker.record_timing("eval"):
                        try:
                            pre_eval_results = self.astraflow.notify_version(
                                version=0, run_eval=True,
                            )
                        except Exception as e:
                            logger.warning(
                                "[pre-train] notify_version failed: %s — "
                                "skipping initial eval, continuing",
                                e,
                            )
                            pre_eval_results = None
                    print(
                        "[Trainer] [pre-train] Initial eval complete",
                        flush=True,
                    )
                    # Log pre-train eval to wandb at step=0 via a direct wandb.log
                    # call so the stats_logger monotonic guard doesn't push the
                    # first training step's commit to wandb step 1.
                    if pre_eval_results and isinstance(pre_eval_results, dict):
                        self._log_pre_train_eval_to_wandb(pre_eval_results)
                dist.barrier(group=self.actor.cpu_group)

            for global_step in range(start_step, max_steps):
                if (
                    config.total_train_steps is not None
                    and global_step >= config.total_train_steps
                ):
                    break
                epoch = global_step // steps_per_epoch
                step = global_step % steps_per_epoch

                with stats_tracker.record_timing("train_step_total"):
                    # Capture for a trainer-local rollout_wait_fraction
                    # that replaces the service-global EWMA (which is
                    # contaminated when multiple trainers call /batch on
                    # the same AstraFlow service). See Step 7 below.
                    _step_t0 = time.monotonic()

                    # Step 1: Get batch from AstraFlow + distribute
                    if self._is_rank0:
                        print(
                            f"[Trainer] [step {global_step}] Waiting for data ...",
                            flush=True,
                        )
                    _batch_t0 = time.monotonic()
                    with (
                        stats_tracker.record_timing("rollout"),
                        perf_tracer.trace_scope(
                            "train.rollout",
                            category=Category.COMPUTE,
                            args={"global_step": global_step, "epoch_step": step},
                        ),
                    ):
                        batch, buffer_stats = self.prepare_batch_from_buffer(
                            timeout=None,
                            granularity=granularity or self.config.actor.group_size,
                            version=global_step,
                        )
                    _batch_elapsed = time.monotonic() - _batch_t0
                    if self._is_rank0:
                        print(
                            f"[Trainer] [step {global_step}] Data collected in "
                            f"{_batch_elapsed:.1f}s",
                            flush=True,
                        )

                    # All-zero-adv batch (filter dropped everything): skip
                    # PPO compute and LR scheduler tick for this step, but
                    # still publish a new weight version so the AstraFlow
                    # service's version barrier advances. Without this the
                    # next step's GET /batch?version=N+1 blocks on
                    # `service_version < trainer_version` (see
                    # dataflow/service.py: get_batch) until the HTTP request
                    # times out and NCCL watchdogs fire on other ranks.
                    # Re-publishing unchanged weights under v=N+1 is a no-op
                    # for correctness — RaaS pulls the same parameters.
                    if batch is None:
                        if self._is_rank0:
                            print(
                                f"[Trainer] [step {global_step}] all-zero-adv "
                                f"batch, skipping training compute",
                                flush=True,
                            )
                            if buffer_stats:
                                for k, v in buffer_stats.items():
                                    scope, _, metric = k.partition("/")
                                    if scope and metric:
                                        with stats_tracker.scope(scope):
                                            stats_tracker.scalar(**{metric: v})

                        new_version = global_step + 1
                        if self._is_rank0:
                            print(
                                f"[Trainer] [step {global_step}] (skip) "
                                f"Offloading weights to WeightManager ...",
                                flush=True,
                            )
                        with stats_tracker.record_timing("update_weights"):
                            if self.config.actor.use_lora:
                                from peft import get_peft_model_state_dict
                                from torch.distributed.checkpoint.state_dict import (
                                    get_model_state_dict,
                                    StateDictOptions,
                                )

                                options = StateDictOptions(
                                    full_state_dict=True, cpu_offload=True
                                )
                                state_dict = get_model_state_dict(
                                    self.actor.model, options=options
                                )
                                state_dict = get_peft_model_state_dict(
                                    self.actor.model, state_dict=state_dict
                                )
                                self.weight_manager.offload(
                                    state_dict.items(),
                                    version=new_version,
                                    rank=dist.get_rank() if dist.is_initialized() else 0,
                                    world_size=dist.get_world_size() if dist.is_initialized() else 1,
                                )
                            else:
                                self.weight_manager.offload(
                                    self._get_named_params_for_offload(),
                                    version=new_version,
                                    rank=dist.get_rank() if dist.is_initialized() else 0,
                                    world_size=dist.get_world_size() if dist.is_initialized() else 1,
                                )
                        if self._is_rank0:
                            print(
                                f"[Trainer] [step {global_step}] (skip) "
                                f"TCP buffer ready, version={new_version}",
                                flush=True,
                            )

                        self.actor.set_version(new_version)
                        if self.critic is not None:
                            self.critic.set_version(new_version)

                        if self._is_rank0:
                            if (
                                hasattr(self, "weight_manager")
                                and self.weight_manager is not None
                            ):
                                self.weight_manager.wait_delta_ready()
                            print(
                                f"[Trainer] [step {global_step}] (skip) "
                                f"Notifying AstraFlow version={new_version} "
                                f"(async) ...",
                                flush=True,
                            )
                            try:
                                self.astraflow.notify_version_async(
                                    version=new_version,
                                )
                            except Exception as e:
                                logger.warning(
                                    "[step %d] (skip) notify_version_async "
                                    "failed: %s — continuing (RaaS will catch "
                                    "up on next bump)",
                                    global_step, e,
                                )

                        dist.barrier(group=self.actor.cpu_group)

                        # Collective across all ranks.
                        self._export_and_commit_stats(
                            epoch=epoch,
                            epoch_step=step,
                            global_step=global_step,
                        )
                        perf_tracer.save(step=global_step)
                        continue

                    # Step 2: Train step (PPO compute)
                    if self.critic is not None:
                        with (
                            stats_tracker.record_timing("critic_values"),
                            perf_tracer.trace_scope(
                                "train.compute_values",
                                category=Category.COMPUTE,
                                args={"global_step": global_step},
                            ),
                        ):
                            values = self.critic.compute_values(batch)
                            batch["values"] = values
                            log_gpu_stats("critic values")

                    if config.actor.recompute_logprob:
                        with (
                            stats_tracker.record_timing("recompute_logp"),
                            perf_tracer.trace_scope(
                                "train.recompute_logp",
                                category=Category.COMPUTE,
                                args={"global_step": global_step},
                            ),
                        ):
                            logp = self.actor.compute_logp(batch)
                            batch["prox_logp"] = logp
                            log_gpu_stats("recompute logp")

                    if self.ref is not None:
                        with (
                            stats_tracker.record_timing("ref_logp"),
                            perf_tracer.trace_scope(
                                "train.ref_logp",
                                category=Category.COMPUTE,
                                args={"global_step": global_step},
                            ),
                        ):
                            batch["ref_logp"] = self.ref.compute_logp(batch)
                            log_gpu_stats("ref logp")

                    with (
                        stats_tracker.record_timing("compute_advantage"),
                        perf_tracer.trace_scope(
                            "train.compute_advantage",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        self.actor.compute_advantages_with_normalized_reward(batch)
                        log_gpu_stats("compute advantages")

                    # Release cached GPU memory before PPO backward pass.
                    # The ref logp forward pass leaves ~37 GB in the caching
                    # allocator's free list; without releasing it the PPO
                    # backward can OOM on the FSDP all-gather.
                    _t_clear = time.monotonic()
                    clear_memory()
                    _t_clear_dur = time.monotonic() - _t_clear
                    log_gpu_stats("after clear_memory (pre-PPO)")
                    if self._is_rank0:
                        print(
                            f"[Trainer] [step {global_step}] clear_memory before PPO took {_t_clear_dur:.3f}s",
                            flush=True,
                        )

                    with (
                        stats_tracker.record_timing("train_step"),
                        perf_tracer.trace_scope(
                            "train.ppo_update",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        self.actor.ppo_update(batch)
                        self.actor.step_lr_scheduler()
                        log_gpu_stats("ppo update")

                    if self.critic is not None:
                        with (
                            stats_tracker.record_timing("critic_train_step"),
                            perf_tracer.trace_scope(
                                "train.critic_ppo_update",
                                category=Category.COMPUTE,
                                args={"global_step": global_step},
                            ),
                        ):
                            self.critic.ppo_update(batch)
                            self.critic.step_lr_scheduler()
                            log_gpu_stats("ppo critic update")

                    # Offload weights to WeightManager (all ranks).
                    # WM handles GPU→CPU copy, buffer swap, and sender
                    # notification. Training continues immediately after.
                    if self._is_rank0:
                        print(
                            f"[Trainer] [step {global_step}] Offloading weights to WeightManager ...",
                            flush=True,
                        )
                    new_version = global_step + 1
                    with (
                        stats_tracker.record_timing("update_weights"),
                        perf_tracer.trace_scope(
                            "train.update_weights",
                            category=Category.COMM,
                            args={"global_step": global_step},
                        ),
                    ):
                        if self.config.actor.use_lora:
                            from peft import get_peft_model_state_dict
                            from torch.distributed.checkpoint.state_dict import (
                                get_model_state_dict,
                                StateDictOptions,
                            )

                            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                            state_dict = get_model_state_dict(self.actor.model, options=options)
                            logger.info("[DEBUG-OFFLOAD] full state_dict keys (%d): %s", len(state_dict), list(state_dict.keys())[:10])
                            state_dict = get_peft_model_state_dict(
                                self.actor.model, state_dict=state_dict
                            )
                            logger.info("[DEBUG-OFFLOAD] after peft filter keys (%d): %s", len(state_dict), list(state_dict.keys())[:10])
                            wt_metrics = self.weight_manager.offload(
                                state_dict.items(),
                                version=new_version,
                                rank=dist.get_rank() if dist.is_initialized() else 0,
                                world_size=dist.get_world_size() if dist.is_initialized() else 1,
                            )
                        else:
                            wt_metrics = self.weight_manager.offload(
                                self._get_named_params_for_offload(),
                                version=new_version,
                                rank=dist.get_rank() if dist.is_initialized() else 0,
                                world_size=dist.get_world_size() if dist.is_initialized() else 1,
                            )
                    if self._is_rank0:
                        print(
                            f"[Trainer] [step {global_step}] TCP buffer ready, "
                            f"version={new_version}",
                            flush=True,
                        )

                    # Set internal version
                    self.actor.set_version(global_step + 1)
                    if self.critic is not None:
                        self.critic.set_version(global_step + 1)

                    # Step 5: Save buffer + checkpoint
                    _save_t0 = time.monotonic()
                    if self._is_rank0:
                        print(f"[Trainer] [step {global_step}] Saving checkpoint ...", flush=True)
                    with (
                        stats_tracker.record_timing("save"),
                        perf_tracer.trace_scope(
                            "train.save",
                            category=Category.IO,
                            args={"global_step": global_step},
                        ),
                    ):
                        _t_hf = time.monotonic()
                        self._save_hf(
                            epoch=epoch,
                            epoch_step=step,
                            global_step=global_step,
                        )
                        if self._is_rank0:
                            print(
                                f"[Trainer] [step {global_step}] _save_hf done in "
                                f"{time.monotonic() - _t_hf:.1f}s",
                                flush=True,
                            )

                    with (
                        stats_tracker.record_timing("checkpoint_for_recover"),
                        perf_tracer.trace_scope(
                            "train.checkpoint",
                            category=Category.IO,
                            args={"global_step": global_step},
                        ),
                    ):
                        prev_saved_step = self.recover_handler.last_saved_global_step
                        _t_recover = time.monotonic()
                        self._save_recover_checkpoint(
                            epoch=epoch,
                            epoch_step=step,
                            global_step=global_step,
                        )
                        if self._is_rank0:
                            print(
                                f"[Trainer] [step {global_step}] _save_recover_checkpoint done in "
                                f"{time.monotonic() - _t_recover:.1f}s "
                                f"(saved={self.recover_handler.last_saved_global_step != prev_saved_step})",
                                flush=True,
                            )
                        # Save AstraFlow buffer only when a recover checkpoint
                        # was actually written (same freq_steps gate).
                        if (
                            self._is_rank0
                            and self.recover_handler.last_saved_global_step != prev_saved_step
                        ):
                            _t_buf = time.monotonic()
                            self.astraflow.save_buffer()
                            print(
                                f"[Trainer] [step {global_step}] save_buffer done in "
                                f"{time.monotonic() - _t_buf:.1f}s",
                                flush=True,
                            )
                    if self._is_rank0:
                        _save_elapsed = time.monotonic() - _save_t0
                        print(
                            f"[Trainer] [step {global_step}] Checkpoint saved in "
                            f"{_save_elapsed:.1f}s",
                            flush=True,
                        )

                    # Step 6: Notify AstraFlow of new version.
                    # Trainer decides whether to run eval based on evaluator.freq_steps.
                    # When eval is requested, use synchronous notification so training
                    # blocks until eval completes.  Otherwise use async (fire-and-forget).
                    eval_results = None
                    if self._is_rank0:
                        version = global_step + 1
                        eval_freq = getattr(config.evaluator, "freq_steps", None)
                        should_eval = (
                            eval_freq is not None
                            and version > 0
                            and version % eval_freq == 0
                        )

                        # Wait for async delta compute before notifying
                        # (ensures delta is ready when RaaS pulls).
                        if hasattr(self, "weight_manager") and self.weight_manager is not None:
                            self.weight_manager.wait_delta_ready()

                        if should_eval:
                            print(
                                f"[Trainer] [step {global_step}] Notifying AstraFlow "
                                f"version={version} (sync, eval) ...",
                                flush=True,
                            )
                            with stats_tracker.record_timing("eval"):
                                try:
                                    eval_results = self.astraflow.notify_version(
                                        version=version, run_eval=True,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "[step %d] notify_version (sync, eval) "
                                        "failed: %s — skipping eval, continuing",
                                        global_step, e,
                                    )
                                    eval_results = None
                            print(
                                f"[Trainer] [step {global_step}] AstraFlow eval "
                                f"complete",
                                flush=True,
                            )
                        else:
                            print(
                                f"[Trainer] [step {global_step}] Notifying AstraFlow "
                                f"version={version} (async) ...",
                                flush=True,
                            )
                            try:
                                self.astraflow.notify_version_async(
                                    version=version,
                                )
                            except Exception as e:
                                logger.warning(
                                    "[step %d] notify_version_async failed: %s — "
                                    "continuing (RaaS will catch up on next bump)",
                                    global_step, e,
                                )
                            print(
                                f"[Trainer] [step {global_step}] AstraFlow version "
                                f"notification submitted",
                                flush=True,
                            )
                    _barrier_t0 = time.monotonic()
                    dist.barrier(group=self.actor.cpu_group)
                    if self._is_rank0:
                        _barrier_elapsed = time.monotonic() - _barrier_t0
                        if _barrier_elapsed > 1.0:
                            print(
                                f"[Trainer] [step {global_step}] Barrier waited "
                                f"{_barrier_elapsed:.1f}s",
                                flush=True,
                            )

                    # Step 7: Log buffer + eval + weight transfer stats, then commit all to wandb
                    if self._is_rank0:
                        # Override the service-supplied rollout_wait_fraction
                        # with a trainer-local, per-step value. The service
                        # version is a shared EWMA across all trainers calling
                        # /batch on the same AstraFlow, which degenerates when
                        # two trainers interleave and pegs at 0.95. Our
                        # numerator is the actual /batch wait time for *this*
                        # trainer's step; the denominator is this step's full
                        # wall time — an honest per-step share.
                        _step_elapsed = time.monotonic() - _step_t0
                        if _step_elapsed > 0.0:
                            if buffer_stats is None:
                                buffer_stats = {}
                            buffer_stats["timeperf/rollout_wait_fraction"] = (
                                _batch_elapsed / _step_elapsed
                            )

                        # Weight transfer metrics from offload()
                        if wt_metrics:
                            for k, v in wt_metrics.items():
                                scope, _, metric = k.partition("/")
                                if scope and metric:
                                    with stats_tracker.scope(scope):
                                        stats_tracker.scalar(**{metric: v})

                        # Buffer stats from get_batch
                        if buffer_stats:
                            for k, v in buffer_stats.items():
                                scope, _, metric = k.partition("/")
                                if scope and metric:
                                    with stats_tracker.scope(scope):
                                        stats_tracker.scalar(**{metric: v})

                        # Weight transfer mode from notify_version response
                        wt_info = self.astraflow.get_last_weight_transfer_info()
                        if wt_info and isinstance(wt_info, dict):
                            with stats_tracker.scope("weight_transfer"):
                                stats_tracker.scalar(
                                    use_full=float(wt_info.get("use_full", 1)),
                                )

                        # Eval stats from notify_version
                        if eval_results and isinstance(eval_results, dict):
                            datasets = eval_results.get("datasets", {})
                            all_ks = {
                                int(ds["pass_k"])
                                for ds in datasets.values()
                                if ds.get("pass_k") is not None
                            }
                            has_pass_at_k = any(k > 1 for k in all_ks)
                            for name, ds in datasets.items():
                                avg_k = ds.get("pass_k")
                                avg_metric_name = (
                                    f"avg@{int(avg_k)}"
                                    if avg_k is not None
                                    else "avg@k"
                                )
                                with stats_tracker.scope(f"eval-avg/{name}"):
                                    stats_tracker.scalar(
                                        **{avg_metric_name: ds.get("avg@k", 0.0)},
                                    )
                                pass_at_k = ds.get("pass@k")
                                pass_k = ds.get("pass_k")
                                if pass_at_k is not None and pass_k is not None and int(pass_k) > 1:
                                    with stats_tracker.scope(f"eval-pass/{name}"):
                                        stats_tracker.scalar(
                                            **{f"pass@{int(pass_k)}": pass_at_k},
                                        )
                            if "overall_avg@k" in eval_results:
                                with stats_tracker.scope("eval-avg"):
                                    stats_tracker.scalar(
                                        overall_avg=eval_results["overall_avg@k"],
                                    )
                                if "overall_pass@k" in eval_results and has_pass_at_k:
                                    metric_name = (
                                        f"overall_pass@{next(iter(all_ks))}"
                                        if len(all_ks) == 1
                                        else "overall_pass@k"
                                    )
                                    with stats_tracker.scope("eval-pass"):
                                        stats_tracker.scalar(
                                            **{metric_name: eval_results["overall_pass@k"]},
                                        )

                    with perf_tracer.trace_scope(
                        "train.log_stats",
                        category=Category.INSTR,
                        args={"global_step": global_step},
                    ):
                        self._export_and_commit_stats(
                            epoch=epoch,
                            epoch_step=step,
                            global_step=global_step,
                        )

                perf_tracer.save(step=global_step)
            # Loop exited cleanly — record so _shutdown_and_exit knows
            # this is a real completion (vs an exception path).
            self._completed_normally = True
        except Exception as _exc:
            import traceback
            print(f"[Trainer] EXCEPTION in training loop: {_exc}", flush=True)
            traceback.print_exc()
            raise
        finally:
            self._shutdown_and_exit()

    def _shutdown_and_exit(self) -> None:
        """Send shutdown signals to services and hard-exit the process.

        Called from the training loop's ``finally`` block. After sending
        shutdown signals, uses ``os._exit`` to terminate immediately.
        We cannot call ``super().close()`` (actor/critic destroy) because
        those require NCCL communication with all ranks, and other ranks
        may have already exited — causing a hang.

        Each rank exits independently — no barrier, because a barrier
        can hang if any rank exits before others arrive.
        """
        if self._closed:
            return
        self._closed = True

        if self._is_rank0:
            if self._completed_normally:
                # Real completion — release the cluster.
                # Wait for any in-flight async version notification to finish
                # before sending shutdown — avoids killing inference engines
                # while a weight load is still in progress.
                self.astraflow.drain_pending_notifications()
                print(
                    "[Trainer] Training complete — sending shutdown to services ...",
                    flush=True,
                )
                self.astraflow.shutdown_service()
                self._shutdown_raas()
                print("[Trainer] Shutdown signals sent.", flush=True)
            else:
                # Trainer crashed — leave AstraFlow + RaaS up so a fresh
                # trainer can reconnect (hot-swappable architecture).
                print(
                    "[Trainer] Trainer exiting on error — leaving AstraFlow + RaaS "
                    "running for reconnect.",
                    flush=True,
                )

        self.astraflow.close()
        self.stats_logger.close()
        perf_tracer.save(force=True)

        # Kill child processes (e.g. sender_agent) that would otherwise
        # become orphans and hold ports.
        import multiprocessing
        for child in multiprocessing.active_children():
            child.kill()

        # Hard-exit: avoids hanging on NCCL cleanup in actor.destroy()
        # and prevents torchrun from misinterpreting the exit.
        import os as _os
        _os._exit(0)

    def close(self):
        self._shutdown_and_exit()

    def _shutdown_raas(self) -> None:
        """Send shutdown to RaaS service via HTTP."""
        import requests as _requests

        for url in self._tcp_raas_urls:
            try:
                logger.info("Sending shutdown to RaaS at %s", url)
                _requests.post(
                    f"{url.rstrip('/')}/shutdown",
                    data=b"",
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=10.0,
                )
            except Exception as exc:
                logger.info("RaaS shutdown request finished (exc=%s)", exc)
