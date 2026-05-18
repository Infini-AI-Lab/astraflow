"""AstraFlow HTTP service.

Flask-based HTTP server that exposes the AstraFlow data pipeline as a
service.  Trainers communicate with this service via these main endpoints:

- ``GET  /batch``              — get a training batch (blocks)
- ``POST /notify_version``     — update version (TCP weight transfer)

RaaS instances are managed via a global ``RaaSPool``:

- ``POST /register_raas``   — dynamically add a RaaS instance
- ``POST /deregister_raas`` — remove a RaaS instance
- ``GET  /raas_pool``        — pool status

RaaS instances are added at runtime via the ``/register_raas`` endpoint.
"""

from __future__ import annotations

import logging
import inspect
import os
import threading
from typing import Any

from flask import Flask, Response, jsonify, request

from astraflow.dataflow import AstraFlow
from astraflow.dataflow.eval_manager import EvalManager
from astraflow.dataflow.raas2_engine import (
    dumps_object,
    loads_object,
)
from astraflow.dataflow.raas_pool import RaaSPool
from astraflow.dataflow.service_config import AgentConfig, ServiceConfig

logger = logging.getLogger(__name__)


app = Flask(__name__)


class AstraFlowService:
    """Central service managing per-agent AstraFlow instances."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.flows: dict[str, AstraFlow] = {}
        # Single global RaaSPool shared by all agents.
        self.raas_pool = RaaSPool(
            heartbeat_interval=config.heartbeat_interval,
            heartbeat_max_failures=config.heartbeat_max_failures,
            raas_initialize_timeout=config.raas_initialize_timeout,
        )
        self.agent_configs: dict[str, AgentConfig] = {}
        self.versions: dict[str, int] = {}
        self.eval_manager = EvalManager(
            timeout=config.eval.timeout if config.eval else None,
        )
        self._locks: dict[str, threading.Lock] = {}
        # Readiness tracking: data acquisition starts only when both
        # RaaS and trainer are confirmed ready for an agent.
        self._raas_ready: set[str] = set()
        self._trainer_ready: set[str] = set()
        self._started: set[str] = set()
        # Per-trainer batch sizes, keyed by (agent_name, model_id).
        # For single-model mode model_id is None, key is (agent_name, None).
        self._train_batch_size: dict[tuple[str, str | None], int] = {}

        # Multi-model trainer tracking: discovered from trainer_ready calls.
        # Maps agent_name → set of model_ids that have connected.
        self._registered_model_ids: dict[str, set[str]] = {}
        # Per-model version tracking: (agent_name, model_id) → version
        self._model_versions: dict[tuple[str, str], int] = {}
        # Per-model sender endpoints: (agent_name, model_id) → "host:port"
        self._sender_endpoints: dict[tuple[str, str], str] = {}
        # Models that have actually called /ready (as opposed to being
        # pre-populated via expected_model_ids with a default version of 0).
        self._connected_model_ids: set[tuple[str, str]] = set()

        # Version barrier: each model does its own weight load independently,
        # but all trainers must reach the same version before proceeding to
        # the next training step.
        self._pending_versions: dict[tuple[str, str], int] = {}
        self._pending_eval: dict[tuple[str, str], bool] = {}
        self._version_barrier_lock = threading.Lock()
        self._version_barrier_cond = threading.Condition(self._version_barrier_lock)
        self._version_barrier_generation = 0
        self._barrier_eval_results = None

        # NOTE: Per-model buffers are now used instead of a shared batch cache.
        # Each model's buffer has loss_mask pre-applied at ingest time, so
        # trainers pull independently from their own buffer.

        # Balance report: accumulated stats since last report (reset on each report).
        self._balance_stats_lock = threading.Lock()
        self._balance_consumed: int = 0
        self._balance_stale_skipped: int = 0
        self._balance_iterations: int = 0  # number of batch pulls since last report
        # EWMA of get_batch wait time and full step time (seconds).  Kept
        # across reports so the per-step timeperf/rollout_wait_fraction
        # logged to wandb decays smoothly instead of spiking at reset.
        self._ewma_alpha: float = 0.1
        self._ewma_wait: float = 0.0
        self._ewma_step: float = 0.0
        self._ewma_initialized: bool = False
        # Window-exact sums of wait_sec and step_time (paired per call).
        # Reset each balance report so the report shows the true ratio
        # of (time spent waiting) / (step time) for the just-finished
        # window, while EWMA above powers the per-step wandb metric.
        self._window_wait_sum: float = 0.0
        self._window_step_sum: float = 0.0
        self._window_step_count: int = 0
        # Monotonic time of the previous get_batch completion.  Used to
        # measure step_time = t1 - prev_t1.  Set to None after eval to
        # drop the eval gap from the step_time sample.  NOT cleared at
        # report time — so the first call of a new window still gets a
        # valid step_time from the previous window's last completion.
        self._last_batch_t1: float | None = None
        # Wall-clock window tracking for the report display.
        import time as _time
        self._balance_window_start: float = _time.monotonic()
        self._balance_eval_time: float = 0.0  # accumulated eval seconds in window

        # Auto-save balance report to disk.
        self._balance_report_freq = config.balance_report_freq
        self._balance_report_dir: str | None = None
        if config.checkpoint_dir and self._balance_report_freq > 0:
            # Place balance_reports/ as sibling of checkpoints/
            self._balance_report_dir = os.path.join(
                os.path.dirname(config.checkpoint_dir), "balance_reports"
            )
        self._balance_last_saved_version: int = 0

    def register_agent(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        rollout_dataloader: Any = None,
    ) -> None:
        """Register an agent with its RaaS engine and AstraFlow buffer.

        Parameters
        ----------
        agent_name : str
            Unique agent identifier.
        agent_config : AgentConfig
            Configuration for this agent.
        rollout_dataloader : StatefulDataLoader | None
            Dataloader for rollout data. If None and ``agent_config.rollout_dataset``
            is set, the dataloader will be created automatically from the config.
        """
        self.agent_configs[agent_name] = agent_config
        self._locks[agent_name] = threading.Lock()
        self.versions[agent_name] = 0

        # Mark this agent as RaaS-ready; the global pool handles availability.
        self._raas_ready.add(agent_name)

        # Create rollout dataloader from config if not provided
        if rollout_dataloader is None and agent_config.rollout_dataset is not None:
            rollout_dataloader = self._create_rollout_dataloader(agent_config)

        # Create AstraFlow buffer
        from astraflow.dataflow.utils import NormConfig

        reward_norm = None
        if agent_config.reward_norm is not None:
            reward_norm = NormConfig(**agent_config.reward_norm)

        flow = AstraFlow(
            rollout=self.raas_pool,
            rollout_dataloader=rollout_dataloader,
            workflow_spec=agent_config.workflow_spec,
            buffer_size=agent_config.buffer_size,
            buffer_debug=True,
            filter_fn=agent_config.filter_function,
            max_staleness=agent_config.max_staleness,
            replay_max_size=agent_config.replay_size,
            replay_selection_fn=agent_config.selection_fn,
            reward_norm=reward_norm,
            expected_model_ids=agent_config.expected_model_ids,
            curator=agent_config.curator,
            curator_args=agent_config.curator_args,
        )
        self.flows[agent_name] = flow

        # Restore buffer from checkpoint if available
        self.load_buffer(agent_name)

        # Configure eval if eval datasets are provided
        if agent_config.eval_datasets is not None:
            tokenizer = self._load_tokenizer(agent_config)
            eval_datasets = self._create_eval_datasets(agent_config, tokenizer)
            self.eval_manager.configure_agent(
                agent_name=agent_name,
                eval_datasets=eval_datasets,
            )

        # Pre-populate expected model_ids so the version barrier knows the
        # full set before trainers connect.
        if agent_config.expected_model_ids:
            self._registered_model_ids[agent_name] = set(agent_config.expected_model_ids)
            for mid in agent_config.expected_model_ids:
                self._model_versions[(agent_name, mid)] = 0
            print(
                f"[{agent_name}] Expecting trainers for model_ids: "
                f"{agent_config.expected_model_ids}",
                flush=True,
            )

        logger.info(
            "Registered agent %s (pool_size=%d)",
            agent_name,
            self.raas_pool.size(),
        )

    def _load_tokenizer(self, agent_config: AgentConfig) -> Any:
        """Load tokenizer from agent config."""
        tokenizer_path = agent_config.tokenizer_path
        if tokenizer_path is None:
            raise ValueError(
                "tokenizer_path must be set in agent config to create datasets"
            )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _create_rollout_dataloader(self, agent_config: AgentConfig) -> Any:
        """Create rollout dataloader from agent config.

        The AstraFlow service runs as a single process, so ``rank=0,
        world_size=1`` — no DP sharding for data acquisition.
        """
        ds_cfg = agent_config.rollout_dataset
        tokenizer = self._load_tokenizer(agent_config)
        dataset = _create_dataset_from_config(ds_cfg, tokenizer)

        batch_size = ds_cfg.get("batch_size", 1)
        return _create_dataloader(dataset, batch_size=batch_size)

    def _create_eval_datasets(
        self,
        agent_config: AgentConfig,
        tokenizer: Any,
    ) -> dict[str, tuple[Any, int, Any]]:
        """Create eval datasets from agent config.

        Returns dict of {name: (dataset, repeat, workflow_spec)} for the
        EvalManager.  Workflow specs are resolved from ``eval_workflows``
        (new style) or ``eval_workflow_specs`` (deprecated fallback).
        """
        eval_workflows = agent_config.eval_workflows
        legacy_specs = agent_config.eval_workflow_specs

        if eval_workflows is None and legacy_specs is not None:
            # Backward compat: convert old-style eval_workflow_specs to
            # the new eval_workflows format.  The old format keyed specs
            # by "{dataset}_{run_idx}" or "default".  We collapse them
            # to per-dataset specs by preferring "{dataset}_0" then
            # "default".
            logger.warning(
                "eval_workflow_specs is deprecated — migrate to "
                "eval_workflows + per-dataset eval_workflow"
            )
            eval_workflows = {}
            default_spec = legacy_specs.get("default")
            for ds_name in agent_config.eval_datasets:
                run0_key = f"{ds_name}_0"
                spec = legacy_specs.get(run0_key, default_spec)
                if spec is not None:
                    eval_workflows[ds_name] = spec

        eval_datasets: dict[str, tuple[Any, int, Any]] = {}
        for name, ds_cfg in agent_config.eval_datasets.items():
            ds_cfg = dict(ds_cfg)  # shallow copy to avoid mutating config
            repeat = ds_cfg.pop("repeat", ds_cfg.pop("k", 1))
            wf_name = ds_cfg.pop("eval_workflow", None)

            if wf_name is not None:
                if eval_workflows is None or wf_name not in eval_workflows:
                    raise ValueError(
                        f"Eval dataset '{name}' references eval_workflow "
                        f"'{wf_name}', but it is not defined in eval_workflows"
                    )
                wf = eval_workflows[wf_name]
            elif eval_workflows is not None and name in eval_workflows:
                # Backward compat: auto-converted legacy specs keyed by
                # dataset name.
                wf = eval_workflows[name]
            else:
                raise ValueError(
                    f"Eval dataset '{name}' has no eval_workflow assigned "
                    f"and no legacy eval_workflow_specs fallback is available"
                )

            dataset = _create_dataset_from_config(ds_cfg, tokenizer)
            eval_datasets[name] = (dataset, repeat, wf)

        return eval_datasets

    def start_agent(self, agent_name: str) -> None:
        """Start data acquisition for an agent."""
        self.flows[agent_name].start()
        print(f"[{agent_name}] Both ready — data acquisition started!", flush=True)

    def start_all(self) -> None:
        """Start data acquisition for all registered agents."""
        for name in self.flows:
            self.start_agent(name)

    def stop_all(self) -> None:
        """Stop all agents."""
        for name, flow in self.flows.items():
            flow.close()
            logger.info("Stopped agent %s", name)

    def _try_start(self, agent_name: str) -> None:
        """Start data acquisition if both RaaS and all expected trainers are ready."""
        if agent_name in self._started:
            return
        if agent_name not in self._raas_ready:
            return
        if agent_name not in self._trainer_ready:
            return

        # If expected_model_ids is configured, wait for all of them.
        agent_cfg = self.agent_configs.get(agent_name)
        expected = agent_cfg.expected_model_ids if agent_cfg else None
        if expected is not None:
            registered = self._registered_model_ids.get(agent_name, set())
            missing = set(expected) - registered
            if missing:
                print(
                    f"[{agent_name}] Waiting for trainers: "
                    f"registered={registered}, expected={set(expected)}, "
                    f"missing={missing}",
                    flush=True,
                )
                return

        # Recovery weight loads are handled per-model in trainer_ready().
        self._started.add(agent_name)
        self.start_agent(agent_name)

    def _buffer_checkpoint_path(self, agent_name: str) -> str | None:
        """Return the buffer checkpoint file path, or None if not configured."""
        if self.config.checkpoint_dir is None:
            return None
        return os.path.join(self.config.checkpoint_dir, f"{agent_name}_buffer.pkl")

    def save_buffer(self, agent_name: str) -> None:
        """Save buffer state for an agent. No-op if checkpoint_dir is not set."""
        path = self._buffer_checkpoint_path(agent_name)
        if path is None:
            return
        self.flows[agent_name].save_buffer(path)

    def load_buffer(self, agent_name: str) -> bool:
        """Load buffer state for an agent. Returns False if no checkpoint found."""
        path = self._buffer_checkpoint_path(agent_name)
        if path is None:
            return False
        return self.flows[agent_name].load_buffer(path)

    def trainer_ready(
        self,
        agent_name: str,
        train_batch_size: int | None = None,
        model_id: str | None = None,
        sender_endpoint: str | None = None,
        recovered_version: int | None = None,
    ) -> None:
        """Mark a trainer as ready for this agent.

        Called when the trainer sends ``POST /ready/{agent_name}``.
        If RaaS is also ready, data acquisition starts immediately.

        Parameters
        ----------
        train_batch_size : int | None
            Number of examples (sequences) per training batch.
        model_id : str | None
            Model identifier for multi-model training.  Multiple trainers
            can connect to the same agent with different ``model_id`` values.
            When None, single-model mode is assumed.
        sender_endpoint : str | None
            TCP weight transfer sender agent endpoint (``"host:port"``).
            AstraFlow forwards this to RaaS during coordinated version
            notifications so RaaS knows where to pull each model's weights.
        recovered_version : int | None
            If the trainer recovered from a checkpoint, the version to
            resume from.  Sets the internal version counter so that
            staleness filtering works correctly after recovery.
        """
        trainer_key = (agent_name, model_id)
        if train_batch_size is not None:
            self._train_batch_size[trainer_key] = train_batch_size
            print(
                f"[{agent_name}:{model_id}] train_batch_size={train_batch_size}",
                flush=True,
            )

        # If recovering from a checkpoint, set the service version to the
        # recovered version.  This prevents a version jump (0 → N) that
        # would cause all buffered data to be evicted as stale.
        if recovered_version is not None:
            prev = self.versions.get(agent_name, 0)
            self.versions[agent_name] = recovered_version
            # Update pool version so new rollout data is tagged correctly.
            self.raas_pool.set_version_local(recovered_version)
            # Seed the acquisition curator with the recovered version too.
            try:
                flow_for_hook = self.flows.get(agent_name)
                if flow_for_hook is not None:
                    flow_for_hook.data_acquisition.notify_version_changed(
                        recovered_version
                    )
            except Exception:
                logger.exception(
                    "[%s] curator notify_version_changed (recovery) failed",
                    agent_name,
                )
            print(
                f"[{agent_name}] Recovery: version set to {recovered_version} "
                f"(was {prev})",
                flush=True,
            )

        # Register model_id for version barrier tracking.
        # Single-model trainers (model_id=None) are normalised to "default"
        # so that the same coordinated weight-load path handles both cases.
        effective_model_id = model_id if model_id is not None else "default"
        # Validate against expected_model_ids if configured
        agent_cfg = self.agent_configs.get(agent_name)
        expected = agent_cfg.expected_model_ids if agent_cfg else None
        if expected is not None and effective_model_id not in expected:
            logger.warning(
                "Trainer registered with unexpected model_id=%r "
                "(expected=%s) for agent %s",
                effective_model_id,
                expected,
                agent_name,
            )
        if agent_name not in self._registered_model_ids:
            self._registered_model_ids[agent_name] = set()
        self._registered_model_ids[agent_name].add(effective_model_id)
        init_version = recovered_version if recovered_version is not None else 0
        self._model_versions[(agent_name, effective_model_id)] = init_version
        if sender_endpoint is not None:
            self._sender_endpoints[(agent_name, effective_model_id)] = sender_endpoint
        self._connected_model_ids.add((agent_name, effective_model_id))
        print(
            f"[{agent_name}] Trainer connected (model_id={effective_model_id}, "
            f"sender={sender_endpoint}, "
            f"registered={self._registered_model_ids[agent_name]}, "
            f"version={init_version})",
            flush=True,
        )

        # In multi-model setups, all models must start at the same version.
        # If one model recovered from a checkpoint (e.g. v=25) while another
        # starts fresh (v=0), the version barrier will deadlock because the
        # models can never reach the same version simultaneously.  Fail fast.
        # Only check models that have actually called /ready — skip models
        # that were pre-populated via expected_model_ids but haven't
        # connected yet (their default v=0 is not a real version).
        other_versions = {
            mid: self._model_versions.get((agent_name, mid), 0)
            for mid in self._registered_model_ids.get(agent_name, set())
            if mid != effective_model_id
            and (agent_name, mid) in self._connected_model_ids
        }
        if other_versions:
            mismatched = {
                mid: v for mid, v in other_versions.items()
                if v != init_version
            }
            if mismatched:
                msg = (
                    f"[{agent_name}:{effective_model_id}] Version mismatch at "
                    f"startup: this model at v={init_version}, but other "
                    f"model(s) at {mismatched}. All models must start at the "
                    f"same version — check recovery checkpoints."
                )
                print(msg, flush=True)
                raise RuntimeError(msg)

        # Per-model recovery weight load: each trainer pushes its own
        # recovered weights to RaaS independently as it connects.
        if recovered_version is not None and recovered_version > 0 and sender_endpoint:
            print(
                f"[{agent_name}:{effective_model_id}] Recovery: loading "
                f"weights (version={recovered_version}) ...",
                flush=True,
            )
            try:
                self._trigger_raas_weight_load_single(
                    agent_name, effective_model_id, recovered_version,
                )
                print(
                    f"[{agent_name}:{effective_model_id}] Recovery: weight "
                    f"load complete",
                    flush=True,
                )
            except Exception:
                logger.exception(
                    "Recovery: weight load failed for %s:%s",
                    agent_name, effective_model_id,
                )

        self._trainer_ready.add(agent_name)
        self._try_start(agent_name)

    def get_batch(
        self,
        agent_name: str,
        model_id: str | None = None,
        trainer_version: int | None = None,
    ) -> dict[str, Any]:
        """Get a training batch + buffer stats for an agent.

        Returns ``{"batch": <tensor_dict>, "buffer_stats": <dict>}``.
        The trainer logs ``buffer_stats`` to wandb.

        In multi-model mode, each model has its own per-model buffer with
        ``loss_mask`` pre-applied at ingest time, so trainers pull
        independently without any shared cache or deepcopy.

        If ``trainer_version`` is provided and this is a multi-model agent,
        blocks until all other trainers have caught up so that no trainer
        can race ahead.

        Parameters
        ----------
        model_id : str | None
            If provided, pulls from the per-model buffer for this model.
        trainer_version : int | None
            The trainer's current global step.
        """
        # Block if this trainer is ahead of the service version (other
        # trainers haven't completed their version barrier yet).
        if trainer_version is not None and model_id is not None:
            service_version = self.versions.get(agent_name, 0)
            if trainer_version > service_version:
                print(
                    f"[{agent_name}:{model_id}] Trainer at v={trainer_version} "
                    f"ahead of service v={service_version}, waiting ...",
                    flush=True,
                )
                with self._version_barrier_cond:
                    while self.versions.get(agent_name, 0) < trainer_version:
                        self._version_barrier_cond.wait(timeout=5.0)
                print(
                    f"[{agent_name}:{model_id}] Service caught up to "
                    f"v={self.versions.get(agent_name, 0)}",
                    flush=True,
                )

        version = self._get_version(agent_name, model_id)
        print(f"[{agent_name}:{model_id}] Trainer requesting batch (v={version}) ...", flush=True)

        import time as _time
        t0 = _time.monotonic()
        batch, buffer_stats = self._pull_batch(
            agent_name, model_id, version,
            self.flows[agent_name], self.agent_configs[agent_name],
        )
        t1 = _time.monotonic()
        wait_sec = t1 - t0

        # Update EWMAs of wait time and step time, then expose the
        # rollout_wait_fraction via buffer_stats so the trainer can log it
        # to wandb under timeperf/.  EWMA (vs the old cumulative ratio)
        # avoids the 0.95 spike right after a balance-report reset where
        # the window's wall time was essentially just the wait itself.
        with self._balance_stats_lock:
            self._balance_consumed += int(buffer_stats.get("buffer/consumed", 0))
            self._balance_stale_skipped += int(buffer_stats.get("buffer/skipped_stale", 0))
            self._balance_iterations += 1

            if self._last_batch_t1 is not None:
                step_time = t1 - self._last_batch_t1
                # Window-exact paired sums for the balance report.
                self._window_wait_sum += wait_sec
                self._window_step_sum += step_time
                self._window_step_count += 1
                # Smoothed EWMA for the per-step wandb metric.
                if self._ewma_initialized:
                    a = self._ewma_alpha
                    self._ewma_wait = a * wait_sec + (1.0 - a) * self._ewma_wait
                    self._ewma_step = a * step_time + (1.0 - a) * self._ewma_step
                else:
                    self._ewma_wait = wait_sec
                    self._ewma_step = step_time
                    self._ewma_initialized = True
            self._last_batch_t1 = t1

            if self._ewma_initialized and self._ewma_step > 0.0:
                rollout_wait_fraction = min(self._ewma_wait / self._ewma_step, 0.95)
            else:
                rollout_wait_fraction = 0.0

        buffer_stats["timeperf/rollout_wait_fraction"] = rollout_wait_fraction

        print(f"[{agent_name}:{model_id}] Batch served (v={version})", flush=True)
        return {"batch": batch, "buffer_stats": buffer_stats}

    def _pull_batch(
        self,
        agent_name: str,
        model_id: str | None,
        version: int,
        flow: Any,
        agent_cfg: Any,
    ) -> tuple[dict, dict]:
        """Pull a fresh batch from the buffer and collect stats.

        In multi-model mode, ``model_id`` routes to the per-model buffer.
        """
        trainer_key = (agent_name, model_id)
        batch_size = self._train_batch_size.get(
            trainer_key,
            self._train_batch_size.get((agent_name, None), 16),
        )
        batch, _metadatas = flow.get_training_batch(
            expected_sample_count=batch_size,
            replay_ratio=agent_cfg.replay_ratio,
            timeout=None,
            current_version=version,
            model_id=model_id,
        )

        # Collect buffer stats for trainer's wandb logging.
        # All stats are per-model (pushed from acquisition into each
        # per-model buffer), so no shared-state race conditions.
        acq = flow.get_and_reset_acquisition_stats(model_id=model_id)
        consume = flow.get_and_reset_consume_stats(model_id=model_id)

        total = acq.get("total", 0)
        accepted = acq.get("accepted", 0)
        pre_count = int(acq.get("pre_filter_reward_count", 0))
        pre_sum = float(acq.get("pre_filter_reward_sum", 0.0))
        post_count = int(acq.get("post_filter_reward_count", 0))
        post_sum = float(acq.get("post_filter_reward_sum", 0.0))

        buffer_stats = {
            "buffer/buffer_size": float(flow.size(model_id)),
            "buffer/replay_buffer_size": float(flow.replay_size(model_id)),
            "buffer/accepted": float(accepted),
            "buffer/filtered": float(acq.get("filtered", 0)),
            "buffer/total": float(total),
            "buffer/accept_rate": (accepted / total) * 100.0 if total > 0 else 0.0,
            "buffer/consumed": float(consume.get("consumed", 0)),
            "buffer/skipped_stale": float(consume.get("skipped_stale", 0)),
        }
        # Only include reward means when we have data — avoids logging
        # misleading 0.0 on the first step after recovery (buffer serves
        # pre-loaded data before new acquisition stats accumulate).
        if pre_count > 0:
            buffer_stats["rollout/pre_filter_reward_mean"] = pre_sum / pre_count
        if post_count > 0:
            buffer_stats["rollout/post_filter_reward_mean"] = post_sum / post_count

        # Include workflow-defined agent metrics (prefixed with "agent/").
        # Each per-model buffer has its own copy of the accumulated metrics,
        # so every model trainer gets the same values independently.
        agent_metrics = flow.get_and_reset_agent_metrics(model_id=model_id)
        for k, v in agent_metrics.items():
            buffer_stats[f"agent/{k}"] = v

        # RaaS pool stats — logged to wandb under "raas/" section.
        buffer_stats["raas/pool_size"] = float(self.raas_pool.size())
        buffer_stats["raas/total_gpus"] = float(self.raas_pool.total_gpu_count())

        # Curator (selective rollout) stats — only emitted when enabled.
        # Drained per /batch call so the windowed view matches one trainer step.
        try:
            if flow.data_acquisition.has_curator():
                cur = flow.data_acquisition.get_curator_stats(reset=True)
                selected = float(cur.get("selected", 0))
                rejected = float(cur.get("rejected", 0))
                forced = float(cur.get("forced", 0))
                buffer_stats["acquisition/selected"] = selected
                buffer_stats["acquisition/rejected"] = rejected
                buffer_stats["acquisition/forced"] = forced
                # Fraction of curator decisions in this window that said
                # "no" (excludes force-submits, since those override the
                # curator's decision). Range [0, 1].
                seen = selected + rejected
                buffer_stats["acquisition/reject_ratio"] = (
                    rejected / seen if seen > 0 else 0.0
                )
                # Optional curator-defined telemetry (e.g. GRESO's p_easy /
                # p_hard / observed_*_ratio). Each key is namespaced under
                # "acquisition/" so wandb groups them with the counters.
                tele = flow.data_acquisition.get_curator_telemetry()
                for k, v in tele.items():
                    buffer_stats[f"acquisition/{k}"] = float(v)
        except Exception:
            logger.exception("failed to drain curator stats")

        return batch, buffer_stats

    def generate_balance_report(self, agent_name: str = "default") -> str:
        """Generate a text balance report for elastic RaaS scaling decisions.

        The report has two parts:
        1. Window-level timing, production, and scaling decision.
        2. Per-RaaS instance layout with throughput/gpu.

        Decision rule (three-zone with dead band):
          - If ``rollout_wait_fraction > WAIT_HIGH`` (0.10): scale up via
            ``G_target = ceil(G / (1 - rollout_wait_fraction))``.
          - Else if ``rollout_wait_fraction < WAIT_LOW`` (0.05) and we saw
            both production and consumption: scale down via
            ``G_target = min(G, ceil(G * consumed / entered * SHRINK_PAD))``.
          - Else (WAIT_LOW <= wait <= WAIT_HIGH, "dead zone"): hold.

        ``stale_skipped`` is reported as advisory only and does not enter
        the scaling math.
        """
        import math

        WAIT_HIGH = 0.10
        WAIT_LOW = 0.05
        SHRINK_PAD = 1.10

        flow = self.flows.get(agent_name)
        if flow is None:
            return f"Error: unknown agent '{agent_name}'"

        # --- Part 1: window aggregates ---
        # Get per-RaaS stats (resets each report).
        per_raas = flow.get_per_raas_stats(reset=True)
        instances = self.raas_pool.list_instances()
        total_gpus = self.raas_pool.total_gpu_count()

        # Sum produced/accepted from per-RaaS stats.
        total_produced = sum(s.get("produced", 0) for s in per_raas.values())
        total_entered = sum(s.get("accepted", 0) for s in per_raas.values())

        # Snapshot window counters and reset.  EWMA state is kept across
        # reports (powers the per-step wandb metric); window sums are
        # reset here so the report shows the exact ratio for this window.
        import time as _time
        now = _time.monotonic()
        with self._balance_stats_lock:
            window_consumed = self._balance_consumed
            window_stale = self._balance_stale_skipped
            window_iterations = self._balance_iterations
            window_wall_time = now - self._balance_window_start
            window_eval_time = self._balance_eval_time
            window_wait_sum = self._window_wait_sum
            window_step_sum = self._window_step_sum
            window_step_count = self._window_step_count
            # Reset window counters (EWMA state preserved).
            self._balance_consumed = 0
            self._balance_stale_skipped = 0
            self._balance_iterations = 0
            self._balance_window_start = now
            self._balance_eval_time = 0.0
            self._window_wait_sum = 0.0
            self._window_step_sum = 0.0
            self._window_step_count = 0

        avg_wait = (
            window_wait_sum / window_step_count if window_step_count > 0 else 0.0
        )
        accept_rate = total_entered / total_produced if total_produced > 0 else 0.0
        throughput_per_gpu = total_entered / total_gpus if total_gpus > 0 else 0.0
        produce_consume_ratio = (
            total_entered / window_consumed if window_consumed > 0 else float("inf")
        )
        stale_rate = window_stale / total_entered if total_entered > 0 else 0.0

        # --- Timing and rollout_wait_fraction ---
        # rollout_wait_fraction = sum(wait_sec) / sum(step_time) over all
        # paired samples in this window.  Each pair is (wait_k, step_k)
        # where step_k = t1_k - t1_{k-1}, so the ratio is exactly the
        # share of wall time the trainer spent blocked in get_batch for
        # the steps in this window.  First call after service start /
        # after eval is dropped (no prev_t1 available) — at most one
        # sample lost per eval boundary.  This is used both for display
        # and the scaling decision below.
        training_time = max(window_wall_time - window_eval_time, 1.0)
        avg_step_time = (
            window_step_sum / window_step_count if window_step_count > 0 else 0.0
        )
        if window_step_sum > 0.0:
            rollout_wait_fraction = min(window_wait_sum / window_step_sum, 0.95)
        else:
            rollout_wait_fraction = 0.0

        # --- Scaling decision (three-zone with dead band [WAIT_LOW, WAIT_HIGH]) ---
        if rollout_wait_fraction > WAIT_HIGH and total_gpus > 0:
            branch = "scale_up"
            G_target = math.ceil(total_gpus / (1.0 - rollout_wait_fraction))
        elif (
            rollout_wait_fraction < WAIT_LOW
            and total_entered > 0
            and window_consumed > 0
            and total_gpus > 0
        ):
            branch = "scale_down"
            G_target = min(
                total_gpus,
                math.ceil(
                    total_gpus * (window_consumed / total_entered) * SHRINK_PAD
                ),
            )
        else:
            branch = "hold"
            G_target = total_gpus

        estimated_delta = G_target - total_gpus
        weight_active = self.raas_pool.is_weight_transfer_active()

        lines = [
            f"--- Window (last {window_iterations} iterations) ---",
            f"wall_time_sec          : {window_wall_time:.1f}",
            f"eval_time_sec          : {window_eval_time:.1f}",
            f"training_time_sec      : {training_time:.1f}",
            f"avg_step_time_sec      : {avg_step_time:.3f}",
            f"avg_batch_wait_sec     : {avg_wait:.3f}",
            f"rollout_wait_fraction  : {rollout_wait_fraction:.3f}",
            "",
            "--- Production ---",
            f"total_raas_gpus        : {total_gpus}",
            f"produced               : {total_produced}",
            f"entered                : {total_entered}",
            f"  accept_rate          : {accept_rate:.4f}",
            f"consumed               : {window_consumed}",
            f"stale_skipped          : {window_stale}",
            f"  stale_rate           : {stale_rate:.3f}",
            f"throughput_per_gpu     : {throughput_per_gpu:.2f}",
            f"produce_consume_ratio  : {produce_consume_ratio:.3f}",
            "",
            "--- Scaling decision ---",
            f"branch                 : {branch}",
            f"G_target               : {G_target}",
            f"estimated_delta_gpus   : {estimated_delta:+d}",
            f"weight_transfer_active : {str(weight_active).lower()}",
            "",
        ]

        # --- Part 2: Per-instance layout ---
        # Build gpu_count lookup from instances list.
        gpu_map = {inst["uid"]: inst.get("gpu_count", 0) for inst in instances}
        all_uids = set(per_raas.keys()) | set(gpu_map.keys())

        lines.append(
            f"--- RaaS Instance Layout (last {window_iterations} iterations) ---"
        )
        header = (
            f"{'uid':<16}{'gpus':>6}{'produced':>12}{'accepted':>12}"
            f"{'accept_rate':>14}{'throughput/gpu':>16}{'status':>10}"
        )
        lines.append(header)

        total_inst = 0
        total_inst_gpus = 0
        total_inst_produced = 0
        total_inst_accepted = 0

        for uid in sorted(all_uids):
            gpus = gpu_map.get(uid, 0)
            stats = per_raas.get(uid, {"produced": 0, "accepted": 0, "filtered": 0})
            produced = stats["produced"]
            accepted = stats["accepted"]
            ar = accepted / produced if produced > 0 else 0.0
            tpg = accepted / gpus if gpus > 0 else 0.0
            status = "healthy"
            # Check if instance is suspect or missing from current pool.
            inst_info = next((i for i in instances if i["uid"] == uid), None)
            if inst_info is None:
                status = "gone"
            elif inst_info.get("suspect"):
                status = "suspect"

            lines.append(
                f"{uid:<16}{gpus:>6}{produced:>12}{accepted:>12}"
                f"{ar:>14.3f}{tpg:>16.1f}{status:>10}"
            )
            total_inst += 1
            total_inst_gpus += gpus
            total_inst_produced += produced
            total_inst_accepted += accepted

        lines.append("---")
        lines.append(
            f"Total: {total_inst} instances, {total_inst_gpus} GPUs, "
            f"{total_inst_produced} produced, {total_inst_accepted} accepted"
        )

        return "\n".join(lines)

    def _save_balance_report(self, agent_name: str, version: int) -> None:
        """Auto-save balance report to disk."""
        try:
            os.makedirs(self._balance_report_dir, exist_ok=True)
            report = self.generate_balance_report(agent_name)
            path = os.path.join(
                self._balance_report_dir, f"balance_report_v{version}.txt"
            )
            with open(path, "w") as f:
                f.write(report)
            self._balance_last_saved_version = version
            print(
                f"[AstraFlow] Balance report saved: {path}\n{report}",
                flush=True,
            )
        except Exception:
            logger.warning(
                "Failed to save balance report for v=%d", version, exc_info=True
            )

    def _get_expected_model_ids(self, agent_name: str) -> list[str] | None:
        """Return the expected model_ids list for an agent, or None if single-model."""
        agent_cfg = self.agent_configs.get(agent_name)
        if agent_cfg and agent_cfg.expected_model_ids:
            return agent_cfg.expected_model_ids
        return None

    def _get_version(self, agent_name: str, model_id: str | None = None) -> int:
        """Get the current version for an agent/model_id pair."""
        if model_id is not None:
            return self._model_versions.get((agent_name, model_id), 0)
        return self.versions.get(agent_name, 0)

    def _trigger_raas_weight_load_single(
        self,
        agent_name: str,
        model_id: str,
        version: int,
    ) -> dict[str, Any]:
        """Trigger weight load for a single model on all RaaS instances."""
        import time as _time

        sender_endpoint = self._sender_endpoints.get((agent_name, model_id), "")
        if not sender_endpoint:
            logger.warning(
                "No sender_endpoint for %s:%s, skipping weight load",
                agent_name, model_id,
            )
            return {}

        _t0 = _time.monotonic()
        print(
            f"[{agent_name}:{model_id}] notify_version v={version} "
            f"to pool (size={self.raas_pool.size()}) ...",
            flush=True,
        )
        result = self.raas_pool.notify_version(model_id, version, sender_endpoint)
        _elapsed = _time.monotonic() - _t0
        print(
            f"[{agent_name}:{model_id}] notify_version v={version} completed "
            f"in {_elapsed:.1f}s across {len(result)} instance(s)",
            flush=True,
        )
        return result

    def catchup_raas(self, uid: str) -> dict[str, Any]:
        """Immediately sync a newly registered RaaS with current weights.

        Iterates over all known (agent, model) versions and pushes weights
        to the specified RaaS instance. This avoids the new RaaS serving
        stale base-model rollouts until the next training step completes.
        """
        import time as _time

        # Get the engine for this uid from the pool
        with self.raas_pool._lock:
            engine = self.raas_pool._engines.get(uid)
        if engine is None:
            logger.warning("catchup_raas: uid=%s not in pool, skipping", uid)
            return {}

        results: dict[str, Any] = {}
        for (agent_name, model_id), version in self._model_versions.items():
            if version <= 0:
                continue
            sender = self._sender_endpoints.get((agent_name, model_id))
            if not sender:
                continue
            print(
                f"[catchup] Syncing uid={uid} model={model_id} to v={version} "
                f"from {sender} ...",
                flush=True,
            )
            _t0 = _time.monotonic()
            try:
                result = self.raas_pool._notify_one_model(
                    uid, engine, model_id, version, sender,
                )
                _elapsed = _time.monotonic() - _t0
                print(
                    f"[catchup] uid={uid} model={model_id} v={version} "
                    f"synced in {_elapsed:.1f}s",
                    flush=True,
                )
                results[model_id] = result
            except Exception as exc:
                logger.error(
                    "catchup_raas: failed for uid=%s model=%s: %s",
                    uid, model_id, exc,
                )
                results[model_id] = {"ok": False, "error": str(exc)}
        return results

    def notify_version(
        self,
        agent_name: str,
        version: int,
        run_eval: bool = False,
        model_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Update internal version for TCP weight transfer mode.

        Two phases:
        1. **Per-model weight load** — each model independently pulls its
           weights to RaaS (no waiting for other models).
        2. **Version barrier** — all trainers must reach the same version
           before any of them can proceed to the next training step.
           The last trainer to arrive (leader) runs eval if requested.
        """
        effective_model_id = model_id if model_id is not None else "default"

        # Phase 1: Per-model weight load
        # For eval steps, DEFER weight loading to the Phase 2 leader so that
        # RaaS is not put in an inconsistent state (some models updated,
        # others not) while the version barrier waits.  Loading eagerly here
        # caused a deadlock in multi-model setups: the first model's sync
        # weight load disrupted rollout generation, starving the second
        # model of data and preventing it from ever reaching the barrier.
        # For normal (non-eval) steps, fire-and-forget — the staleness
        # filter (max_staleness) handles any briefly-stale rollouts.
        weight_transfer_info = None
        if run_eval:
            print(
                f"[{agent_name}:{effective_model_id}] notify_version v={version} "
                f"— weight load deferred (eval, barrier-first) ...",
                flush=True,
            )
        else:
            import threading
            print(
                f"[{agent_name}:{effective_model_id}] notify_version v={version} "
                f"— loading weights (async, fire-and-forget) ...",
                flush=True,
            )
            threading.Thread(
                target=self._trigger_raas_weight_load_single,
                args=(agent_name, effective_model_id, version),
                daemon=True,
            ).start()

        # Update per-model version tracking
        self._model_versions[(agent_name, effective_model_id)] = version

        # Phase 2: Version barrier — wait for all models to reach this version
        expected_model_ids = self._registered_model_ids.get(agent_name, set())
        is_leader = False

        with self._version_barrier_cond:
            my_gen = self._version_barrier_generation
            self._pending_versions[(agent_name, effective_model_id)] = version
            if run_eval:
                self._pending_eval[(agent_name, effective_model_id)] = True

            # Check if all registered model_ids have reported the same version
            pending_for_agent = {
                mid: v
                for (aname, mid), v in self._pending_versions.items()
                if aname == agent_name
            }
            all_same = (
                len(pending_for_agent) >= len(expected_model_ids)
                and len(set(pending_for_agent.values())) == 1
            )
            if all_same:
                is_leader = True

        eval_results = None
        if is_leader:
            print(
                f"[{agent_name}:{effective_model_id}] version barrier: "
                f"all models at v={version}",
                flush=True,
            )

            # Check if ANY model in this agent requested eval.
            any_eval = any(
                self._pending_eval.get((agent_name, mid), False)
                for mid in expected_model_ids
            )

            # ── Eval sequence ──
            # The order is critical for correctness:
            #   1. Pause data acquisition (stop new task submissions)
            #   2. Clear suspects (ensure RaaS visible for reset)
            #   3. Reset training engine (cancel workflow coroutines,
            #      which otherwise re-submit SGLang requests after abort)
            #   4. Load ALL models' weights (fast — 0 inflight)
            #   5. Run eval
            #   6. Resume data acquisition
            if any_eval:
                import time as _time
                _eval_t0 = _time.monotonic()
                flow = self.flows[agent_name]

                # Step 1: Pause data acquisition
                print(
                    f"[{agent_name}:{effective_model_id}] eval: pausing "
                    f"data acquisition ...",
                    flush=True,
                )
                flow.pause()

                # Step 2: Clear suspect flags so reset_training_engine
                # can see the RaaS instances.
                self.raas_pool.clear_all_suspects(
                    reason="eval-leader pre-reset"
                )

                # Step 3: Cancel all workflow tasks inside RaaS.
                # This kills the arun_episode coroutines that would
                # otherwise re-submit requests to SGLang after
                # pause_generation aborts them.  Without this, the
                # event loop stays saturated and weight loads take
                # 5+ minutes instead of ~30s.
                print(
                    f"[{agent_name}:{effective_model_id}] eval: resetting "
                    f"training engine ...",
                    flush=True,
                )
                try:
                    self.raas_pool.reset_training_engine(timeout=10.0)
                except Exception as reset_exc:
                    logger.warning(
                        "[%s] reset_training_engine raised: %s",
                        agent_name,
                        reset_exc,
                        exc_info=True,
                    )

                # Step 4: Load ALL models' weights synchronously.
                # With 0 inflight (reset killed everything), the
                # pause→load→resume cycle is fast (~15s per model).
                # Skip at version=0: RaaS already serves the initial checkpoint
                # (pre-training eval_at_start path), no trainer-side sender is
                # ready yet, so a pull would stall or fail.
                for mid in sorted(expected_model_ids):
                    if version == 0:
                        print(
                            f"[{agent_name}:{mid}] skipping weight load at v=0 "
                            f"(initial checkpoint already loaded)",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{agent_name}:{mid}] loading weights v={version} "
                        f"(sync, eval-leader) ...",
                        flush=True,
                    )
                    load_result = self._trigger_raas_weight_load_single(
                        agent_name, mid, version,
                    )
                    if mid == effective_model_id and isinstance(load_result, dict):
                        for uid, uid_result in load_result.items():
                            if isinstance(uid_result, dict):
                                pr = uid_result.get("pull_result", {})
                                if isinstance(pr, dict) and "mode" in pr:
                                    weight_transfer_info = {
                                        "use_full": 1 if pr["mode"] == "full" else 0,
                                    }
                                    break

            # Update service-level version
            self.versions[agent_name] = version
            self.raas_pool.set_version_local(version)

            # Forward to the agent's acquisition curator so it can
            # invalidate version-dependent state. No-op if no curator.
            try:
                flow_for_hook = self.flows.get(agent_name)
                if flow_for_hook is not None:
                    flow_for_hook.data_acquisition.notify_version_changed(version)
            except Exception:
                logger.exception(
                    "[%s] curator notify_version_changed failed",
                    agent_name,
                )

            # Auto-save balance report at configured frequency.
            if (
                self._balance_report_dir
                and self._balance_report_freq > 0
                and version > 0
                and version % self._balance_report_freq == 0
                and version > self._balance_last_saved_version
            ):
                self._save_balance_report(agent_name, version)

            # Step 5: Run eval
            if any_eval:
                try:
                    eval_results = self.eval_manager.run_eval(
                        agent_name, self.raas_pool
                    )
                except RuntimeError as e:
                    if "no healthy RaaS" in str(e):
                        logger.warning(
                            "Eval skipped — no healthy RaaS instance: %s", e
                        )
                        print(
                            f"[{agent_name}:{effective_model_id}] eval SKIPPED "
                            f"(no healthy RaaS instance)",
                            flush=True,
                        )
                        eval_results = {}
                    else:
                        flow.resume()
                        raise
                # Step 6: Resume data acquisition
                flow.resume()
                # Accumulate eval wall-clock time for the balance report
                # so the time-based GPU estimator can subtract it.  Also
                # invalidate _last_batch_t1 so the next get_batch does
                # not fold the eval gap into its step_time EWMA sample.
                with self._balance_stats_lock:
                    self._balance_eval_time += _time.monotonic() - _eval_t0
                    self._last_batch_t1 = None
                print(
                    f"[{agent_name}:{effective_model_id}] eval complete, resumed",
                    flush=True,
                )

            # Store eval_results so non-leader can read it
            with self._version_barrier_cond:
                self._barrier_eval_results = eval_results
                self._pending_versions = {
                    k: v for k, v in self._pending_versions.items()
                    if k[0] != agent_name
                }
                self._pending_eval = {
                    k: v for k, v in self._pending_eval.items()
                    if k[0] != agent_name
                }
                self._version_barrier_generation += 1
                self._version_barrier_cond.notify_all()
        else:
            print(
                f"[{agent_name}:{effective_model_id}] version barrier: "
                f"waiting for all models ...",
                flush=True,
            )
            with self._version_barrier_cond:
                while self._version_barrier_generation == my_gen:
                    self._version_barrier_cond.wait()
                eval_results = self._barrier_eval_results
            print(
                f"[{agent_name}:{effective_model_id}] version barrier: "
                f"released (v={version})",
                flush=True,
            )

        return eval_results, weight_transfer_info


# Global service instance — set by ``create_app()`` or ``__main__``.
_service: AstraFlowService | None = None


def _get_service() -> AstraFlowService:
    if _service is None:
        raise RuntimeError("AstraFlowService not initialized")
    return _service


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------


@app.route("/ready", methods=["POST"])
def trainer_ready():
    """Trainer signals it is ready.

    AstraFlow starts data acquisition once both RaaS and trainer are ready.
    """
    service = _get_service()
    agent_name = "default"
    if agent_name not in service.flows:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

    # Parse optional trainer payload (train_batch_size, model_id, etc.)
    train_batch_size = None
    model_id = None
    sender_endpoint = None
    recovered_version = None
    if request.data:
        try:
            payload = loads_object(request.data)
            if isinstance(payload, dict):
                train_batch_size = payload.get("train_batch_size")
                model_id = payload.get("model_id")
                sender_endpoint = payload.get("sender_endpoint")
                recovered_version = payload.get("recovered_version")
        except Exception:
            pass  # Old clients send empty body — that's fine

    service.trainer_ready(
        agent_name,
        train_batch_size=train_batch_size,
        model_id=model_id,
        sender_endpoint=sender_endpoint,
        recovered_version=recovered_version,
    )
    return Response(
        dumps_object({"ok": True}),
        content_type="application/octet-stream",
    )


@app.route("/batch", methods=["GET"])
def get_batch():
    """Serve a training batch. Blocks until data available."""
    service = _get_service()
    agent_name = "default"
    if agent_name not in service.flows:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

    model_id = request.args.get("model_id")
    trainer_version = request.args.get("version", type=int)
    batch = service.get_batch(
        agent_name, model_id=model_id, trainer_version=trainer_version,
    )
    return Response(
        dumps_object(batch),
        content_type="application/octet-stream",
    )


@app.route("/save_buffer", methods=["POST"])
def save_buffer():
    """Trigger buffer checkpoint."""
    service = _get_service()
    agent_name = "default"
    if agent_name not in service.flows:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

    service.save_buffer(agent_name)
    return Response(
        dumps_object({"ok": True}),
        content_type="application/octet-stream",
    )


@app.route("/notify_version", methods=["POST"])
def notify_version():
    """Update version for TCP weight transfer mode. Optionally run eval."""
    service = _get_service()
    agent_name = "default"
    if agent_name not in service.flows:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

    data = loads_object(request.data)
    version = data["version"]
    run_eval = data.get("run_eval", False)
    model_id = data.get("model_id")

    eval_results, weight_transfer_info = service.notify_version(
        agent_name,
        version=version,
        run_eval=run_eval,
        model_id=model_id,
    )
    resp_data = {"ok": True, "eval_results": eval_results}
    if weight_transfer_info:
        resp_data["weight_transfer_info"] = weight_transfer_info
    return Response(
        dumps_object(resp_data),
        content_type="application/octet-stream",
    )


@app.route("/status", methods=["GET"])
def status():
    """Health check with per-agent readiness info and pool status."""
    service = _get_service()
    agents_status = {}
    for name in service.flows:
        agents_status[name] = {
            "raas_ready": name in service._raas_ready,
            "trainer_ready": name in service._trainer_ready,
            "started": name in service._started,
            "version": service.versions.get(name, 0),
        }
    return jsonify({
        "status": "ready",
        "eval_running": service.eval_manager._eval_running,
        "agents": agents_status,
        "raas_pool": {
            "size": service.raas_pool.size(),
            "version": service.raas_pool._version,
            "instances": service.raas_pool.list_instances(),
        },
    })


@app.route("/register_raas", methods=["POST"])
def register_raas():
    """Dynamically register a RaaS instance into the global pool.

    Request body (JSON): ``{"uid": str, "raas_url": str}``

    The ``uid`` is a unique identifier assigned by the launcher (e.g. the
    hostname or a UUID).  If a RaaS with the same uid is already registered
    it is replaced.
    """
    service = _get_service()
    data = request.get_json(force=True)
    if not data or "uid" not in data or "raas_url" not in data:
        return jsonify({"error": "Request body must contain 'uid' and 'raas_url'"}), 400

    uid: str = data["uid"]
    raas_url: str = data["raas_url"]
    gpu_count: int | None = data.get("gpu_count")

    try:
        service.raas_pool.register(uid, raas_url, gpu_count=gpu_count)
    except Exception as exc:
        logger.exception("Failed to register RaaS uid=%s url=%s", uid, raas_url)
        return jsonify({"error": str(exc)}), 500

    # Immediate weight catch-up: sync the new RaaS with current model
    # weights in a background thread so the HTTP response isn't blocked.
    import threading
    threading.Thread(
        target=service.catchup_raas,
        args=(uid,),
        name=f"catchup-{uid}",
        daemon=True,
    ).start()

    return jsonify({
        "ok": True,
        "uid": uid,
        "pool_size": service.raas_pool.size(),
    })


@app.route("/deregister_raas", methods=["POST"])
def deregister_raas():
    """Remove a RaaS instance from the global pool.

    Request body (JSON): ``{"uid": str, "shutdown": bool (optional)}``

    If ``shutdown`` is true, the RaaS process is sent a ``/shutdown``
    request so it terminates and frees GPU resources.
    """
    service = _get_service()
    data = request.get_json(force=True)
    if not data or "uid" not in data:
        return jsonify({"error": "Request body must contain 'uid'"}), 400

    uid: str = data["uid"]
    shutdown: bool = data.get("shutdown", False)
    service.raas_pool.deregister(uid, shutdown=shutdown)
    return jsonify({
        "ok": True,
        "uid": uid,
        "pool_size": service.raas_pool.size(),
    })


@app.route("/raas_pool", methods=["GET"])
def raas_pool_status():
    """Return status of all RaaS instances in the global pool."""
    service = _get_service()
    return jsonify({
        "size": service.raas_pool.size(),
        "version": service.raas_pool._version,
        "instances": service.raas_pool.list_instances(),
    })



@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Gracefully shut down the AstraFlow service and its RaaS pool.

    Called by the trainer when training completes. Broadcasts shutdown
    to all registered RaaS engines (so remote engines on other nodes
    terminate cleanly), then terminates the Flask process.
    """
    service = _get_service()
    logger.info("Shutdown requested — terminating process ...")
    print("=" * 60, flush=True)
    print("[AstraFlow] Shutdown requested — terminating process.", flush=True)
    print("=" * 60, flush=True)

    # Tell every registered RaaS to stop. Local engines die with our
    # parent shell's cleanup trap, but remote engines on other nodes
    # only learn of shutdown via this broadcast. Bounded timeout so we
    # don't stall indefinitely if an engine is unreachable.
    try:
        statuses = service.raas_pool.shutdown_all(per_engine_timeout=5.0)
        if statuses:
            print(f"[AstraFlow] RaaSPool shutdown_all: {statuses}", flush=True)
    except Exception:
        logger.exception("shutdown_all failed; proceeding to exit")

    # Schedule hard exit on a background thread. os._exit() terminates
    # all threads regardless of state — safe because the broadcast
    # above has already fired and engines no longer need our services.
    def _exit_soon():
        import time as _time
        _time.sleep(2.0)  # Allow shutdown_all callbacks + Flask response
        os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()

    return Response(
        dumps_object({"ok": True}),
        content_type="application/octet-stream",
    )


def _import_function(import_path: str) -> Any:
    """Import a function from a dotted path like ``module.path:function_name``."""
    module_path, _, func_name = import_path.rpartition(":")
    if not module_path or not func_name:
        raise ValueError(
            f"Invalid dataset_fn format: {import_path!r}. "
            f"Expected 'module.path:function_name'"
        )
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _create_dataset_from_config(ds_cfg: dict[str, Any], tokenizer: Any) -> Any:
    """Create a dataset from a config dict using ``dataset_fn``.

    The ``dataset_fn`` field is a Python import path like
    ``"astraflow.dataflow.dataset.deepscaler:get_deepscaler_rl_dataset"``.
    Extra fields in ``ds_cfg`` are forwarded as kwargs when supported by
    the target dataset function.
    """
    dataset_fn_path = ds_cfg.get("dataset_fn")
    if dataset_fn_path is None:
        raise ValueError(
            "rollout_dataset/eval_datasets entries must specify 'dataset_fn' — "
            "a Python import path like 'astraflow.dataset.deepscaler:get_deepscaler_rl_dataset'"
        )
    dataset_fn = _import_function(dataset_fn_path)
    kwargs = {
        k: v
        for k, v in ds_cfg.items()
        if k not in {"dataset_fn", "batch_size", "k", "repeat"}
    }
    kwargs.setdefault("tokenizer", tokenizer)

    sig = inspect.signature(dataset_fn)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return dataset_fn(**kwargs)

    filtered_kwargs = {
        name: kwargs[name] for name in sig.parameters if name in kwargs
    }
    return dataset_fn(**filtered_kwargs)


def _create_dataloader(dataset: Any, batch_size: int = 1) -> Any:
    """Create a StatefulDataLoader for the AstraFlow service.

    The service runs as a single process (rank=0, world_size=1), so no
    distributed sharding is needed.
    """
    from torch.utils.data import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            drop_last=True,
        ),
        drop_last=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )


def create_app(service: AstraFlowService) -> Flask:
    """Create a Flask app with the given service instance.

    Use this for programmatic startup or testing.
    """
    global _service
    _service = service
    return app
