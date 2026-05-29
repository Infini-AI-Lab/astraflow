from __future__ import annotations

import asyncio
import logging as _stdlib_logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from astraflow.raas.api.alloc_mode import AllocationMode
from astraflow.raas.api.cli_args import InferenceEngineConfig
from astraflow.raas.engine.sglang_remote import SGLangEngine
from astraflow.raas.engine.vllm_remote import VLLMEngine
from astraflow.raas.platforms import current_platform
from astraflow.raas.utils import logging
from astraflow.raas.utils.network import find_free_ports, gethostip
from astraflow.core.workflow.api.engine_api import EngineGroup
from astraflow.core.workflow.registry import get_reward, get_workflow

_base_logger = logging.getLogger(__name__)
logger = _base_logger  # replaced with adapter after engine_id is known


class RaaS3Manager:
    """Simplified async-native rollout-as-a-service manager.

    2-layer design: RaaS3Manager -> workflow.arun_episode(engine, data)
    No intermediate orchestration layers. All state is accessed from a
    single event loop.
    """

    def __init__(
        self,
        engine_factories: dict[str, Any] | None = None,
        service_port: int | None = None,
        service_host: str | None = None,
    ):
        self._engine_factories = engine_factories or {
            "sglang": SGLangEngine,
            "vllm": VLLMEngine,
        }
        self._service_port = service_port
        # Keep a stable service identity for instance-id matching.
        self._service_host = service_host or "0.0.0.0"

        # Multi-engine support: model_id -> engine.
        # Single-model mode uses "default" as the key.
        self._engines: dict[str, Any] = {}
        self._eval_engines: dict[str, Any] = {}
        self._engine_id: str | None = None
        self._backend: str | None = None  # legacy (single-model only)

        # Workflow registry
        self._workflows: dict[str, Any] = {}

        # Per-model workflow construction context
        self._gconfigs: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        # Legacy single-model aliases (set during single-model bootstrap)
        self._gconfig: Any | None = None
        self._tokenizer: Any | None = None

        # Pause control — created lazily
        self._running: asyncio.Event | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._max_concurrency: int = 0

        # Training tasks
        self._running_tasks: dict[int, asyncio.Task] = {}
        self._completed_results: dict[int, Any] = {}
        self._next_task_id: int = 0
        # Bumped by reset_training_engine.  Each training task captures
        # the current epoch at submit time; _on_task_done drops the
        # result if the epoch has advanced, preventing stragglers from
        # re-populating _completed_results after a wipe.
        self._reset_epoch: int = 0
        # Default: honor reset-on-eval unless the rollout config
        # explicitly disables it during bootstrap.
        self._reset_training_on_eval: bool = True

        # Eval tasks
        self._eval_running_tasks: dict[int, asyncio.Task] = {}
        self._eval_completed_results: dict[int, Any] = {}
        self._next_eval_task_id: int = 0
        self._eval_semaphore: asyncio.Semaphore | None = None
        self._max_eval_concurrency: int = 0

        # Bootstrap state
        self._status: str = "idle"
        self._status_message: str = ""

        # Rollout-manager state for TCP weight transfer
        self._rollout_instances: dict[str, dict] = {}
        self._weight_versions: dict[str, int] = {}  # per-model version tracking

        # Per-model TCP receivers (lazy init on first weight pull per model).
        # Each model's sender is a separate process, so each needs its own
        # receiver with its own TCP session and registration.
        self._tcp_receivers: dict[str, Any] = {}
        # Per-model weight-update lock: serialises the entire
        # pull-to-disk → pause → load → resume cycle so that two
        # concurrent notify_version calls for the same model cannot
        # race on the same safetensors file (which causes Bus errors
        # in sglang's mmap reader).
        self._weight_update_locks: dict[str, asyncio.Lock] = {}
        # Delta transfer: periodic full sync interval (set during bootstrap).
        # Every N-th step uses full transfer for resync. 0 = never force full.
        self._delta_full_sync_interval: int = 0

        # Eval / weight-load mutual exclusion (created lazily on event loop).
        # acquire_eval_lock / release_eval_lock let the trainer hold the lock
        # across the entire save+eval window, blocking new weight loads.

        # Periodic generation stats logger
        self._gen_stats_task: asyncio.Task | None = None

        # Background engine health monitor (started after bootstrap).
        self._health_monitor_task: asyncio.Task | None = None
        self._weight_update_in_progress: bool = False
        # Wall-clock start time of the current weight update, or 0.0 when
        # idle. Used by the engine-health monitor: if a weight update has
        # been "in progress" for longer than the grace window, we assume
        # the workers died silently (the flag would otherwise never clear
        # because pause/load/resume never returned) and force a probe.
        self._weight_update_started_at: float = 0.0

        # Adaptive availability state (driven by sglang /metrics).
        # Defaults leave adaptive disabled until bootstrap reads the rollout
        # config. ``_metrics_cache`` is ``(timestamp, per_engine_list)``.
        self._adaptive_enabled: bool = False
        self._target_per_dp: int = 4
        self._step_size: int = 4
        self._load_cache_ttl_s: float = 0.1
        self._metrics_cache: tuple[float, list] = (0.0, [])
        self._metrics_cache_ok: bool = False
        self._load_poll_lock: asyncio.Lock | None = None
        # Last-known-good snapshot preserved across refresh failures so the
        # controller can fail closed (stale cache is better than no data).
        # Cleared on weight-update transitions so post-update /availability
        # refetches fresh sglang state.
        self._last_good_snapshot: list | None = None
        self._last_good_snapshot_at: float = 0.0
        self._max_stale_age_s: float = 10.0

    @property
    def _tag(self) -> str:
        """Short prefix for print() log lines, e.g. ``[raas-1]``."""
        return f"[{self._engine_id or 'RaaS3'}]"

    # -- Backward-compat properties for single-engine access ---------------

    @property
    def _engine(self) -> Any | None:
        """Default engine (backward compat for single-model mode)."""
        if "default" in self._engines:
            return self._engines["default"]
        # Fall back to the sole engine if there's exactly one
        if len(self._engines) == 1:
            return next(iter(self._engines.values()))
        return None

    @_engine.setter
    def _engine(self, value: Any | None) -> None:
        if value is None:
            self._engines.pop("default", None)
        else:
            self._engines["default"] = value

    @property
    def _eval_engine(self) -> Any | None:
        if "default" in self._eval_engines:
            return self._eval_engines["default"]
        if len(self._eval_engines) == 1:
            return next(iter(self._eval_engines.values()))
        return None

    @_eval_engine.setter
    def _eval_engine(self, value: Any | None) -> None:
        if value is None:
            self._eval_engines.pop("default", None)
        else:
            self._eval_engines["default"] = value

    def _build_engine_group(self, eval: bool = False) -> EngineGroup:
        """Build an EngineGroup from the current engine set."""
        engines = self._eval_engines if eval else self._engines
        return EngineGroup(engines)

    # -- Async state -------------------------------------------------------

    def _ensure_async_state(self) -> None:
        """Create asyncio primitives on the current event loop."""
        if self._running is None:
            self._running = asyncio.Event()
            self._running.set()
        if self._load_poll_lock is None:
            self._load_poll_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap(
        self,
        *,
        config: Any,
        allocation_mode: dict | AllocationMode,
        engine_id: str = "default",
    ) -> dict[str, Any]:
        """Create engine, launch servers, and initialize."""
        self._ensure_async_state()

        if self._status == "ready":
            return self.get_status()

        self._status = "starting"
        self._status_message = ""
        self._engine_id = engine_id

        # Tag all subsequent log messages with the engine-id so that
        # raas-1 / raas-2 logs are distinguishable.
        global logger
        _eid = engine_id
        adapter = _stdlib_logging.LoggerAdapter(_base_logger, {})
        adapter.process = lambda msg, kw: (f"[{_eid}] {msg}", kw)
        logger = adapter

        try:
            alloc = self._resolve_allocation_mode(allocation_mode)
            backend = alloc.gen_backend
            if backend not in self._engine_factories:
                raise ValueError(
                    f"Unsupported inference backend: {backend}. "
                    f"Expected one of {sorted(self._engine_factories.keys())}."
                )
            self._backend = backend

            rollout_config = deepcopy(config.rollout)
            train_dp_size = self._get_train_dp_size(alloc)
            init_kwargs = {
                "engine_id": engine_id,
                "train_data_parallel_size": train_dp_size,
            }

            # Create engine
            engine = self._engine_factories[backend](rollout_config)
            self._engine = engine

            # Launch servers in parallel via thread pool
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._launch_inference_servers,
                engine,
                backend,
                config,
                alloc,
            )

            # Initialize engine
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: engine.initialize(**init_kwargs)
            )

            # Create eval engine — shares the same servers
            eval_config = deepcopy(rollout_config)
            eval_config.max_head_offpolicyness = int(1e12)
            eval_engine = self._engine_factories[backend](eval_config)
            addrs = self._engine._engine.addresses
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: eval_engine.initialize(
                    engine_id="eval",
                    addr=addrs,
                    train_data_parallel_size=train_dp_size,
                ),
            )
            self._eval_engine = eval_engine
            logger.info("Eval engine initialized, sharing servers with training engine")

            # Set max concurrency and create semaphore
            self._max_concurrency = self._resolve_max_concurrency(
                rollout_config, train_dp_size
            )
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

            # Eval concurrency cap (independent of rollout cap; OOM-safety knob)
            self._max_eval_concurrency = max(
                1, int(getattr(rollout_config, "max_concurrent_evals", 128))
            )
            self._eval_semaphore = asyncio.Semaphore(self._max_eval_concurrency)

            # Stash adaptive-availability knobs from rollout_config.
            self._load_adaptive_knobs(rollout_config)

            # Stash the reset_training_on_eval flag.  When false,
            # reset_training_engine returns a ready no-op instead of
            # cancelling in-flight training tasks.  Lets users with
            # very high eval frequency opt out of the wasted-compute
            # tradeoff without changing the service-side call path.
            self._reset_training_on_eval = bool(
                getattr(rollout_config, "reset_training_on_eval", True)
            )

            # Stash gconfig and tokenizer for spec-based workflow construction
            self._gconfig = deepcopy(config.gconfig)
            tokenizer_path = config.tokenizer_path
            if tokenizer_path:
                from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

                self._tokenizer = load_hf_tokenizer(tokenizer_path)
                logger.info(
                    "Loaded tokenizer from %s for workflow construction",
                    tokenizer_path,
                )

            self._status = "ready"
            self._status_message = ""

            # Start periodic generation stats logger
            if self._gen_stats_task is None:
                self._gen_stats_task = asyncio.create_task(
                    self._log_generation_stats_loop()
                )

            # Start background engine health monitor (fail-fast on dead SGLang).
            if self._health_monitor_task is None:
                self._health_monitor_task = asyncio.create_task(
                    self._engine_health_monitor()
                )

        except Exception as exc:
            self._status = "error"
            self._status_message = repr(exc)
            if self._engine is not None:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._engine.destroy
                    )
                except Exception:
                    logger.exception("Failed to destroy engine after bootstrap error.")
            self._engine = None
            raise

        return self.get_status()

    async def bootstrap_from_yaml(
        self,
        *,
        config_paths: list[str],
        overrides: list[str] | None = None,
        engine_id: str = "default",
    ) -> dict[str, Any]:
        """Load config from yaml file(s) and bootstrap."""
        from astraflow.raas.api.cli_args import RaaSConfig, load_expr_config

        argv = []
        for p in config_paths:
            argv += ["--config", p]
        argv += overrides or []
        config, _ = load_expr_config(argv, RaaSConfig)

        # Multi-model bootstrap: only use when there are 2+ models.
        # Single-model entries in config.models are handled by the
        # normal bootstrap path with an engine alias (see below).
        models = getattr(config, "models", None) or {}
        if len(models) >= 1:
            return await self.bootstrap_multi_model(
                config=config,
                allocation_mode=config.allocation_mode,
                engine_id=engine_id,
            )
        result = await self.bootstrap(
            config=config,
            allocation_mode=config.allocation_mode,
            engine_id=engine_id,
        )
        return result

    async def bootstrap_multi_model(
        self,
        *,
        config: Any,
        allocation_mode: dict | AllocationMode,
        engine_id: str = "default",
    ) -> dict[str, Any]:
        """Bootstrap multiple engines from ``config.models``.

        Each key in ``config.models`` maps to a named inference allocation
        in *allocation_mode* and produces its own engine.
        """
        from astraflow.raas.api.cli_args import ModelSpec
        from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

        self._ensure_async_state()
        if self._status == "ready":
            return self.get_status()

        self._status = "starting"
        self._status_message = ""
        self._engine_id = engine_id

        try:
            alloc = self._resolve_allocation_mode(allocation_mode)
            models: dict[str, Any] = config.models

            # Resolve ModelSpec from raw dicts if needed (Hydra returns dicts)
            from astraflow.raas.api.cli_args import (
                GenerationHyperparameters,
                SGLangConfig,
                vLLMConfig,
            )

            resolved: dict[str, ModelSpec] = {}
            for mid, spec in models.items():
                if isinstance(spec, dict):
                    # Convert nested dicts to their dataclass types
                    if "sglang" in spec and isinstance(spec["sglang"], dict):
                        spec["sglang"] = SGLangConfig(**spec["sglang"])
                    if "vllm" in spec and isinstance(spec["vllm"], dict):
                        spec["vllm"] = vLLMConfig(**spec["vllm"])
                    if "gconfig" in spec and isinstance(spec["gconfig"], dict):
                        spec["gconfig"] = GenerationHyperparameters(**spec["gconfig"])
                    spec = ModelSpec(**spec)
                else:
                    # Already a ModelSpec but nested fields may still be dicts
                    if isinstance(getattr(spec, "sglang", None), dict):
                        spec.sglang = SGLangConfig(**spec.sglang)
                    if isinstance(getattr(spec, "vllm", None), dict):
                        spec.vllm = vLLMConfig(**spec.vllm)
                    if isinstance(getattr(spec, "gconfig", None), dict):
                        spec.gconfig = GenerationHyperparameters(**spec.gconfig)
                resolved[mid] = spec

            total_max_concurrency = 0
            loop = asyncio.get_event_loop()

            # For single-model-in-multi-model-config (e.g. smoke test),
            # fall back to the first inference allocation if names don't match.
            inference_allocs = alloc._get_inference_allocations()

            # ── Phase 1: Pre-compute per-model configs (sequential, cheap) ──
            gpu_offset = 0
            model_plans: list[dict[str, Any]] = []

            for idx, (model_id, spec) in enumerate(resolved.items()):
                backend = spec.backend
                if backend not in self._engine_factories:
                    raise ValueError(
                        f"Unsupported backend {backend!r} for model {model_id!r}"
                    )

                # Get per-model allocation: try by name first, fall back to
                # positional index for unnamed allocations.
                try:
                    model_alloc = alloc[model_id]
                except KeyError:
                    if idx < len(inference_allocs):
                        model_alloc = inference_allocs[idx]
                        logger.info(
                            "No named allocation for %r, using inference alloc #%d",
                            model_id, idx,
                        )
                    else:
                        raise

                # Build rollout config from model spec, falling back to top-level
                # rollout config when the per-model spec has only defaults.
                if spec.rollout and spec.rollout.max_concurrent_rollouts is not None:
                    rollout_config = deepcopy(spec.rollout)
                else:
                    rollout_config = deepcopy(config.rollout)
                train_dp_size = self._get_train_dp_size(alloc)

                engine = self._engine_factories[backend](rollout_config)
                self._engines[model_id] = engine

                model_config = self._build_model_launch_config(
                    config, spec, model_id
                )
                model_alloc_mode = AllocationMode(allocations=[model_alloc])
                n_gpus_this_model = (
                    model_alloc_mode.gen.dp_size
                    * model_alloc_mode.gen_instance_size
                )

                # Per-model gconfig & tokenizer (no I/O, safe to do here).
                # Use per-model gconfig (populated via model_base merge).
                gconfig = deepcopy(spec.gconfig)
                self._gconfigs[model_id] = gconfig
                tok_path = spec.tokenizer_path or spec.model_path or config.tokenizer_path
                if tok_path:
                    self._tokenizers[model_id] = load_hf_tokenizer(tok_path)

                total_max_concurrency += self._resolve_max_concurrency(
                    rollout_config, train_dp_size
                )

                model_plans.append({
                    "model_id": model_id,
                    "backend": backend,
                    "engine": engine,
                    "model_config": model_config,
                    "model_alloc_mode": model_alloc_mode,
                    "gpu_offset": gpu_offset,
                    "rollout_config": rollout_config,
                    "train_dp_size": train_dp_size,
                    "spec": spec,
                })
                gpu_offset += n_gpus_this_model

            # ── Phase 2: Launch all inference servers in parallel ──
            logger.info(
                "Launching %d model engines in parallel ...",
                len(model_plans),
            )
            launch_coros = [
                loop.run_in_executor(
                    None,
                    self._launch_inference_servers,
                    plan["engine"],
                    plan["backend"],
                    plan["model_config"],
                    plan["model_alloc_mode"],
                    plan["gpu_offset"],
                )
                for plan in model_plans
            ]
            await asyncio.gather(*launch_coros)

            # ── Phase 3: Initialize all engines in parallel ──
            init_coros = [
                loop.run_in_executor(
                    None,
                    lambda e=plan["engine"], mid=plan["model_id"], dp=plan["train_dp_size"]: (
                        e.initialize(
                            engine_id=mid,
                            train_data_parallel_size=dp,
                        )
                    ),
                )
                for plan in model_plans
            ]
            await asyncio.gather(*init_coros)

            # ── Phase 4: Create eval engines in parallel ──
            eval_init_coros = []
            for plan in model_plans:
                model_id = plan["model_id"]
                backend = plan["backend"]
                engine = plan["engine"]
                rollout_config = plan["rollout_config"]
                train_dp_size = plan["train_dp_size"]

                eval_config = deepcopy(rollout_config)
                eval_config.max_head_offpolicyness = int(1e12)
                eval_engine = self._engine_factories[backend](eval_config)
                addrs = engine._engine.addresses
                self._eval_engines[model_id] = eval_engine

                eval_init_coros.append(
                    loop.run_in_executor(
                        None,
                        lambda ee=eval_engine, a=addrs, mid=model_id, dp=train_dp_size: (
                            ee.initialize(
                                engine_id=f"eval_{mid}",
                                addr=a,
                                train_data_parallel_size=dp,
                            )
                        ),
                    )
                )
            await asyncio.gather(*eval_init_coros)

            for plan in model_plans:
                logger.info(
                    "Bootstrapped model %r: backend=%s",
                    plan["model_id"], plan["backend"],
                )

            # Use first model's gconfig/tokenizer as default for workflow construction
            first_id = next(iter(resolved))
            self._gconfig = self._gconfigs.get(first_id)
            self._tokenizer = self._tokenizers.get(first_id)

            self._max_concurrency = total_max_concurrency
            self._semaphore = asyncio.Semaphore(self._max_concurrency)

            # Eval concurrency cap (independent of rollout cap; OOM-safety knob)
            self._max_eval_concurrency = max(
                1, int(getattr(config.rollout, "max_concurrent_evals", 128))
            )
            self._eval_semaphore = asyncio.Semaphore(self._max_eval_concurrency)

            # Stash adaptive-availability knobs from the top-level rollout
            # config (multi-model uses config.rollout, same as single-model).
            self._load_adaptive_knobs(deepcopy(config.rollout))
            self._reset_training_on_eval = bool(
                getattr(config.rollout, "reset_training_on_eval", True)
            )

            # Delta config from RaaSConfig
            self._delta_full_sync_interval = getattr(
                config, "delta_full_sync_interval", 0,
            )
            if self._delta_full_sync_interval > 0:
                print(
                    f"[RaaS] Delta full sync interval: every "
                    f"{self._delta_full_sync_interval} steps",
                    flush=True,
                )

            self._status = "ready"
            self._status_message = ""

            if self._gen_stats_task is None:
                self._gen_stats_task = asyncio.create_task(
                    self._log_generation_stats_loop()
                )

            # Start background engine health monitor (fail-fast on dead SGLang).
            if self._health_monitor_task is None:
                self._health_monitor_task = asyncio.create_task(
                    self._engine_health_monitor()
                )

        except Exception as exc:
            self._status = "error"
            self._status_message = repr(exc)
            for eid, eng in list(self._engines.items()):
                try:
                    await loop.run_in_executor(None, eng.destroy)
                except Exception:
                    logger.exception("Failed to destroy engine %s", eid)
            self._engines.clear()
            raise

        return self.get_status()

    @staticmethod
    def _build_model_launch_config(
        base_config: Any, spec: Any, model_id: str
    ) -> Any:
        """Build a config-like object for _launch_inference_servers.

        Merges per-model overrides (sglang/vllm/tokenizer) with the
        base config so _build_server_args can find the right fields.
        """
        from types import SimpleNamespace

        cfg = SimpleNamespace()
        cfg.sglang = deepcopy(spec.sglang) if spec.sglang else deepcopy(base_config.sglang)
        cfg.vllm = deepcopy(spec.vllm) if spec.vllm else deepcopy(base_config.vllm)
        cfg.cluster = base_config.cluster
        cfg.weight_transfer_mode = getattr(base_config, "weight_transfer_mode", "tcp")

        # Set model path on the correct backend config
        if spec.backend == "sglang" and spec.model_path:
            cfg.sglang.model_path = spec.model_path
        elif spec.backend == "vllm" and spec.model_path:
            cfg.vllm.model = spec.model_path

        return cfg

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    def register_workflow(
        self,
        workflow_id: str,
        workflow_cls: str,
        reward_fn: str | None = None,
        gconfig_overrides: dict[str, Any] | None = None,
        **workflow_kwargs: Any,
    ) -> dict[str, Any]:
        """Register a workflow by spec.

        Resolves ``workflow_cls`` and ``reward_fn`` from ``workflow.registry``
        and constructs the workflow with the engine's tokenizer and gconfig.
        """
        logger.info(
            "register_workflow called: workflow_id=%r, "
            "workflow_cls=%r, reward_fn=%r, gconfig_overrides=%s, "
            "workflow_kwargs keys=%s",
            workflow_id,
            workflow_cls,
            reward_fn,
            gconfig_overrides,
            list(workflow_kwargs.keys()),
        )
        cls = get_workflow(workflow_cls)
        logger.info("Resolved workflow_cls=%r -> %s", workflow_cls, cls)
        if reward_fn is not None:
            reward = get_reward(reward_fn)
            logger.info("Resolved reward_fn=%r -> %s", reward_fn, reward)
            workflow_kwargs["reward_fn"] = reward
        if self._gconfig is None:
            raise RuntimeError(
                "No gconfig available. Was bootstrap() called with a config "
                "that has gconfig and tokenizer_path?"
            )
        gconfig = deepcopy(self._gconfig)
        if gconfig_overrides:
            logger.info("Applying gconfig_overrides=%s", gconfig_overrides)
            gconfig = gconfig.new(**gconfig_overrides)
        workflow_kwargs.setdefault("tokenizer", self._tokenizer)
        workflow_kwargs.setdefault("gconfig", gconfig)
        # Pass per-model gconfigs so multi-model workflows can use
        # different generation settings per model.
        if self._gconfigs and "gconfigs" not in workflow_kwargs:
            per_model = {
                mid: deepcopy(gc).new(**gconfig_overrides) if gconfig_overrides else deepcopy(gc)
                for mid, gc in self._gconfigs.items()
            }
            # Only pass if the workflow accepts it (avoid breaking single-model workflows)
            import inspect
            sig = inspect.signature(cls.__init__)
            if "gconfigs" in sig.parameters:
                workflow_kwargs["gconfigs"] = per_model
        # Pass per-model tokenizers so multi-model workflows with different
        # model families can tokenize prompts correctly per engine.
        if self._tokenizers and "tokenizers" not in workflow_kwargs:
            import inspect
            sig = inspect.signature(cls.__init__)
            if "tokenizers" in sig.parameters:
                workflow_kwargs["tokenizers"] = dict(self._tokenizers)
        logger.info(
            "Constructing workflow with kwargs keys=%s...",
            list(workflow_kwargs.keys()),
        )
        wf_instance = cls(**workflow_kwargs)
        self._workflows[workflow_id] = wf_instance
        logger.info("Workflow %r constructed successfully", workflow_id)
        logger.info("Registered workflow %r", workflow_id)
        return {
            "workflow_id": workflow_id,
            "registered": True,
            "total_workflows": len(self._workflows),
        }

    # ------------------------------------------------------------------
    # Submit — direct asyncio.Task, no intermediate layers
    # ------------------------------------------------------------------

    async def submit(self, data: dict[str, Any], workflow_id: str = "default") -> int:
        """Submit a task. Returns task_id immediately."""
        self._ensure_async_state()
        if self._status != "ready":
            raise RuntimeError(
                f"Cannot submit: manager status is '{self._status}', expected 'ready'."
            )
        if workflow_id not in self._workflows:
            raise KeyError(
                f"Unknown workflow_id: {workflow_id!r}. "
                f"Registered: {list(self._workflows.keys())}"
            )

        task_id = self._next_task_id
        self._next_task_id += 1
        workflow = self._workflows[workflow_id]
        engine_or_group = self._build_engine_group() if len(self._engines) > 1 else self._engine
        # Snapshot current reset epoch so _on_task_done can detect
        # whether this task was submitted before a reset that
        # happened while it was running.
        epoch = self._reset_epoch

        async def _run():
            await self._running.wait()  # blocks during pause
            async with self._semaphore:  # concurrency limit
                async with engine_or_group.managed_session():
                    result = await workflow.arun_episode(engine_or_group, data)
            # Auto-propagate prompt-level metadata into the result so it
            # survives the pull on the AstraFlow side. Workflows that
            # already populate these keys are respected (no overwrite).
            # Used by AstraFlow's curator (selective rollout) to key
            # per-prompt feedback by query_id, and by stats by source.
            if isinstance(result, dict):
                for key in ("query_id", "source", "prompt_id"):
                    if key in data and key not in result:
                        result[key] = data[key]
            return result

        task = asyncio.create_task(_run())
        task.add_done_callback(lambda t: self._on_task_done(task_id, epoch, t))
        self._running_tasks[task_id] = task
        logger.debug(
            "RaaS3 submit: task_id=%d (inflight=%d)",
            task_id,
            len(self._running_tasks),
        )
        return task_id

    def _on_task_done(
        self, task_id: int, epoch: int, task: asyncio.Task
    ) -> None:
        """Callback when an asyncio.Task completes."""
        self._running_tasks.pop(task_id, None)
        # If the reset epoch has advanced since this task was submitted,
        # the training engine was wiped while we were running — drop the
        # result to avoid polluting the freshly-cleared _completed_results
        # dict and tripping the pool's task_failure suspect threshold.
        if epoch != self._reset_epoch:
            return
        try:
            self._completed_results[task_id] = task.result()
        except BaseException as exc:
            # BaseException catches CancelledError (Python 3.9+)
            self._completed_results[task_id] = {"ok": False, "error": repr(exc)}

    # ------------------------------------------------------------------
    # Pull completed
    # ------------------------------------------------------------------

    async def pull_completed(
        self,
        max_items: int = 256,
        timeout: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Pull completed task results."""
        items = self._drain_completed(self._completed_results, max_items)
        if not items and timeout > 0:
            await asyncio.sleep(min(timeout, 0.1))
            items = self._drain_completed(self._completed_results, max_items)
        return items

    @staticmethod
    def _drain_completed(
        results: dict[int, Any], max_items: int
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tid in list(results):
            items.append({"task_id": tid, "result": results.pop(tid)})
            if len(items) >= max_items:
                break
        return items

    # ------------------------------------------------------------------
    # Availability / Status
    # ------------------------------------------------------------------

    def _load_adaptive_knobs(self, rollout_config: InferenceEngineConfig) -> None:
        """Stash adaptive availability knobs from the rollout config.

        Called once at bootstrap from both the single-model and multi-model
        paths. Safe to call before or after ``_ensure_async_state``.
        """
        self._adaptive_enabled = bool(
            getattr(rollout_config, "enable_adaptive_availability", False)
        )
        self._target_per_dp = int(
            getattr(rollout_config, "target_waiting_queue_per_dp", 4)
        )
        self._step_size = int(
            getattr(rollout_config, "adaptive_step_size", 4)
        )
        ttl_ms = int(getattr(rollout_config, "load_cache_ttl_ms", 100))
        self._load_cache_ttl_s = max(0.001, ttl_ms / 1000.0)
        # Reset caches so the first /availability call after bootstrap refetches.
        self._metrics_cache = (0.0, [])
        self._metrics_cache_ok = False
        self._last_good_snapshot = None
        self._last_good_snapshot_at = 0.0
        if self._adaptive_enabled:
            logger.info(
                "%s Adaptive availability ON: target_per_dp=%d step=%d "
                "cache_ttl=%.0fms max_stale_age=%.1fs",
                self._tag, self._target_per_dp, self._step_size,
                self._load_cache_ttl_s * 1000, self._max_stale_age_s,
            )

    async def _refresh_metrics_cache_if_stale(self) -> bool:
        """Refresh the sglang ``/metrics`` cache if older than TTL.

        Returns True iff the refresh produced usable data from at least one
        sglang server in the *current* poll. Uses a single-flight
        ``asyncio.Lock`` with double-check so concurrent /availability calls
        share one poll rather than fanning out independently.

        **Fail-closed invariant:** on total failure (all engines / all servers
        failed), this does NOT clobber ``_last_good_snapshot``. The caller
        prefers the last-good snapshot (up to ``_max_stale_age_s``) over
        starting blind. ``_metrics_cache`` is updated with the failed result
        purely to advance the TTL timestamp (avoids a refresh-storm when
        sglang is down).
        """
        if self._weight_update_in_progress:
            # Engines are paused for weight update — skip the poll; caller
            # returns available=0 for the whole update window.
            return False

        now = time.monotonic()
        cached_at, _ = self._metrics_cache
        if now - cached_at < self._load_cache_ttl_s and self._metrics_cache_ok:
            return True

        assert self._load_poll_lock is not None, (
            "Load poll lock not initialized; _ensure_async_state must run first."
        )
        async with self._load_poll_lock:
            # Re-check under lock — another coroutine may have refreshed.
            now = time.monotonic()
            cached_at, _ = self._metrics_cache
            if now - cached_at < self._load_cache_ttl_s and self._metrics_cache_ok:
                return True

            per_engine: list[Any] = []
            any_server_succeeded = False
            _failed_engines: list[str] = []
            _ok_engines: list[str] = []
            for mid, engine in self._engines.items():
                inner = getattr(engine, "_engine", engine)
                # Prefer sync get_metrics_sync (runs in a thread, immune
                # to event-loop congestion) over async aget_metrics.
                if hasattr(inner, "get_metrics_sync"):
                    try:
                        results = await asyncio.to_thread(
                            inner.get_metrics_sync, 1.0,
                        )
                    except Exception as exc:
                        logger.warning(
                            "get_metrics_sync failed for engine %s: %s",
                            mid, exc,
                        )
                        _failed_engines.append(mid)
                        per_engine.append(None)
                        continue
                elif hasattr(inner, "aget_metrics"):
                    try:
                        results = await inner.aget_metrics(total_timeout=1.0)
                    except Exception as exc:
                        logger.warning(
                            "aget_metrics failed for engine %s: %s",
                            mid, exc,
                        )
                        _failed_engines.append(mid)
                        per_engine.append(None)
                        continue
                else:
                    per_engine.append(None)  # Unsupported backend (e.g. vLLM)
                    continue
                # ``results`` is a list parallel to ``inner.addresses``.
                # Entries are dict|None; any non-None dict counts as success.
                if any(r is not None for r in results):
                    any_server_succeeded = True
                    _ok_engines.append(mid)
                else:
                    _failed_engines.append(f"{mid}(all-None)")
                per_engine.append({"engine_id": mid, "results": results})

            self._metrics_cache = (now, per_engine)
            self._metrics_cache_ok = any_server_succeeded
            if any_server_succeeded:
                # Only overwrite last-good when we actually have new data.
                self._last_good_snapshot = per_engine
                self._last_good_snapshot_at = now
            if _failed_engines:
                _snap_age = (
                    f"{(now - self._last_good_snapshot_at)*1000:.0f}ms"
                    if self._last_good_snapshot is not None else "None"
                )
                print(
                    f"[RaaS-avail-debug] refresh: ok={_ok_engines} "
                    f"failed={_failed_engines} snapshot_age={_snap_age} "
                    f"stale_limit={self._max_stale_age_s}s",
                    flush=True,
                )
            return any_server_succeeded

    async def get_availability(self) -> dict[str, Any]:
        """Return capacity information for client-side throttling.

        When adaptive availability is disabled, behavior matches the legacy
        path: ``available = max_concurrency - inflight``.

        When enabled, drives ``available`` off sglang's live waiting-queue
        depth via ``/metrics`` (Prometheus). Logic is queue-only bang-bang:
        if the summed waiting queue across all DPs is at/above the effective
        target (``target_per_dp * total_dps``), return 0; otherwise return a
        small constant (``step_size``). The feedback loop at 10 Hz handles
        pacing. The static semaphore stays as a hard safety ceiling.

        **Fail-closed invariant:** in the adaptive path this function never
        returns ``hard_cap`` as the fallback. If the fresh refresh fails but
        a recent (<``_max_stale_age_s``) last-good snapshot exists, the
        controller acts on that stale data with ``fallback_kind='stale'``.
        If there's no usable snapshot at all, it returns ``available=0``
        with ``fallback_kind='blind'`` so the producer pauses rather than
        floods while telemetry is down.

        KV fraction (``sglang:token_usage``) is parsed and reported in the
        response as ``max_kv_frac``/``per_dp_kv`` but does NOT factor into
        the decision in v1 — observability only. Promotion to a control
        signal is a follow-up if production data shows KV pressure without
        queue growth.
        """
        inflight = len(self._running_tasks)
        hard_cap = max(0, self._max_concurrency - inflight)

        # Fast path: adaptive disabled -> legacy behavior.
        if not self._adaptive_enabled:
            return {
                "max_concurrency": self._max_concurrency,
                "queued": 0,
                "inflight": inflight,
                "available": hard_cap,
                "adaptive_enabled": False,
            }

        # Weight update in progress: refuse new work, drop both the fresh
        # cache and the last-good snapshot so the first post-update poll
        # refetches a fresh view of sglang.
        if self._weight_update_in_progress:
            self._metrics_cache = (0.0, [])
            self._metrics_cache_ok = False
            self._last_good_snapshot = None
            self._last_good_snapshot_at = 0.0
            print(
                f"[RaaS-avail-debug] available=0 reason=weight_update inflight={inflight}",
                flush=True,
            )
            return {
                "max_concurrency": self._max_concurrency,
                "queued": 0,
                "inflight": inflight,
                "available": 0,
                "adaptive_enabled": True,
                "note": "weight_update",
            }

        ok = await self._refresh_metrics_cache_if_stale()

        # Pick snapshot source for the decision.
        snapshot: list | None
        snapshot_age_ms: float
        fallback_kind: str | None
        if ok:
            snapshot = self._metrics_cache[1]
            snapshot_age_ms = (time.monotonic() - self._metrics_cache[0]) * 1000
            fallback_kind = None
        elif (
            self._last_good_snapshot is not None
            and (time.monotonic() - self._last_good_snapshot_at)
            < self._max_stale_age_s
        ):
            # Fail-closed: prefer stale but recent snapshot over starting blind.
            snapshot = self._last_good_snapshot
            snapshot_age_ms = (
                time.monotonic() - self._last_good_snapshot_at
            ) * 1000
            fallback_kind = "stale"
            print(
                f"[RaaS-avail-debug] using stale snapshot age={snapshot_age_ms:.0f}ms inflight={inflight}",
                flush=True,
            )
        else:
            # Fail-closed: no recent data at all, pause the producer.
            _snap_info = (
                f"snapshot_age={(time.monotonic() - self._last_good_snapshot_at)*1000:.0f}ms"
                if self._last_good_snapshot is not None
                else "snapshot=None"
            )
            print(
                f"[RaaS-avail-debug] available=0 reason=blind inflight={inflight} "
                f"{_snap_info} stale_limit={self._max_stale_age_s}s",
                flush=True,
            )
            return {
                "max_concurrency": self._max_concurrency,
                "queued": 0,
                "inflight": inflight,
                "available": 0,
                "adaptive_enabled": True,
                "hard_cap": hard_cap,
                "fallback_kind": "blind",
            }

        # Aggregate across engines -> servers -> DPs using Prometheus values.
        # Each server_metrics is a dict {metric_name: float} from aget_metrics.
        total_target = 0
        total_waiting = 0
        max_kv_frac = 0.0
        per_dp_waiting: list[int] = []
        per_dp_kv: list[float] = []
        for eng in snapshot:
            if eng is None:
                continue
            results = eng.get("results") or []
            for server_metrics in results:
                if server_metrics is None:
                    continue  # Failed server: contribute nothing.
                if not isinstance(server_metrics, dict):
                    continue
                total_target += self._target_per_dp
                waiting = int(server_metrics.get("sglang:num_queue_reqs", 0))
                kv_frac = float(server_metrics.get("sglang:token_usage", 0.0))
                total_waiting += waiting
                if kv_frac > max_kv_frac:
                    max_kv_frac = kv_frac
                per_dp_waiting.append(waiting)
                per_dp_kv.append(kv_frac)

        # Queue-only bang-bang decision. max_kv_frac is reported but not used.
        if total_target == 0:
            # Snapshot contains no usable server data (all failed). Treat as
            # blind — don't fall back to hard_cap.
            return {
                "max_concurrency": self._max_concurrency,
                "queued": 0,
                "inflight": inflight,
                "available": 0,
                "adaptive_enabled": True,
                "hard_cap": hard_cap,
                "fallback_kind": fallback_kind or "no_dps",
                "snapshot_age_ms": snapshot_age_ms,
            }
        elif total_waiting >= total_target:
            soft_cap = 0
            note = "at_target"
        else:
            soft_cap = self._step_size
            note = "below_target"

        available = min(hard_cap, soft_cap)
        return {
            "max_concurrency": self._max_concurrency,
            "queued": total_waiting,
            "inflight": inflight,
            "available": available,
            "adaptive_enabled": True,
            "hard_cap": hard_cap,
            "soft_cap": soft_cap,
            "total_target": total_target,
            "total_waiting": total_waiting,
            "max_kv_frac": max_kv_frac,       # observability only
            "per_dp_waiting": per_dp_waiting,
            "per_dp_kv": per_dp_kv,           # observability only
            "snapshot_age_ms": snapshot_age_ms,
            "note": note,
            "fallback_kind": fallback_kind,
        }

    # Cached engine health — avoids hammering SGLang on every /status poll.
    # NEVER probe synchronously on the event-loop thread: each sync probe
    # is up to 5 s × N servers, and with ≥6 servers that's enough to blow
    # through the RaaS self-register 10 s budget and freeze heartbeat/pull.
    # The sync entrypoint ``_check_engines_healthy`` only reads the cache
    # and schedules a non-blocking background refresh; actual probes run
    # inside ``_refresh_engine_health_async`` via executor workers.
    _engine_health_cache: tuple[float, bool] = (0.0, True)
    _ENGINE_HEALTH_CACHE_TTL = 5.0  # seconds
    _health_refresh_task: "asyncio.Task | None" = None

    def _check_engines_healthy(self) -> bool:
        """Return cached engine health; kick off background refresh if stale.

        Non-blocking: never does sync IO on the event-loop thread. The
        cache default is ``True`` so the very first post-bootstrap call
        (before any refresh has completed) does not false-alarm — if
        bootstrap got this far, engines came up healthy. Subsequent calls
        see the refreshed value once the background task populates it.
        """
        if self._weight_update_in_progress:
            elapsed = time.monotonic() - self._weight_update_started_at
            if elapsed < self._WEIGHT_UPDATE_GRACE_SEC:
                # Engines are legitimately paused; trust them.
                return True
            # Stuck: fall through to cache read. The monitor's force-probe
            # within ~10s will refresh the cache to reflect actual death.
        now = time.monotonic()
        cached_at, cached_result = self._engine_health_cache
        if now - cached_at < self._ENGINE_HEALTH_CACHE_TTL:
            return cached_result

        # Cache stale (or cold). Schedule a background refresh; return the
        # last known value so the caller never blocks on sync HTTP.
        self._schedule_health_refresh()
        return cached_result

    def _schedule_health_refresh(self) -> None:
        """Start a background async refresh of the engine-health cache.

        Idempotent: if a refresh is already in flight, does nothing.
        Safe to call from sync code as long as there is an event loop
        running in the current thread (the FastAPI request handler is).
        """
        if (
            self._health_refresh_task is not None
            and not self._health_refresh_task.done()
        ):
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        self._health_refresh_task = loop.create_task(
            self._refresh_engine_health_async()
        )

    async def _refresh_engine_health_async(self) -> bool:
        """Probe every SGLang engine's /health in parallel via executor.

        Two complementary probes per engine:

        - HTTP /health on each server address (the front-door check; fast
          but only verifies the SGLang HTTP server itself is responsive).
        - Subprocess liveness — walk each entrypoint's process tree via
          ``/proc`` and fail if any descendant is a zombie. Catches the
          SGLang-specific failure where the watchdog kills its own
          scheduler/detokenizer subprocesses but the HTTP front-end keeps
          running, leaving /health falsely OK.

        The per-server probes are still blocking ``requests.get`` calls,
        but they run in executor worker threads so they can't freeze the
        event loop. Wall time is ``max(per-server)``, not ``sum``. Updates
        ``_engine_health_cache`` with the combined result.
        """
        loop = asyncio.get_event_loop()
        tasks: list = []
        addrs: list[tuple[str, str]] = []
        for mid, engine in self._engines.items():
            inner = getattr(engine, "_engine", engine)
            for addr in getattr(inner, "addresses", []):
                addrs.append((mid, addr))
                tasks.append(
                    loop.run_in_executor(
                        None, inner.check_health, f"http://{addr}",
                    )
                )
            # Subprocess liveness — uses the special address tag so the
            # result-handling loop below can route it to the right log
            # message. Pure /proc reads, runs in executor like the others.
            addrs.append((mid, "<subprocess>"))
            tasks.append(
                loop.run_in_executor(None, inner.check_subprocesses_alive)
            )
        if not tasks:
            return True
        results = await asyncio.gather(*tasks, return_exceptions=True)
        healthy = True
        for (mid, addr), r in zip(addrs, results):
            if addr == "<subprocess>":
                ok = isinstance(r, tuple) and r[0] is True
                if not ok:
                    healthy = False
                    reason = (
                        r[1] if isinstance(r, tuple) and len(r) >= 2
                        else repr(r)
                    )
                    logger.warning(
                        "%s subprocess liveness check failed: model=%s "
                        "reason=%s",
                        self._tag, mid, reason,
                    )
            else:
                ok = r is True
                if not ok:
                    healthy = False
                    logger.warning(
                        "%s engine health check failed: "
                        "model=%s addr=%s result=%r",
                        self._tag, mid, addr, r,
                    )
        self._engine_health_cache = (time.monotonic(), healthy)
        return healthy

    def get_status(self) -> dict[str, Any]:
        """Return manager status snapshot with live engine health."""
        status = self._status
        message = self._status_message

        # If bootstrap completed but engines are now dead, report error.
        if status == "ready" and self._engines and not self._check_engines_healthy():
            status = "error"
            message = "SGLang engine(s) failed health check"

        return {
            "status": status,
            "engine_id": self._engine_id,
            "backend": self._backend,
            "models": list(self._engines.keys()),
            "message": message,
            "max_concurrency": self._max_concurrency,
            "queued": 0,
            "inflight": len(self._running_tasks),
            "pending_futures": len(self._completed_results),
            "workflows": list(self._workflows.keys()),
        }

    # ------------------------------------------------------------------
    # Background engine health monitor
    # ------------------------------------------------------------------

    _HEALTH_MONITOR_INTERVAL = 10.0  # seconds between checks
    # sglang 0.5.12's /health round-trips through the scheduler, which is
    # saturated for ~30-40s during the initial unchunked prefill of ~2048
    # reqs/engine, so the old 3-strike (30s) watchdog false-positive-killed a
    # busy-but-alive engine before the first rollout batch. A crashed engine
    # refuses connections instantly, so dead-engine detection time is
    # INTERVAL * MAX_FAILURES = ~50s here; the 20s probe timeout only extends
    # cycles for an alive-but-slow engine (which we want to tolerate, up to
    # ~100s worst case). 5 strikes covers the ~35-40s prefill ramp (a slow but
    # eventually-200 /health resets the counter) while catching a real death
    # in ~50s.
    _HEALTH_MONITOR_MAX_FAILURES = 5  # consecutive failures before exit
    # Maximum time a weight update is allowed to legitimately stall the
    # engine before the monitor force-probes anyway. A normal full pull +
    # apply + load runs ~60-70s end-to-end, deltas ~30-40s; 90s is a
    # generous upper bound — anything beyond it suggests the workers
    # died silently mid-update and the flag will never clear.
    _WEIGHT_UPDATE_GRACE_SEC = 90.0

    async def _engine_health_monitor(self) -> None:
        """Periodically check SGLang engine health. Exit process if dead.

        Runs as a background asyncio task after bootstrap completes.
        If all engines are unreachable for ``_HEALTH_MONITOR_MAX_FAILURES``
        consecutive checks (~30s), the process exits so the orchestrator
        (Slurm/k8s) can reboot it.
        """
        consecutive_failures = 0
        while True:
            await asyncio.sleep(self._HEALTH_MONITOR_INTERVAL)
            skip_reason = None
            if self._status != "ready":
                skip_reason = "status_not_ready"
            elif self._eval_running_tasks:
                skip_reason = "eval_running"
            elif self._weight_update_in_progress:
                elapsed = time.monotonic() - self._weight_update_started_at
                if elapsed < self._WEIGHT_UPDATE_GRACE_SEC:
                    skip_reason = "weight_update_in_progress"
                else:
                    # Weight update has stalled past the grace window. The
                    # workers most likely died silently mid-update — force
                    # a probe so the cache reflects reality and the pool
                    # can deregister this instance.
                    logger.warning(
                        "%s weight update in progress for %.1fs (>%.1fs "
                        "grace) — forcing health probe in case workers "
                        "died silently",
                        self._tag, elapsed, self._WEIGHT_UPDATE_GRACE_SEC,
                    )
            if skip_reason is not None:
                consecutive_failures = 0
                continue

            # Force a real probe here (can't rely on the sync cache-reader
            # entrypoint — that only returns stale data and kicks off a
            # background refresh). The monitor's whole job is to drive
            # periodic fresh probes, so it awaits the async path directly.
            if await self._refresh_engine_health_async():
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.warning(
                    "%s Engine health monitor: failure %d/%d",
                    self._tag,
                    consecutive_failures,
                    self._HEALTH_MONITOR_MAX_FAILURES,
                )
                if consecutive_failures >= self._HEALTH_MONITOR_MAX_FAILURES:
                    elapsed = consecutive_failures * self._HEALTH_MONITOR_INTERVAL
                    msg = (
                        f"SGLang engine(s) unreachable for {elapsed:.0f}s "
                        f"({consecutive_failures} consecutive health check failures). "
                        f"Exiting so orchestrator can reboot."
                    )
                    logger.critical("%s %s", self._tag, msg)
                    print(f"{self._tag} FATAL: {msg}", flush=True)
                    self._status = "error"
                    self._status_message = msg
                    # Give a moment for logs to flush, then hard-exit.
                    await asyncio.sleep(1.0)
                    os._exit(1)

    # ------------------------------------------------------------------
    # Pause / Resume (generation-only)
    # ------------------------------------------------------------------

    async def pause_generation(self) -> dict[str, str]:
        """Pause all inference servers (stop accepting new requests)."""
        loop = asyncio.get_event_loop()
        print(f"{self._tag} pause_generation: stopping inference servers...", flush=True)
        for mid, engine in self._engines.items():
            engine.set_generation_paused(True)
            await loop.run_in_executor(None, engine.pause_generation)
        print(f"{self._tag} pause_generation: done", flush=True)
        return {"status": "generation_paused"}

    async def continue_generation(self) -> dict[str, str]:
        """Resume all inference servers (start accepting requests again)."""
        loop = asyncio.get_event_loop()
        print(f"{self._tag} continue_generation: restarting inference servers...", flush=True)
        for mid, engine in self._engines.items():
            await loop.run_in_executor(None, engine.continue_generation)
            engine.set_generation_paused(False)
        print(f"{self._tag} continue_generation: done", flush=True)
        return {"status": "generation_running"}

    # ------------------------------------------------------------------
    # Training-engine reset (called before each eval window)
    # ------------------------------------------------------------------

    async def reset_training_engine(
        self, timeout: float = 5.0
    ) -> dict[str, Any]:
        """Wipe the training engine so the shared inference servers are
        quiescent and ready for the next eval window.

        Steps
        -----
        a. Bump ``_reset_epoch`` so any straggler ``_on_task_done``
           callbacks drop their results instead of polluting the
           freshly-cleared ``_completed_results``.
        b. Cancel every asyncio.Task in ``_running_tasks`` and
           ``await`` their cleanup.  Cancellation propagates through
           ``arun_episode → agenerate → aiohttp.request``; the
           ``managed_session`` ``finally`` closes the ``ClientSession``
           which sends a TCP disconnect.  SGLang aborts the request on
           disconnect.  After the ``agenerate`` try/finally fix this
           also cleans up the ``_inflight_per_server`` counter.
        c. Verify quiescence by reading ``sglang:num_running_reqs``
           from each server's ``/metrics`` — best-effort, doesn't gate.
        d. Clear Python-side bookkeeping and (as defense-in-depth
           against future regressions in ``agenerate``) zero the
           per-server inflight counters.

        Why no ``pause_generation``
        ---------------------------
        We deliberately do NOT call ``pause_generation``/``continue_generation``
        from this path.  ``notify_version`` (the trainer's blocking call)
        already runs the weight-update flow, which itself calls
        ``pause_generation`` immediately before reset arrives — within
        ~50 ms.  Calling ``pause_generation`` again here would be a
        second abort cycle ~50 ms after the first, which wedges
        SGLang's detokenizer thread (verified empirically: SGLang
        stops sending detokenizer heartbeats and the ``_get_load``
        path goes silent).  We rely on the cancel→aiohttp-disconnect
        path instead, which gives a soft drain without confusing
        SGLang's abort state machine.
        """
        self._ensure_async_state()
        _t0 = time.monotonic()

        # Honor the opt-out flag: when disabled, return a ready no-op so
        # the service-side call path stays unconditional but cheap.
        if not self._reset_training_on_eval:
            logger.info(
                "reset_training_engine: disabled by config "
                "(reset_training_on_eval=False), returning no-op"
            )
            return {
                "cancelled": 0,
                "stragglers": 0,
                "dropped_results": 0,
                "sglang_running": 0,
                "ready_for_eval": True,
                "reset_epoch": self._reset_epoch,
                "elapsed": 0.0,
                "disabled": True,
            }

        # (a) Bump epoch first so any callback that fires during the
        #     gather() below will already see the new epoch and drop.
        self._reset_epoch += 1
        epoch = self._reset_epoch

        # (b) Cancel Python-side training tasks and wait for unwind.
        tasks = list(self._running_tasks.values())
        for t in tasks:
            t.cancel()
        stragglers = 0
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                stragglers = sum(1 for t in tasks if not t.done())
                logger.warning(
                    "reset_training_engine: %d/%d tasks did not drain "
                    "within %.1fs — they will finish in the background "
                    "but their results will be dropped by epoch check",
                    stragglers,
                    len(tasks),
                    timeout,
                )

        # (c) Verify SGLang is actually idle on every server.  Best-effort:
        #     we report the number but do not gate readiness on it, since
        #     the cancel→disconnect path is async and SGLang's metrics
        #     scrape lags by a scheduler tick.
        running = 0
        try:
            for engine in self._engines.values():
                metrics_list = await engine.aget_metrics(total_timeout=1.0)
                for m in metrics_list:
                    if m is None:
                        continue
                    running += int(m.get("sglang:num_running_reqs", 0))
        except Exception as exc:
            logger.warning(
                "reset_training_engine: aget_metrics failed (continuing): %s",
                exc,
            )
            running = -1

        # (e) Wipe Python-side state.
        cancelled = len(tasks)
        dropped = len(self._completed_results)
        self._running_tasks.clear()
        self._completed_results.clear()
        # Defense-in-depth.  With the agenerate try/finally fix, cancelled
        # tasks already decrement _inflight_per_server correctly — zeroing
        # here protects against future regressions in that path.
        for engine in self._engines.values():
            inner = getattr(engine, "_engine", None)
            if inner is None:
                continue
            inflight = getattr(inner, "_inflight_per_server", None)
            if not isinstance(inflight, dict):
                continue
            for addr in list(inflight):
                inflight[addr] = 0

        # We no longer hard-drain SGLang from this path (see docstring),
        # so num_running_reqs may briefly be non-zero as cancelled
        # requests finish their current forward pass.  Report the count
        # for visibility but always return ready=True — the asyncio
        # cancellation has already removed the Python-side state and
        # any in-flight tail will free its KV slot within a tick or two.
        logger.info(
            "reset_training_engine: epoch=%d cancelled=%d stragglers=%d "
            "dropped_results=%d sglang_running=%s elapsed=%.2fs",
            epoch,
            cancelled,
            stragglers,
            dropped,
            running,
            time.monotonic() - _t0,
        )
        return {
            "cancelled": cancelled,
            "stragglers": stragglers,
            "dropped_results": dropped,
            "sglang_running": running,
            "ready_for_eval": True,
            "reset_epoch": epoch,
            "elapsed": time.monotonic() - _t0,
        }

    # ------------------------------------------------------------------
    # Eval lifecycle
    # ------------------------------------------------------------------

    async def eval_start(self) -> dict[str, str]:
        """Reset eval tracking state before submitting eval tasks."""
        # Cancel any orphaned tasks from a previous eval round
        for task in self._eval_running_tasks.values():
            task.cancel()
        self._eval_running_tasks.clear()
        self._eval_completed_results.clear()
        self._next_eval_task_id = 0
        logger.info("RaaS3 eval_start: eval state reset")
        return {"status": "eval_started"}

    async def eval_end(self) -> dict[str, str]:
        """Clear eval tracking state after all eval results are collected."""
        # Cancel any remaining eval tasks
        for task in self._eval_running_tasks.values():
            task.cancel()
        self._eval_running_tasks.clear()
        self._eval_completed_results.clear()
        self._next_eval_task_id = 0
        logger.info("RaaS3 eval_end: eval state cleared")
        return {"status": "eval_ended"}

    async def eval_submit(
        self, data: dict[str, Any], workflow_id: str = "default"
    ) -> int:
        """Submit an eval task directly as an asyncio.Task."""
        self._ensure_async_state()
        if not self._eval_engines:
            raise RuntimeError("Cannot eval_submit: eval engine is not initialized.")
        if workflow_id not in self._workflows:
            raise KeyError(
                f"Unknown workflow_id: {workflow_id!r}. "
                f"Registered: {list(self._workflows.keys())}"
            )

        task_id = self._next_eval_task_id
        self._next_eval_task_id += 1
        workflow = self._workflows[workflow_id]
        eval_engine_or_group = (
            self._build_engine_group(eval=True)
            if len(self._eval_engines) > 1
            else self._eval_engine
        )

        async def _run():
            async with self._eval_semaphore:  # cap concurrent eval prefills
                async with eval_engine_or_group.managed_session():
                    return await workflow.arun_episode(eval_engine_or_group, data)

        task = asyncio.create_task(_run())
        task.add_done_callback(lambda t: self._on_eval_task_done(task_id, t))
        self._eval_running_tasks[task_id] = task
        logger.debug(
            "RaaS3 eval_submit: task_id=%d (inflight=%d, eval_cap=%d)",
            task_id,
            len(self._eval_running_tasks),
            self._max_eval_concurrency,
        )
        return task_id

    def _on_eval_task_done(self, task_id: int, task: asyncio.Task) -> None:
        """Callback when an eval asyncio.Task completes."""
        self._eval_running_tasks.pop(task_id, None)
        try:
            self._eval_completed_results[task_id] = task.result()
        except BaseException as exc:
            # BaseException catches CancelledError (Python 3.9+)
            self._eval_completed_results[task_id] = {"ok": False, "error": repr(exc)}

    async def eval_pull(
        self,
        max_items: int = 256,
        timeout: float = 0.0,
    ) -> dict[str, Any]:
        """Pull completed eval results."""
        items = self._drain_completed(self._eval_completed_results, max_items)
        if not items and timeout > 0:
            await asyncio.sleep(min(timeout, 0.1))
            items = self._drain_completed(self._eval_completed_results, max_items)
        return {
            "items": items,
            "inflight": len(self._eval_running_tasks),
            "pending": len(self._eval_completed_results),
            "total_submitted": self._next_eval_task_id,
        }

    # ------------------------------------------------------------------
    # Weight updates — TCP pull path
    # ------------------------------------------------------------------

    def _get_engine(self, model_id: str = "default") -> Any:
        """Resolve engine by model_id."""
        if model_id not in self._engines:
            raise KeyError(
                f"Unknown model_id: {model_id!r}. "
                f"Available: {list(self._engines)}"
            )
        return self._engines[model_id]

    async def notify_version(
        self,
        model_id: str,
        version: int,
        sender_endpoint: str,
    ) -> dict[str, Any]:
        """Per-model version notification from AstraFlow.

        Pulls weights for a single model from its sender agent and loads
        them into that model's inference engine.  Each model is handled
        independently — no coordination with other models.

        Parameters
        ----------
        model_id : str
            Which model to update (e.g. ``"model0"``).
        version : int
            New weight version.
        sender_endpoint : str
            ``"host:port"`` of this model's sender agent.
        """
        local_version = self._weight_versions.get(model_id, 0)
        logger.info(
            "notify_version: model=%s v=%d (local=%d) endpoint=%s",
            model_id, version, local_version, sender_endpoint,
        )

        if version <= local_version:
            return {
                "ok": True,
                "model_id": model_id,
                "pulled": False,
                "reason": f"version={version} <= local={local_version}",
            }

        if not sender_endpoint:
            return {
                "ok": False,
                "model_id": model_id,
                "pulled": False,
                "reason": "no sender_endpoint",
            }

        # Serialise the entire weight-update cycle per model so that
        # two concurrent notify_version calls cannot race on the same
        # safetensors file (the root cause of sglang Bus-error crashes).
        lock = self._weight_update_locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._weight_update_locks[model_id] = lock

        async with lock:
            return await self._do_weight_update(
                model_id, version, sender_endpoint,
            )

    async def _do_weight_update(
        self,
        model_id: str,
        version: int,
        sender_endpoint: str,
    ) -> dict[str, Any]:
        """Execute the weight-update cycle (must be called under lock).

        Separated from ``notify_version`` so the lock scope covers the
        full pull → pause → load → resume cycle.
        """
        import time as _time

        # Re-check version under the lock: an earlier update that was
        # queued ahead of us may have already advanced past our version.
        local_version = self._weight_versions.get(model_id, 0)
        if version <= local_version:
            logger.info(
                "notify_version: model=%s v=%d already loaded (local=%d) "
                "after acquiring lock, skipping",
                model_id, version, local_version,
            )
            return {
                "ok": True,
                "model_id": model_id,
                "pulled": False,
                "reason": f"version={version} <= local={local_version} (after lock)",
            }

        # Phase 1: Pull weights to disk
        loop = asyncio.get_event_loop()
        t_start = _time.monotonic()
        try:
            pull_result = await loop.run_in_executor(
                None, self._pull_weights_to_disk, sender_endpoint, model_id,
            )
        except Exception as exc:
            logger.error(
                "notify_version: pull failed for %s: %s",
                model_id, exc, exc_info=True,
            )
            return {"ok": False, "model_id": model_id, "reason": str(exc)}

        if not pull_result.get("ok"):
            return {"ok": False, "model_id": model_id, "pull_result": pull_result}

        t_pull = _time.monotonic()

        # Phase 2: Load into this model's engine only
        shm_path = pull_result["shm_path"]
        use_lora = pull_result.get("use_lora", False)
        engine = self._get_engine(model_id)

        self._weight_update_in_progress = True
        self._weight_update_started_at = _time.monotonic()
        # Invalidate adaptive metrics cache AND last-good snapshot: anything
        # polled before the pause is stale, and we must not let the stale
        # fallback path use pre-pause data after the resume.
        self._metrics_cache = (0.0, [])
        self._metrics_cache_ok = False
        self._last_good_snapshot = None
        self._last_good_snapshot_at = 0.0
        print(
            f"[RaaS-avail-debug] snapshot WIPED (pre-pause) model={model_id} v={version}",
            flush=True,
        )
        t_pause = t_load = t_resume = _time.monotonic()
        try:
            await loop.run_in_executor(None, engine.pause_generation)
            t_pause = _time.monotonic()
            await loop.run_in_executor(
                None, engine.load_weights_from_path, shm_path, use_lora,
            )
            t_load = _time.monotonic()
            await loop.run_in_executor(None, engine.continue_generation)
            t_resume = _time.monotonic()
        finally:
            # Always clear weight-update state, even on exception, so
            # /availability doesn't get stuck at "weight_update" forever
            # if pause/load/resume raises (e.g. sglang scheduler crash).
            self._weight_update_in_progress = False
            self._weight_update_started_at = 0.0
            # Second invalidation: sglang schedulers may have been reset
            # by the pause/resume, so the first post-update poll refetches.
            self._metrics_cache = (0.0, [])
            self._metrics_cache_ok = False
            self._last_good_snapshot = None
            self._last_good_snapshot_at = 0.0
            print(
                f"[RaaS-avail-debug] snapshot WIPED (post-resume) model={model_id} v={version} "
                f"weight_update_in_progress=False",
                flush=True,
            )

        # Update per-model version tracking
        self._weight_versions[model_id] = version
        try:
            engine.set_version(version)
        except Exception:
            logger.warning(
                "notify_version: set_version failed for %s",
                model_id, exc_info=True,
            )

        # Sync LoRA state to eval engines
        if use_lora:
            for eval_eng in self._eval_engines.values():
                inner = getattr(eval_eng, "_engine", eval_eng)
                inner.lora_initialized = True

        _timing = (
            f"notify_version: loaded {model_id} v={version}  "
            f"| lora={use_lora}  "
            f"| pull={t_pull - t_start:.2f}s pause={t_pause - t_pull:.2f}s "
            f"load={t_load - t_pause:.2f}s resume={t_resume - t_load:.2f}s "
            f"total={t_resume - t_start:.2f}s"
        )
        logger.info(_timing)
        print(f"{self._tag} {_timing}", flush=True)

        return {
            "ok": True,
            "model_id": model_id,
            "version": version,
            "pull_result": pull_result,
        }

    def _pull_weights_to_disk(
        self, sender_http_endpoint: str, model_id: str = "default"
    ) -> dict:
        """Pull weights via TCP (full) or HTTP (delta) and save as safetensors.

        Each ``model_id`` gets its own ``RaaSWeightReceiver`` (with its own
        TCP session, ZMQ listener, and sender registration) and its own shm
        subdirectory so that multi-model pulls don't overwrite each other.

        RaaS decides which mode to use based on sender capabilities and
        local state. Delta is used when the sender has a delta ready whose
        base version matches our local version.

        Does NOT load into the engine — caller is responsible for that.
        Returns ``{"ok": True, "shm_path": ..., "version": ..., "pull_time": ...}``
        on success.
        """
        import requests as http_requests
        import time as _time
        from astraflow.raas.server.tcp_receiver import RaaSWeightReceiver

        host, port_str = sender_http_endpoint.rsplit(":", 1)
        port = int(port_str)
        sender_url = f"http://{host}:{port}"

        # Lazy init: one receiver per model_id, each registered with its
        # own sender and writing to its own shm subdirectory.
        receiver = self._tcp_receivers.get(model_id)
        if receiver is None:
            logger.info(
                "Initializing RaaS TCP receiver for model=%s sender=%s:%d",
                model_id, host, port,
            )
            instance_tag = self._engine_id or str(self._service_port or "default")
            shm_subdir = os.path.join(
                RaaSWeightReceiver.DEFAULT_SHM_DIR, instance_tag, model_id
            )
            receiver = RaaSWeightReceiver(shm_dir=shm_subdir)
            receiver.start(
                sender_http_endpoint,
                identity_port=self._service_port or 0,
            )
            self._tcp_receivers[model_id] = receiver

        instance_id = receiver._get_local_ip() + f":{self._service_port or 0}"

        # Decide transfer mode based on sender capabilities + local state
        mode = self._choose_transfer_mode(sender_url, model_id)

        t_start = _time.monotonic()

        if mode == "delta":
            result = self._pull_delta(
                sender_url, instance_id, model_id, receiver,
            )
            # If delta failed or was refused, fall back to full
            if not result.get("ok"):
                fallback = result.get("fallback", "")
                reason = result.get("reason", "unknown")
                logger.info(
                    "_pull_weights_to_disk: delta refused for model=%s "
                    "(reason=%s), falling back to full",
                    model_id, reason,
                )
                mode = "full"
                result = self._pull_full(
                    sender_url, instance_id, model_id, receiver,
                )
        else:
            result = self._pull_full(
                sender_url, instance_id, model_id, receiver,
            )

        t_done = _time.monotonic()
        result["pull_time"] = t_done - t_start
        result["mode"] = mode
        logger.info(
            "_pull_weights_to_disk: model=%s mode=%s saved to %s in %.2fs",
            model_id, mode, result.get("shm_path", "?"), t_done - t_start,
        )
        return result

    def _choose_transfer_mode(
        self, sender_url: str, model_id: str,
    ) -> str:
        """Decide whether to use delta or full transfer for this pull."""
        import requests as http_requests

        try:
            resp = http_requests.get(
                f"{sender_url}/get_capabilities", timeout=5,
            )
            resp.raise_for_status()
            caps = resp.json()
        except Exception as e:
            logger.info(
                "_choose_transfer_mode: model=%s get_capabilities failed: %s, using full",
                model_id, e,
            )
            return "full"

        local_v = self._weight_versions.get(model_id, 0)
        logger.info(
            "_choose_transfer_mode: model=%s local_v=%d caps=%s",
            model_id, local_v, caps,
        )

        if "delta" not in caps.get("strategies", []):
            return "full"
        if not caps.get("delta_ready", False):
            logger.info(
                "_choose_transfer_mode: model=%s delta not ready, using full",
                model_id,
            )
            return "full"
        if local_v == 0:
            logger.info(
                "_choose_transfer_mode: model=%s first pull (local_v=0), using full",
                model_id,
            )
            return "full"
        # Periodic full sync: every N-th version uses full for resync.
        # Check target version (local + 1) so that full sync aligns with
        # the version being pulled (e.g. v50, v100, ...) rather than the
        # version after (v51, v101, ...).
        if (
            self._delta_full_sync_interval > 0
            and (local_v + 1) % self._delta_full_sync_interval == 0
        ):
            logger.info(
                "_choose_transfer_mode: model=%s periodic full sync "
                "(local_v=%d, target_v=%d, interval=%d), using full",
                model_id, local_v, local_v + 1,
                self._delta_full_sync_interval,
            )
            return "full"
        # Delta is usable when delta_base_version matches our local version.
        delta_base = caps.get("delta_base_version", -1)
        delta_ver = caps.get("delta_version", -1)
        sender_ver = caps.get("version", -1)
        if local_v != delta_base:
            logger.info(
                "_choose_transfer_mode: model=%s base mismatch "
                "(local=%d, delta_base=%d, delta_ver=%d, sender_ver=%d), using full",
                model_id, local_v, delta_base, delta_ver, sender_ver,
            )
            return "full"
        logger.info(
            "_choose_transfer_mode: model=%s using DELTA "
            "(local_v=%d → delta_v=%d, base=%d, sender_v=%d)",
            model_id, local_v, delta_ver, delta_base, sender_ver,
        )
        return "delta"

    def _pull_full(
        self,
        sender_url: str,
        instance_id: str,
        model_id: str,
        receiver: Any,
    ) -> dict:
        """Pull full weights via TCP (existing path)."""
        import requests as http_requests
        import time as _time

        t0 = _time.monotonic()
        resp = http_requests.post(
            f"{sender_url}/request_transfer",
            json={"instance_id": instance_id, "mode": "full"},
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
        t_http = _time.monotonic()

        if not result.get("ok"):
            raise RuntimeError(
                f"Weight sender refused full transfer for model={model_id}, "
                f"instance_id={instance_id}: {result}"
            )

        receiver.wait_for_transfer()
        t_tcp = _time.monotonic()
        save_result = receiver.save_as_safetensors()
        t_save = _time.monotonic()

        print(
            f"[RaaS] Full pull profile: model={model_id}, "
            f"http_request={t_http - t0:.3f}s, "
            f"tcp_wait={t_tcp - t_http:.3f}s, "
            f"save_safetensors={t_save - t_tcp:.3f}s, "
            f"total={t_save - t0:.3f}s, "
            f"size={receiver.buffer.length / (1024 * 1024):.1f} MB",
            flush=True,
        )

        result["shm_path"] = save_result["shm_path"]
        result["use_lora"] = save_result.get("use_lora", False)
        result["tcp_time"] = t_tcp - t0
        result["save_time"] = t_save - t_tcp
        result["tcp_size_mb"] = (receiver.buffer.length if receiver.buffer else 0) / (1024 * 1024)
        return result

    def _pull_delta(
        self,
        sender_url: str,
        instance_id: str,
        model_id: str,
        receiver: Any,
    ) -> dict:
        """Pull sparse delta via TCP and apply to existing safetensors.

        The sender transfers delta bytes via the same TCP engine used for
        full transfers. The receiver gets the bytes in its existing buffer,
        then applies them as a delta patch to the safetensors file.
        """
        import requests as http_requests
        import time as _time

        t0 = _time.monotonic()

        # Step 1: HTTP request to sender (triggers TCP transfer on sender side)
        resp = http_requests.post(
            f"{sender_url}/request_transfer",
            json={"instance_id": instance_id, "mode": "delta"},
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()
        t_http = _time.monotonic()

        if not result.get("ok"):
            return result  # caller handles fallback

        # Step 2: Wait for TCP transfer completion via ZMQ signal
        receiver.wait_for_transfer()
        t_tcp = _time.monotonic()

        # Step 3: Read delta bytes from receiver buffer
        delta_size = result.get("delta_size", 0)
        delta_bytes = bytes(receiver.buffer.buffer[:delta_size].numpy())
        t_read = _time.monotonic()

        # Step 4: Apply delta to the receiver's persistent in-RAM
        # weights buffer, then write the result to a fresh safetensors
        # file and atomic-rename. Done in-process — see
        # ``RaaSWeightReceiver.apply_delta_and_save`` for the rationale
        # behind this approach (avoids any MAP_SHARED hazard with
        # SGLang's prior mmap of the safetensors file).
        #
        # If the apply raises, the persistent buffer state may be
        # inconsistent. Unlink the on-disk safetensors so the next pull
        # falls back to a full transfer, which re-populates the buffer.
        try:
            # Sender response uses "version" for the target/applied version.
            target_v = int(result.get("version", -1))
            shm_path = receiver.apply_delta_and_save(
                delta_bytes, target_version=target_v,
            )
        except Exception as exc:
            logger.error(
                "_pull_delta: apply_delta_and_save failed for model=%s: %s",
                model_id, exc, exc_info=True,
            )
            sf_path = os.path.join(receiver.shm_dir, "model.safetensors")
            try:
                os.unlink(sf_path)
                logger.info(
                    "_pull_delta: unlinked possibly-corrupt %s; next "
                    "pull will fall back to full transfer",
                    sf_path,
                )
            except OSError:
                pass
            return {
                "ok": False,
                "fallback": "apply_failed",
                "reason": f"apply_delta_and_save failed: {exc}",
            }
        t_patch = _time.monotonic()

        print(
            f"[RaaS] Delta pull profile: model={model_id}, "
            f"http_request={t_http - t0:.3f}s, "
            f"tcp_wait={t_tcp - t_http:.3f}s, "
            f"read_buf={t_read - t_tcp:.3f}s, "
            f"apply_patch={t_patch - t_read:.3f}s, "
            f"total={t_patch - t0:.3f}s, "
            f"delta_size={delta_size / (1024 * 1024):.1f} MB",
            flush=True,
        )

        result["shm_path"] = shm_path
        result["tcp_time"] = t_tcp - t0
        result["patch_time"] = t_patch - t_tcp
        result["tcp_size_mb"] = delta_size / (1024 * 1024)
        return result

    def _get_self_endpoint(self) -> str:
        """Return this RaaS instance's HTTP endpoint for identification."""
        for ep in self._rollout_instances:
            return ep
        host = self._service_host
        port = self._service_port or 5000
        return f"http://{host}:{port}"

    # ------------------------------------------------------------------
    # Periodic generation stats
    # ------------------------------------------------------------------

    async def _log_generation_stats_loop(self) -> None:
        """Periodically log per-DP generation stats (every 10s)."""
        while True:
            await asyncio.sleep(10)
            try:
                for mid, engine in self._engines.items():
                    inner = getattr(engine, "_engine", None)
                    if inner is None or not hasattr(inner, "get_generation_stats"):
                        continue
                    stats = inner.get_generation_stats()
                    if not stats:
                        continue
                    inflight_parts = []
                    completed_parts = []
                    total_inflight = 0
                    total_completed = 0
                    for idx, (addr, s) in enumerate(stats.items()):
                        inflight_parts.append(str(s["inflight"]))
                        completed_parts.append(str(s["completed"]))
                        total_inflight += s["inflight"]
                        total_completed += s["completed"]
                    n_dps = len(stats)
                    prefix = f"[{mid}] " if len(self._engines) > 1 else ""
                    msg = (
                        f"{prefix}{n_dps} DPs | inflight: "
                        f"{'/'.join(inflight_parts)} (total {total_inflight}) | "
                        f"completed: {'/'.join(completed_parts)} "
                        f"(total {total_completed})"
                    )
                    logger.info("[RaaS3 GenStats] %s", msg)
                # Adaptive availability summary (only when enabled). Prefer
                # the fresh cache; fall back to the last-good snapshot so we
                # keep logging useful numbers during brief telemetry stalls.
                if self._adaptive_enabled:
                    if self._metrics_cache_ok:
                        cached_at, per_engine = self._metrics_cache
                        src = "fresh"
                    elif self._last_good_snapshot is not None:
                        cached_at = self._last_good_snapshot_at
                        per_engine = self._last_good_snapshot
                        src = "stale"
                    else:
                        cached_at = 0.0
                        per_engine = None
                        src = "blind"
                    if per_engine is not None:
                        total_waiting = 0
                        total_target = 0
                        max_kv_frac = 0.0
                        for eng in per_engine:
                            if eng is None:
                                continue
                            for srv in eng.get("results") or []:
                                if not isinstance(srv, dict):
                                    continue
                                total_waiting += int(
                                    srv.get("sglang:num_queue_reqs", 0)
                                )
                                kv = float(srv.get("sglang:token_usage", 0.0))
                                if kv > max_kv_frac:
                                    max_kv_frac = kv
                                total_target += self._target_per_dp
                        logger.info(
                            "[RaaS Adaptive] total_waiting=%d/%d "
                            "max_kv_frac=%.2f target_per_dp=%d step=%d "
                            "src=%s cache_age=%.0fms",
                            total_waiting, total_target, max_kv_frac,
                            self._target_per_dp, self._step_size, src,
                            (time.monotonic() - cached_at) * 1000,
                        )
                    else:
                        logger.info(
                            "[RaaS Adaptive] src=blind (no telemetry yet)"
                        )
            except Exception:
                logger.debug("_log_generation_stats_loop: error", exc_info=True)

    # ------------------------------------------------------------------
    # Destroy
    # ------------------------------------------------------------------

    async def destroy(self) -> dict[str, Any]:
        """Destroy engine and clean up all state."""
        # Cancel generation stats logger
        if self._gen_stats_task is not None:
            self._gen_stats_task.cancel()
            self._gen_stats_task = None

        # Cancel all running training tasks
        for task in list(self._running_tasks.values()):
            task.cancel()
        for task in list(self._running_tasks.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._running_tasks.clear()

        # Cancel all running eval tasks
        for task in list(self._eval_running_tasks.values()):
            task.cancel()
        for task in list(self._eval_running_tasks.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._eval_running_tasks.clear()

        loop = asyncio.get_event_loop()
        for eid, eng in self._eval_engines.items():
            try:
                await loop.run_in_executor(None, eng.destroy)
            except Exception:
                logger.exception("Failed to destroy eval engine %s", eid)

        for eid, eng in self._engines.items():
            try:
                await loop.run_in_executor(None, eng.destroy)
            except Exception:
                logger.exception("Failed to destroy engine %s", eid)

        engine_id = self._engine_id
        self._eval_engines.clear()
        self._engines.clear()
        self._engine_id = None
        self._backend = None
        self._workflows.clear()
        self._completed_results.clear()
        self._eval_completed_results.clear()
        self._next_eval_task_id = 0
        self._running = None
        self._semaphore = None
        self._eval_semaphore = None
        self._max_eval_concurrency = 0
        self._next_task_id = 0
        self._max_concurrency = 0
        self._status = "idle"
        self._status_message = ""

        return {"engine_id": engine_id, "destroyed": True}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_allocation_mode(
        raw: str | dict | AllocationMode,
    ) -> AllocationMode:
        return AllocationMode.resolve(raw)

    @staticmethod
    def _get_train_dp_size(allocation_mode: AllocationMode) -> int | None:
        train = allocation_mode.train
        if train is None:
            return None
        return train.dp_size

    @staticmethod
    def _resolve_max_concurrency(
        config: InferenceEngineConfig,
        train_data_parallel_size: int | None,
    ) -> int:
        configured = config.max_concurrent_rollouts
        if configured is None:
            configured = config.consumer_batch_size
        dp_size = max(1, int(train_data_parallel_size or 1))
        return max(1, int(configured) // dp_size)

    def _launch_inference_servers(
        self,
        engine: Any,
        backend: str,
        config: Any,
        allocation_mode: AllocationMode,
        gpu_offset: int = 0,
    ) -> None:
        """Launch backend inference servers according to generation topology.

        Parameters
        ----------
        gpu_offset : int
            Skip the first ``gpu_offset`` visible devices before assigning
            GPUs to servers.  Used by ``bootstrap_multi_model`` to give each
            model its own slice of the GPU pool.
        """
        n_servers = allocation_mode.gen.dp_size
        gpus_per_server = allocation_mode.gen_instance_size
        if n_servers <= 0 or gpus_per_server <= 0:
            raise ValueError(
                f"Invalid inference topology: "
                f"n_servers={n_servers}, gpus_per_server={gpus_per_server}."
            )

        total_fallback = max(
            n_servers * gpus_per_server + gpu_offset,
            int(getattr(config.cluster, "n_gpus_per_node", 0) or 0),
        )
        visible_devices = self._extract_visible_devices(total_fallback)
        # Skip already-claimed devices
        visible_devices = visible_devices[gpu_offset:]
        required = n_servers * gpus_per_server
        if len(visible_devices) < required:
            raise RuntimeError(
                f"Not enough visible devices. "
                f"need={required}, visible={len(visible_devices)}."
            )

        logger.info(
            "Launching inference servers: backend=%s n_servers=%s gpus_per_server=%s",
            backend,
            n_servers,
            gpus_per_server,
        )

        env_key = current_platform.device_control_env_var or "CUDA_VISIBLE_DEVICES"
        try:
            host = gethostip()
        except OSError:
            host = "127.0.0.1"
            logger.warning("Failed to resolve host IP. Falling back to %s.", host)

        reserved_ports: set[int] = set()
        if self._service_port is not None:
            reserved_ports.add(self._service_port)
        launch_jobs: list[tuple[int, list[str], dict[str, Any]]] = []
        for server_idx in range(n_servers):
            start = server_idx * gpus_per_server
            end = start + gpus_per_server
            server_devices = visible_devices[start:end]
            server_args = self._build_server_args(
                backend=backend,
                config=config,
                allocation_mode=allocation_mode,
                server_idx=server_idx,
            )
            server_port = find_free_ports(1, exclude_ports=reserved_ports)[0]
            reserved_ports.add(server_port)
            server_args["host"] = host
            server_args["port"] = server_port
            launch_env: dict[str, str] = {env_key: ",".join(server_devices)}

            tcp_mode = getattr(config, "weight_transfer_mode", "tcp") == "tcp"
            if tcp_mode:
                if not self._service_port:
                    raise RuntimeError(
                        "TCP weight transfer requires a valid RaaS service port"
                    )
                rollout_mgr_url = f"http://127.0.0.1:{self._service_port}"
                # TCP v2: only pass rollout_manager_address for registration.
                # No receiver agent runs inside SGLang — RaaS handles TCP
                # receiving and loads weights via native /update_weights_from_disk.
                server_args["rollout_manager_address"] = rollout_mgr_url
                launch_env["ASTRAFLOW_AUTOPATCH"] = "true"

            server_args["__launch_env__"] = launch_env
            launch_jobs.append((server_idx, server_devices, server_args))

        from concurrent.futures import as_completed

        weight_transfer_enabled = (
            getattr(config, "weight_transfer_mode", "tcp") == "tcp"
        )

        with ThreadPoolExecutor(max_workers=n_servers) as pool:
            futures = [pool.submit(engine.launch_server, job[2]) for job in launch_jobs]
            future_to_job = {
                future: job for future, job in zip(futures, launch_jobs, strict=True)
            }
            for future in as_completed(future_to_job):
                server_idx, server_devices, s_args = future_to_job[future]
                future.result()
                logger.info(
                    "Launched inference server %s/%s with devices=%s",
                    server_idx + 1,
                    n_servers,
                    ",".join(server_devices),
                )
                # Directly register the rollout instance so _get_self_endpoint()
                # can return the real SGLang host:port rather than falling back
                # to 0.0.0.0:service_port.
                #
                # The HTTP-callback path inside HttpServerPatch.patched_launch_server
                # cannot work here because uvicorn does not accept new connections
                # until all startup events have completed, and _launch_inference_servers
                # runs *inside* the startup event (via run_in_executor).
                if weight_transfer_enabled:
                    endpoint = f"http://{s_args['host']}:{s_args['port']}"
                    self._rollout_instances[endpoint] = {
                        "host": s_args["host"],
                        "port": s_args["port"],
                        "weight_version": 0,
                    }
                    logger.info(
                        "Registered rollout instance (direct, no HTTP callback): %s",
                        endpoint,
                    )

    @staticmethod
    def _build_server_args(
        *,
        backend: str,
        config: Any,
        allocation_mode: AllocationMode,
        server_idx: int,
    ) -> dict[str, Any]:
        """Build backend-specific server launch args for one server instance."""
        if backend == "sglang":
            from astraflow.raas.api.cli_args import SGLangConfig

            sglang_config = deepcopy(config.sglang)
            if (
                hasattr(sglang_config, "random_seed")
                and sglang_config.random_seed is not None
            ):
                sglang_config.random_seed += server_idx
            return SGLangConfig.build_args(
                sglang_config=sglang_config,
                tp_size=allocation_mode.gen.tp_size,
                base_gpu_id=0,
            )
        if backend == "vllm":
            from astraflow.raas.api.cli_args import vLLMConfig

            vllm_config = deepcopy(config.vllm)
            if hasattr(vllm_config, "seed") and vllm_config.seed is not None:
                vllm_config.seed += server_idx
            return vLLMConfig.build_args(
                vllm_config=vllm_config,
                tp_size=allocation_mode.gen.tp_size,
                pp_size=allocation_mode.gen.pp_size,
            )
        raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def _extract_visible_devices(total_fallback: int) -> list[str]:
        """Read visible device ids from env or build a sequential fallback list."""
        env_key = current_platform.device_control_env_var or "CUDA_VISIBLE_DEVICES"
        env_value = os.environ.get(env_key, "").strip()
        if env_value:
            return [x.strip() for x in env_value.split(",") if x.strip()]
        return [str(i) for i in range(total_fallback)]
