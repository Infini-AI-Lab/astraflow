"""Configuration dataclasses for the AstraFlow HTTP service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for one agent/trainer connecting to AstraFlow."""

    workflow_spec: dict[str, Any] = field(default_factory=dict)
    """Rollout workflow specification."""

    buffer_size: int = 65536
    """Maximum number of samples in the fresh buffer."""

    filter_function: str | None = None
    """Name of a registered buffer filter function."""

    max_staleness: int | None = None
    """Maximum version staleness for samples; older samples are discarded."""

    replay_size: int | None = None
    """Maximum number of samples in the replay buffer."""

    replay_ratio: float = 0.0
    """Fraction of each training batch to fill from replay buffer."""

    selection_fn: str | None = None
    """Name of a registered replay selection function."""

    reward_norm: dict[str, Any] | None = None
    """Reward normalization config (passed to AstraFlow)."""

    tokenizer_path: str | None = None
    """Path to tokenizer (HuggingFace model name or local path)."""

    data_root: str | None = None
    """Root directory for pre-downloaded datasets (offline mode).

    When set, every entry in ``rollout_dataset`` and ``eval_datasets``
    that does not already specify ``offline_dir`` gets one auto-derived
    as ``f"{data_root}/{name}"`` — where ``name`` is the dict key for
    eval datasets, and the value of ``dataset_name`` (falling back to
    the dataset_fn module name) for the rollout dataset.

    Use ``examples/math/offline/download_math_datasets.py`` to populate
    this directory.
    """

    rollout_dataset: dict[str, Any] | None = None
    """Dataset config for rollout data acquisition.

    Uses ``dataset_fn`` (a Python import path) to create the dataset.

    Example::

        rollout_dataset:
          dataset_fn: "astraflow.dataflow.dataset.deepscaler:get_deepscaler_rl_dataset"
          max_length: 2000
          batch_size: 1
    """

    eval_datasets: dict[str, Any] | None = None
    """Eval dataset configurations.

    Each entry uses ``dataset_fn`` to create the dataset.

    Example::

        eval_datasets:
          math500:
            dataset_fn: "astraflow.dataflow.dataset.math500:get_math500_test_dataset"
            max_length: 2000
            repeat: 1
    """

    eval_workflows: dict[str, dict[str, Any]] | None = None
    """Named eval workflow definitions.

    Each entry defines a reusable workflow spec that can be referenced
    by eval datasets via the ``eval_workflow`` key.

    Example::

        eval_workflows:
          code_eval:
            workflow_cls: "livecodebench_single_turn"
            reward_fn: "livecodebench_reward"
            gconfig_overrides:
              temperature: 0.6
              n_samples: 1
    """

    eval_workflow_specs: dict[str, dict[str, Any]] | None = None
    """Deprecated. Use ``eval_workflows`` + per-dataset ``eval_workflow`` instead."""

    eval_freq_versions: int | None = None
    """Run evaluation every N model versions. None disables periodic eval."""

    expected_model_ids: list[str] | None = None
    """List of model_ids that must register before data acquisition starts.

    When set, AstraFlow waits for all listed model_ids to call
    ``trainer_ready`` before starting the submit/collect threads.
    When None (default), data acquisition starts as soon as the first
    trainer connects (backward-compatible single-model behavior).
    """

    curator: str | None = None
    """Name of a registered ``PromptCurator`` for selective rollout.

    When set, ``DataAcquisition`` consults
    ``curator.should_submit(data, version=...)`` for every prompt before
    submitting to RaaS, and feeds rollout outcomes back via
    ``curator.update(...)``.  When ``None`` (default), every prompt is
    submitted in dataloader order — bit-for-bit identical to behavior
    before this option existed.
    """

    curator_args: dict[str, Any] = field(default_factory=dict)
    """Constructor kwargs for the curator named in ``curator``."""


@dataclass
class EvalConfig:
    """Global eval configuration for the AstraFlow service."""

    timeout: float = 3600
    """Timeout in seconds for each eval run. Default 3600s (1 hour)."""


@dataclass
class ServiceConfig:
    """Top-level AstraFlow service configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    agent: AgentConfig = field(default_factory=AgentConfig)
    """Single agent configuration (flattened from the YAML)."""
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint_dir: str | None = None
    """Directory for buffer checkpoints. If None, buffer is not persisted."""

    # Global RaaS pool configuration.
    heartbeat_interval: float = 30.0
    """Seconds between heartbeat polls for the global RaaSPool."""

    heartbeat_max_failures: int = 30
    """Consecutive heartbeat failures before a RaaS instance is auto-deregistered.

    Weight updates temporarily saturate the RaaS event loop (30-40s per
    model), causing transient heartbeat timeouts.  With two models doing
    back-to-back updates, the unresponsive window can span several
    minutes.  30 failures × 30s interval = 15 min tolerance.
    """

    raas_initialize_timeout: float = 120.0
    """Maximum seconds to wait for a newly registered RaaS to become ready."""

    balance_report_freq: int = 10
    """Auto-save balance report every N training versions. 0 disables."""
