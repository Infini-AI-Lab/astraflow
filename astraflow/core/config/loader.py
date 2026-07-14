"""Unified config loader for AstraFlow.

Config format:
    experiment.yaml has named sections: experiment, raas, astraflow, trainer
    raas.yaml is a separate hardware config merged with experiment.yaml
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _set_if_missing(d: dict, key: str, value: Any) -> None:
    """Set d[key] = value only if key is not already in d."""
    if key not in d:
        d[key] = value


# ---------------------------------------------------------------------------
# Multi-config loading
# ---------------------------------------------------------------------------


def load_and_merge_configs(config_paths: list[str]) -> dict:
    """Load multiple YAML files and deep-merge them in order.

    Later files override earlier ones.  Returns the merged dict.
    """
    result: dict = {}
    for p in config_paths:
        path = Path(p).absolute()
        assert path.exists(), f"Config file {path} does not exist."
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        result = _deep_merge(result, data)
    return result


# ---------------------------------------------------------------------------
# Auto-propagation
# ---------------------------------------------------------------------------


def _auto_propagate_trainer(
    experiment: dict,
    raas_section: dict,
    trainer: dict,
    is_multi_model: bool = False,
) -> dict:
    """Apply auto-propagation rules to a trainer config dict.

    Fills in fields from ``experiment`` and ``raas`` sections
    when they are not explicitly set in the trainer dict.
    """
    trainer = deepcopy(trainer)
    exp = experiment

    # -- Top-level experiment fields --
    _set_if_missing(trainer, "experiment_name", exp.get("experiment_name"))
    _set_if_missing(trainer, "trial_name", exp.get("trial_name"))
    _set_if_missing(trainer, "tokenizer_path", exp.get("tokenizer_path"))
    _set_if_missing(trainer, "seed", exp.get("seed"))
    _set_if_missing(trainer, "weight_transfer_mode", exp.get("weight_transfer_mode"))
    wts = exp.get("weight_transfer_strategies")
    if wts is not None:
        _set_if_missing(trainer, "weight_transfer_strategies", wts)

    fileroot = exp.get("fileroot", "")
    model_id = trainer.get("model_id")
    model_path = exp.get("model_path")
    dtype = exp.get("dtype")

    # -- Multi-model trial_name suffix (only when using trainer_base pattern) --
    if is_multi_model and model_id and "trial_name" in trainer:
        base_trial = trainer["trial_name"]
        if not base_trial.endswith(f"-{model_id}"):
            trainer["trial_name"] = f"{base_trial}-{model_id}"

    trial_name = trainer.get("trial_name", "")

    # -- Resolve fileroot with actual values --
    if fileroot:
        resolved_fileroot = fileroot.replace(
            "${experiment.experiment_name}", exp.get("experiment_name", "")
        ).replace(
            "${experiment.trial_name}", exp.get("trial_name", "")
        )
    else:
        resolved_fileroot = ""

    # -- Model-specific propagation via model_id --
    models = raas_section.get("models", {})
    model_spec = models.get(model_id, {}) if model_id else {}
    model_gconfig = model_spec.get("gconfig", {})
    # model_path: prefer model-specific path, fall back to experiment.model_path
    resolved_model_path = model_spec.get("path", model_path)

    # -- actor --
    if "actor" not in trainer:
        trainer["actor"] = {}
    actor = trainer["actor"]
    _set_if_missing(actor, "path", resolved_model_path)
    _set_if_missing(actor, "dtype", dtype)
    if model_gconfig.get("max_new_tokens"):
        _set_if_missing(actor, "max_new_tokens", model_gconfig["max_new_tokens"])
    _set_if_missing(actor, "init_from_scratch", False)
    _set_if_missing(actor, "disable_dropout", True)

    # -- ref --
    if "ref" not in trainer:
        trainer["ref"] = {}
    ref = trainer["ref"]
    _set_if_missing(ref, "path", resolved_model_path)
    _set_if_missing(ref, "dtype", dtype)
    _set_if_missing(ref, "init_from_scratch", False)
    _set_if_missing(ref, "disable_dropout", True)
    _set_if_missing(ref, "optimizer", None)

    # -- cluster --
    if "cluster" not in trainer:
        trainer["cluster"] = {}
    cluster = trainer["cluster"]
    _set_if_missing(cluster, "fileroot", resolved_fileroot)
    if "name_resolve" not in cluster:
        nr_root = resolved_fileroot
        if model_id:
            nr_root = os.path.join(resolved_fileroot, f"name_resolve_{model_id}")
        else:
            nr_root = os.path.join(resolved_fileroot, "name_resolve")
        cluster["name_resolve"] = {"type": "nfs", "nfs_record_root": nr_root}

    # -- saver / recover fileroot --
    # For multi-model setups, each model gets its own checkpoint subtree
    # so that trainer_model0 and trainer_model1 don't overwrite each other.
    ckpt_fileroot = resolved_fileroot
    if is_multi_model and model_id:
        ckpt_fileroot = os.path.join(resolved_fileroot, model_id)

    if "saver" not in trainer:
        trainer["saver"] = {}
    saver = trainer["saver"]
    _set_if_missing(saver, "fileroot", ckpt_fileroot)
    _set_if_missing(saver, "freq_epochs", None)
    _set_if_missing(saver, "freq_steps", None)
    _set_if_missing(saver, "freq_secs", None)

    # -- recover --
    if "recover" not in trainer:
        trainer["recover"] = {}
    recover = trainer["recover"]
    _set_if_missing(recover, "fileroot", ckpt_fileroot)

    # -- stats_logger --
    if "stats_logger" not in trainer:
        trainer["stats_logger"] = {}
    sl = trainer["stats_logger"]
    _set_if_missing(sl, "experiment_name", exp.get("experiment_name"))
    _set_if_missing(sl, "trial_name", trial_name)
    _set_if_missing(sl, "fileroot", resolved_fileroot)
    if "wandb" not in sl:
        sl["wandb"] = {}
    wandb = sl["wandb"]
    _set_if_missing(wandb, "project", exp.get("experiment_name"))
    _set_if_missing(wandb, "name", trial_name)

    # -- Common defaults --
    _set_if_missing(trainer, "enable_offload", False)
    _set_if_missing(trainer, "total_train_epochs", 1000000)

    return trainer


# ---------------------------------------------------------------------------
# Config extraction for each component
# ---------------------------------------------------------------------------


def _find_base_key(trainer_key: str, raw: dict) -> str | None:
    """Find a base config key for inheritance.

    For ``trainer_key="trainer_model0"``, looks for ``"trainer_base"`` in *raw*.
    """
    parts = trainer_key.rsplit("_", 1)
    if len(parts) == 2:
        candidate = f"{parts[0]}_base"
        if candidate in raw and candidate != trainer_key:
            return candidate
    return None


def load_trainer_config(raw: dict, trainer_key: str | None = None) -> dict:
    """Extract trainer config with auto-propagation.

    Args:
        raw: The full merged YAML dict (must have ``experiment`` key).
        trainer_key: Which trainer section to use. If None, uses ``trainer``.
            For multi-model: ``trainer_model0``, ``trainer_model1``, etc.

    Returns:
        A flat dict ready to be loaded into a trainer config dataclass.
    """
    experiment = raw.get("experiment", {})
    raas_section = raw.get("raas", {})
    trainer_key = trainer_key or "trainer"

    if trainer_key not in raw:
        raise KeyError(
            f"Trainer key '{trainer_key}' not found in config. "
            f"Available keys: {list(raw.keys())}"
        )

    trainer = dict(raw[trainer_key])

    base_key = _find_base_key(trainer_key, raw)
    is_multi_model = base_key is not None
    if is_multi_model:
        trainer = _deep_merge(raw[base_key], trainer)

    trainer = _auto_propagate_trainer(
        experiment, raas_section, trainer, is_multi_model=is_multi_model
    )

    # Pass engine dict directly as allocation_mode
    engine = trainer.pop("engine", None)
    if engine and "allocation_mode" not in trainer:
        trainer["allocation_mode"] = engine

    return trainer


def load_dataflow_config(raw: dict) -> dict:
    """Extract AstraFlow dataflow service config.

    Returns a dict with service-level fields + an ``agent`` sub-dict
    built from the ``dataflow`` section.
    """
    experiment = raw.get("experiment", {})
    dataflow = raw.get("dataflow", {})
    raas_section = raw.get("raas", {})

    # Auto-derive expected_model_ids from raas.models keys
    models = raas_section.get("models", {})
    model_ids = list(models.keys()) if models else None

    # Build agent config from dataflow section
    agent_fields = dict(dataflow)
    host = agent_fields.pop("host", "0.0.0.0")
    port = agent_fields.pop("port", 8000)

    _set_if_missing(agent_fields, "expected_model_ids", model_ids)
    _set_if_missing(agent_fields, "tokenizer_path", experiment.get("tokenizer_path"))

    # Propagate dump_dir into workflow_spec so it reaches the workflow constructor
    dump_dir = agent_fields.pop("dump_dir", None)
    if dump_dir is not None:
        # Resolve ${experiment.*} interpolation variables
        dump_dir = dump_dir.replace(
            "${experiment.experiment_name}", experiment.get("experiment_name", "")
        ).replace(
            "${experiment.trial_name}", experiment.get("trial_name", "")
        )
        ws = agent_fields.get("workflow_spec", {})
        _set_if_missing(ws, "dump_dir", dump_dir)
        agent_fields["workflow_spec"] = ws

    # Map buffer sub-dict to flat agent fields
    buffer = agent_fields.pop("buffer", {})
    if buffer:
        _set_if_missing(agent_fields, "buffer_size", buffer.get("size"))
        _set_if_missing(agent_fields, "replay_size", buffer.get("replay_size"))
        _set_if_missing(agent_fields, "replay_ratio", buffer.get("replay_ratio"))
        _set_if_missing(agent_fields, "max_staleness", buffer.get("max_staleness"))
        _set_if_missing(agent_fields, "queue_order", buffer.get("queue_order"))
        _set_if_missing(agent_fields, "filter_function", buffer.get("filter_function"))

    # Extract service-level config from agent_fields (they're in the
    # dataflow: YAML section but belong to ServiceConfig, not AgentConfig).
    balance_report_freq = agent_fields.pop("balance_report_freq", None)

    # Derive checkpoint_dir
    experiment_name = experiment.get("experiment_name", "")
    trial_name = experiment.get("trial_name", "")
    fileroot = experiment.get("fileroot", "")
    if fileroot:
        resolved_fileroot = fileroot.replace(
            "${experiment.experiment_name}", experiment_name
        ).replace(
            "${experiment.trial_name}", trial_name
        )
    else:
        resolved_fileroot = ""
    checkpoint_dir = os.path.join(resolved_fileroot, "checkpoints") if resolved_fileroot else None

    result = {
        "host": host,
        "port": port,
        "agent": agent_fields,
        "checkpoint_dir": checkpoint_dir,
        "experiment": experiment,
    }
    if balance_report_freq is not None:
        result["balance_report_freq"] = balance_report_freq
    return result


def load_raas_config(raw: dict) -> dict:
    """Extract RaaS config from merged experiment + raas files.

    The ``raw`` dict comes from merging experiment.yaml and raas.yaml.
    Extracts experiment-level fields + raas section + hardware overrides.

    Returns a flat dict ready to be loaded into a RaaSConfig dataclass.
    """
    experiment = raw.get("experiment", {})
    raas_section = raw.get("raas", {})

    result = {}

    # Inject experiment fields
    result["tokenizer_path"] = experiment.get("tokenizer_path")
    result["seed"] = experiment.get("seed")
    result["weight_transfer_mode"] = experiment.get("weight_transfer_mode")

    # Models: merge model specs from raas section
    models = raas_section.get("models", {})
    model_path = experiment.get("model_path")
    dtype = experiment.get("dtype")

    # Build full model specs (inject experiment-level defaults)
    resolved_models = {}
    for mid, mspec in models.items():
        m = deepcopy(mspec) if isinstance(mspec, dict) else {}
        _set_if_missing(m, "model_path", model_path)
        _set_if_missing(m, "tokenizer_path", experiment.get("tokenizer_path"))
        if "sglang" not in m:
            m["sglang"] = {}
        sglang = m["sglang"]
        _set_if_missing(sglang, "model_path", model_path)
        _set_if_missing(sglang, "random_seed", experiment.get("seed"))
        _set_if_missing(sglang, "dtype", dtype)
        resolved_models[mid] = m

    result["models"] = resolved_models

    # Copy top-level hardware fields from raas.yaml (these override)
    for key in ("cluster", "sglang", "vllm", "rollout",
                "delta_full_sync_interval"):
        if key in raw and key not in ("experiment", "raas", "dataflow",
                                       "trainer", "trainer_base"):
            result[key] = raw[key]

    # Also check raas section for these (from experiment.yaml raas section)
    for key in ("rollout", "delta_full_sync_interval"):
        if key in raas_section and key not in result:
            result[key] = raas_section[key]

    # Pass engine dict directly as allocation_mode
    engine = raw.get("engine", None)
    if engine and "allocation_mode" not in result:
        result["allocation_mode"] = engine

    # Deep-merge sglang: raas section sglang (from experiment.yaml) as base,
    # top-level sglang (from raas.yaml) as override
    raas_sglang = raas_section.get("sglang", {})
    top_sglang = raw.get("sglang", {})
    if raas_sglang or top_sglang:
        merged_sglang = _deep_merge(raas_sglang, top_sglang)
        for mid, m in resolved_models.items():
            m["sglang"] = _deep_merge(merged_sglang, m.get("sglang", {}))
        result["sglang"] = merged_sglang

    return result
