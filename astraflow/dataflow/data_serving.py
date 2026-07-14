"""Data-serving component for AstraFlow.

This component owns fresh and replay storage and exposes serving APIs used by
trainers and higher-level orchestrators.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from typing import Any, Protocol

import torch

from astraflow.dataflow.replay_selectors import ReplaySelectionFn, get_replay_selection
from astraflow.dataflow.rollout_buffer import RolloutBuffer
from astraflow.dataflow.utils import NormConfig, concat_padded_tensors, get_batch_size

logger = logging.getLogger(__name__)


def _parse_model_index(model_id: str) -> int | None:
    """Extract numeric index from model_id string (e.g. 'model0' -> 0)."""
    m = re.match(r"model(\d+)$", model_id)
    if m:
        return int(m.group(1))
    return None


class DataServing(Protocol):
    """Serving interface for fresh/replay rollout data."""

    def put(
        self,
        batch: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bool: ...

    def get(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> dict[str, Any] | None: ...

    def get_with_metadata(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None: ...

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None: ...

    def get_replay_batch(
        self,
        batch_size: int,
        ids: list[int] | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None: ...

    def get_training_batch(
        self,
        *,
        expected_sample_count: int,
        replay_ratio: float,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...

    def replay_size(self) -> int: ...

    def size(self) -> int: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

    def get_and_reset_put_stats(self) -> dict[str, int]: ...

    def get_and_reset_consume_stats(self) -> dict[str, int]: ...

    def accumulate_acquisition_stats(
        self, stats: dict[str, float | int]
    ) -> None: ...

    def get_and_reset_acquisition_stats(
        self, model_id: str | None = None
    ) -> dict[str, float | int]: ...

    def close(self) -> None: ...

    def is_closed(self) -> bool: ...


def _empty_acquisition_stats() -> dict[str, float | int]:
    """Zeroed stats dict for metrics pushed from DataAcquisition."""
    return {
        "accepted": 0,
        "filtered": 0,
        "total": 0,
        "pre_filter_reward_sum": 0.0,
        "pre_filter_reward_count": 0,
        "post_filter_reward_sum": 0.0,
        "post_filter_reward_count": 0,
    }


class AstraDataServing:
    """RolloutBuffer-backed data-serving implementation.

    Fresh samples are stored in the main queue, while consumed samples are
    tracked in replay storage managed by ``RolloutBuffer``.
    """

    def __init__(
        self,
        *,
        buffer_size: int = 65536,
        buffer_debug: bool = False,
        reward_norm: NormConfig | None = None,
        max_staleness: int | None = None,
        replay_max_size: int | None = None,
        replay_selection_fn: ReplaySelectionFn | str | None = None,
        queue_order: str = "edf",
    ):
        resolved_replay_selection_fn = self._resolve_replay_selection_fn(
            replay_selection_fn
        )
        self.buffer = RolloutBuffer(
            max_size=buffer_size,
            debug=buffer_debug,
            reward_norm=reward_norm,
            max_staleness=max_staleness,
            replay_max_size=replay_max_size,
            replay_selection_fn=resolved_replay_selection_fn,
            queue_order=queue_order,
        )
        self._acq_stats_lock = threading.Lock()
        self._acq_stats: dict[str, float | int] = _empty_acquisition_stats()
        self._agent_metrics_lock = threading.Lock()
        self._agent_metrics_accum: dict[str, list[float]] = {}

    @staticmethod
    def _resolve_replay_selection_fn(
        replay_selection_fn: ReplaySelectionFn | str | None,
    ) -> ReplaySelectionFn | None:
        """Resolve replay-selection policy from callable or registry name."""
        if replay_selection_fn is None:
            return None
        if callable(replay_selection_fn):
            return replay_selection_fn
        if isinstance(replay_selection_fn, str):
            try:
                return get_replay_selection(replay_selection_fn)
            except ValueError as e:
                logger.warning("%s. Using default replay selection.", e)
                return None
        logger.warning(
            "Unsupported replay_selection_fn type %s. Using default replay selection.",
            type(replay_selection_fn),
        )
        return None

    def put(
        self,
        batch: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bool:
        return self.buffer.put(batch, metadata=metadata, timeout=timeout)

    def get(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> dict[str, Any] | None:
        return self.buffer.get(timeout=timeout, current_version=current_version)

    def get_with_metadata(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        return self.buffer.get_with_metadata(
            timeout=timeout,
            current_version=current_version,
        )

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        result = self.buffer.get_batch(
            batch_size=batch_size,
            timeout=timeout,
            current_version=current_version,
        )
        if self.buffer.debug:
            got = get_batch_size(result[0]) if result is not None else 0
            print(
                f"AstraFlow serving: fresh get_batch requested={batch_size}, got={got}",
                flush=True,
            )
        return result

    def get_replay_batch(
        self,
        batch_size: int,
        ids: list[int] | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        result = self.buffer.get_replay_batch(
            batch_size=batch_size,
            ids=ids,
            current_version=current_version,
        )
        return result

    def get_training_batch(
        self,
        *,
        expected_sample_count: int,
        replay_ratio: float,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build one training batch by mixing fresh and replay samples."""
        replay_target = int(expected_sample_count * replay_ratio)
        replay_target = min(max(replay_target, 0), expected_sample_count)

        replay_batch = None
        replay_metadatas: list[dict[str, Any]] = []
        if replay_target > 0:
            replay_result = self.get_replay_batch(
                batch_size=replay_target,
                current_version=current_version,
            )
            if replay_result is not None:
                replay_batch, replay_metadatas = replay_result

        replay_count = get_batch_size(replay_batch) if replay_batch is not None else 0
        fresh_target = expected_sample_count - replay_count
        if fresh_target > 0:
            result = self.get_batch(
                batch_size=fresh_target,
                timeout=timeout,
                current_version=current_version,
            )
            if result is None:
                if self.is_closed():
                    raise RuntimeError("Rollout buffer is closed")
                raise RuntimeError(
                    f"Timeout waiting for {fresh_target} examples from buffer "
                    f"(timeout={timeout})"
                )
            batch, metadatas = result
        else:
            batch = None
            metadatas = []

        if batch is not None and replay_batch is not None:
            batch = concat_padded_tensors([batch, replay_batch])
            metadatas = metadatas + replay_metadatas
        elif replay_batch is not None:
            batch = replay_batch
            metadatas = replay_metadatas

        if batch is None:
            raise RuntimeError("Failed to construct training batch from serving layer")
        if self.buffer.debug:
            final_count = get_batch_size(batch)
            print(
                "AstraFlow serving: "
                f"mixed get_training_batch expected={expected_sample_count}, "
                f"replay_target={replay_target}, replay_got={replay_count}, "
                f"fresh_target={fresh_target}, final_got={final_count}",
                flush=True,
            )
        return batch, metadatas

    def replay_size(self) -> int:
        return self.buffer.replay_size()

    def size(self) -> int:
        return self.buffer.size()

    def state_dict(self) -> dict[str, Any]:
        return self.buffer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.buffer.load_state_dict(state_dict)

    def get_and_reset_put_stats(self) -> dict[str, int]:
        return self.buffer.get_and_reset_put_stats()

    def get_and_reset_consume_stats(self) -> dict[str, int]:
        return self.buffer.get_and_reset_consume_stats()

    def accumulate_acquisition_stats(
        self, stats: dict[str, float | int]
    ) -> None:
        """Accumulate stats pushed from data acquisition (thread-safe)."""
        with self._acq_stats_lock:
            for k, v in stats.items():
                if k in self._acq_stats:
                    self._acq_stats[k] = type(self._acq_stats[k])(
                        self._acq_stats[k] + v
                    )

    def get_and_reset_acquisition_stats(self) -> dict[str, float | int]:
        """Get and reset all acquisition-pushed stats."""
        with self._acq_stats_lock:
            stats = dict(self._acq_stats)
            self._acq_stats = _empty_acquisition_stats()
        return stats

    def accumulate_agent_metrics(
        self, metrics: dict[str, float]
    ) -> None:
        """Accumulate workflow-defined agent metrics (thread-safe).

        Each metric value is appended to a per-key list.  The final
        aggregated value (mean) is computed at retrieval time.
        """
        with self._agent_metrics_lock:
            for name, value in metrics.items():
                if name not in self._agent_metrics_accum:
                    self._agent_metrics_accum[name] = []
                self._agent_metrics_accum[name].append(float(value))

    def get_and_reset_agent_metrics(self) -> dict[str, float]:
        """Get aggregated agent metrics and reset the accumulator.

        Returns a dict of ``{metric_name: mean_value}``.
        """
        with self._agent_metrics_lock:
            result: dict[str, float] = {}
            for name, values in self._agent_metrics_accum.items():
                if values:
                    result[name] = sum(values) / len(values)
            self._agent_metrics_accum = {}
        return result

    def close(self) -> None:
        self.buffer.close()

    def is_closed(self) -> bool:
        return self.buffer.is_closed()


_DEFAULT_MODEL_ID = "default"


class MultiModelDataServing:
    """Per-model buffer management for training.

    Each model gets its own ``AstraDataServing`` (fresh + replay buffer).
    When a trajectory is ingested via ``put()``, it is copied into each
    relevant model's buffer with the ``loss_mask`` pre-filtered so only
    that model's tokens are trainable.  Models with zero tokens in a
    trajectory are skipped entirely.

    For single-model training, pass ``model_ids=None`` (or omit it).
    A single buffer keyed by ``"default"`` is created and ``put()``
    stores data without any loss_mask filtering — equivalent to the
    old ``AstraDataServing`` behavior.

    Parameters
    ----------
    model_ids : list[str] | None
        Model identifiers (e.g. ``["model0", "model1"]``).
        ``None`` for single-model mode.
    buffer_size : int
        Default fresh buffer size for each model.
    buffer_debug : bool
        Enable debug logging on buffers.
    reward_norm : NormConfig | None
        Reward normalization config.
    max_staleness : int | None
        Default max staleness for each model.
    replay_max_size : int | None
        Default replay buffer size for each model.
    replay_selection_fn : ReplaySelectionFn | str | None
        Default replay selection function.
    per_model_config : dict[str, dict] | None
        Optional per-model overrides.  Keys are model_ids, values are dicts
        with optional keys ``buffer_size``, ``max_staleness``,
        ``replay_max_size``, ``queue_order``.
    """

    def __init__(
        self,
        model_ids: list[str] | None = None,
        *,
        buffer_size: int = 65536,
        buffer_debug: bool = False,
        reward_norm: NormConfig | None = None,
        max_staleness: int | None = None,
        replay_max_size: int | None = None,
        replay_selection_fn: ReplaySelectionFn | str | None = None,
        queue_order: str = "edf",
        per_model_config: dict[str, dict] | None = None,
    ):
        if model_ids is not None and len(model_ids) > 0:
            self.model_ids = list(model_ids)
            self._is_multi_model = True
        else:
            self.model_ids = [_DEFAULT_MODEL_ID]
            self._is_multi_model = False

        self.buffers: dict[str, AstraDataServing] = {}
        for mid in self.model_ids:
            cfg = (per_model_config or {}).get(mid, {})
            buf = AstraDataServing(
                buffer_size=cfg.get("buffer_size", buffer_size),
                buffer_debug=buffer_debug,
                reward_norm=reward_norm,
                max_staleness=cfg.get("max_staleness", max_staleness),
                replay_max_size=cfg.get("replay_max_size", replay_max_size),
                replay_selection_fn=replay_selection_fn,
                queue_order=cfg.get("queue_order", queue_order),
            )
            buf.buffer.label = f"[{mid}] " if self._is_multi_model else ""
            self.buffers[mid] = buf

    @property
    def buffer(self) -> RolloutBuffer | None:
        """Compatibility alias for single-model callers expecting .buffer."""
        if not self._is_multi_model:
            return self.buffers[_DEFAULT_MODEL_ID].buffer
        return None

    def _resolve_model_id(self, model_id: str | None) -> str:
        """Map None → default key for single-model mode."""
        if model_id is None:
            return _DEFAULT_MODEL_ID
        return model_id

    def put(
        self,
        batch: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Dispatch one example to each relevant model's buffer.

        In single-model mode, stores directly without loss_mask filtering.
        In multi-model mode, checks ``model_ids`` tensor and dispatches a
        filtered copy to each relevant model's buffer.
        """
        # Single-model fast path — no copying, no filtering
        if not self._is_multi_model:
            return self.buffers[_DEFAULT_MODEL_ID].put(batch, metadata, timeout)

        model_ids_tensor = batch.get("model_ids")
        loss_mask = batch.get("loss_mask")

        if model_ids_tensor is None or loss_mask is None:
            # No model_ids — put into all buffers with original loss_mask
            success = True
            for buf in self.buffers.values():
                if not buf.put(copy.deepcopy(batch), metadata, timeout):
                    success = False
            return success

        success = True
        for mid, buf in self.buffers.items():
            model_idx = _parse_model_index(mid)
            if model_idx is None:
                logger.warning(
                    "Cannot parse model index from model_id=%r, skipping", mid
                )
                continue
            model_mask = (model_ids_tensor == model_idx).to(loss_mask.dtype)
            if model_mask.sum() == 0:
                continue  # Skip — this model has no tokens in this example
            batch_copy = copy.deepcopy(batch)
            batch_copy["loss_mask"] = loss_mask * model_mask
            if not buf.put(batch_copy, metadata, timeout):
                success = False
        return success

    def get(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> dict[str, Any] | None:
        """Pop one sample from the default buffer (single-model compat)."""
        return self.buffers[_DEFAULT_MODEL_ID].get(
            timeout=timeout, current_version=current_version,
        )

    def get_with_metadata(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Pop one sample + metadata from the default buffer."""
        return self.buffers[_DEFAULT_MODEL_ID].get_with_metadata(
            timeout=timeout, current_version=current_version,
        )

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Pop a batch from the default buffer (single-model compat)."""
        return self.buffers[_DEFAULT_MODEL_ID].get_batch(
            batch_size=batch_size, timeout=timeout,
            current_version=current_version,
        )

    def get_replay_batch(
        self,
        batch_size: int,
        ids: list[int] | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Sample from the default replay buffer (single-model compat)."""
        return self.buffers[_DEFAULT_MODEL_ID].get_replay_batch(
            batch_size=batch_size, ids=ids,
            current_version=current_version,
        )

    def get_training_batch(
        self,
        model_id: str | None = None,
        *,
        expected_sample_count: int,
        replay_ratio: float,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Pull a training batch from a specific model's buffer."""
        key = self._resolve_model_id(model_id)
        if key not in self.buffers:
            raise KeyError(
                f"Unknown model_id={model_id!r}, expected one of {self.model_ids}"
            )
        return self.buffers[key].get_training_batch(
            expected_sample_count=expected_sample_count,
            replay_ratio=replay_ratio,
            timeout=timeout,
            current_version=current_version,
        )

    def size(self, model_id: str | None = None) -> int:
        """Return fresh-buffer sample count.

        If model_id is given, return that model's count.
        If None: in single-model mode returns the default buffer count;
        in multi-model mode returns the total across all buffers.
        """
        if model_id is not None:
            return self.buffers[self._resolve_model_id(model_id)].size()
        if not self._is_multi_model:
            return self.buffers[_DEFAULT_MODEL_ID].size()
        return sum(buf.size() for buf in self.buffers.values())

    def replay_size(self, model_id: str | None = None) -> int:
        """Return replay sample count (same semantics as ``size``)."""
        if model_id is not None:
            return self.buffers[self._resolve_model_id(model_id)].replay_size()
        if not self._is_multi_model:
            return self.buffers[_DEFAULT_MODEL_ID].replay_size()
        return sum(buf.replay_size() for buf in self.buffers.values())

    def get_and_reset_put_stats(
        self, model_id: str | None = None
    ) -> dict[str, int]:
        """Get and reset put stats.  If model_id is None, aggregate all."""
        if model_id is not None:
            return self.buffers[self._resolve_model_id(model_id)].get_and_reset_put_stats()
        combined: dict[str, int] = {}
        for buf in self.buffers.values():
            for k, v in buf.get_and_reset_put_stats().items():
                combined[k] = combined.get(k, 0) + v
        return combined

    def get_and_reset_consume_stats(
        self, model_id: str | None = None
    ) -> dict[str, int]:
        """Get and reset consume stats.  If model_id is None, aggregate all."""
        if model_id is not None:
            return self.buffers[self._resolve_model_id(model_id)].get_and_reset_consume_stats()
        combined: dict[str, int] = {}
        for buf in self.buffers.values():
            for k, v in buf.get_and_reset_consume_stats().items():
                combined[k] = combined.get(k, 0) + v
        return combined

    def accumulate_agent_metrics(
        self, metrics: dict[str, float]
    ) -> None:
        """Push agent metrics into all per-model buffers."""
        for buf in self.buffers.values():
            buf.accumulate_agent_metrics(metrics)

    def get_and_reset_agent_metrics(
        self, model_id: str | None = None
    ) -> dict[str, float]:
        """Get and reset agent metrics. If model_id given, per-model only."""
        if model_id is not None:
            return self.buffers[self._resolve_model_id(model_id)].get_and_reset_agent_metrics()
        # Single-model mode: just return the default buffer's metrics.
        if not self._is_multi_model:
            return self.buffers[_DEFAULT_MODEL_ID].get_and_reset_agent_metrics()
        # Multi-model without specific model_id: aggregate all (shouldn't
        # normally happen — callers should specify model_id).
        combined: dict[str, list[float]] = {}
        for buf in self.buffers.values():
            for k, v in buf.get_and_reset_agent_metrics().items():
                combined.setdefault(k, []).append(v)
        return {k: sum(vs) / len(vs) for k, vs in combined.items()}

    def accumulate_acquisition_stats(
        self, stats: dict[str, float | int]
    ) -> None:
        """Push acquisition stats into all per-model buffers."""
        for buf in self.buffers.values():
            buf.accumulate_acquisition_stats(stats)

    def get_and_reset_acquisition_stats(
        self, model_id: str | None = None
    ) -> dict[str, float | int]:
        """Get and reset acquisition stats. If model_id given, per-model only."""
        if model_id is not None:
            return self.buffers[self._resolve_model_id(model_id)].get_and_reset_acquisition_stats()
        combined: dict[str, float | int] = _empty_acquisition_stats()
        for buf in self.buffers.values():
            s = buf.get_and_reset_acquisition_stats()
            for k in combined:
                combined[k] = type(combined[k])(combined[k] + s.get(k, 0))
        return combined

    def state_dict(self) -> dict[str, Any]:
        """Serialize all per-model buffer states."""
        return {
            "per_model": {
                mid: buf.state_dict() for mid, buf in self.buffers.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore per-model buffer states.

        Handles two checkpoint formats:
        - New format: ``{"per_model": {"model0": {...}, "model1": {...}}}``
        - Old single-buffer format (backward compat): loads the same state
          into all per-model buffers so training can resume after migration.
        """
        per_model = state_dict.get("per_model")
        if per_model is not None:
            for mid, buf in self.buffers.items():
                if mid in per_model:
                    buf.load_state_dict(per_model[mid])
        elif "buffer" in state_dict or "max_size" in state_dict:
            # Old single-buffer checkpoint — load into all per-model buffers
            logger.warning(
                "Loading old single-buffer checkpoint into %d per-model buffers. "
                "All models will share the same initial buffer contents.",
                len(self.buffers),
            )
            for buf in self.buffers.values():
                buf.load_state_dict(state_dict)

    def close(self) -> None:
        for buf in self.buffers.values():
            buf.close()

    def is_closed(self) -> bool:
        return any(buf.is_closed() for buf in self.buffers.values())
