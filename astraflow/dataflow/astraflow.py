"""AstraFlow orchestration over data acquisition and data serving.

`AstraFlow` composes two components:
- data acquisition: rollout generation and ingestion into serving.
- data serving: fresh/replay buffering and trainer-facing retrieval APIs.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Callable
from typing import Any

from torchdata.stateful_dataloader import StatefulDataLoader

from astraflow.dataflow.buffer_filters import BufferFilterFn

logger = logging.getLogger(__name__)
from astraflow.dataflow.data_acquisition import (
    AstraDataAcquisition,
    DataAcquisition,
)
from astraflow.dataflow.data_serving import (
    AstraDataServing,
    DataServing,
    MultiModelDataServing,
)
from astraflow.dataflow.replay_selectors import ReplaySelectionFn
from astraflow.dataflow.utils import NormConfig


class AstraFlow:
    """Data-flow orchestrator with explicit acquisition and serving components."""

    def __init__(
        self,
        rollout: Any,
        rollout_dataloader: StatefulDataLoader,
        workflow_spec: dict[str, Any],
        *,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        buffer_size: int = 65536,
        buffer_debug: bool = False,
        reward_norm: NormConfig | None = None,
        filter_fn: BufferFilterFn | str | None = None,
        max_staleness: int | None = None,
        replay_max_size: int | None = None,
        replay_selection_fn: ReplaySelectionFn | str | None = None,
        producer_error_backoff: float = 0.5,
        data_serving: DataServing | None = None,
        data_acquisition: DataAcquisition | None = None,
        expected_model_ids: list[str] | None = None,
        per_model_buffer_config: dict[str, dict] | None = None,
        curator: Any = None,
        curator_args: dict[str, Any] | None = None,
    ):
        """Initialize AstraFlow with acquisition and serving components."""
        self.rollout = rollout
        self.rollout_dataloader = rollout_dataloader
        self.workflow_spec = workflow_spec
        self.should_accept_fn = should_accept_fn

        if data_serving is None:
            data_serving = MultiModelDataServing(
                model_ids=expected_model_ids,
                buffer_size=buffer_size,
                buffer_debug=buffer_debug,
                reward_norm=reward_norm,
                max_staleness=max_staleness,
                replay_max_size=replay_max_size,
                replay_selection_fn=replay_selection_fn,
                per_model_config=per_model_buffer_config,
            )
        self.data_serving = data_serving

        if data_acquisition is None:
            data_acquisition = AstraDataAcquisition(
                rollout=rollout,
                rollout_dataloader=rollout_dataloader,
                workflow_spec=workflow_spec,
                filter_fn=filter_fn,
                curator=curator,
                curator_args=curator_args,
                publish_fn=self.data_serving.put,
                data_serving=self.data_serving,
                debug=buffer_debug,
                error_backoff=producer_error_backoff,
            )
        self.data_acquisition = data_acquisition

        # Compatibility alias for callers expecting direct buffer access.
        self.buffer = getattr(self.data_serving, "buffer", None)

    def start(self) -> None:
        """Start data acquisition."""
        self.data_acquisition.start()

    def pause(self) -> None:
        """Pause data acquisition and underlying rollout submission."""
        self.data_acquisition.pause()

    def resume(self) -> None:
        """Resume data acquisition and underlying rollout submission."""
        self.data_acquisition.resume()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop data acquisition."""
        self.data_acquisition.stop(timeout=timeout)

    def close(self) -> None:
        """Close AstraFlow and all owned components."""
        self.stop()
        self.data_serving.close()

    def is_closed(self) -> bool:
        """Return whether the serving component has been closed."""
        return self.data_serving.is_closed()

    def is_running(self) -> bool:
        """Return whether the acquisition component is running."""
        return self.data_acquisition.is_running()

    def get_metrics(self, reset: bool = False) -> dict[str, int | bool]:
        """Get AstraFlow metrics from acquisition and serving components."""
        put_stats = self.data_acquisition.get_ingest_stats(reset=reset)
        # Always use get_and_reset for consume stats — the state_dict path
        # was fragile and format-dependent.
        consume_stats = self.data_serving.get_and_reset_consume_stats() if reset else {}
        producer_stats = self.data_acquisition.get_stats(reset=reset)

        return {
            "buffer_size": self.size(),
            "replay_size": self.replay_size(),
            "is_closed": self.is_closed(),
            "is_paused": self.data_acquisition.is_paused(),
            "is_running": self.is_running(),
            "accepted": int(put_stats.get("accepted", 0)),
            "filtered": int(put_stats.get("filtered", 0)),
            "total": int(put_stats.get("total", 0)),
            "consumed": int(consume_stats.get("consumed", 0)),
            "skipped_stale": int(consume_stats.get("skipped_stale", 0)),
            "producer_batches": int(producer_stats.get("producer_batches", 0)),
            "producer_errors": int(producer_stats.get("producer_errors", 0)),
            "producer_put_failures": int(
                producer_stats.get("producer_put_failures", 0)
            ),
        }

    # Serving-compatible APIs for easy integration.
    def put(
        self,
        batch: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Insert one batch into the serving component."""
        return self.data_serving.put(batch, metadata=metadata, timeout=timeout)

    def get(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> dict[str, Any] | None:
        """Pop one sample from fresh data serving."""
        return self.data_serving.get(timeout=timeout, current_version=current_version)

    def get_with_metadata(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Pop one sample and metadata from fresh data serving."""
        return self.data_serving.get_with_metadata(
            timeout=timeout,
            current_version=current_version,
        )

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Pop a batch of fresh samples and metadata."""
        return self.data_serving.get_batch(
            batch_size=batch_size,
            timeout=timeout,
            current_version=current_version,
        )

    def get_replay_batch(
        self,
        batch_size: int,
        ids: list[int] | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Sample a batch from replay serving."""
        return self.data_serving.get_replay_batch(
            batch_size=batch_size,
            ids=ids,
            current_version=current_version,
        )

    def get_training_batch(
        self,
        *,
        expected_sample_count: int,
        replay_ratio: float,
        timeout: float | None = None,
        current_version: int | None = None,
        model_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Get one trainer-ready batch mixed from fresh and replay buffers.

        ``model_id`` routes to the corresponding per-model buffer.
        In single-model mode it can be omitted (routes to the default buffer).
        """
        return self.data_serving.get_training_batch(
            model_id,
            expected_sample_count=expected_sample_count,
            replay_ratio=replay_ratio,
            timeout=timeout,
            current_version=current_version,
        )

    def replay_size(self, model_id: str | None = None) -> int:
        """Return replay sample count."""
        return self.data_serving.replay_size(model_id)

    def size(self, model_id: str | None = None) -> int:
        """Return fresh-buffer sample count."""
        return self.data_serving.size(model_id)

    def state_dict(self) -> dict[str, Any]:
        """Serialize serving state plus acquisition counters."""
        state = self.data_serving.state_dict()
        state["producer_stats"] = self.data_acquisition.get_stats(reset=False)
        state["ingest_stats"] = self.data_acquisition.get_ingest_stats(reset=False)
        state["reward_stats"] = self.data_acquisition.get_reward_stats(reset=False)
        state["curator"] = self.data_acquisition.get_curator_state()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore serving state and acquisition counters."""
        self.data_serving.load_state_dict(state_dict)
        self.data_acquisition.set_stats(state_dict.get("producer_stats"))
        self.data_acquisition.set_ingest_stats(state_dict.get("ingest_stats"))
        self.data_acquisition.set_reward_stats(state_dict.get("reward_stats"))
        self.data_acquisition.load_curator_state(state_dict.get("curator"))

    def save_buffer(self, path: str) -> None:
        """Save buffer state to disk.

        The underlying ``RolloutBuffer.state_dict()`` holds the buffer lock
        for the entire snapshot, so concurrent puts and gets are blocked
        during serialization. No pause is needed.
        """
        state = {
            "buffer": self.state_dict(),
            "dataloader": self.rollout_dataloader.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Buffer saved to %s", path)

    def load_buffer(self, path: str) -> bool:
        """Load buffer state from disk.

        Should be called **before** ``start()`` so that producers have not
        yet begun writing into the buffer.

        Returns True if the buffer was loaded, False if the file was not found.
        """
        if not os.path.exists(path):
            logger.info("No buffer checkpoint at %s, starting fresh.", path)
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.load_state_dict(state["buffer"])
        if "dataloader" in state:
            self.rollout_dataloader.load_state_dict(state["dataloader"])
        logger.info(
            "Buffer loaded from %s (size=%d, replay=%d)",
            path,
            self.size(),
            self.replay_size(),
        )
        return True

    def get_and_reset_acquisition_stats(
        self, model_id: str | None = None
    ) -> dict[str, float | int]:
        """Fetch and reset acquisition stats from data_serving (per-model)."""
        return self.data_serving.get_and_reset_acquisition_stats(model_id)

    def get_and_reset_consume_stats(
        self, model_id: str | None = None
    ) -> dict[str, int]:
        """Fetch and reset serving consume counters."""
        return self.data_serving.get_and_reset_consume_stats(model_id)

    def get_and_reset_agent_metrics(
        self, model_id: str | None = None
    ) -> dict[str, float]:
        """Fetch and reset workflow-defined agent metrics."""
        return self.data_serving.get_and_reset_agent_metrics(model_id)

    def get_per_raas_stats(self, reset: bool = False) -> dict[str, dict[str, int]]:
        """Fetch per-RaaS throughput stats from data acquisition."""
        return self.data_acquisition.get_per_raas_stats(reset=reset)
