"""Standalone AstraFlow rollout buffer.

This implementation is AstraFlow-specific and does not import the core rollout
buffer module.
"""

from __future__ import annotations

import copy
import heapq
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import torch

from astraflow.dataflow.replay_selectors import ReplaySelectionFn, select_latest
from astraflow.dataflow.utils import (
    Normalization,
    NormConfig,
    concat_padded_tensors,
    get_batch_size,
)

logger = logging.getLogger(__name__)


def _slice_tensor_dict(data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Slice tensor-like fields by batch dimension while keeping metadata fields."""
    sliced_data: dict[str, Any] = {}
    batch_size = -1
    if "attention_mask" in data and torch.is_tensor(data["attention_mask"]):
        batch_size = data["attention_mask"].shape[0]
    for key, value in data.items():
        if (torch.is_tensor(value) and value.shape[0] == batch_size) or (
            isinstance(value, Iterable)
            and not isinstance(value, (str, bytes, dict))
            and len(value) == batch_size
        ):
            sliced_data[key] = value[start:end]
        else:
            sliced_data[key] = value
    return sliced_data


#: Valid fresh-queue consumption orders.
QUEUE_ORDERS = ("fifo", "edf")


class RolloutBuffer:
    """Thread-safe storage for fresh and replay rollout examples.

    The fresh queue is a priority heap whose consumption order is set by
    ``queue_order``:

    - ``"edf"`` (default, earliest deadline first): ascending
      ``min_version``, i.e.
      the sample closest to its staleness deadline
      (``min_version + max_staleness``) is consumed first.
    - ``"fifo"``: arrival order — the historical deque behavior; set
      explicitly for comparison runs.

    Why edf is the default: long generations
      enter the buffer with the least remaining staleness budget; FIFO makes
      them wait behind fresher short samples and expire, biasing training
      data toward short/easy prompts. EDF consumes them first instead. The
      per-token staleness invariant is unchanged: any consumed sample still
      satisfies ``current_version - min_version <= max_staleness``.

    Ties (same priority) are broken by arrival order, so samples of one
    rollout group — ingested contiguously with the same ``min_version`` —
    stay contiguous under both orders.
    """

    def __init__(
        self,
        max_size: int = 65536,
        debug: bool = False,
        reward_norm: NormConfig | None = None,
        filter_fn: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
        max_staleness: int | None = None,
        replay_max_size: int | None = None,
        replay_selection_fn: ReplaySelectionFn | None = None,
        queue_order: str = "edf",
    ):
        # Filtering has been moved to data-acquisition layer. Keep this argument
        # for backward compatibility with older call sites.
        del filter_fn
        self.max_size = max_size
        if replay_max_size is None:
            replay_max_size = max_size
        self.replay_max_size = replay_max_size
        self.debug = debug
        self.label = ""

        if queue_order not in QUEUE_ORDERS:
            logger.warning(
                "Unknown queue_order %r; falling back to 'edf'. Valid: %s",
                queue_order,
                QUEUE_ORDERS,
            )
            queue_order = "edf"
        self.queue_order = queue_order

        # Fresh queue: heap of (priority, seq_no, example, metadata).
        # priority is 0.0 for fifo (order degenerates to arrival seq_no) and
        # min_version for edf. seq_no is a monotonic arrival counter.
        self._heap: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        self._seq_counter = 0
        self._replay_buffer: deque[dict[str, Any]] = deque(maxlen=replay_max_size)
        self._replay_metadata: deque[dict[str, Any]] = deque(maxlen=replay_max_size)

        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

        self.reward_norm = Normalization(reward_norm) if reward_norm else None
        self.max_staleness = max_staleness
        self.replay_selection_fn = (
            replay_selection_fn if replay_selection_fn is not None else select_latest
        )
        self._put_stats = {"accepted": 0, "filtered": 0, "total": 0, "evicted": 0}
        self._consume_stats = {
            "consumed": 0,
            "skipped_stale": 0,
            "consumed_len_sum": 0,
            "skipped_stale_len_sum": 0,
        }

    def _priority(self, metadata: dict[str, Any]) -> float:
        """Heap priority for a fresh example (lower = consumed earlier)."""
        if self.queue_order == "edf":
            min_v = metadata.get("min_version")
            if isinstance(min_v, (int, float)):
                return float(min_v)
            # No version info: the sample can never be staleness-dropped, so
            # consume it eagerly rather than risk starving it behind an
            # endless stream of versioned samples.
            return float("-inf")
        return 0.0

    @staticmethod
    def _seq_len(example: dict[str, Any]) -> int:
        """Token length of a single-example dict (0 if unknown)."""
        mask = example.get("attention_mask")
        try:
            if torch.is_tensor(mask):
                return int(mask.sum().item())
        except Exception:
            pass
        return 0

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        reward_score = rewards
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)
        return reward_score

    def _clone_for_state(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: self._clone_for_state(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone_for_state(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_for_state(v) for v in value)
        return copy.deepcopy(value)

    def put(
        self,
        batch: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bool:
        del timeout
        with self._not_full:
            if self._closed:
                if self.debug:
                    print("RolloutBuffer.put: Buffer is closed, cannot add batch")
                return False

            if "rewards" in batch:
                batch["normalized_rewards"] = self._normalize_rewards(batch["rewards"])

            batch_size = get_batch_size(batch)
            example_metadata = metadata if metadata is not None else {}
            inserted_count = 0
            evicted_count = 0

            priority = self._priority(example_metadata)
            for i in range(batch_size):
                example = _slice_tensor_dict(batch, i, i + 1)
                if len(self._heap) >= self.max_size:
                    # Evict the head: under fifo the oldest arrival (historic
                    # behavior), under edf the most staleness-critical sample
                    # (the one that would expire soonest anyway).
                    heapq.heappop(self._heap)
                    evicted_count += 1
                heapq.heappush(
                    self._heap,
                    (priority, self._seq_counter, example, example_metadata),
                )
                self._seq_counter += 1
                inserted_count += 1

            self._put_stats["accepted"] += inserted_count
            self._put_stats["total"] += batch_size
            self._put_stats["evicted"] += evicted_count

            if inserted_count > 0:
                self._not_empty.notify()
            return True

    def get_and_reset_put_stats(self) -> dict[str, int]:
        with self._lock:
            stats = dict(self._put_stats)
            self._put_stats = {"accepted": 0, "filtered": 0, "total": 0, "evicted": 0}
            return stats

    def get_and_reset_consume_stats(self) -> dict[str, int]:
        with self._lock:
            stats = dict(self._consume_stats)
            self._consume_stats = {
                "consumed": 0,
                "skipped_stale": 0,
                "consumed_len_sum": 0,
                "skipped_stale_len_sum": 0,
            }
            return stats

    def get(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> dict[str, Any] | None:
        result = self.get_with_metadata(
            timeout=timeout, current_version=current_version
        )
        if result is None:
            return None
        batch, _ = result
        return batch

    def _pop_one(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[float, int, dict[str, Any], dict[str, Any]] | None:
        """Pop the highest-priority fresh entry, dropping expired ones.

        Returns the full heap entry ``(priority, seq_no, example, metadata)``
        so callers can push it back verbatim. Blocks while the queue is
        empty; returns None only when the buffer is closed.
        """
        del timeout
        import time as time_module

        with self._not_empty:
            while True:
                if self._closed:
                    if self.debug:
                        print("RolloutBuffer.get: Buffer is closed, returning None")
                    return None

                if len(self._heap) == 0:
                    if self.debug:
                        print("RolloutBuffer.get: Buffer is empty, waiting...")
                    self._not_empty.wait()
                    time_module.sleep(0.1)
                    continue

                _, _, example, metadata = self._heap[0]
                if current_version is not None and self.max_staleness is not None:
                    min_v = metadata.get("min_version")
                    if min_v is not None and isinstance(min_v, (int, float)):
                        version_gap = current_version - int(min_v)
                        if version_gap > self.max_staleness:
                            heapq.heappop(self._heap)
                            self._consume_stats["skipped_stale"] += 1
                            self._consume_stats["skipped_stale_len_sum"] += (
                                self._seq_len(example)
                            )
                            if self.debug:
                                print(
                                    f"{self.label}RolloutBuffer: skipped stale example "
                                    f"(min_version={int(min_v)}, current_version={current_version}, "
                                    f"gap={version_gap}, max_staleness={self.max_staleness})",
                                    flush=True,
                                )
                            continue

                entry = heapq.heappop(self._heap)
                self._consume_stats["consumed"] += 1
                self._consume_stats["consumed_len_sum"] += self._seq_len(entry[2])
                self._not_full.notify()
                return entry

    def get_with_metadata(
        self,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        entry = self._pop_one(timeout=timeout, current_version=current_version)
        if entry is None:
            return None
        _, _, example, metadata = entry
        return example, metadata

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        import time as time_module

        start_time = time_module.time() if timeout is not None else None

        if timeout is not None:
            first_entry = self._pop_one(
                timeout=min(timeout, 5.0),
                current_version=current_version,
            )
        else:
            first_entry = self._pop_one(
                timeout=None,
                current_version=current_version,
            )

        if first_entry is None:
            if self.debug:
                print(
                    "RolloutBuffer.get_batch: No examples available, returning None",
                    flush=True,
                )
            return None

        entries = [first_entry]
        if self.debug:
            print(
                f"{self.label}RolloutBuffer.get_batch: Collected {len(entries)}/{batch_size} fresh examples",
                flush=True,
            )

        if timeout is not None:
            start_time = time_module.time()

        for _ in range(1, batch_size):
            remaining_timeout = None
            if timeout is not None:
                elapsed = time_module.time() - start_time
                remaining_timeout = max(0.0, timeout - elapsed)
                if remaining_timeout <= 0:
                    break

            entry = self._pop_one(
                timeout=remaining_timeout,
                current_version=current_version,
            )
            if entry is None:
                if self.debug:
                    print(
                        "RolloutBuffer.get_batch: _pop_one returned None while collecting",
                        flush=True,
                    )
                break

            entries.append(entry)
            if self.debug and (len(entries) == batch_size or len(entries) % 10 == 0):
                print(
                    f"{self.label}RolloutBuffer.get_batch: Collected {len(entries)}/{batch_size} fresh examples",
                    flush=True,
                )

            if len(entries) < batch_size and len(entries) % 8 == 0:
                time_module.sleep(0.05)

        if len(entries) < batch_size:
            if self.debug:
                print(
                    f"RolloutBuffer.get_batch: Only collected {len(entries)}/{batch_size}, putting back",
                    flush=True,
                )
            with self._lock:
                # Push entries back verbatim: original (priority, seq_no) keys
                # restore the exact consumption order under both queue modes.
                for entry in entries:
                    heapq.heappush(self._heap, entry)
                self._not_empty.notify_all()
            return None

        examples = [entry[2] for entry in entries]
        metadatas = [entry[3] for entry in entries]

        with self._lock:
            for example, metadata in zip(examples, metadatas):
                self._replay_buffer.append(example)
                self._replay_metadata.append(
                    self._build_replay_metadata(metadata, current_version)
                )

        combined_batch = concat_padded_tensors(examples)
        if self.debug:
            print(
                f"RolloutBuffer.get_batch: Built fresh batch with {len(examples)} examples",
                flush=True,
            )
        return combined_batch, metadatas

    def replay_size(self) -> int:
        with self._lock:
            return len(self._replay_buffer)

    def get_replay_with_metadata(
        self,
        batch_size: int = 1,
        ids: list[int] | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        with self._lock:
            if not self._replay_buffer:
                return None

            buffer_size = len(self._replay_buffer)
            indices = (
                self.replay_selection_fn(buffer_size, batch_size)
                if ids is None
                else ids
            )
            normalized_indices = self._normalize_replay_indices(indices, buffer_size)
            if not normalized_indices:
                return None

            examples = []
            metadatas = []
            for index in normalized_indices:
                examples.append(self._replay_buffer[index])
                if current_version is None:
                    metadatas.append(dict(self._replay_metadata[index]))
                else:
                    updated = self._record_train_version(
                        self._replay_metadata[index],
                        current_version,
                    )
                    self._replay_metadata[index] = updated
                    metadatas.append(dict(updated))

        combined_batch = concat_padded_tensors(examples)
        return combined_batch, metadatas

    def get_replay_batch(
        self,
        batch_size: int,
        ids: list[int] | None = None,
        current_version: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        return self.get_replay_with_metadata(
            batch_size=batch_size,
            ids=ids,
            current_version=current_version,
        )

    def _build_replay_metadata(
        self,
        metadata: dict[str, Any],
        current_version: int | None,
    ) -> dict[str, Any]:
        replay_metadata = dict(metadata)
        return self._record_train_version(replay_metadata, current_version)

    def _record_train_version(
        self,
        metadata: dict[str, Any],
        current_version: int | None,
    ) -> dict[str, Any]:
        if current_version is None:
            return metadata

        train_versions = metadata.get("train_versions")
        if isinstance(train_versions, list):
            train_versions = list(train_versions)
        elif train_versions is None:
            train_versions = []
        else:
            train_versions = [train_versions]

        train_versions.append(current_version)
        metadata["train_versions"] = train_versions
        return metadata

    def _normalize_replay_indices(
        self,
        indices: list[int],
        buffer_size: int,
    ) -> list[int]:
        if not indices:
            return []
        normalized = []
        for index in indices:
            if not isinstance(index, int):
                raise ValueError(f"Replay index must be int, got {type(index)}")
            if index < 0 or index >= buffer_size:
                raise ValueError(
                    f"Replay index {index} out of range for buffer size {buffer_size}"
                )
            normalized.append(index)
        return normalized

    def _aggregate_metadata(self, metadatas: list[dict[str, Any]]) -> dict[str, Any]:
        if not metadatas:
            return {}

        aggregated: dict[str, Any] = {}

        versions: list[int | float] = []
        for meta in metadatas:
            if "version" in meta:
                version = meta["version"]
                if isinstance(version, (int, float)):
                    versions.append(version)
                elif isinstance(version, list):
                    versions.extend([v for v in version if isinstance(v, (int, float))])

        if versions:
            aggregated["min_version"] = min(versions)
            aggregated["max_version"] = max(versions)

        min_versions = [
            meta.get("min_version")
            for meta in metadatas
            if "min_version" in meta and isinstance(meta["min_version"], (int, float))
        ]
        max_versions = [
            meta.get("max_version")
            for meta in metadatas
            if "max_version" in meta and isinstance(meta["max_version"], (int, float))
        ]

        if min_versions:
            aggregated["min_version"] = (
                min(aggregated["min_version"], min(min_versions))
                if "min_version" in aggregated
                else min(min_versions)
            )

        if max_versions:
            aggregated["max_version"] = (
                max(aggregated["max_version"], max(max_versions))
                if "max_version" in aggregated
                else max(max_versions)
            )

        zero_adv_values = [
            meta.get("zero_adv")
            for meta in metadatas
            if "zero_adv" in meta and isinstance(meta["zero_adv"], (int, float))
        ]
        if zero_adv_values:
            aggregated["zero_adv"] = 1 if all(v == 1 for v in zero_adv_values) else 0

        for meta in metadatas:
            for key, value in meta.items():
                if (
                    key not in ("version", "min_version", "max_version", "zero_adv")
                    and key not in aggregated
                ):
                    aggregated[key] = value

        return aggregated

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            # Persist fresh entries in consumption order. Priorities are NOT
            # persisted — they are recomputed from metadata on load, so a
            # checkpoint saved under one queue_order loads correctly under
            # another.
            ordered = sorted(self._heap, key=lambda e: (e[0], e[1]))
            return {
                "max_size": self.max_size,
                "replay_max_size": self.replay_max_size,
                "queue_order": self.queue_order,
                "buffer": [self._clone_for_state(e[2]) for e in ordered],
                "metadata": [self._clone_for_state(e[3]) for e in ordered],
                "seq_nos": [e[1] for e in ordered],
                "seq_counter": self._seq_counter,
                "replay_buffer": [
                    self._clone_for_state(x) for x in self._replay_buffer
                ],
                "replay_metadata": [
                    self._clone_for_state(x) for x in self._replay_metadata
                ],
                "closed": bool(self._closed),
                "put_stats": dict(self._put_stats),
                "consume_stats": dict(self._consume_stats),
            }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._lock:
            buffer_items = state_dict.get("buffer", [])
            metadata_items = state_dict.get("metadata", [])
            replay_items = state_dict.get("replay_buffer", [])
            replay_metadata_items = state_dict.get("replay_metadata", [])
            # Old checkpoints (deque era) have no seq_nos: assign arrival
            # counters in list order, which preserves their FIFO order.
            seq_nos = state_dict.get("seq_nos")
            if not seq_nos or len(seq_nos) != len(buffer_items):
                seq_nos = list(range(len(buffer_items)))

            self._heap = []
            for seq_no, example, metadata in zip(
                seq_nos, buffer_items, metadata_items
            ):
                example = self._clone_for_state(example)
                metadata = self._clone_for_state(metadata)
                heapq.heappush(
                    self._heap,
                    (self._priority(metadata), int(seq_no), example, metadata),
                )
            self._seq_counter = int(
                state_dict.get(
                    "seq_counter",
                    (max(seq_nos) + 1) if seq_nos else 0,
                )
            )
            self._replay_buffer = deque(
                [self._clone_for_state(x) for x in replay_items],
                maxlen=self.replay_max_size,
            )
            self._replay_metadata = deque(
                [self._clone_for_state(x) for x in replay_metadata_items],
                maxlen=self.replay_max_size,
            )
            self._closed = bool(state_dict.get("closed", False))
            self._put_stats = {
                "accepted": 0,
                "filtered": 0,
                "total": 0,
                "evicted": 0,
                **dict(state_dict.get("put_stats", {})),
            }
            self._consume_stats = {
                "consumed": 0,
                "skipped_stale": 0,
                "consumed_len_sum": 0,
                "skipped_stale_len_sum": 0,
                **dict(state_dict.get("consume_stats", {})),
            }

            if len(self._heap) > 0:
                self._not_empty.notify_all()
            if len(self._heap) < self.max_size:
                self._not_full.notify_all()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


__all__ = ["RolloutBuffer"]
