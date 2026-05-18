"""Data-acquisition component for AstraFlow.

This component continuously acquires rollout batches from the rollout engine
and forwards accepted samples to the serving layer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import requests
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from astraflow.dataflow.buffer_filters import BufferFilterFn, get_filter
from astraflow.dataflow.prompt_curators import (
    PromptCurator,
    RolloutOutcome,
    resolve_curator,
)
from astraflow.dataflow.utils import cycle_dataloader

logger = logging.getLogger(__name__)

# Debug instrumentation: when env var ASTRAFLOW_PRODUCER_DEBUG=1, log a
# deterministic per-prompt digest at submit and ingest. Used to A/B the
# curator codepath against no-curator. No effect when unset.
_DEBUG_PRODUCER = os.environ.get("ASTRAFLOW_PRODUCER_DEBUG", "") == "1"
_DEBUG_SUBMIT_COUNTER = 0
_DEBUG_SUBMIT_LOCK = threading.Lock()
_DEBUG_INGEST_COUNTER = 0
_DEBUG_INGEST_LOCK = threading.Lock()

# Maximum consecutive rejections by the curator before the submit loop
# force-accepts one prompt to keep training fed. Prevents a buggy curator
# from silently starving training.
MAX_REJECTS_PER_TICK = 256


class DataAcquisition(Protocol):
    """Acquisition interface that controls rollout production lifecycle."""

    def start(self) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self, timeout: float = 30.0) -> None: ...

    def close(self, timeout: float = 30.0) -> None: ...

    def is_running(self) -> bool: ...

    def is_paused(self) -> bool: ...

    def get_stats(self, reset: bool = False) -> dict[str, int]: ...

    def set_stats(self, stats: dict[str, Any] | None) -> None: ...

    def get_ingest_stats(self, reset: bool = False) -> dict[str, int]: ...

    def set_ingest_stats(self, stats: dict[str, Any] | None) -> None: ...

    def get_reward_stats(self, reset: bool = False) -> dict[str, float | int]: ...

    def set_reward_stats(self, stats: dict[str, Any] | None) -> None: ...


SUBMIT_CONCURRENCY = 8


class AstraDataAcquisition:
    """Threaded acquisition loop that forwards batches to serving."""

    def __init__(
        self,
        *,
        rollout: Any,
        rollout_dataloader: StatefulDataLoader,
        workflow_spec: dict[str, Any],
        publish_fn: Callable[
            [dict[str, Any], dict[str, Any] | None, float | None], bool
        ],
        data_serving: Any | None = None,
        filter_fn: BufferFilterFn | str | None = None,
        curator: PromptCurator | str | None = None,
        curator_args: dict[str, Any] | None = None,
        debug: bool = False,
        error_backoff: float = 0.5,
        publish_timeout: float | None = 0.1,
    ):
        self.rollout = rollout
        self.rollout_dataloader = rollout_dataloader
        self.workflow_spec = workflow_spec

        self._publish_fn = publish_fn
        self._data_serving = data_serving
        self._filter_fn = self._resolve_filter_fn(filter_fn)
        self._curator = resolve_curator(curator, curator_args)
        self._curator_lock = threading.Lock()
        # Warmup epoch tracking. The curator's adaptive controllers stay
        # frozen until DataAcquisition fires notify_warmup_complete(),
        # which happens after one full dataloader epoch of post-pre-fill
        # samples has been emitted from _get_next_sample(). The pre-fill
        # phase is excluded by latching _training_started on the first
        # notify_version_changed() call (i.e. when the trainer publishes
        # its first weight version).
        self._training_started: bool = False
        self._post_prefill_samples_emitted: int = 0
        self._warmup_notified: bool = False
        self._warmup_target: int | None = None
        try:
            ds = getattr(rollout_dataloader, "dataset", None)
            if ds is not None:
                n = len(ds)
                if n > 0:
                    self._warmup_target = int(n)
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "could not derive warmup target from dataloader.dataset: %s; "
                "curator warmup notification will not fire — curator-side "
                "warmup-gated logic stays frozen indefinitely",
                exc,
            )
        self._current_version: int = 0
        self._version_lock = threading.Lock()
        self._debug = debug
        self._error_backoff = error_backoff
        self._publish_timeout = publish_timeout

        self._producer_thread: threading.Thread | None = None
        self._submit_thread: threading.Thread | None = None
        self._collect_thread: threading.Thread | None = None
        self._producer_thread_lock = threading.Lock()
        self._producer_stop = threading.Event()
        self._paused = threading.Event()
        self._sample_iter_lock = threading.Lock()
        self._sample_iter = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "producer_batches": 0,
            "producer_errors": 0,
            "producer_put_failures": 0,
        }
        self._submit_executor = ThreadPoolExecutor(max_workers=SUBMIT_CONCURRENCY)
        # Monotonic counter for group_id — each arun_episode result gets a unique id.
        self._group_id_counter = 0
        self._group_id_lock = threading.Lock()
        self._ingest_stats: dict[str, int] = {
            "accepted": 0,
            "filtered": 0,
            "total": 0,
        }
        # Curator-side counters (pre-rollout selection). Distinct from
        # ``_ingest_stats`` which counts post-rollout filter outcomes.
        self._curator_stats: dict[str, int] = {
            "selected": 0,
            "rejected": 0,
            "forced": 0,
        }
        self._reward_stats: dict[str, float | int] = self._build_empty_reward_stats()
        # Per-RaaS instance throughput stats for balance report.
        self._per_raas_stats: dict[str, dict[str, int]] = {}
        self._per_raas_stats_lock = threading.Lock()

    def _next_group_id(self) -> int:
        """Return a monotonically increasing group id (thread-safe)."""
        with self._group_id_lock:
            gid = self._group_id_counter
            self._group_id_counter += 1
            return gid

    @staticmethod
    def _build_empty_reward_stats() -> dict[str, float | int]:
        return {
            "pre_filter_reward_sum": 0.0,
            "pre_filter_reward_count": 0,
            "pre_filter_reward_min": 0.0,
            "pre_filter_reward_max": 0.0,
            "post_filter_reward_sum": 0.0,
            "post_filter_reward_count": 0,
            "post_filter_reward_min": 0.0,
            "post_filter_reward_max": 0.0,
        }

    def _accumulate_reward_stats(
        self,
        rewards: torch.Tensor | None,
        *,
        prefix: str,
    ) -> None:
        if rewards is None:
            return
        count = int(rewards.numel())
        if count <= 0:
            return
        reward_sum = float(rewards.sum().item())
        reward_min = float(rewards.min().item())
        reward_max = float(rewards.max().item())
        count_key = f"{prefix}_reward_count"
        sum_key = f"{prefix}_reward_sum"
        min_key = f"{prefix}_reward_min"
        max_key = f"{prefix}_reward_max"
        with self._stats_lock:
            prev_count = int(self._reward_stats[count_key])
            self._reward_stats[count_key] = prev_count + count
            self._reward_stats[sum_key] = (
                float(self._reward_stats[sum_key]) + reward_sum
            )
            if prev_count == 0:
                self._reward_stats[min_key] = reward_min
                self._reward_stats[max_key] = reward_max
            else:
                self._reward_stats[min_key] = min(
                    float(self._reward_stats[min_key]), reward_min
                )
                self._reward_stats[max_key] = max(
                    float(self._reward_stats[max_key]), reward_max
                )
        # Push to data_serving for per-model tracking
        if self._data_serving is not None:
            self._data_serving.accumulate_acquisition_stats({
                count_key: count,
                sum_key: reward_sum,
            })

    def _supports_raas_service_mode(self) -> bool:
        return (
            hasattr(self.rollout, "get_raas_availability")
            and hasattr(self.rollout, "submit_auto")
            and hasattr(self.rollout, "pull_completed")
        )

    def _iter_samples(self):
        for data in cycle_dataloader(self.rollout_dataloader):
            yield from data

    def _get_next_sample(self) -> dict[str, Any] | None:
        with self._sample_iter_lock:
            if self._sample_iter is None:
                self._sample_iter = self._iter_samples()
            try:
                sample = next(self._sample_iter)
            except StopIteration:
                self._sample_iter = self._iter_samples()
                try:
                    sample = next(self._sample_iter)
                except StopIteration:
                    return None
        self._maybe_notify_warmup_complete()
        return sample

    def _maybe_notify_warmup_complete(self) -> None:
        """Fire ``curator.notify_warmup_complete()`` exactly once after one
        full dataloader epoch of post-pre-fill samples has been emitted.

        Called from ``_get_next_sample`` after a sample is successfully
        produced. Cheap fast-path when already notified or when no curator
        is configured.
        """
        if self._warmup_notified:
            return
        if self._curator is None or self._warmup_target is None:
            return
        if not self._training_started:
            return
        self._post_prefill_samples_emitted += 1
        if self._post_prefill_samples_emitted < self._warmup_target:
            return
        self._warmup_notified = True
        if not hasattr(self._curator, "notify_warmup_complete"):
            return
        try:
            with self._curator_lock:
                self._curator.notify_warmup_complete()
            logger.info(
                "[curator] notify_warmup_complete fired after %d post-pre-fill "
                "samples (target=%d, one full dataloader epoch)",
                self._post_prefill_samples_emitted,
                self._warmup_target,
            )
        except Exception:
            logger.exception("curator.notify_warmup_complete raised; ignoring")

    def _resolve_filter_fn(
        self, filter_fn: BufferFilterFn | str | None
    ) -> BufferFilterFn | None:
        """Resolve filter policy from callable or registry name."""
        if filter_fn is None:
            return None
        if callable(filter_fn):
            return filter_fn
        if isinstance(filter_fn, str):
            try:
                return get_filter(filter_fn)
            except ValueError as e:
                logger.warning("%s. Using default KeepAllFilter.", e)
                return None
        logger.warning(
            "Unsupported filter_fn type %s. Using default KeepAllFilter.",
            type(filter_fn),
        )
        return None

    @staticmethod
    def build_metadata(batch: dict[str, Any]) -> dict[str, Any]:
        """Build lightweight metadata used for filtering, replay, and metrics."""
        metadata: dict[str, Any] = {}

        versions = batch.get("versions")
        if versions is not None:
            try:
                valid_versions = versions[versions > 0]
                if valid_versions.numel() > 0:
                    min_v = int(valid_versions.min().item())
                    max_v = int(versions.max().item())
                else:
                    min_v = 0
                    max_v = int(versions.max().item())
                metadata["min_version"] = min_v
                metadata["max_version"] = max_v
            except Exception:
                metadata = {}

        rewards = batch.get("rewards")
        if rewards is not None:
            try:
                if torch.is_tensor(rewards):
                    rewards_flat = rewards.flatten()
                    if len(rewards_flat) > 0:
                        metadata["zero_adv"] = (
                            1 if torch.allclose(rewards_flat, rewards_flat[0]) else 0
                        )
                    else:
                        metadata["zero_adv"] = 1
                else:
                    metadata["zero_adv"] = 0
            except Exception:
                metadata["zero_adv"] = 0

        return metadata

    def _ingest_one_result(
        self,
        batch: dict[str, Any] | None,
        raas_uid: str | None = None,
    ) -> None:
        # Upstream rejection (e.g., should_accept_fn) is represented as None.
        if batch is None:
            with self._stats_lock:
                self._stats["producer_batches"] += 1
            return

        self._ingest_structured_result(batch, raas_uid=raas_uid)

    @staticmethod
    def _get_seq_model_key(seq: dict[str, Any]) -> int:
        """Extract model index from a sequence's model_ids tensor.

        Returns the first non-negative model_id value, or -1 if none found
        (single-model case where model_ids may not exist).
        """
        mid = seq.get("model_ids")
        if mid is None or not torch.is_tensor(mid):
            return -1
        flat = mid.flatten()
        positive = flat[flat >= 0]
        if positive.numel() == 0:
            return -1
        return int(positive[0].item())

    def _ingest_structured_result(
        self,
        result: dict[str, Any],
        raas_uid: str | None = None,
    ) -> None:
        """Ingest a structured result with per-model filtering.

        Expected format::

            {
                "n_trajs": int,
                "rewards": Tensor[n_trajs],  # trajectory-level rewards
                "trajectories": [
                    {"sequences": [seq_dict, ...], "stats": {...}},
                    ...
                ],
                "agent_metrics": {str: float, ...},  # optional workflow metrics
            }

        Sequences may have per-sequence ``rewards`` already set (multi-model
        workflows).  If not, the trajectory-level reward is assigned.

        Sequences are grouped by model (via ``model_ids`` tensor).  Filtering
        (e.g. ``filter_zero_adv``) is applied **per model group** independently
        so that one model's data can be kept while another's is filtered.
        """
        # Extract and forward workflow-defined agent metrics (if present).
        agent_metrics = result.pop("agent_metrics", None)
        if agent_metrics and self._data_serving is not None:
            self._data_serving.accumulate_agent_metrics(agent_metrics)

        traj_rewards = result["rewards"]  # Tensor[n_trajs]
        if not torch.is_tensor(traj_rewards):
            traj_rewards = torch.tensor(traj_rewards)
        traj_rewards = traj_rewards.detach().float()
        n_trajs = len(traj_rewards)

        self._accumulate_reward_stats(traj_rewards, prefix="pre_filter")

        # Extract prompt_id from the result if available.
        prompt_id = None
        for key in ("prompt_id", "query_id", "task_id", "id", "qid"):
            prompt_id = result.get(key)
            if prompt_id is not None:
                break
        prompt_id_str = str(prompt_id) if prompt_id is not None else ""

        # Assign a unique group_id for this rollout.
        group_id = self._next_group_id()

        # Flatten all sequences and stamp group_id / prompt_id / rewards.
        all_sequences: list[dict[str, Any]] = []
        for i, traj in enumerate(result["trajectories"]):
            sequences = traj.get("sequences", [])
            reward_val = float(traj_rewards[i].item())
            for seq in sequences:
                # Use per-sequence reward if set by workflow, else trajectory reward
                if "rewards" not in seq:
                    seq["rewards"] = torch.tensor([reward_val])
                seq["group_id"] = torch.tensor([group_id], dtype=torch.long)
                seq["prompt_id"] = prompt_id_str
                all_sequences.append(seq)

        # Group sequences by model.
        model_groups: dict[int, list[dict[str, Any]]] = {}
        for seq in all_sequences:
            model_key = self._get_seq_model_key(seq)
            model_groups.setdefault(model_key, []).append(seq)

        # Per-model filtering and ingestion.
        total_seqs = 0
        accepted_seqs = 0
        filtered_seqs = 0
        per_model_accepted: dict[int, int] = {}

        for model_key, seqs in model_groups.items():
            # Collect rewards for this model group to check variance.
            group_rewards = torch.tensor(
                [float(s["rewards"].flatten()[0].item()) for s in seqs]
            )

            # Build per-model metadata for filtering.
            metadata: dict[str, Any] = {}
            if len(group_rewards) > 0:
                metadata["zero_adv"] = (
                    1 if torch.allclose(group_rewards, group_rewards[0]) else 0
                )
            else:
                metadata["zero_adv"] = 1

            total_seqs += len(seqs)

            # Apply filter per model group.
            if self._filter_fn is not None and not self._filter_fn({}, metadata):
                filtered_seqs += len(seqs)
                continue

            # Stamp per-group reward mean/std on each seq while the full
            # per-prompt × per-model group is in scope. The trainer reads
            # data["group_reward_mean"] / data["group_reward_std"] and
            # normalizes locally — avoiding the partial-group bug when a
            # group is split across DP ranks (multi-turn / variable-size
            # workflows such as actor_and_verify and asearcher).
            #
            # Use unbiased (N-1) std to match Normalization defaults
            # (NormConfig.std_unbiased=True). For singleton groups,
            # fall back to std=1.0 to match Normalization's special case.
            n_g = len(group_rewards)
            if n_g >= 2:
                g_mean = float(group_rewards.mean().item())
                g_std = float(group_rewards.std(unbiased=True).item())
            elif n_g == 1:
                g_mean = float(group_rewards[0].item())
                g_std = 1.0
            else:
                g_mean = 0.0
                g_std = 1.0

            # Publish each sequence to the buffer.
            model_accepted = 0
            for seq in seqs:
                seq["group_reward_mean"] = torch.tensor([g_mean])
                seq["group_reward_std"] = torch.tensor([g_std])
                seq_metadata = dict(metadata)
                version_meta = self.build_metadata(seq)
                for k in ("min_version", "max_version"):
                    if k in version_meta:
                        seq_metadata[k] = version_meta[k]
                success = self._publish_fn(seq, seq_metadata, self._publish_timeout)
                accepted_seqs += 1
                model_accepted += 1
                if not success:
                    with self._stats_lock:
                        self._stats["producer_put_failures"] += 1
            per_model_accepted[model_key] = model_accepted

        # Curator feedback: one update per ingested rollout. Uses
        # trajectory-level rewards (pre-model-split) so the outcome is
        # per-prompt, regardless of how the workflow groups by model.
        if self._curator is not None and prompt_id_str:
            n_total = traj_rewards.numel()
            if n_total >= 2:
                _g_mean = float(traj_rewards.mean().item())
                _g_std = float(traj_rewards.std(unbiased=True).item())
            elif n_total == 1:
                _g_mean = float(traj_rewards[0].item())
                _g_std = 1.0
            else:
                _g_mean = 0.0
                _g_std = 1.0
            _zero_adv = (
                1
                if n_total > 0 and torch.allclose(traj_rewards, traj_rewards[0])
                else 0
            )
            with self._version_lock:
                _ver = self._current_version
            outcome = RolloutOutcome(
                query_id=prompt_id_str,
                rewards=traj_rewards.detach().cpu(),
                g_mean=_g_mean,
                g_std=_g_std,
                zero_adv=_zero_adv,
                n_trajs=n_trajs,
                version=_ver,
                source=result.get("source"),
            )
            try:
                with self._curator_lock:
                    self._curator.update(outcome)
            except Exception:
                logger.exception("curator.update raised; ignoring")

        self._accumulate_reward_stats(traj_rewards, prefix="post_filter")

        with self._stats_lock:
            self._ingest_stats["total"] += total_seqs
            self._ingest_stats["accepted"] += accepted_seqs
            self._ingest_stats["filtered"] += filtered_seqs
            self._stats["producer_batches"] += 1
        # Track per-RaaS produced/accepted/filtered counts (sequence-level).
        if raas_uid:
            with self._per_raas_stats_lock:
                if raas_uid not in self._per_raas_stats:
                    self._per_raas_stats[raas_uid] = {
                        "produced": 0,
                        "accepted": 0,
                        "filtered": 0,
                    }
                self._per_raas_stats[raas_uid]["produced"] += total_seqs
                self._per_raas_stats[raas_uid]["accepted"] += accepted_seqs
                self._per_raas_stats[raas_uid]["filtered"] += filtered_seqs
        if self._data_serving is not None:
            self._data_serving.accumulate_acquisition_stats(
                {"accepted": accepted_seqs, "filtered": filtered_seqs, "total": total_seqs}
            )

        if _DEBUG_PRODUCER:
            with self._version_lock:
                _ver_now = self._current_version
            _r_sum = float(traj_rewards.sum().item()) if n_trajs > 0 else 0.0
            _r_mean = float(traj_rewards.mean().item()) if n_trajs > 0 else 0.0
            _r_std = float(traj_rewards.std().item()) if n_trajs > 1 else 0.0
            global _DEBUG_INGEST_COUNTER
            with _DEBUG_INGEST_LOCK:
                _DEBUG_INGEST_COUNTER += 1
                _iseq = _DEBUG_INGEST_COUNTER
            print(
                f"[DBG-INGEST] iseq={_iseq} qid={prompt_id_str or '<noid>'} "
                f"v_now={_ver_now} n={n_trajs} "
                f"r_sum={_r_sum:.4f} r_mean={_r_mean:.4f} r_std={_r_std:.4f} "
                f"accepted={accepted_seqs} filtered={filtered_seqs}",
                flush=True,
            )

        if self._debug and n_trajs > 0:
            version_info = ""
            for seq in all_sequences:
                v = seq.get("versions")
                if v is not None and torch.is_tensor(v):
                    v_flat = v.flatten()
                    valid_v = v_flat[v_flat > 0]
                    if valid_v.numel() > 0:
                        version_info = f", weight_v={int(valid_v[0].item())}"
                        break
            if prompt_id is not None:
                version_info += f", prompt={prompt_id}"
            with self._stats_lock:
                cum_info = (
                    f", cum_accepted={self._ingest_stats['accepted']}"
                    f", cum_filtered={self._ingest_stats['filtered']}"
                    f", cum_total={self._ingest_stats['total']}"
                )
            per_model_info = ", ".join(
                f"m{k}={v}" for k, v in sorted(per_model_accepted.items())
            )
            print(
                f"[AstraFlow] filter: "
                f"accepted={accepted_seqs} ({per_model_info}), "
                f"filtered={filtered_seqs}, "
                f"n_trajs={n_trajs}, total_seqs={total_seqs}, "
                f"rewards={traj_rewards.tolist()}, "
                f"reward_mean={traj_rewards.mean().item():.4f}, "
                f"reward_std={traj_rewards.std().item():.4f}"
                f"{version_info}{cum_info}",
                flush=True,
            )

    def _producer_loop_batchwise(self) -> None:
        while not self._producer_stop.is_set():
            if self._paused.is_set():
                time.sleep(0.1)
                continue

            try:
                batch = self.rollout.prepare_batch(
                    self.rollout_dataloader,
                    workflow_spec=self.workflow_spec,
                )
                self._ingest_one_result(batch)
            except Exception as exc:
                if self._producer_stop.is_set():
                    break
                with self._stats_lock:
                    self._stats["producer_errors"] += 1
                logger.error(
                    "AstraFlow acquisition iteration failed: %s",
                    exc,
                    exc_info=True,
                )
                self._producer_stop.wait(self._error_backoff)

    def _submit_one_auto(self, data: dict[str, Any]) -> None:
        """Submit a single sample via submit_auto. Used as a worker target."""
        if self._paused.is_set() or self._producer_stop.is_set():
            return
        t0 = time.monotonic()
        try:
            self.rollout.submit_auto(
                data=data,
                workflow_spec=self.workflow_spec,
            )
        except requests.exceptions.ReadTimeout:
            logger.warning(
                "AstraFlow submit_auto timed out after %.1fs (server busy), skipping",
                time.monotonic() - t0,
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                "AstraFlow submit_auto connection error after %.1fs (server unreachable), skipping",
                time.monotonic() - t0,
            )

    def _submit_loop_raas_service(self) -> None:
        max_submit_per_tick = 256
        _heartbeat_t = time.monotonic()
        _heartbeat_submitted = 0
        _heartbeat_zero_count = 0
        _dbg_last_t = time.monotonic()
        _dbg_notes: dict[str, int] = {}
        _dbg_total_waiting_sum = 0
        _dbg_hard_cap_sum = 0
        _dbg_soft_cap_sum = 0
        _dbg_samples = 0
        _dbg_per_dp_snapshot: list | None = None
        _dbg_last_avail_ms = 0.0
        _dbg_last_submit_ms = 0.0
        _dbg_tick_count = 0
        _dbg_sample_none = 0

        while not self._producer_stop.is_set():
            if self._paused.is_set():
                time.sleep(0.1)
                continue

            # Periodic heartbeat: every 10s, log submit thread activity.
            _now = time.monotonic()
            if _now - _heartbeat_t >= 10.0:
                print(
                    f"[AstraFlow-submit-heartbeat] submitted={_heartbeat_submitted} "
                    f"avail_zero_ticks={_heartbeat_zero_count} "
                    f"elapsed={_now - _heartbeat_t:.1f}s",
                    flush=True,
                )
                _heartbeat_submitted = 0
                _heartbeat_zero_count = 0
                _heartbeat_t = _now

            # Fine-grained debug every 5s.
            if _now - _dbg_last_t >= 5.0 and _dbg_samples > 0:
                _notes_str = ",".join(f"{k}={v}" for k, v in _dbg_notes.items())
                print(
                    f"[AstraFlow-submit-debug] ticks={_dbg_tick_count} samples={_dbg_samples} "
                    f"avg_total_waiting={_dbg_total_waiting_sum / _dbg_samples:.1f} "
                    f"avg_hard_cap={_dbg_hard_cap_sum / _dbg_samples:.1f} "
                    f"avg_soft_cap={_dbg_soft_cap_sum / _dbg_samples:.1f} "
                    f"notes={{{_notes_str}}} "
                    f"last_per_dp_waiting={_dbg_per_dp_snapshot} "
                    f"last_avail_call_ms={_dbg_last_avail_ms:.1f} "
                    f"last_submit_ms={_dbg_last_submit_ms:.1f} "
                    f"sample_none={_dbg_sample_none}",
                    flush=True,
                )
                _dbg_last_t = _now
                _dbg_notes = {}
                _dbg_total_waiting_sum = 0
                _dbg_hard_cap_sum = 0
                _dbg_soft_cap_sum = 0
                _dbg_samples = 0
                _dbg_tick_count = 0
                _dbg_sample_none = 0

            try:
                _t_tick = time.monotonic()
                _n, _dbg_info = self._submit_tick_debug(max_submit_per_tick)
                _dbg_tick_count += 1
                if _dbg_info is not None:
                    _dbg_samples += 1
                    _dbg_total_waiting_sum += int(_dbg_info.get("total_waiting", 0))
                    _dbg_hard_cap_sum += int(_dbg_info.get("hard_cap", 0))
                    _dbg_soft_cap_sum += int(_dbg_info.get("soft_cap", 0))
                    _note = str(_dbg_info.get("note", "?"))
                    _dbg_notes[_note] = _dbg_notes.get(_note, 0) + 1
                    _dbg_per_dp_snapshot = _dbg_info.get("per_dp_waiting")
                    _dbg_last_avail_ms = _dbg_info.get("avail_ms", 0.0)
                    _dbg_last_submit_ms = _dbg_info.get("submit_ms", 0.0)
                    if _dbg_info.get("sample_none"):
                        _dbg_sample_none += 1
                _heartbeat_submitted += _n if _n else 0
                if _n == 0:
                    _heartbeat_zero_count += 1
            except Exception as exc:
                if self._producer_stop.is_set():
                    break
                with self._stats_lock:
                    self._stats["producer_errors"] += 1
                logger.error(
                    "AstraFlow submit tick failed: %s",
                    exc,
                    exc_info=True,
                )
                self._producer_stop.wait(self._error_backoff)

    def _submit_tick_debug(self, max_submit_per_tick: int):
        """Debug wrapper: returns (n_submitted, info_dict)."""
        t_avail = time.monotonic()
        info: dict = {}
        try:
            availability = self.rollout.get_raas_availability()
        except Exception:
            return 0, None
        info["avail_ms"] = (time.monotonic() - t_avail) * 1000
        # RaaSPool wraps per-instance dicts under "per_instance"; unpack first one.
        _per = availability.get("per_instance") or {}
        _inst = next(iter(_per.values()), {}) if _per else availability
        info["note"] = _inst.get("note", "?")
        info["available"] = int(availability.get("available", 0))
        info["hard_cap"] = int(_inst.get("hard_cap", -1))
        info["soft_cap"] = int(_inst.get("soft_cap", -1))
        info["total_waiting"] = int(_inst.get("total_waiting", -1))
        info["total_target"] = int(_inst.get("total_target", -1))
        info["inflight"] = int(_inst.get("inflight", -1))
        info["per_dp_waiting"] = _inst.get("per_dp_waiting")
        info["fallback_kind"] = _inst.get("fallback_kind")

        available_slots = info["available"]
        submit_budget = max(0, min(available_slots, max_submit_per_tick))
        batch: list[dict[str, Any]] = []
        # Read current version once per tick (curator may use it).
        with self._version_lock:
            ver = self._current_version
        miss_count = 0  # consecutive curator rejections; force-submit at MAX
        while submit_budget > 0 and not self._producer_stop.is_set():
            if self._paused.is_set():
                break
            data = self._get_next_sample()
            if data is None:
                info["sample_none"] = True
                break

            if self._curator is None:
                accepted = True
            else:
                try:
                    with self._curator_lock:
                        accepted = self._curator.should_submit(data, version=ver)
                except Exception:
                    logger.exception("curator.should_submit raised; submitting prompt")
                    accepted = True

            if accepted:
                batch.append(data)
                submit_budget -= 1
                miss_count = 0
                if self._curator is not None:
                    with self._stats_lock:
                        self._curator_stats["selected"] += 1
                if _DEBUG_PRODUCER:
                    from astraflow.workflow.utils.data import resolve_prompt_id as _rpi
                    _qid = _rpi(data) or "<noid>"
                    global _DEBUG_SUBMIT_COUNTER
                    with _DEBUG_SUBMIT_LOCK:
                        _DEBUG_SUBMIT_COUNTER += 1
                        _seq = _DEBUG_SUBMIT_COUNTER
                    print(f"[DBG-SUBMIT] seq={_seq} qid={_qid} v={ver}", flush=True)
            else:
                miss_count += 1
                with self._stats_lock:
                    self._curator_stats["rejected"] += 1
                if miss_count >= MAX_REJECTS_PER_TICK:
                    logger.warning(
                        "curator rejection storm (%d in a row); "
                        "force-submitting one prompt",
                        miss_count,
                    )
                    batch.append(data)
                    submit_budget -= 1
                    miss_count = 0
                    with self._stats_lock:
                        self._curator_stats["forced"] += 1
        t_submit = time.monotonic()
        if batch:
            try:
                list(self._submit_executor.map(self._submit_one_auto, batch))
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ):
                pass
            except Exception as exc:
                logger.error("AstraFlow submit_auto failed: %s", exc, exc_info=True)
        info["submit_ms"] = (time.monotonic() - t_submit) * 1000
        if not batch:
            time.sleep(0.1)
        return len(batch), info

    def _submit_tick(self, max_submit_per_tick: int) -> int:
        """Single iteration of the submit loop. Returns count submitted."""
        t_avail = time.monotonic()
        try:
            availability = self.rollout.get_raas_availability()
            available_slots = int(availability.get("available", 0))
            submit_budget = max(0, min(available_slots, max_submit_per_tick))
            _fallback = availability.get("fallback_kind")
            if available_slots == 0 and _fallback:
                print(
                    f"[AstraFlow-submit-debug] available=0 fallback={_fallback} "
                    f"inflight={availability.get('inflight', '?')} "
                    f"elapsed={time.monotonic() - t_avail:.3f}s",
                    flush=True,
                )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            logger.warning(
                "AstraFlow availability check failed after %.1fs (server busy), retrying",
                time.monotonic() - t_avail,
            )
            self._producer_stop.wait(self._error_backoff)
            return 0
        except Exception as exc:
            with self._stats_lock:
                self._stats["producer_errors"] += 1
            logger.error(
                "AstraFlow failed to query RaaS availability: %s",
                exc,
                exc_info=True,
            )
            self._producer_stop.wait(self._error_backoff)
            return 0

        # Gather samples first, then submit in parallel.
        batch: list[dict[str, Any]] = []
        while submit_budget > 0 and not self._producer_stop.is_set():
            if self._paused.is_set():
                break
            data = self._get_next_sample()
            if data is None:
                break
            batch.append(data)
            submit_budget -= 1

        if batch:
            try:
                list(self._submit_executor.map(self._submit_one_auto, batch))
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ):
                # Already logged in _submit_one_auto; don't print traceback again
                pass
            except Exception as exc:
                with self._stats_lock:
                    self._stats["producer_errors"] += 1
                logger.error(
                    "AstraFlow submit_auto failed: %s",
                    exc,
                    exc_info=True,
                )
                return 0

        if not batch:
            time.sleep(0.1)
        return len(batch)

    def _collect_loop_raas_service(self) -> None:
        max_collect_per_tick = 512
        collect_timeout = 0.1
        _collect_call_count = 0
        _last_heartbeat = time.monotonic()
        _total_ingested = 0
        while not self._producer_stop.is_set():
            if self._paused.is_set():
                now = time.monotonic()
                if now - _last_heartbeat >= 30.0:
                    logger.info(
                        "AstraFlow collect thread alive (paused), ingested=%d",
                        _total_ingested,
                    )
                    _last_heartbeat = now
                time.sleep(0.1)
                continue

            try:
                completed = self.rollout.pull_completed(
                    max_items=max_collect_per_tick,
                    timeout=collect_timeout,
                )
                _collect_call_count += 1
                if _collect_call_count <= 20 or (_collect_call_count % 50 == 0):
                    print(
                        f"[AstraFlow] pull #{_collect_call_count}: "
                        f"got {len(completed) if completed else 0} items, "
                        f"total ingested={_total_ingested}",
                        flush=True,
                    )
            except Exception as exc:
                if self._producer_stop.is_set():
                    break
                with self._stats_lock:
                    self._stats["producer_errors"] += 1
                logger.error(
                    "AstraFlow pull_completed failed: %s",
                    exc,
                    exc_info=True,
                )
                self._producer_stop.wait(self._error_backoff)
                continue

            now = time.monotonic()
            if now - _last_heartbeat >= 30.0:
                print(
                    f"[AstraFlow] Collect heartbeat: "
                    f"calls={_collect_call_count}, ingested={_total_ingested}",
                    flush=True,
                )
                _last_heartbeat = now

            if not completed:
                time.sleep(0.02)
                continue

            for item in completed:
                if not isinstance(item, dict):
                    with self._stats_lock:
                        self._stats["producer_errors"] += 1
                    logger.error(
                        "Invalid completed payload from RaaS: %s",
                        item,
                    )
                    continue

                ok = bool(item.get("ok", True))
                if not ok:
                    with self._stats_lock:
                        self._stats["producer_errors"] += 1
                    logger.error(
                        "RaaS task failed (task_id=%s): %s",
                        item.get("task_id"),
                        item.get("error"),
                    )
                    continue

                try:
                    raas_uid = item.get("_raas_uid")
                    self._ingest_one_result(item.get("result"), raas_uid=raas_uid)
                    _total_ingested += 1
                except Exception as exc:
                    with self._stats_lock:
                        self._stats["producer_errors"] += 1
                    logger.error(
                        "AstraFlow ingest failed for task_id=%s: %s",
                        item.get("task_id"),
                        exc,
                        exc_info=True,
                    )

    def _run_submit_loop(self) -> None:
        print("[AstraFlow] Submit thread started", flush=True)
        try:
            self._submit_loop_raas_service()
        except Exception:
            logger.exception("AstraFlow RaaS submit thread crashed")
        finally:
            print("[AstraFlow] Submit thread stopped", flush=True)

    def _run_collect_loop(self) -> None:
        print("[AstraFlow] Collect thread started", flush=True)
        try:
            self._collect_loop_raas_service()
        except Exception:
            logger.exception("AstraFlow RaaS collect thread crashed")
        finally:
            print("[AstraFlow] Collect thread stopped", flush=True)

    def _producer_loop(self) -> None:
        logger.info("AstraFlow data-acquisition thread started")
        try:
            self._producer_loop_batchwise()
        finally:
            logger.info("AstraFlow data-acquisition thread stopped")

    def start(self) -> None:
        with self._producer_thread_lock:
            running_threads = [
                t
                for t in (
                    self._producer_thread,
                    self._submit_thread,
                    self._collect_thread,
                )
                if t is not None and t.is_alive()
            ]
            if running_threads:
                logger.warning("AstraFlow data-acquisition thread is already running")
                return
            self._producer_stop.clear()
            self._sample_iter = None
            if self._supports_raas_service_mode():
                self._submit_thread = threading.Thread(
                    target=self._run_submit_loop,
                    daemon=True,
                )
                self._collect_thread = threading.Thread(
                    target=self._run_collect_loop,
                    daemon=True,
                )
                self._submit_thread.start()
                self._collect_thread.start()
                logger.info(
                    "AstraFlow acquisition started with decoupled RaaS submit/collect threads."
                )
            else:
                self._producer_thread = threading.Thread(
                    target=self._producer_loop,
                    daemon=True,
                )
                self._producer_thread.start()

    def pause(self) -> None:
        self._paused.set()
        # Don't block on in-flight submits — they may be stuck waiting for
        # the HTTP response (up to request_timeout). Instead, shut down
        # the old executor in the background and create a new one immediately.
        old_executor = self._submit_executor
        self._submit_executor = ThreadPoolExecutor(max_workers=SUBMIT_CONCURRENCY)
        old_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("AstraDataAcquisition paused: submit executor replaced")

    def resume(self) -> None:
        self._paused.clear()

    def stop(self, timeout: float = 30.0) -> None:
        with self._producer_thread_lock:
            managed_threads = [
                t
                for t in (
                    self._producer_thread,
                    self._submit_thread,
                    self._collect_thread,
                )
                if t is not None
            ]
            if not managed_threads:
                return
            self._producer_stop.set()
            for t in managed_threads:
                if not t.is_alive():
                    continue
                t.join(timeout=timeout)
                if t.is_alive():
                    logger.warning(
                        "AstraFlow data-acquisition thread did not stop within timeout"
                    )
            self._producer_thread = None
            self._submit_thread = None
            self._collect_thread = None
            self._submit_executor.shutdown(wait=False, cancel_futures=True)
            self._submit_executor = ThreadPoolExecutor(max_workers=SUBMIT_CONCURRENCY)

    def close(self, timeout: float = 30.0) -> None:
        self.stop(timeout=timeout)

    def is_running(self) -> bool:
        threads = (
            self._producer_thread,
            self._submit_thread,
            self._collect_thread,
        )
        return any(t is not None and t.is_alive() for t in threads)

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def get_stats(self, reset: bool = False) -> dict[str, int]:
        with self._stats_lock:
            stats = dict(self._stats)
            if reset:
                self._stats = {
                    "producer_batches": 0,
                    "producer_errors": 0,
                    "producer_put_failures": 0,
                }
        return stats

    def set_stats(self, stats: dict[str, Any] | None) -> None:
        default_stats = {
            "producer_batches": 0,
            "producer_errors": 0,
            "producer_put_failures": 0,
        }
        normalized = dict(default_stats)
        if stats is not None:
            for key in normalized:
                value = stats.get(key, normalized[key])
                normalized[key] = int(value)
        with self._stats_lock:
            self._stats = normalized

    def get_ingest_stats(self, reset: bool = False) -> dict[str, int]:
        with self._stats_lock:
            stats = dict(self._ingest_stats)
            if reset:
                self._ingest_stats = {
                    "accepted": 0,
                    "filtered": 0,
                    "total": 0,
                }
        return stats

    def set_ingest_stats(self, stats: dict[str, Any] | None) -> None:
        default_stats = {
            "accepted": 0,
            "filtered": 0,
            "total": 0,
        }
        normalized = dict(default_stats)
        if stats is not None:
            for key in normalized:
                value = stats.get(key, normalized[key])
                normalized[key] = int(value)
        with self._stats_lock:
            self._ingest_stats = normalized

    def get_per_raas_stats(self, reset: bool = False) -> dict[str, dict[str, int]]:
        """Return per-RaaS throughput stats.

        Returns a dict of ``{uid: {produced, accepted, filtered}}``.
        """
        with self._per_raas_stats_lock:
            stats = {uid: dict(s) for uid, s in self._per_raas_stats.items()}
            if reset:
                self._per_raas_stats = {}
        return stats

    def get_reward_stats(self, reset: bool = False) -> dict[str, float | int]:
        with self._stats_lock:
            stats = dict(self._reward_stats)
            if reset:
                self._reward_stats = self._build_empty_reward_stats()
        return stats

    def set_reward_stats(self, stats: dict[str, Any] | None) -> None:
        defaults = self._build_empty_reward_stats()
        normalized: dict[str, float | int] = dict(defaults)
        if stats is not None:
            for key, default_value in defaults.items():
                raw = stats.get(key, default_value)
                if isinstance(default_value, int):
                    normalized[key] = int(raw)
                else:
                    normalized[key] = float(raw)
        with self._stats_lock:
            self._reward_stats = normalized

    # ------------------------------------------------------------------
    # Curator (selective rollout) — opt-in, no-op when curator is None.
    # ------------------------------------------------------------------

    def notify_version_changed(self, version: int) -> None:
        """Forward a weight-version bump to the curator (and stash it).

        Called by ``AstraFlowService.notify_version`` after the service
        commits a new version. The curator receives the hook so it can
        invalidate any version-dependent state; the stashed value is
        passed to ``should_submit`` on subsequent submit ticks.

        The first call latches ``_training_started=True``, which starts
        the warmup-epoch counter ticking in ``_get_next_sample``. This
        excludes the pre-fill burst (samples emitted before the trainer
        published its first weight version) from the epoch count.
        """
        with self._version_lock:
            self._current_version = int(version)
        if not self._training_started:
            self._training_started = True
            logger.info(
                "[curator] training_started=True at version=%s; warmup epoch "
                "counter armed (target=%s post-pre-fill samples)",
                version,
                self._warmup_target,
            )
        if self._curator is None:
            return
        try:
            with self._curator_lock:
                self._curator.on_version_changed(int(version))
        except Exception:
            logger.exception("curator.on_version_changed raised; ignoring")

    def get_curator_stats(self, reset: bool = False) -> dict[str, int]:
        """Return ``{selected, rejected, forced}`` counters."""
        with self._stats_lock:
            stats = dict(self._curator_stats)
            if reset:
                self._curator_stats = {
                    "selected": 0,
                    "rejected": 0,
                    "forced": 0,
                }
        return stats

    def has_curator(self) -> bool:
        """Whether selective rollout is enabled."""
        return self._curator is not None

    def get_curator_telemetry(self) -> dict[str, float]:
        """Return curator's optional telemetry dict (e.g. adaptive state).

        Empty dict if no curator or the curator does not implement
        ``get_telemetry()``.
        """
        if self._curator is None:
            return {}
        if not hasattr(self._curator, "get_telemetry"):
            return {}
        try:
            with self._curator_lock:
                return dict(self._curator.get_telemetry())
        except Exception:
            logger.exception("curator.get_telemetry raised; returning empty")
            return {}

    def get_curator_state(self) -> dict[str, Any]:
        """Snapshot the curator's persistent state for checkpointing.

        Empty dict if no curator or the curator has no ``state_dict``.
        """
        if self._curator is None or not hasattr(self._curator, "state_dict"):
            return {}
        try:
            with self._curator_lock:
                return self._curator.state_dict()
        except Exception:
            logger.exception("curator.state_dict raised; returning empty")
            return {}

    def load_curator_state(self, state: dict[str, Any] | None) -> None:
        """Restore the curator's persistent state from a checkpoint.

        If the restored state shows the curator was already past warmup
        (``adjustment_armed=True``), short-circuit our warmup tracking so
        we don't re-count an unnecessary epoch and re-fire the (idempotent)
        notify_warmup_complete().
        """
        if not state:
            return
        if self._curator is None or not hasattr(self._curator, "load_state_dict"):
            return
        try:
            with self._curator_lock:
                self._curator.load_state_dict(state)
        except Exception:
            logger.exception("curator.load_state_dict raised; ignoring")
            return
        if bool(state.get("adjustment_armed", False)):
            self._warmup_notified = True
            self._training_started = True
