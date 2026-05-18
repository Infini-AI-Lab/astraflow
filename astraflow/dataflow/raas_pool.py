"""Global thread-safe pool of RaaS2InferenceEngine instances.

``RaaSPool`` wraps N ``RaaS2InferenceEngine`` instances behind the same
duck-typed interface that ``AstraDataAcquisition`` expects.  It is shared
by all agents registered with ``AstraFlowService``.

Key design points
-----------------
- **Single global pool** — all agents share the same pool because every
  RaaS replica bootstraps all agent workflows.
- **Capacity-based routing** — ``submit_auto`` routes to the instance with
  the highest last-known available slot count.
- **Parallel collect** — ``pull_completed`` polls all live instances in
  parallel and merges results.
- **Broadcast-only version notify** — ``notify_all_versions`` fans out to
  all instances in parallel; each RaaS independently handles pause/resume.
- **Suspect-and-confirm failure detection** — data-path methods
  (``pull_completed``, ``get_raas_availability``) mark failing instances as
  *suspect* and wake the heartbeat thread.  Only the heartbeat thread
  increments failure counts and deregisters, keeping the two concerns
  cleanly separated.
- **Lazy workflow registration** — workflow specs are registered on-demand
  on first use via ``RaaS2InferenceEngine._ensure_workflow_registered``.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from astraflow.dataflow.raas2_engine import (
    RaaS2InferenceEngine,
    dumps_object,
    loads_object,
)

logger = logging.getLogger(__name__)

# The train_worker import chain (raas2_engine → io_struct → cli_args)
# replaces Logger.root and Logger.manager at import time via a custom
# dictConfig call.  This leaves loggers created *after* the swap (including
# this one) orphaned from the root handler installed by basicConfig().
# Detect this situation and attach a StreamHandler directly so that
# messages are not silently dropped.
def _ensure_logger_has_handler(lg: logging.Logger) -> None:
    """Add a StreamHandler if the logger's ancestor chain has no handlers."""
    c: logging.Logger | None = lg
    while c:
        if c.handlers:
            return
        if not c.propagate:
            break
        c = c.parent  # type: ignore[assignment]
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)

_ensure_logger_has_handler(logger)

_NOTIFY_TIMEOUT_SEC = 660.0
_AVAILABILITY_TIMEOUT_SEC = 5.0
_HEARTBEAT_HTTP_TIMEOUT_SEC = 30.0

# Log routing decisions every N submit_auto calls to avoid spamming logs.
_SUBMIT_LOG_INTERVAL = 50

# Mark a RaaS suspect after this many consecutive task failures (ok=False).
_TASK_FAILURE_SUSPECT_THRESHOLD = 5


class RaaSPool:
    """Global, thread-safe pool of ``RaaS2InferenceEngine`` instances.

    Parameters
    ----------
    heartbeat_interval:
        Seconds between heartbeat polls (default 10).
    heartbeat_max_failures:
        Consecutive heartbeat failures before auto-deregister (default 2).
    raas_initialize_timeout:
        Maximum seconds to wait for a newly registered RaaS to become ready
        (default 60).
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        heartbeat_max_failures: int = 10,
        raas_initialize_timeout: float = 60.0,
    ) -> None:
        self._engines: dict[str, RaaS2InferenceEngine] = {}  # uid -> engine
        self._lock = threading.RLock()
        self._version = 0

        # GPU count per uid — reported by each RaaS at registration time.
        self._gpu_counts: dict[str, int] = {}

        # Last-known available slot count per uid; updated by
        # get_raas_availability and decremented speculatively by submit_auto.
        self._last_availability: dict[str, int] = {}

        # Failure tracking — only the heartbeat thread increments these.
        self._failure_counts: dict[str, int] = {}

        # Weight transfer in-flight tracking.
        self._weight_transfer_active = False
        self._weight_transfer_lock = threading.Lock()
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_max_failures = heartbeat_max_failures
        self._raas_initialize_timeout = raas_initialize_timeout

        self._submit_count = 0  # for periodic routing log

        # Per-RaaS consecutive task failure counts.  Incremented when
        # pull_completed returns items with ok=False.  Reset to 0 when
        # a successful task result is received.  If a RaaS exceeds
        # _TASK_FAILURE_SUSPECT_THRESHOLD consecutive failures it is
        # marked suspect so the heartbeat thread can confirm.
        self._task_failure_counts: dict[str, int] = {}

        # Suspect set — data-path methods add uids here when an HTTP call
        # fails.  The heartbeat thread checks suspects via /health and
        # deregisters if confirmed dead.  Data-path methods skip suspect
        # uids to avoid log spam and wasted round-trips.
        self._suspect_uids: set[str] = set()
        self._suspect_event = threading.Event()  # wakes heartbeat early

        # Thread pool for parallel fan-out operations.
        self._executor = ThreadPoolExecutor(
            max_workers=64, thread_name_prefix="raas-pool"
        )

        # Heartbeat monitor thread.
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="raas-pool-heartbeat",
        )
        self._heartbeat_thread.start()

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def register(
        self,
        uid: str,
        raas_url: str,
        versions: dict[str, int] | None = None,
        sender_endpoints: dict[str, str] | None = None,
        gpu_count: int | None = None,
    ) -> None:
        """Add a RaaS instance to the global pool.

        Steps
        -----
        1. If ``uid`` is already registered, clean up the old engine.
        2. Create a ``RaaS2InferenceEngine`` and wait until ready.
        3. Add to the live instance dict immediately.
        4. The new RaaS will receive weights on the next regular
           ``notify_version`` broadcast from the training loop — no
           special catch-up needed.  This uses the exact same code path
           as all other RaaS instances.

        Workflow specs are registered lazily on first use via
        ``RaaS2InferenceEngine._ensure_workflow_registered``.

        Parameters
        ----------
        versions:
            Current ``{model_id: version}`` mapping (unused, kept for
            API compatibility).
        sender_endpoints:
            Current ``{model_id: "host:port"}`` mapping (unused, kept
            for API compatibility).
        gpu_count:
            Number of GPUs available to this RaaS instance.  Used by
            the balance report to compute per-GPU throughput.
        """
        with self._lock:
            if uid in self._engines:
                logger.warning(
                    "RaaSPool: uid=%s already registered, replacing", uid
                )
                old_engine = self._engines.pop(uid)
                self._failure_counts.pop(uid, None)
                self._task_failure_counts.pop(uid, None)
                self._last_availability.pop(uid, None)
                self._gpu_counts.pop(uid, None)
                try:
                    old_engine.destroy()
                except Exception:
                    pass

        logger.info(
            "RaaSPool: registering uid=%s at %s (timeout=%.0fs)",
            uid,
            raas_url,
            self._raas_initialize_timeout,
        )
        engine = RaaS2InferenceEngine(service_url=raas_url)
        engine.initialize(max_wait=self._raas_initialize_timeout, verbose=True)

        # Add to pool immediately.  The new RaaS gets weights via the
        # normal notify_version broadcast on the next training step —
        # the same code path used by every other RaaS instance.
        with self._lock:
            self._engines[uid] = engine
            self._failure_counts[uid] = 0
            self._task_failure_counts[uid] = 0
            if gpu_count is not None and gpu_count > 0:
                self._gpu_counts[uid] = gpu_count

        pool_size = len(self._engines)
        logger.info(
            "RaaSPool: uid=%s registered (pool size=%d, gpu_count=%s)",
            uid,
            pool_size,
            gpu_count,
        )
        print(
            f"[RaaSPool] +++ RaaS registered: uid={uid}  "
            f"gpu_count={gpu_count}  pool_size={pool_size}  "
            f"instances=[{', '.join(self._engines.keys())}]",
            flush=True,
        )

    def deregister(
        self, uid: str, reason: str = "manual", shutdown: bool = False,
    ) -> None:
        """Remove a RaaS instance from the pool.

        Parameters
        ----------
        reason:
            Human-readable reason for deregistration (e.g.
            ``"heartbeat_timeout"``, ``"manual"``).
        shutdown:
            If True, send ``POST /shutdown`` to the RaaS process so it
            terminates and frees GPU resources.  Used by elastic scaling.
        """
        with self._lock:
            engine = self._engines.pop(uid, None)
            self._failure_counts.pop(uid, None)
            self._task_failure_counts.pop(uid, None)
            self._last_availability.pop(uid, None)
            self._gpu_counts.pop(uid, None)
            self._suspect_uids.discard(uid)
            pool_size = len(self._engines)
            remaining_uids = list(self._engines.keys())

        if engine is None:
            logger.warning(
                "RaaSPool: deregister called for unknown uid=%s", uid
            )
            return

        logger.info(
            "RaaSPool: uid=%s deregistered — reason=%s, shutdown=%s, "
            "pool size=%d, remaining=%s",
            uid,
            reason,
            shutdown,
            pool_size,
            remaining_uids,
        )
        print(
            f"[RaaSPool] --- RaaS deregistered: uid={uid}  reason={reason}  "
            f"shutdown={shutdown}  pool_size={pool_size}  "
            f"remaining=[{', '.join(remaining_uids)}]",
            flush=True,
        )
        if pool_size == 0:
            logger.warning(
                "RaaSPool: pool is now empty — submit/collect will be no-ops "
                "until a new RaaS instance registers"
            )

        # Send /shutdown to the RaaS process so it terminates and frees
        # GPU resources.  Used by elastic scaling to reclaim GPUs.
        if shutdown and engine is not None:
            try:
                url = f"{engine.service_url}/shutdown"
                logger.info(
                    "RaaSPool: sending shutdown to uid=%s (%s)", uid, url
                )
                requests.post(url, timeout=10)
            except Exception:
                logger.warning(
                    "RaaSPool: shutdown request failed for uid=%s "
                    "(process may already be dead)",
                    uid,
                )

        # Destroy the local engine proxy when the RaaS is presumed dead
        # (e.g. heartbeat timeout).  For manual deregistration without
        # shutdown the RaaS process is still alive and may rejoin later.
        if reason != "manual" or shutdown:
            try:
                engine.destroy()
            except Exception:
                pass

    def list_instances(self) -> list[dict[str, Any]]:
        """Return status snapshot of all registered instances."""
        with self._lock:
            return [
                {
                    "uid": uid,
                    "url": engine.service_url,
                    "version": engine._version,
                    "initialized": engine._initialized,
                    "failure_count": self._failure_counts.get(uid, 0),
                    "last_availability": self._last_availability.get(uid, 0),
                    "suspect": uid in self._suspect_uids,
                    "gpu_count": self._gpu_counts.get(uid, 0),
                }
                for uid, engine in self._engines.items()
            ]

    def total_gpu_count(self) -> int:
        """Return total GPU count across all registered instances."""
        with self._lock:
            return sum(self._gpu_counts.values())

    def size(self) -> int:
        """Return number of registered instances."""
        with self._lock:
            return len(self._engines)

    def _get_live_engines(self) -> list[tuple[str, RaaS2InferenceEngine]]:
        """Return a stable snapshot of live (uid, engine) pairs."""
        with self._lock:
            return list(self._engines.items())

    def clear_all_suspects(self, reason: str = "") -> None:
        """Clear suspect flags and reset failure counts for all instances.

        Called by the eval leader after successfully loading weights on
        all RaaS instances.  The weight load itself confirms the RaaS is
        responsive, so any suspect flags from transient data-path
        failures during the load window should be cleared before eval
        checks for healthy engines.
        """
        with self._lock:
            cleared = list(self._suspect_uids)
            self._suspect_uids.clear()
            for uid in self._failure_counts:
                self._failure_counts[uid] = 0
        if cleared:
            logger.info(
                "RaaSPool: cleared %d suspect(s) %s (reason=%s)",
                len(cleared), cleared, reason or "manual",
            )

    def _mark_suspect(self, uid: str, context: str) -> None:
        """Flag *uid* as suspect and wake the heartbeat thread to confirm.

        Called by data-path methods (``pull_completed``,
        ``get_raas_availability``) when an HTTP call to *uid* fails.  The
        data path never deregisters directly — it only flags the instance
        so the heartbeat thread can do a ``/health`` check and decide.

        While suspect, the uid is skipped by data-path methods to avoid
        log spam and wasted HTTP round-trips.
        """
        with self._lock:
            if uid not in self._engines:
                return  # already deregistered
            if uid in self._suspect_uids:
                return  # already flagged
            self._suspect_uids.add(uid)
        logger.warning(
            "RaaSPool: uid=%s marked suspect (context=%s) "
            "— waking heartbeat for confirmation",
            uid,
            context,
        )
        self._suspect_event.set()  # wake heartbeat thread immediately

    def _get_healthy_engines(
        self,
    ) -> list[tuple[str, RaaS2InferenceEngine]]:
        """Return live engines excluding suspect ones."""
        with self._lock:
            return [
                (uid, eng)
                for uid, eng in self._engines.items()
                if uid not in self._suspect_uids
            ]

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def set_version_local(self, version: int, model_id: str | None = None) -> None:
        """Record the trainer's latest version on the pool (display only)."""
        with self._lock:
            self._version = version

    def _notify_one_model(
        self,
        uid: str,
        engine: RaaS2InferenceEngine,
        model_id: str,
        version: int,
        sender_endpoint: str,
    ) -> dict[str, Any]:
        """Send ``/notify_version`` for a single model to a single RaaS instance."""
        payload = dumps_object(
            {"model_id": model_id, "version": version, "sender_endpoint": sender_endpoint}
        )
        url = f"{engine.service_url}/notify_version"
        t0 = time.monotonic()
        logger.info(
            "RaaSPool: sending notify_version model=%s v=%d to uid=%s (%s)",
            model_id, version, uid, url,
        )
        resp = requests.post(
            url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
        resp.raise_for_status()
        result = loads_object(resp.content)
        logger.info(
            "RaaSPool: uid=%s model=%s weight load complete in %.1fs",
            uid, model_id, time.monotonic() - t0,
        )
        return result

    def notify_version(
        self,
        model_id: str,
        version: int,
        sender_endpoint: str,
    ) -> dict[str, Any]:
        """Fan-out ``/notify_version`` for a single model to all live RaaS instances.

        Each RaaS independently pulls weights from the sender and loads
        them into that model's engine.  The overall latency is
        ``max(all instances)`` rather than ``sum``.
        """
        engines = self._get_live_engines()
        if not engines:
            logger.warning(
                "RaaSPool.notify_version: pool is empty, skipping"
            )
            return {}

        with self._weight_transfer_lock:
            self._weight_transfer_active = True

        futs = {
            uid: self._executor.submit(
                self._notify_one_model, uid, engine,
                model_id, version, sender_endpoint,
            )
            for uid, engine in engines
        }

        t0 = time.monotonic()
        combined: dict[str, Any] = {}
        ok_uids: list[str] = []
        fail_uids: list[str] = []
        for uid, fut in futs.items():
            try:
                result = fut.result(timeout=_NOTIFY_TIMEOUT_SEC)
                combined[uid] = result
                ok_uids.append(uid)
                loaded_version = result.get("version") if isinstance(result, dict) else None
                if loaded_version is not None:
                    with self._lock:
                        eng = self._engines.get(uid)
                        if eng is not None:
                            eng._version = loaded_version
            except Exception as exc:
                logger.error(
                    "RaaSPool: notify_version failed for uid=%s model=%s: %s",
                    uid, model_id, exc,
                )
                combined[uid] = {"ok": False, "error": str(exc)}
                fail_uids.append(uid)

        with self._weight_transfer_lock:
            self._weight_transfer_active = False

        logger.info(
            "RaaSPool: notify_version model=%s v=%d completed in %.1fs — "
            "%d ok [%s]%s",
            model_id, version, time.monotonic() - t0,
            len(ok_uids), ", ".join(ok_uids),
            f", {len(fail_uids)} failed [{', '.join(fail_uids)}]"
            if fail_uids else "",
        )
        return combined

    def is_weight_transfer_active(self) -> bool:
        """Return whether a weight transfer is currently in flight."""
        with self._weight_transfer_lock:
            return self._weight_transfer_active

    # ------------------------------------------------------------------
    # AstraDataAcquisition duck-typed interface
    # ------------------------------------------------------------------

    def get_raas_availability(self) -> dict[str, Any]:
        """Return aggregate availability across all healthy (non-suspect)
        instances.

        Queries instances in parallel and caches per-uid counts for use by
        ``submit_auto``.  Suspect instances are skipped.
        """
        engines = self._get_healthy_engines()
        if not engines:
            return {"available": 0, "total": 0, "instances": 0}

        futs = {
            uid: self._executor.submit(engine.get_raas_availability)
            for uid, engine in engines
        }

        total_available = 0
        total_capacity = 0
        active = 0
        per_instance: dict[str, Any] = {}

        for uid, fut in futs.items():
            try:
                avail = fut.result(timeout=_AVAILABILITY_TIMEOUT_SEC)
                count = int(avail.get("available", 0))
                with self._lock:
                    self._last_availability[uid] = count
                total_available += count
                total_capacity += int(avail.get("total", 0))
                active += 1
                per_instance[uid] = avail
            except Exception:
                logger.debug(
                    "RaaSPool: availability check failed for uid=%s",
                    uid,
                    exc_info=True,
                )
                with self._lock:
                    self._last_availability[uid] = 0
                self._mark_suspect(uid, "availability")

        logger.debug(
            "RaaSPool: availability — %d/%d slots across %d instance(s)%s",
            total_available,
            total_capacity,
            active,
            " | " + ", ".join(
                f"{uid}={per_instance[uid].get('available', '?')}"
                for uid in per_instance
            ) if per_instance else "",
        )
        return {
            "available": total_available,
            "total": total_capacity,
            "instances": active,
            "per_instance": per_instance,
        }

    def submit_auto(
        self,
        data: dict[str, Any],
        workflow_spec: dict[str, Any],
    ) -> int:
        """Submit a single sample to the best-available RaaS instance.

        Routing is based on last-known availability (populated by the most
        recent ``get_raas_availability`` call).  The chosen instance's
        slot count is decremented speculatively to spread subsequent
        submissions within the same tick.

        Workflow spec registration is handled lazily by each engine on first
        use via ``RaaS2InferenceEngine._ensure_workflow_registered``.
        """
        engines_snapshot = dict(self._get_healthy_engines())
        if not engines_snapshot:
            logger.debug(
                "RaaSPool: submit_auto skipped — no healthy instances"
            )
            return -1

        # Pick instance with highest last-known available count.
        with self._lock:
            candidates = [
                (uid, self._last_availability.get(uid, 0))
                for uid in engines_snapshot
            ]

        # Sort descending by available count; uid is stable tiebreaker.
        candidates.sort(key=lambda x: (-x[1], x[0]))
        best_uid = candidates[0][0]

        # Speculative decrement to spread submissions across instances
        # when multiple concurrent calls arrive during the same tick.
        with self._lock:
            if self._last_availability.get(best_uid, 0) > 0:
                self._last_availability[best_uid] -= 1

        engine = engines_snapshot[best_uid]

        with self._lock:
            self._submit_count += 1
            count = self._submit_count

        if count % _SUBMIT_LOG_INTERVAL == 1:
            logger.info(
                "RaaSPool: submit #%d → uid=%s (avail=%d)%s",
                count,
                best_uid,
                candidates[0][1],
                " | candidates: " + ", ".join(
                    f"{uid}={a}" for uid, a in candidates
                ) if len(candidates) > 1 else "",
            )
        else:
            logger.debug(
                "RaaSPool: submit #%d → uid=%s (avail=%d)",
                count,
                best_uid,
                candidates[0][1],
            )

        return engine.submit_auto(data, workflow_spec)

    def pull_completed(
        self,
        max_items: int = 256,
        timeout: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Pull completed task results from all healthy instances in parallel.

        Each instance receives a budget of ``max_items // n_instances``
        to avoid over-consuming from any single instance.  Suspect
        instances are skipped — the heartbeat thread will confirm and
        either clear or deregister them.
        """
        engines = self._get_healthy_engines()
        if not engines:
            return []

        n = max(1, len(engines))
        per_instance_max = max(1, max_items // n)
        collect_timeout = max(timeout + 5.0, 10.0)

        futs = {
            uid: self._executor.submit(
                engine.pull_completed, per_instance_max, timeout
            )
            for uid, engine in engines
        }

        results: list[dict[str, Any]] = []
        per_uid_counts: dict[str, int] = {}
        for uid, fut in futs.items():
            try:
                items = fut.result(timeout=collect_timeout)
                # Tag each item with source RaaS uid for per-instance stats.
                for item in items:
                    item["_raas_uid"] = uid
                per_uid_counts[uid] = len(items)
                results.extend(items)
            except Exception:
                per_uid_counts[uid] = -1  # -1 signals error
                logger.warning(
                    "RaaSPool: pull_completed failed for uid=%s", uid
                )
                self._mark_suspect(uid, "pull_completed")

        # --- Task-failure circuit breaker ---
        # Track per-uid consecutive task failures.  If a RaaS returns
        # many ok=False results in a row, mark it suspect so the
        # heartbeat thread can confirm and deregister.
        per_uid_failures: dict[str, int] = {}
        per_uid_successes: dict[str, int] = {}
        for item in results:
            uid = item.get("_raas_uid")
            if uid is None:
                continue
            if bool(item.get("ok", True)):
                per_uid_successes[uid] = per_uid_successes.get(uid, 0) + 1
            else:
                per_uid_failures[uid] = per_uid_failures.get(uid, 0) + 1

        suspect_uids: list[str] = []
        with self._lock:
            for uid in per_uid_successes:
                self._task_failure_counts[uid] = 0
            for uid, count in per_uid_failures.items():
                if uid in per_uid_successes:
                    # Mixed results — some tasks succeeded, reset counter.
                    continue
                self._task_failure_counts[uid] = (
                    self._task_failure_counts.get(uid, 0) + count
                )
                if self._task_failure_counts[uid] >= _TASK_FAILURE_SUSPECT_THRESHOLD:
                    logger.warning(
                        "RaaSPool: uid=%s has %d consecutive task failures "
                        "(threshold=%d) — marking suspect",
                        uid,
                        self._task_failure_counts[uid],
                        _TASK_FAILURE_SUSPECT_THRESHOLD,
                    )
                    self._task_failure_counts[uid] = 0
                    suspect_uids.append(uid)

        for uid in suspect_uids:
            self._mark_suspect(uid, "task_failures")

        if results:
            logger.debug(
                "RaaSPool: pull_completed — %d items total | %s",
                len(results),
                ", ".join(f"{uid}={c}" for uid, c in per_uid_counts.items()),
            )
        return results

    # ------------------------------------------------------------------
    # Training-engine reset (called before each eval window)
    # ------------------------------------------------------------------

    def reset_training_engine(
        self, timeout: float = 5.0
    ) -> dict[str, Any]:
        """Fan out ``reset_training_engine`` to every healthy engine
        in parallel, block until all have returned, and mark any
        not-ready engine suspect so eval and the post-eval training
        resume both skip it.

        Blocking semantics
        ------------------
        Parallel submission via the pool's ThreadPoolExecutor, so
        total wall time is ``max(per-engine reset time)`` rather than
        the sum.  The join loop blocks until every engine has
        returned or individually timed out; the per-engine future
        timeout prevents one hung RaaS from wedging the whole
        training loop.

        Partial failure
        ---------------
        An engine that returns ``ready_for_eval=False`` (or whose
        HTTP call errors outright) is passed to ``_mark_suspect`` so
        that:

        1. ``_get_eval_engines`` excludes it from the upcoming eval
           round.
        2. All data-path methods (submit, pull_completed,
           get_raas_availability) skip it when training resumes.
        3. The heartbeat thread wakes immediately via
           ``_suspect_event`` and probes ``/status`` to decide
           whether the engine returns to rotation or is deregistered.

        No manual intervention is needed to keep a failed RaaS from
        influencing future training.
        """
        with self._lock:
            engines = [
                (uid, eng)
                for uid, eng in self._engines.items()
                if eng._initialized and uid not in self._suspect_uids
            ]

        if not engines:
            logger.warning(
                "RaaSPool: reset_training_engine called with 0 healthy "
                "engines — eval will also have 0 targets"
            )
            return {}

        logger.info(
            "RaaSPool: reset_training_engine fan-out to %d engine(s): %s",
            len(engines),
            [uid for uid, _ in engines],
        )

        futs = {
            uid: self._executor.submit(eng.reset_training_engine, timeout)
            for uid, eng in engines
        }

        results: dict[str, Any] = {}
        for uid, fut in futs.items():
            try:
                results[uid] = fut.result(timeout=timeout + 10.0)
            except Exception as exc:
                logger.error(
                    "RaaSPool: reset_training_engine failed for uid=%s: %s",
                    uid,
                    exc,
                    exc_info=True,
                )
                results[uid] = {
                    "ready_for_eval": False,
                    "error": repr(exc),
                }

        # Mark not-ready engines suspect so both eval and the
        # post-eval training resume skip them.  The heartbeat thread
        # will confirm via /status and either reinstate or deregister.
        not_ready = [
            uid for uid, r in results.items() if not r.get("ready_for_eval")
        ]
        for uid in not_ready:
            self._mark_suspect(uid, "reset_failed")

        ready_count = len(results) - len(not_ready)
        logger.info(
            "RaaSPool: reset_training_engine complete — "
            "ready=%d not_ready=%d (suspect=%s)",
            ready_count,
            len(not_ready),
            not_ready,
        )
        return results

    # ------------------------------------------------------------------
    # Eval interface — distributed across all live instances
    # ------------------------------------------------------------------

    def _get_eval_engines(self) -> list[tuple[str, RaaS2InferenceEngine]]:
        """Return all healthy engines for eval traffic.

        Excludes suspect uids so eval does not route to an engine that
        just failed reset_training_engine or was flagged by the data
        path.  The heartbeat thread decides whether a suspect returns
        to rotation or is deregistered.
        """
        with self._lock:
            engines = [
                (uid, eng)
                for uid, eng in self._engines.items()
                if eng._initialized and uid not in self._suspect_uids
            ]
        if not engines:
            raise RuntimeError(
                "RaaSPool: no healthy RaaS instance available for eval"
            )
        return engines

    def eval_start(self) -> None:
        """Broadcast eval_start to all live engines in parallel."""
        engines = self._get_eval_engines()
        # Build a GPU-weighted slot list so round-robin submission
        # distributes eval tasks proportionally to each engine's GPU
        # count rather than one-per-engine.  e.g. a 2-GPU + 8-GPU pool
        # yields a length-10 slot list → 8:2 task split.  Engines with
        # unknown gpu_count fall back to 1 slot each.
        slots: list[tuple[str, RaaS2InferenceEngine]] = []
        for uid, eng in engines:
            gpus = max(1, int(self._gpu_counts.get(uid, 1)))
            for _ in range(gpus):
                slots.append((uid, eng))
        logger.info(
            "RaaSPool: eval_start broadcasting to %d engine(s): %s "
            "(gpu-weighted slots=%d, per_uid=%s)",
            len(engines),
            [uid for uid, _ in engines],
            len(slots),
            {uid: int(self._gpu_counts.get(uid, 1)) for uid, _ in engines},
        )
        # Reset pool-level eval tracking state.
        self._eval_engines: list[tuple[str, RaaS2InferenceEngine]] = engines
        self._eval_slots: list[tuple[str, RaaS2InferenceEngine]] = slots
        self._eval_rr_idx = 0  # round-robin index over _eval_slots
        self._eval_next_pool_id = 0
        # Map pool_task_id → (uid, engine_task_id)
        self._eval_task_map: dict[int, tuple[str, int]] = {}
        # Track how many tasks were submitted to each engine uid.
        self._eval_uid_submitted: dict[str, int] = {uid: 0 for uid, _ in engines}

        futs = {
            uid: self._executor.submit(eng.eval_start)
            for uid, eng in engines
        }
        for uid, fut in futs.items():
            try:
                fut.result(timeout=60)
            except Exception:
                logger.error("RaaSPool: eval_start failed for uid=%s", uid, exc_info=True)

    def eval_end(self) -> None:
        """Broadcast eval_end to all live engines in parallel."""
        engines = getattr(self, "_eval_engines", None) or self._get_eval_engines()
        uid_submitted = getattr(self, "_eval_uid_submitted", {})
        logger.info(
            "RaaSPool: eval_end broadcasting to %d engine(s), "
            "total submitted=%d, distribution=%s",
            len(engines),
            sum(uid_submitted.values()),
            dict(uid_submitted),
        )
        futs = {
            uid: self._executor.submit(eng.eval_end)
            for uid, eng in engines
        }
        for uid, fut in futs.items():
            try:
                fut.result(timeout=60)
            except Exception:
                logger.error("RaaSPool: eval_end failed for uid=%s", uid, exc_info=True)
        # Clear pool-level eval state.
        self._eval_engines = []
        self._eval_slots = []
        self._eval_task_map = {}
        self._eval_uid_submitted = {}

    def shutdown_all(self, per_engine_timeout: float = 5.0) -> dict[str, str]:
        """Broadcast POST /shutdown to every registered RaaS in parallel.

        Best-effort: failures are logged, never raised. Called from
        AstraFlow's process /shutdown handler so remote RaaS instances
        on other nodes terminate before this process exits — local
        engines die with the parent shell, but a remote engine launched
        by a separate srun has no other signal to stop on.
        """
        with self._lock:
            engines = list(self._engines.items())
        if not engines:
            return {}
        logger.info(
            "RaaSPool: shutdown_all broadcasting to %d engine(s): %s",
            len(engines), [uid for uid, _ in engines],
        )

        def _post(uid_url: tuple[str, str]) -> str:
            _, url = uid_url
            try:
                requests.post(f"{url}/shutdown", timeout=per_engine_timeout)
                return "ok"
            except Exception as exc:
                return f"failed: {exc.__class__.__name__}"

        futs = {
            uid: self._executor.submit(_post, (uid, eng.service_url))
            for uid, eng in engines
        }
        statuses: dict[str, str] = {}
        for uid, fut in futs.items():
            try:
                statuses[uid] = fut.result(timeout=per_engine_timeout + 1.0)
            except Exception as exc:
                statuses[uid] = f"future-error: {exc.__class__.__name__}"
            logger.info("RaaSPool: shutdown %s -> %s", uid, statuses[uid])
        return statuses

    def submit(
        self,
        data: dict[str, Any],
        workflow_spec: dict[str, Any],
    ) -> int:
        """Submit one eval sample via GPU-weighted round-robin.

        _eval_slots is a slot list built at eval_start where each engine
        appears ``gpu_count`` times, so strict round-robin over it yields
        a GPU-proportional task distribution (e.g. a 2-GPU + 8-GPU pool
        gets 2:8 tasks per 10 submissions).
        """
        slots = self._eval_slots
        if not slots:
            raise RuntimeError("RaaSPool: eval_start() not called or no engines")

        # GPU-weighted round-robin pick.
        idx = self._eval_rr_idx % len(slots)
        self._eval_rr_idx += 1
        uid, engine = slots[idx]

        engine_task_id = engine.submit(data, workflow_spec)
        pool_task_id = self._eval_next_pool_id
        self._eval_next_pool_id += 1
        self._eval_task_map[pool_task_id] = (uid, engine_task_id)
        self._eval_uid_submitted[uid] = self._eval_uid_submitted.get(uid, 0) + 1
        return pool_task_id

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[dict[str, Any] | None]:
        """Wait for eval results from all engines in parallel, merge results.

        Polls each engine that received tasks for its share of results,
        then remaps engine-level task_ids back to pool-level task_ids.
        """
        engines = self._eval_engines
        if not engines:
            raise RuntimeError("RaaSPool: eval_start() not called or no engines")

        # Build reverse map: engine_task_id → pool_task_id per uid.
        uid_reverse: dict[str, dict[int, int]] = {}
        for pool_id, (uid, eng_id) in self._eval_task_map.items():
            uid_reverse.setdefault(uid, {})[eng_id] = pool_id

        logger.info(
            "RaaSPool: eval wait for %d results across %d engine(s): %s",
            count,
            len(engines),
            {uid: n for uid, n in self._eval_uid_submitted.items()},
        )

        def _wait_one(uid: str, engine: RaaS2InferenceEngine) -> list[dict[str, Any]]:
            n = self._eval_uid_submitted.get(uid, 0)
            if n == 0:
                return []
            # Always return partial results on timeout instead of raising.
            results = engine.wait(n, timeout=timeout, raise_timeout=False)
            rmap = uid_reverse.get(uid, {})
            for r in results:
                if r is not None and "task_id" in r:
                    r["task_id"] = rmap.get(r["task_id"], r["task_id"])
            return results

        futs = {
            uid: self._executor.submit(_wait_one, uid, eng)
            for uid, eng in engines
        }

        # Wait with generous margin beyond the per-engine timeout.
        pool_timeout = (timeout or 3600) + 60
        all_results: list[dict[str, Any] | None] = []
        for uid, fut in futs.items():
            try:
                all_results.extend(fut.result(timeout=pool_timeout))
            except Exception:
                logger.error(
                    "RaaSPool: eval wait failed for uid=%s", uid, exc_info=True
                )

        logger.info(
            "RaaSPool: eval wait collected %d/%d results", len(all_results), count
        )
        return all_results

    # ------------------------------------------------------------------
    # Heartbeat monitor
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Background daemon: poll ``/health`` on each instance.

        Runs on a regular interval *and* wakes early when the data path
        flags a suspect via ``_mark_suspect``.  This is the **only** place
        that increments ``_failure_counts`` and calls ``deregister``.
        """
        logger.info(
            "RaaSPool heartbeat thread started "
            "(interval=%.1fs, max_failures=%d)",
            self._heartbeat_interval,
            self._heartbeat_max_failures,
        )
        self._hb_tick_count = 0
        self._hb_last_pool_size = 0
        while not self._stop_event.is_set():
            try:
                self._heartbeat_tick()
            except Exception:
                logger.exception("RaaSPool heartbeat tick crashed unexpectedly")

            # Wait for the next regular tick, OR wake early if a suspect
            # is flagged by the data path.
            self._suspect_event.wait(timeout=self._heartbeat_interval)
            self._suspect_event.clear()

    # Log a summary every ~60s worth of ticks instead of every tick.
    _HB_SUMMARY_EVERY = 6

    def _heartbeat_tick(self) -> None:
        """Single heartbeat pass — separated for exception safety."""
        self._hb_tick_count += 1
        engines = self._get_live_engines()
        pool_size = len(engines)

        # Log on pool-size changes or every _HB_SUMMARY_EVERY ticks (quiet otherwise).
        pool_changed = pool_size != self._hb_last_pool_size
        if engines and (
            pool_changed
            or self._hb_tick_count % self._HB_SUMMARY_EVERY == 1
        ):
            with self._lock:
                n_suspect = len(self._suspect_uids)
                instance_details = []
                for uid, eng in engines:
                    v = getattr(eng, "_version", "?")
                    avail = self._last_availability.get(uid, "?")
                    instance_details.append(f"{uid}(v={v},avail={avail})")
            summary = (
                f"[RaaSPool] pool_size={pool_size} | "
                + " | ".join(instance_details)
            )
            if n_suspect:
                summary += f" | suspects={n_suspect}"
            print(summary, flush=True)
            logger.info(
                "RaaSPool heartbeat: pool size=%d, instances=[%s]%s",
                pool_size,
                ", ".join(uid for uid, _ in engines),
                f", suspects={n_suspect}" if n_suspect else "",
            )
        self._hb_last_pool_size = pool_size

        for uid, engine in engines:
            with self._lock:
                if uid not in self._engines:
                    continue
            try:
                # Use /status (not /health) so the check reflects actual
                # engine health.  /health is a trivial OK that stays up even
                # when the inference workers behind the manager are dead.
                resp = requests.get(
                    f"{engine.service_url}/status",
                    timeout=_HEARTBEAT_HTTP_TIMEOUT_SEC,
                )
                resp.raise_for_status()
                status_data = resp.json()
                svc_status = status_data.get("status", "unknown")
                if svc_status not in ("ready", "loading"):
                    raise RuntimeError(
                        f"RaaS status={svc_status}: "
                        f"{status_data.get('message', '')}"
                    )
                # Healthy — reset failure count and clear suspect flag.
                with self._lock:
                    self._failure_counts[uid] = 0
                    was_suspect = uid in self._suspect_uids
                    self._suspect_uids.discard(uid)
                if was_suspect:
                    logger.info(
                        "RaaSPool: uid=%s suspect cleared — healthy",
                        uid,
                    )
            except Exception:
                with self._lock:
                    if uid not in self._engines:
                        continue
                    self._failure_counts[uid] = (
                        self._failure_counts.get(uid, 0) + 1
                    )
                    failures = self._failure_counts[uid]

                if failures >= self._heartbeat_max_failures:
                    logger.warning(
                        "RaaSPool: uid=%s confirmed dead "
                        "(%d consecutive heartbeat failures) "
                        "— deregistering",
                        uid,
                        failures,
                    )
                    self.deregister(uid, reason="heartbeat_confirmed")
                else:
                    logger.warning(
                        "RaaSPool: uid=%s heartbeat failed (%d/%d)",
                        uid,
                        failures,
                        self._heartbeat_max_failures,
                    )

    def shutdown(self) -> None:
        """Stop the heartbeat thread and shut down the thread pool."""
        self._stop_event.set()
        self._suspect_event.set()  # unblock heartbeat wait
        self._executor.shutdown(wait=False)
