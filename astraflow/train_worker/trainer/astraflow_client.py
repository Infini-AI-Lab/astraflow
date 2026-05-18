"""HTTP client for the AstraFlow service.

Follows the same pattern as ``RaaS2InferenceEngine`` — pickle-based
serialization over HTTP using ``_dumps`` / ``_loads``.

The trainer uses this client to communicate with the AstraFlow HTTP
service via these endpoints:

- ``GET  /batch``              — get a training batch (blocks)
- ``POST /notify_version``     — notify new weight version (TCP mode)
- ``POST /ready``              — signal trainer readiness
- ``POST /save_buffer``        — trigger buffer checkpoint
- ``POST /shutdown``           — graceful shutdown
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pickle

import requests

try:
    import cloudpickle  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cloudpickle = None


def _dumps(obj):
    if cloudpickle is not None:
        return cloudpickle.dumps(obj)
    return pickle.dumps(obj)


def _loads(blob):
    if cloudpickle is not None:
        return cloudpickle.loads(blob)
    return pickle.loads(blob)

logger = logging.getLogger(__name__)

# AstraFlow control RPCs (save_buffer, non-eval notify_version).
CONTROL_TIMEOUT_SEC = 1800.0

# Eval can run for hours (large datasets, slow RaaS). 12-hour timeout.
EVAL_TIMEOUT_SEC = 43200.0

# get_batch blocks until data is available. Use long timeout.
BATCH_TIMEOUT_SEC = 3600.0


class AstraFlowClient:
    """HTTP client for the AstraFlow service.

    Parameters
    ----------
    service_url : str
        Base URL of the AstraFlow HTTP service (e.g., ``http://host:8000``).
    """

    def __init__(
        self,
        service_url: str,
        model_id: str | None = None,
    ):
        self.service_url = service_url.rstrip("/")
        self.model_id = model_id
        self._session = requests.Session()

    def initialize(self, max_wait: float = 600.0, verbose: bool = True) -> None:
        """Wait for AstraFlow service to be ready by polling ``GET /status``.

        Parameters
        ----------
        max_wait : float
            Maximum seconds to wait before raising RuntimeError.
        verbose : bool
            Whether to log progress.
        """
        poll_interval = 2.0
        max_poll_interval = 10.0
        deadline = time.monotonic() + max_wait
        attempt = 0

        if verbose:
            logger.info(
                "Waiting for AstraFlow service at %s ...", self.service_url
            )

        while True:
            attempt += 1
            try:
                resp = self._session.get(
                    f"{self.service_url}/status",
                    timeout=10.0,
                )
                resp.raise_for_status()
                status = resp.json()
                if status.get("status") == "ready":
                    if verbose:
                        logger.info("AstraFlow service is ready: %s", status)
                    return
                if verbose:
                    logger.info(
                        "AstraFlow status=%s, waiting for ready...",
                        status.get("status", "unknown"),
                    )
            except (requests.ConnectionError, requests.Timeout, OSError) as exc:
                if verbose and (attempt <= 3 or attempt % 10 == 0):
                    logger.warning(
                        "AstraFlow not reachable (attempt %d): %s. Retrying in %.1fs...",
                        attempt,
                        exc,
                        poll_interval,
                    )
            except Exception as exc:
                if verbose and (attempt <= 3 or attempt % 10 == 0):
                    logger.warning(
                        "AstraFlow status check failed (attempt %d): %s",
                        attempt,
                        exc,
                    )

            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"AstraFlow service not ready after {max_wait}s"
                )
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, max_poll_interval)

    def signal_ready(
        self,
        train_batch_size: int | None = None,
        sender_endpoint: str | None = None,
        recovered_version: int | None = None,
    ) -> None:
        """Tell AstraFlow that the trainer is ready.

        AstraFlow starts data acquisition only after both RaaS and the
        trainer have signalled readiness.

        Parameters
        ----------
        train_batch_size : int | None
            Number of examples (sequences) per training step. If provided,
            AstraFlow uses this as the batch size for ``get_batch``.
        sender_endpoint : str | None
            TCP sender endpoint for multi-model weight transfer.
        recovered_version : int | None
            If the trainer recovered from a checkpoint, the version to
            resume from.  AstraFlow uses this to set its internal version
            so that staleness filtering works correctly (avoids a version
            jump from 0 to N that would evict the entire buffer).
        """
        payload: dict[str, Any] = {}
        if train_batch_size is not None:
            payload["train_batch_size"] = train_batch_size
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        if sender_endpoint is not None:
            payload["sender_endpoint"] = sender_endpoint
        if recovered_version is not None:
            payload["recovered_version"] = recovered_version
        # The server-side /ready handler triggers a synchronous RaaS weight
        # load when recovering from a checkpoint.  When multiple models
        # contend for the RaaS weight-load slot (e.g. an ongoing training
        # update on model_i while model_j is still replaying its recovery
        # load), the handler can block well past 300s.  Observed ~546s in a
        # 2-model recovery; give ample headroom.
        resp = self._session.post(
            f"{self.service_url}/ready",
            data=_dumps(payload),
            headers={"Content-Type": "application/octet-stream"},
            timeout=1800.0,
        )
        resp.raise_for_status()
        logger.info(
            "Signalled ready to AstraFlow "
            "(train_batch_size=%s, model_id=%s, sender_endpoint=%s, "
            "recovered_version=%s)",
            train_batch_size,
            self.model_id,
            sender_endpoint,
            recovered_version,
        )

    def get_batch(
        self,
        timeout: float | None = None,
        version: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Get a training batch + buffer stats. Blocks until data is available.

        Parameters
        ----------
        timeout : float | None
            HTTP request timeout. Defaults to ``BATCH_TIMEOUT_SEC``.
        version : int | None
            Trainer's current version. In multi-model mode, the service
            blocks until all trainers are at compatible versions.

        Returns
        -------
        tuple[dict[str, Any], dict[str, float]]
            ``(batch, buffer_stats)`` — the training batch (tensor dict) and
            buffer/filter stats for wandb logging.
        """
        params = {}
        if self.model_id is not None:
            params["model_id"] = self.model_id
        if version is not None:
            params["version"] = version
        resp = self._session.get(
            f"{self.service_url}/batch",
            params=params,
            timeout=timeout or BATCH_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        result = _loads(resp.content)
        if isinstance(result, dict) and "batch" in result:
            return result["batch"], result.get("buffer_stats", {})
        # Backwards compatibility: old service returns raw batch
        return result, {}

    def save_buffer(self) -> None:
        """Trigger buffer checkpoint on the AstraFlow service."""
        resp = self._session.post(
            f"{self.service_url}/save_buffer",
            timeout=CONTROL_TIMEOUT_SEC,
        )
        resp.raise_for_status()

    def notify_version(
        self,
        version: int,
        run_eval: bool = False,
    ) -> dict[str, Any] | None:
        """Notify AstraFlow of a new weight version (TCP transfer mode).

        This does NOT pause or resume data acquisition around the weight
        copy — TCP transfer is non-blocking.
        If ``run_eval`` is True, AstraFlow pauses data acquisition only for
        the duration of eval, then resumes automatically.

        Parameters
        ----------
        version : int
            New model version.
        run_eval : bool
            Whether to trigger eval.

        Returns
        -------
        dict[str, Any] | None
            Eval results if ``run_eval=True``, else None.
        """
        payload: dict[str, Any] = {"version": version, "run_eval": run_eval}
        if self.model_id is not None:
            payload["model_id"] = self.model_id
        resp = self._session.post(
            f"{self.service_url}/notify_version",
            data=_dumps(payload),
            headers={"Content-Type": "application/octet-stream"},
            timeout=EVAL_TIMEOUT_SEC if run_eval else CONTROL_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        result = _loads(resp.content)
        if isinstance(result, dict):
            # Stash weight transfer info for wandb logging
            self._last_weight_transfer_info = result.get("weight_transfer_info")
            return result.get("eval_results")
        return None

    def get_last_weight_transfer_info(self) -> dict[str, Any] | None:
        """Return weight transfer info from the last notify_version call."""
        return getattr(self, "_last_weight_transfer_info", None)

    def notify_version_async(
        self,
        version: int,
        run_eval: bool = False,
    ) -> None:
        """Fire-and-forget version notification (TCP mode).

        Submits ``notify_version`` to a background thread so the trainer
        can continue training immediately.
        """
        import concurrent.futures

        if not hasattr(self, "_notify_executor"):
            self._notify_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="notify_version",
            )
        if not hasattr(self, "_pending_eval_results"):
            self._pending_eval_results = None
        # Wait for any previous notification to finish before submitting a
        # new one — we don't want two version notifications in flight at once.
        # Stash any eval results so they are not lost.
        if hasattr(self, "_notify_future") and self._notify_future is not None:
            try:
                result = self._notify_future.result(timeout=300)
                if result is not None and isinstance(result, dict):
                    self._pending_eval_results = result
            except Exception:
                pass
            self._notify_future = None
        self._notify_future = self._notify_executor.submit(
            self.notify_version, version, run_eval,
        )

    def collect_async_eval_results(self) -> dict[str, Any] | None:
        """Collect eval results from the last async notification, if ready.

        Non-blocking: returns None immediately if the notification is still
        in flight or if there are no results.  Also checks for results
        stashed by ``notify_version_async`` when it drained a previous future.
        """
        # Check stashed results first (from a previous future drained by
        # notify_version_async before we could collect them).
        if hasattr(self, "_pending_eval_results") and self._pending_eval_results is not None:
            result = self._pending_eval_results
            self._pending_eval_results = None
            return result
        if not hasattr(self, "_notify_future") or self._notify_future is None:
            return None
        if not self._notify_future.done():
            return None
        try:
            result = self._notify_future.result(timeout=0)
            self._notify_future = None
            return result
        except Exception:
            self._notify_future = None
            return None

    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        resp = self._session.get(
            f"{self.service_url}/status",
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def drain_pending_notifications(self, timeout: float = 300.0) -> None:
        """Block until any in-flight async version notification completes.

        Must be called before shutdown to avoid a race where the shutdown
        kills inference engines while a ``notify_all_versions`` weight load
        is still in progress on AstraFlow/RaaS.
        """
        if not hasattr(self, "_notify_future") or self._notify_future is None:
            return
        logger.info("Draining pending async version notification ...")
        try:
            self._notify_future.result(timeout=timeout)
        except Exception as exc:
            logger.warning("Pending notification finished with error: %s", exc)
        self._notify_future = None
        logger.info("Pending notification drained.")

    def shutdown_service(self) -> None:
        """Ask the AstraFlow service to shut down gracefully.

        Should only be called by rank 0 after training completes.
        Ignores connection errors (the service may already be gone).
        """
        try:
            logger.info("Sending shutdown to AstraFlow service at %s", self.service_url)
            self._session.post(
                f"{self.service_url}/shutdown",
                timeout=10.0,
            )
        except Exception as exc:
            # Service may exit before we read the response — that's fine.
            logger.info("AstraFlow shutdown request finished (exc=%s)", exc)

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
