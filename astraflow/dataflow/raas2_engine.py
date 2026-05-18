"""RaaS v2 HTTP client for AstraFlow data orchestration.

Implements the duck-typed interface expected by ``AstraDataAcquisition``
(``get_raas_availability``, ``submit_auto``, ``pull_completed``) and
the eval interface used by ``EvalManager``.

Weight transfer uses TCP-based pull from the trainer's sender endpoint.

Unlike the v1 ``RaaSInferenceEngine`` which dispatches all calls via a
generic ``/v1/raas/call`` RPC endpoint, this client talks directly to
dedicated ``/*`` REST endpoints.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pickle

import requests

try:
    import cloudpickle  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cloudpickle = None


def dumps_object(obj):
    if cloudpickle is not None:
        return cloudpickle.dumps(obj)
    return pickle.dumps(obj)


def loads_object(blob):
    if cloudpickle is not None:
        return cloudpickle.loads(blob)
    return pickle.loads(blob)

logger = logging.getLogger(__name__)

MAX_REQUEST_TIMEOUT_SEC = 600.0
# Control RPCs (pause/resume) can take longer than data-plane requests.
CONTROL_TIMEOUT_SEC = 300.0


class RaaS2InferenceEngine:
    """HTTP client that talks to a RaaS v2 service.

    Each RPC creates a fresh HTTP connection — no long-lived sessions are
    kept.  This avoids stale TCP connection issues that arise when
    keep-alive connections go idle between bursts of traffic.
    """

    def __init__(
        self,
        *,
        service_url: str,
        request_timeout: float = 10.0,
    ):
        self.service_url = service_url.rstrip("/")
        self.request_timeout = min(float(request_timeout), MAX_REQUEST_TIMEOUT_SEC)

        # Transparent workflow registration cache: spec JSON → workflow_id
        self._workflow_cache: dict[str, str] = {}
        self._workflow_counter = 0
        self._workflow_lock = threading.Lock()

        self._version = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # RPC helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """POST to ``/<path>`` with pickle serialization.

        Returns the ``result`` field from the server response.
        """
        url = f"{self.service_url}{path}"
        try:
            response = requests.post(
                url,
                data=dumps_object(payload),
                timeout=self.request_timeout,
                headers={"Content-Type": "application/octet-stream"},
            )
        except requests.exceptions.ReadTimeout:
            logger.error("POST %s timed out after %ss", path, self.request_timeout)
            raise
        try:
            decoded = loads_object(response.content)
        except Exception:
            response.raise_for_status()
            raise
        if not isinstance(decoded, dict):
            raise RuntimeError(f"Invalid RaaS v2 response from {path}.")
        if response.status_code >= 400:
            raise RuntimeError(decoded.get("error", f"HTTP {response.status_code}"))
        if not decoded.get("ok"):
            raise RuntimeError(decoded.get("error", "Unknown RaaS v2 error"))
        return decoded.get("result")

    def _post_collect(self, path: str, payload: dict[str, Any]) -> Any:
        """POST for the collect path (pull_completed / eval_pull)."""
        url = f"{self.service_url}{path}"
        response = requests.post(
            url,
            data=dumps_object(payload),
            timeout=self.request_timeout,
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            decoded = loads_object(response.content)
        except Exception:
            response.raise_for_status()
            raise
        if not isinstance(decoded, dict):
            raise RuntimeError(f"Invalid RaaS v2 response from {path}.")
        if response.status_code >= 400:
            raise RuntimeError(decoded.get("error", f"HTTP {response.status_code}"))
        if not decoded.get("ok"):
            raise RuntimeError(decoded.get("error", "Unknown RaaS v2 error"))
        return decoded.get("result")

    def _get(self, path: str) -> dict[str, Any]:
        """GET endpoint that returns JSON."""
        url = f"{self.service_url}{path}"
        response = requests.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        return response.json()

    def _get_rpc(self, path: str) -> Any:
        """GET endpoint that returns binary pickle via ``_encode_ok()``."""
        url = f"{self.service_url}{path}"
        response = requests.get(url, timeout=self.request_timeout)
        try:
            decoded = loads_object(response.content)
        except Exception:
            response.raise_for_status()
            raise
        if not isinstance(decoded, dict):
            raise RuntimeError(f"Invalid RaaS v2 response from {path}.")
        if response.status_code >= 400:
            raise RuntimeError(decoded.get("error", f"HTTP {response.status_code}"))
        if not decoded.get("ok"):
            raise RuntimeError(decoded.get("error", "Unknown RaaS v2 error"))
        return decoded.get("result")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, **kwargs) -> None:
        """Wait for the raas2 service to be ready by polling ``/status``."""
        max_wait = kwargs.get("max_wait", 600.0)
        verbose = kwargs.get("verbose", True)
        poll_interval = 2.0
        max_poll_interval = 10.0
        deadline = time.monotonic() + max_wait
        attempt = 0

        if verbose:
            logger.info("Waiting for RaaS v2 service at %s ...", self.service_url)

        while True:
            attempt += 1
            try:
                status = self._get("/status")
                service_status = status.get("status", "unknown")
                if service_status == "ready":
                    self._initialized = True
                    if verbose:
                        logger.info("RaaS v2 service is ready: %s", status)
                    return
                if service_status == "error":
                    raise RuntimeError(
                        f"RaaS v2 service error: {status.get('message', '')}"
                    )
                # status is "starting", "idle", etc. — keep polling
                if verbose:
                    logger.info(
                        "RaaS v2 status=%s, waiting for ready...", service_status
                    )
            except (requests.ConnectionError, requests.Timeout, OSError) as exc:
                if verbose and (attempt <= 3 or attempt % 10 == 0):
                    logger.warning(
                        "RaaS v2 not reachable (attempt %d): %s. Retrying in %.1fs...",
                        attempt,
                        exc,
                        poll_interval,
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                if verbose and (attempt <= 3 or attempt % 10 == 0):
                    logger.warning(
                        "RaaS v2 status check failed (attempt %d): %s. Retrying...",
                        attempt,
                        exc,
                    )

            if time.monotonic() > deadline:
                raise RuntimeError(f"RaaS v2 service not ready after {max_wait}s")
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, max_poll_interval)

    def destroy(self):
        """Reset client state (engine lifecycle managed by /shutdown)."""
        self._initialized = False
        self._workflow_cache.clear()

    # ------------------------------------------------------------------
    # Workflow registration (transparent, cached)
    # ------------------------------------------------------------------

    def _ensure_workflow_registered(
        self, workflow_spec: dict[str, Any], prefix: str = "auto"
    ) -> str:
        """Register a workflow spec on first use, return cached workflow_id.

        Uses a hashable JSON representation of the spec as cache key.
        """
        import json

        cache_key = json.dumps(workflow_spec, sort_keys=True)
        with self._workflow_lock:
            if cache_key in self._workflow_cache:
                return self._workflow_cache[cache_key]
            workflow_id = f"{prefix}-{self._workflow_counter}"
            self._workflow_counter += 1

        import time as _time

        _t0 = _time.monotonic()
        logger.info(
            "RaaS2 client: registering workflow spec %r as %r...",
            workflow_spec,
            workflow_id,
        )
        payload = {
            "workflow_id": workflow_id,
            "workflow_cls": workflow_spec["workflow_cls"],
            "reward_fn": workflow_spec.get("reward_fn"),
            "gconfig_overrides": workflow_spec.get("gconfig_overrides"),
            "workflow_kwargs": {
                k: v
                for k, v in workflow_spec.items()
                if k not in ("workflow_cls", "reward_fn", "gconfig_overrides")
            },
        }
        url = f"{self.service_url}/register_workflow"
        try:
            response = requests.post(
                url,
                data=dumps_object(payload),
                timeout=self.request_timeout,
                headers={"Content-Type": "application/octet-stream"},
            )
        except requests.exceptions.ReadTimeout:
            logger.error(
                "RaaS2 client: register_workflow spec POST timed out after %.1fs",
                self.request_timeout,
            )
            raise
        decoded = loads_object(response.content)
        if not isinstance(decoded, dict):
            raise RuntimeError("Invalid RaaS v2 response from /register_workflow.")
        if response.status_code >= 400:
            raise RuntimeError(decoded.get("error", f"HTTP {response.status_code}"))
        if not decoded.get("ok"):
            raise RuntimeError(decoded.get("error", "Unknown RaaS v2 error"))

        with self._workflow_lock:
            self._workflow_cache[cache_key] = workflow_id
        logger.info(
            "Registered workflow spec %r on RaaS v2 service (%.2fs).",
            workflow_id,
            _time.monotonic() - _t0,
        )
        return workflow_id

    def register_workflow(
        self, workflow_id: str, workflow_spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Register a workflow by spec dict."""
        payload = {
            "workflow_id": workflow_id,
            "workflow_cls": workflow_spec["workflow_cls"],
            "reward_fn": workflow_spec.get("reward_fn"),
            "gconfig_overrides": workflow_spec.get("gconfig_overrides"),
            "workflow_kwargs": {
                k: v
                for k, v in workflow_spec.items()
                if k not in ("workflow_cls", "reward_fn", "gconfig_overrides")
            },
        }
        return self._post("/register_workflow", payload)

    # ------------------------------------------------------------------
    # AstraDataAcquisition interface (duck-typed)
    # ------------------------------------------------------------------

    def get_raas_availability(self) -> dict[str, Any]:
        """Return capacity info — client uses ``available`` field."""
        return self._get("/availability")

    def submit_auto(
        self,
        data: dict[str, Any],
        workflow_spec: dict[str, Any],
    ) -> int:
        """Submit a single sample. Auto-registers workflow on first call."""
        workflow_id = self._ensure_workflow_registered(workflow_spec)
        result = self._post(
            "/submit",
            {"data": data, "workflow_id": workflow_id},
        )
        return result["task_id"]

    def pull_completed(
        self,
        max_items: int = 256,
        timeout: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Pull completed task results.

        Normalizes the response so each item has top-level ``ok``, ``result``,
        and ``task_id`` fields, matching what ``AstraDataAcquisition`` expects.
        """
        items = self._post_collect(
            "/pull",
            {"max_items": max_items, "timeout": timeout},
        )
        if not isinstance(items, list):
            raise RuntimeError(f"Invalid pull_completed response: {items}")

        normalized: list[dict[str, Any]] = []
        for item in items:
            result = item.get("result")
            # Error payloads from the server have {"ok": False, ...} in result
            if isinstance(result, dict) and "ok" in result and not result["ok"]:
                normalized.append(
                    {
                        "task_id": item.get("task_id"),
                        "ok": False,
                        "error": result.get("error", "unknown"),
                        "result": None,
                    }
                )
            else:
                normalized.append(
                    {
                        "task_id": item.get("task_id"),
                        "ok": True,
                        "result": result,
                    }
                )
        return normalized

    # ------------------------------------------------------------------
    # Training-engine reset (pre-eval)
    # ------------------------------------------------------------------

    def reset_training_engine(self, timeout: float = 5.0) -> dict:
        """Ask the RaaS server to wipe its training engine state.

        Cancels all in-flight training rollouts, drains the underlying
        SGLang servers via /pause_generation, clears the task dicts,
        and verifies num_running_reqs==0.  Called by the pool before
        each eval window so eval runs on a quiescent server.

        Returns the server's response dict, which includes
        ``ready_for_eval`` (bool), ``cancelled``, ``stragglers``,
        ``sglang_running`` and ``reset_epoch``.
        """
        print(
            f"[RaaS2 client] reset_training_engine(timeout={timeout}) sending...",
            flush=True,
        )
        url = f"{self.service_url}/reset_training_engine"
        response = requests.post(
            url,
            data=dumps_object({"timeout": float(timeout)}),
            # Give the HTTP call enough headroom past the server-side
            # cancel+drain timeout so a slow RaaS surfaces a timely
            # error rather than hanging this call.
            timeout=max(self.request_timeout, timeout + 10.0),
            headers={"Content-Type": "application/octet-stream"},
        )
        decoded = loads_object(response.content)
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            raise RuntimeError(
                decoded.get("error", "Unknown error")
                if isinstance(decoded, dict)
                else f"Invalid response from {url}"
            )
        result = decoded.get("result") or {}
        print(
            f"[RaaS2 client] reset_training_engine done: {result}",
            flush=True,
        )
        return result

    # ------------------------------------------------------------------
    # Eval interface
    # ------------------------------------------------------------------

    def eval_start(self):
        """Reset eval tracking state on the server before submitting eval tasks."""
        print("[RaaS2 client] eval_start() sending...", flush=True)
        url = f"{self.service_url}/eval_start"
        response = requests.post(
            url,
            data=dumps_object({}),
            timeout=self.request_timeout,
            headers={"Content-Type": "application/octet-stream"},
        )
        decoded = loads_object(response.content)
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            raise RuntimeError(
                decoded.get("error", "Unknown error")
                if isinstance(decoded, dict)
                else f"Invalid response from {url}"
            )
        print("[RaaS2 client] eval_start() done", flush=True)

    def eval_end(self):
        """Clear eval tracking state on the server after all results collected."""
        print("[RaaS2 client] eval_end() sending...", flush=True)
        url = f"{self.service_url}/eval_end"
        response = requests.post(
            url,
            data=dumps_object({}),
            timeout=self.request_timeout,
            headers={"Content-Type": "application/octet-stream"},
        )
        decoded = loads_object(response.content)
        if not isinstance(decoded, dict) or not decoded.get("ok"):
            raise RuntimeError(
                decoded.get("error", "Unknown error")
                if isinstance(decoded, dict)
                else f"Invalid response from {url}"
            )
        print("[RaaS2 client] eval_end() done", flush=True)

    def submit(
        self,
        data: dict[str, Any],
        workflow_spec: dict[str, Any],
    ) -> int:
        """Submit one eval sample to the RaaS service.

        Auto-registers the workflow on first use.
        """
        workflow_id = self._ensure_workflow_registered(workflow_spec, prefix="eval")
        # Use a generous timeout for eval submissions: the default
        # request_timeout (10s) is too tight when the RaaS event loop
        # has just been reset and is still stabilising after weight
        # loads.  Eval is not latency-critical.
        saved_timeout = self.request_timeout
        self.request_timeout = max(self.request_timeout, 60.0)
        try:
            result = self._post(
                "/eval_submit",
                {"data": data, "workflow_id": workflow_id},
            )
        finally:
            self.request_timeout = saved_timeout
        return result["task_id"]

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[dict[str, Any] | None]:
        """Wait for eval results from the RaaS service.

        Polls ``/eval_pull`` in a loop.  Stops when either:
        1. We have collected ``count`` results (all items returned), OR
        2. The server confirms it received all ``count`` items AND reports
           ``inflight == 0`` and ``pending == 0`` — meaning the engine
           finished processing everything; any missing results are lost.

        Parameters
        ----------
        count : int
            Expected number of results.
        timeout : float | None
            Optional wall-clock timeout. None means no timeout.
        raise_timeout : bool
            Whether to raise TimeoutError on timeout.
        """
        collected: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout if timeout is not None else None
        poll_interval = 0.5
        last_status_time = time.monotonic()
        poll_count = 0

        while True:
            remaining = max(1, count - len(collected))
            try:
                response = self._post_collect(
                    "/eval_pull",
                    {"max_items": remaining, "timeout": poll_interval},
                )
            except requests.exceptions.Timeout:
                # Benign: long-poll returned no new completions within the
                # read timeout. Suppress the traceback — we retry below.
                logger.info(
                    "RaaS2 eval wait: eval_pull long-poll timed out "
                    "(collected=%d/%d), retrying...",
                    len(collected), count,
                )
                time.sleep(poll_interval)
                continue
            except Exception:
                logger.warning(
                    "RaaS2 eval wait: eval_pull request failed, retrying...",
                    exc_info=True,
                )
                time.sleep(poll_interval)
                continue

            if not isinstance(response, dict):
                logger.warning(
                    "RaaS2 eval wait: unexpected response type: %s",
                    type(response),
                )
                time.sleep(poll_interval)
                continue

            items = response.get("items", [])
            inflight = response.get("inflight", -1)
            pending = response.get("pending", -1)
            total_submitted = response.get("total_submitted", -1)

            poll_count += 1
            if items and poll_count % 10 == 0:
                print(
                    f"[RaaS2 eval wait] collected {len(collected) + len(items)}/{count}, "
                    f"inflight={inflight}, pending={pending}, total_submitted={total_submitted}",
                    flush=True,
                )

            for item in items:
                result = item.get("result")
                if isinstance(result, dict) and "ok" in result and not result["ok"]:
                    collected.append(
                        {
                            "task_id": item.get("task_id"),
                            "ok": False,
                            "error": result.get("error", "unknown"),
                            "result": None,
                        }
                    )
                else:
                    collected.append(
                        {
                            "task_id": item.get("task_id"),
                            "ok": True,
                            "result": result,
                        }
                    )

            # Got all expected results.
            if len(collected) >= count:
                print(
                    f"[RaaS2 eval wait] collected all {len(collected)}/{count} results",
                    flush=True,
                )
                break

            # Server received all items and engine finished — some lost.
            if total_submitted >= count and inflight == 0 and pending == 0:
                print(
                    f"[RaaS2 eval wait] engine done, collected {len(collected)}/{count} results (some lost)",
                    flush=True,
                )
                break

            now = time.monotonic()
            if now - last_status_time >= 10.0:
                print(
                    f"[RaaS2 eval wait] collected {len(collected)}/{count}, "
                    f"inflight={inflight}, pending={pending}, total_submitted={total_submitted}",
                    flush=True,
                )
                last_status_time = now

            if deadline is not None and time.monotonic() > deadline:
                msg = (
                    f"RaaS2 eval wait: collected {len(collected)}/{count} "
                    f"results before timeout ({timeout}s), "
                    f"inflight={inflight}, pending={pending}"
                )
                print(f"[RaaS2 eval wait] TIMEOUT: {msg}", flush=True)
                if raise_timeout:
                    raise TimeoutError(msg)
                break

            time.sleep(poll_interval)

        return collected

    # ------------------------------------------------------------------
    # Internal control helper
    # ------------------------------------------------------------------

    def _post_control(self, path: str, payload: dict[str, Any]) -> Any:
        """POST with a longer timeout for control RPCs (pause/resume)."""
        url = f"{self.service_url}{path}"
        print(f"[RaaS2 client] _post_control({path}) sending to {url} ...", flush=True)
        try:
            response = requests.post(
                url,
                data=dumps_object(payload),
                timeout=CONTROL_TIMEOUT_SEC,
                headers={"Content-Type": "application/octet-stream"},
            )
            print(
                f"[RaaS2 client] _post_control({path}) got HTTP {response.status_code}",
                flush=True,
            )
        except requests.exceptions.ReadTimeout:
            print(
                f"[RaaS2 client] _post_control({path}) TIMED OUT after {CONTROL_TIMEOUT_SEC}s",
                flush=True,
            )
            raise
        except Exception as exc:
            print(
                f"[RaaS2 client] _post_control({path}) EXCEPTION: {exc}",
                flush=True,
            )
            raise
        try:
            decoded = loads_object(response.content)
        except Exception:
            response.raise_for_status()
            raise
        if not isinstance(decoded, dict):
            raise RuntimeError(f"Invalid RaaS v2 response from {path}.")
        if response.status_code >= 400:
            raise RuntimeError(decoded.get("error", f"HTTP {response.status_code}"))
        if not decoded.get("ok"):
            raise RuntimeError(decoded.get("error", "Unknown RaaS v2 error"))
        return decoded.get("result")

    def update_weights_from_agent(
        self,
        tensors_meta: list,
        load_format: str | None = None,
        flush_cache: bool = True,
        bootstrap: bool = False,
    ) -> None:
        """Forward update_weights_from_agent to RaaS service."""
        logger.info("RaaS2 client: update_weights_from_agent(bootstrap=%s)", bootstrap)
        try:
            self._post_control(
                "/update_weights_from_agent",
                {
                    "tensors_meta": tensors_meta,
                    "load_format": load_format,
                    "flush_cache": flush_cache,
                    "bootstrap": bootstrap,
                },
            )
        except Exception:
            logger.error(
                "RaaS2 client: update_weights_from_agent failed",
                exc_info=True,
            )
            raise
        logger.info("RaaS2 client: update_weights_from_agent() done")
