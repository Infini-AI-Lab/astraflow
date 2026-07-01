import asyncio
import os
import random
import subprocess
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

# # Prevent HuggingFace tokenizers fork warning — ProcessPoolExecutor forks
# # after tokenizers parallelism is already initialized in the main process.
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from contextlib import asynccontextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any, Protocol

import aiohttp
import requests
import torch.distributed as dist
import uvloop

from astraflow.raas.api.cli_args import InferenceEngineConfig
from astraflow.raas.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
)
from astraflow.raas.utils import logging
from astraflow.raas.utils.http import arequest_with_retry, get_default_connector
from astraflow.raas.utils.launcher import wait_llm_server_addrs
from astraflow.raas.utils.network import find_free_ports, gethostip
from astraflow.raas.utils.perf_tracer import trace_perf
from astraflow.raas.utils.proc import kill_process_tree

RID_CACHE_SIZE = 128

logger = logging.getLogger(__name__)

_session_storage = ContextVar("aiohttp.ClientSession")

# Prometheus metrics the adaptive controller cares about. Anything not in
# this set is discarded by the parser to keep it cheap.
_METRICS_WANTED: frozenset[str] = frozenset(
    {
        "sglang:num_queue_reqs",    # waiting queue depth (decision signal)
        "sglang:num_running_reqs",  # running batch size (observability)
        "sglang:token_usage",       # KV cache fraction (observability in v1)
    }
)


def _parse_prometheus_metrics(
    text: str, wanted: frozenset[str] | set[str]
) -> dict[str, float]:
    """Parse Prometheus text exposition format, extracting only wanted metrics.

    Returns ``{metric_name: sum_of_values_across_labels}``. Metrics with
    multiple labeled series (e.g. one per ``dp_rank`` inside a single server)
    are summed into a single value per metric name. Comment lines, blank
    lines, and metric names not in ``wanted`` are silently discarded.
    Unparseable lines are skipped rather than raising — telemetry should
    never take down the caller.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: ``name{labels...} value``  OR  ``name value``
        if "{" in line:
            name, rest = line.split("{", 1)
            close_idx = rest.rfind("}")
            if close_idx < 0:
                continue  # malformed
            value_str = rest[close_idx + 1 :].strip()
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, value_str = parts
        if name not in wanted:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(value_str)
        except ValueError:
            continue
    return out


def _walk_descendants(pid: int) -> list[int]:
    """Recursively enumerate descendant PIDs of ``pid`` via ``/proc``.

    Reads ``/proc/{pid}/task/{pid}/children`` (Linux ``CONFIG_PROC_CHILDREN``,
    enabled by default). Returns an empty list if the entry is missing or
    the process has already vanished — both are non-fatal for callers that
    just want a "is anybody zombie" check.
    """
    out: list[int] = []
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            children_str = f.read().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return out
    if not children_str:
        return out
    for token in children_str.split():
        try:
            cpid = int(token)
        except ValueError:
            continue
        out.append(cpid)
        out.extend(_walk_descendants(cpid))
    return out


def _read_proc_state(pid: int) -> str | None:
    """Return the State letter (R/S/D/Z/T/...) from ``/proc/{pid}/status``.

    Returns ``None`` if the process is gone or the file is unreadable.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
                    return None
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


class RemoteInfBackendProtocol(Protocol):
    """Protocol defining backend-specific operations for remote inference engines.

    This protocol abstracts the differences between various remote inference servers
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Building HTTP requests with backend-specific formats
    - Parsing backend-specific responses
    - Handling weight updates
    - Managing control flow (pause/resume)
    - Supporting optional features (LoRA)

    Implementations can raise NotImplementedError for unsupported features.
    """

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool
    ) -> HttpRequest:
        """Build HTTP request for text generation.

        Parameters
        ----------
        req : ModelRequest
            The generation request containing input and parameters
        with_lora : bool
            Whether to specify a LoRA to use

        Returns
        -------
        HttpRequest
            The HTTP request with endpoint and payload
        """
        ...

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Parse generation response into standard format.

        Parameters
        ----------
        response : Dict[str, Any]
            The raw JSON response from the server

        Returns
        -------
        HttpGenerationResult
            Parsed result with tokens, logprobs, and stop reason
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        """Get request to pause generation.

        Returns
        -------
        HttpRequest
            The HTTP request to pause generation

        Raises
        ------
        NotImplementedError
            If pause is not supported by this backend
        """
        ...

    def get_resume_request(self) -> HttpRequest:
        """Get request to resume generation.

        Returns
        -------
        HttpRequest
            The HTTP request to resume generation

        Raises
        ------
        NotImplementedError
            If resume is not supported by this backend
        """
        ...

    def get_health_check_request(self) -> HttpRequest:
        """Get the health check request.

        Returns
        -------
        HttpRequest
            The HTTP request for health checks
        """
        ...

    def get_offload_request(self) -> HttpRequest:
        """Get request to offload model memory.

        Returns
        -------
        HttpRequest
            The HTTP request to offload model memory

        Raises
        ------
        NotImplementedError
            If offload is not supported by this backend
        """
        ...

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Get request to onload model memory.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.

        Returns
        -------
        HttpRequest
            The HTTP request to onload model memory

        Raises
        ------
        NotImplementedError
            If onload is not supported by this backend
        """
        ...

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Launch inference server subprocess.

        Parameters
        ----------
        server_args : dict[str, Any]
            Server configuration arguments for build_cmd_from_args

        Returns
        -------
        subprocess.Popen
            The launched server process
        """
        ...


class RemoteInfEngine:
    """HTTP-based remote inference engine (simplified, no orchestration layer).

    Provides HTTP client functionality for communicating with remote inference
    servers. Backend-specific behaviors are delegated to an injected
    RemoteInfBackendProtocol implementation.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : RemoteInfBackendProtocol
        Backend implementation providing server-specific behavior
    """

    def __init__(
        self, config: InferenceEngineConfig, backend: RemoteInfBackendProtocol
    ):
        self.config = config
        self.backend = backend

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self._version = 0

        self.lock = Lock()

        self.lora_initialized = False
        # Versioned LoRA adapter naming: each weight sync loads under a NEW
        # name (``lora_v{seq}``) and we never unload. Unloading an adapter that
        # still has paused/aborted in-flight requests deadlocks on SGLang's
        # ``wait_for_unload`` (aborted requests never release their usage
        # counter). New unique names avoid the unload entirely; SGLang's
        # mem-pool LRU evicts stale adapters from GPU automatically.
        self._lora_seq = 0
        self._current_lora_name: str | None = None

        self._executor: ProcessPoolExecutor | None = None
        self._paused: bool = False
        self.local_server_processes: list[LocalInfServerInfo] = []

        # Per-server generation stats
        self._inflight_per_server: dict[str, int] = {}
        self._completed_per_server: dict[str, int] = {}

    def _create_session(self) -> aiohttp.ClientSession:
        """Create a ClientSession for the current asyncio coroutine.

        Returns
        -------
        aiohttp.ClientSession
            A new client session object
        """
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                sock_connect=self.config.request_timeout,
                connect=self.config.request_timeout,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        )

    @asynccontextmanager
    async def managed_session(self):
        """Provide a managed ClientSession with automatic lifecycle handling.

        Creates a ClientSession, stores it in task-local context for nested
        agenerate() calls to reuse, and ensures proper cleanup.

        Yields
        ------
            None

        Examples
        --------
            async with engine.managed_session():
                result = await engine.agenerate(request)
        """
        session = self._create_session()
        token = _session_storage.set(session)
        try:
            yield
        finally:
            await session.close()
            _session_storage.reset(token)

    def _wait_for_server(self, address):
        """Wait for a server to become healthy."""
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise TimeoutError("server launch failed")

    def check_health(self, base_url):
        """Check if server is healthy.

        Uses a short 5 s timeout — a healthy /health endpoint responds in
        milliseconds; anything slower is effectively unhealthy and should
        not block callers. (Historically this was 30 s, which combined
        with 6 sync probes on the FastAPI event-loop thread was enough
        to blow through the RaaS self-register 10 s budget.)
        """
        try:
            health_req = self.backend.get_health_check_request()
            url = f"{base_url}{health_req.endpoint}"
            response = requests.request(
                health_req.method, url, json=health_req.payload, timeout=20
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def check_subprocesses_alive(self) -> tuple[bool, str]:
        """Verify every locally launched server's process tree is alive.

        Returns ``(True, "")`` if every entrypoint Popen is still running
        and no descendant in any tree is a zombie. Otherwise returns
        ``(False, reason)`` with a human-readable cause.

        This catches the SGLang-specific failure mode where the engine's
        own watchdog kills its scheduler/detokenizer subprocesses while
        the entrypoint HTTP server keeps running. The HTTP /health probe
        cannot detect that — it only checks the front door.

        Cheap: pure ``/proc`` reads, no IPC, no GIL-blocking work. Safe
        to call from any thread.
        """
        for server_info in self.local_server_processes:
            entry_pid = server_info.process.pid
            rc = server_info.process.poll()
            if rc is not None:
                return (
                    False,
                    f"server entrypoint pid={entry_pid} exited with rc={rc}",
                )
            for cpid in _walk_descendants(entry_pid):
                state = _read_proc_state(cpid)
                if state == "Z":
                    return (
                        False,
                        f"zombie descendant pid={cpid} of entrypoint "
                        f"pid={entry_pid}",
                    )
        return True, ""

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        """Initialize the engine by discovering and connecting to servers.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        addr : str | List[str] | None
            Server address(es) to connect to. If None, will auto-discover.
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[Remote Inference Engine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info("Get server addresses from the `addr` argument.")
        elif len(self.local_server_processes) > 0:
            self.addresses = [f"{s.host}:{s.port}" for s in self.local_server_processes]
            self.logger.info("Get server addresses from the local subprocess.")
        elif os.getenv("ASTRAFLOW_LLM_SERVER_ADDRS"):
            # When addr is not provided, fallback to reading addrs from env var
            self.addresses = os.environ["ASTRAFLOW_LLM_SERVER_ADDRS"].split(",")
            self.logger.info("Get server addresses from environment variable.")
        else:
            if (
                self.config.experiment_name is not None
                and self.config.trial_name is not None
            ):
                try:
                    self.addresses = wait_llm_server_addrs(
                        experiment_name=self.config.experiment_name,
                        trial_name=self.config.trial_name,
                        timeout=1,
                    )
                    self.logger.info("Get server addresses from name_resolve.")
                except (TimeoutError, RuntimeError):
                    # RuntimeError happens when name_resolve is not properly configured.
                    pass
        if not self.addresses:
            raise RuntimeError(
                "No configured inference servers. "
                "Please pass in server addresses by arguments "
                "for `initialize` or environment "
                "variable `ASTRAFLOW_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")
        self.executor = ProcessPoolExecutor(max_workers=1)

        # Initialize per-server stats for discovered servers
        for addr_ in self.addresses:
            self._inflight_per_server.setdefault(addr_, 0)
            self._completed_per_server.setdefault(addr_, 0)

    def destroy(self):
        """Destroy the engine and clean up resources."""
        if self._executor is not None:
            self._executor.shutdown()
        if len(self.local_server_processes) > 0:
            self.teardown_server()

    @property
    def executor(self) -> ProcessPoolExecutor:
        """Get the process pool executor of the inference engine."""
        if self._executor is None:
            raise RuntimeError("Executor is not initialized")
        return self._executor

    @executor.setter
    def executor(self, executor: ProcessPoolExecutor):
        """Set the process pool executor of the inference engine."""
        self._executor = executor

    def set_generation_paused(self, paused: bool) -> None:
        """Set the paused flag for in-flight agenerate() calls."""
        self._paused = paused

    def get_generation_stats(self) -> dict[str, dict[str, int]]:
        """Return per-server generation stats (inflight and completed counts)."""
        stats = {}
        for addr in self.addresses:
            stats[addr] = {
                "inflight": self._inflight_per_server.get(addr, 0),
                "completed": self._completed_per_server.get(addr, 0),
            }
        return stats

    def set_version(self, version):
        """Set the current weight version."""
        with self.lock:
            self._version = version

    def get_version(self):
        """Get the current weight version."""
        with self.lock:
            return self._version

    def choose_server(self) -> str:
        """Choose a server based on the scheduling policy.

        Returns
        -------
        str
            Selected server address

        Raises
        ------
        NotImplementedError
            If schedule policy other than round-robin is used
        """
        if self.config.schedule_policy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        logger.debug(
            f"agenerate() ENTRY: rid={req.rid}, "
            f"input_len={len(req.input_ids)}, "
            f"max_new_tokens={req.gconfig.max_new_tokens}, "
            f"n_samples={req.gconfig.n_samples}"
        )

        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Inference engines do not support n_samples > 1. "
                "Please call generate multiple times with n_samples = 1."
            )

        # Validate max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        # Update max_new_tokens in request
        req.gconfig.max_new_tokens = max_new_tokens

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []

        # A single "rid" shares the same server to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Track in-flight requests per server. Increment before the
        # try block so the finally-guarded decrement always matches.
        self._inflight_per_server.setdefault(server_addr, 0)
        self._completed_per_server.setdefault(server_addr, 0)
        self._inflight_per_server[server_addr] += 1

        # Get or create task-local session
        session_cleanup = False
        try:
            session = _session_storage.get()
        except LookupError:
            session = self._create_session()
            session_cleanup = True

        try:
            # Deal with rollout interruption
            stop_reason = None
            iteration = 0
            while (
                stop_reason not in ["stop", "tool_calls", "length"]
                and len(accumulated_output_tokens) < gconfig.max_new_tokens
            ):
                iteration += 1
                logger.debug(
                    f"agenerate() iteration {iteration}, rid={req.rid}, "
                    f"accumulated_tokens={len(accumulated_output_tokens)}, "
                    f"max_new_tokens={gconfig.max_new_tokens}"
                )

                # Request is interrupted, wait for some time to avoid interfering
                # with update weights requests
                while self._paused:
                    await asyncio.sleep(0.5)

                # Build request using backend
                logger.debug(
                    f"agenerate() building HTTP request, rid={req.rid}, "
                    f"iteration={iteration}, server_addr={server_addr}"
                )
                http_req = self.backend.build_generation_request(req, self._current_lora_name)

                # Loop until the generation is complete
                logger.debug(
                    f"agenerate() calling arequest_with_retry, rid={req.rid}, "
                    f"iteration={iteration}, endpoint={http_req.endpoint}"
                )
                result = await arequest_with_retry(
                    session=session,
                    addr=server_addr,
                    endpoint=http_req.endpoint,
                    payload=http_req.payload,
                    method=http_req.method,
                    max_retries=self.config.request_retries,
                    timeout=self.config.request_timeout,
                )
                logger.debug(
                    f"agenerate() received HTTP response, rid={req.rid}, "
                    f"iteration={iteration}, response_size={len(str(result))}"
                )

                # Parse response using backend
                gen_result = self.backend.parse_generation_response(result)
                stop_reason = gen_result.stop_reason

                # Update accumulated outputs
                accumulated_output_tokens.extend(gen_result.output_tokens)
                accumulated_output_logprobs.extend(gen_result.output_logprobs)
                accumulated_versions.extend(
                    [self.get_version()] * len(gen_result.output_tokens)
                )

                # Update request for next iteration
                req.input_ids += gen_result.output_tokens
                req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
                assert req.gconfig.max_new_tokens >= 0, (
                    req.gconfig.max_new_tokens,
                    len(gen_result.output_tokens),
                    len(req.input_ids),
                )

            # Successful path — bump completed counter.
            self._completed_per_server[server_addr] += 1

            # Final abort handling
            if stop_reason == "abort":
                # If stop_reason is "abort", the only reason we exit the loop is
                # len(accumulated_output_tokens) >= gconfig.max_new_tokens
                # so the actual reason is length
                stop_reason = "length"

            latency = time.perf_counter() - start_time

            response = ModelResponse(
                input_tokens=req.input_ids[
                    : len(req.input_ids) - len(accumulated_output_tokens)
                ],
                input_images=req.image_data,
                output_tokens=accumulated_output_tokens,
                output_logprobs=accumulated_output_logprobs,
                output_versions=accumulated_versions,
                stop_reason=stop_reason,
                latency=latency,
                ttft=latency,  # Simplified for non-streaming
                tokenizer=req.tokenizer,
                processor=req.processor,
            )
            return response
        finally:
            # Always decrement on exit — including CancelledError raised
            # mid-while-loop by reset_training_engine.  Without this, every
            # cancelled rollout permanently leaks an inflight count and
            # skews load-balancing on subsequent rounds.
            self._inflight_per_server[server_addr] -= 1
            if session_cleanup:
                await session.close()

    def load_weights_from_path(
        self, path: str, use_lora: bool = False
    ) -> None:
        """Synchronously load weights from ``path`` on all inference servers.

        Used by the TCP weight transfer path after saving weights to
        ``/dev/shm`` as safetensors.

        Caller must call ``pause_generation`` (abort mode) before this
        method so that all inflight requests are drained.

        For full weights: ``/update_weights_from_disk`` includes
        ``abort_all_requests: True`` and ``flush_cache`` internally.

        For LoRA adapters (``use_lora=True``): loads the new adapter under a
        fresh versioned name (``lora_v{seq}``) without explicitly unloading the
        previous one, then flushes the KV cache. SGLang's registry LRU evicts
        old versions once ``max_loaded_loras`` is reached and its mem-pool LRU
        reclaims GPU slots (bounded by ``max_loras_per_batch``); an evicted
        adapter is transparently re-loaded on next use.

        Historically, explicitly unloading an adapter that still had
        paused/aborted in-flight requests deadlocked SGLang's ``wait_for_unload``
        because the adapter's usage counter was never released on abort. That
        leak is now fixed at the source by ``LoRACounterLeakPatch``
        (``astraflow/raas/patch/sglang.py``), so unload/eviction is safe. We keep
        the fresh-name scheme because it stays correct without draining under
        ``lora_update_lock`` on every sync.
        """
        import time as _time

        _t0 = _time.monotonic()

        if use_lora:
            # Load under a NEW versioned name and do NOT explicitly unload the
            # old one. The abort-time usage-counter leak that used to make
            # ``wait_for_unload`` (and thus registry-LRU eviction) hang is fixed
            # by LoRACounterLeakPatch, so eviction is safe; the fresh name also
            # avoids draining under ``lora_update_lock`` on every sync.
            self._lora_seq += 1
            lora_name = f"lora_v{self._lora_seq}"
            logger.info(
                "load_weights_from_path: sending /load_lora_adapter "
                "to %d servers (path=%s, lora_name=%s) ...",
                len(self.addresses),
                path,
                lora_name,
            )
            try:
                load_req = HttpRequest(
                    endpoint="/load_lora_adapter",
                    payload={"lora_name": lora_name, "lora_path": str(path)},
                )
                self._run_request_on_all_servers(load_req)
                self.lora_initialized = True
                self._current_lora_name = lora_name

                # Flush stale KV cache entries computed with old LoRA weights.
                # Safe because caller already paused generation (is_pause=True
                # blocks new requests) and all inflight requests were drained.
                flush_req = HttpRequest(
                    endpoint="/flush_cache",
                    payload={},
                )
                self._run_request_on_all_servers(flush_req)
            except Exception:
                logger.error(
                    "RemoteInfEngine.load_weights_from_path (LoRA) failed "
                    "(path=%s)",
                    path,
                    exc_info=True,
                )
                raise
        else:
            logger.info(
                "load_weights_from_path: sending /update_weights_from_disk "
                "to %d servers (path=%s) ...",
                len(self.addresses),
                path,
            )
            http_req = HttpRequest(
                endpoint="/update_weights_from_disk",
                payload={"model_path": str(path), "abort_all_requests": True},
            )
            try:
                self._run_request_on_all_servers(http_req)
            except Exception:
                logger.error(
                    "RemoteInfEngine.load_weights_from_path failed (path=%s)",
                    path,
                    exc_info=True,
                )
                raise

        logger.info(
            "load_weights_from_path: done in %.2fs (path=%s, use_lora=%s)",
            _time.monotonic() - _t0,
            path,
            use_lora,
        )

    @trace_perf("remote_inf_engine.pause_generation", category="misc")
    def pause_generation(self):
        """Pause request submission for async rollout."""
        print(
            f"[RaaS3 engine] pause_generation: sending to {len(self.addresses)} servers ...",
            flush=True,
        )
        pause_req = self.backend.get_pause_request()
        self._run_request_on_all_servers(pause_req)
        print(
            f"[RaaS3 engine] pause_generation: done, sleeping {self.config.pause_grace_period}s ...",
            flush=True,
        )

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)
        print("[RaaS3 engine] pause_generation: grace period done", flush=True)

    @trace_perf("remote_inf_engine.continue_generation", category="misc")
    def continue_generation(self):
        """Resume request submission for async rollout."""
        resume_req = self.backend.get_resume_request()
        self._run_request_on_all_servers(resume_req)

    def offload(self) -> None:
        """Offload model memory on all servers."""
        offload_req = self.backend.get_offload_request()
        self._run_request_on_all_servers(offload_req)

    def onload(self, tags: list[str] | None = None) -> None:
        """Onload model memory on all servers."""
        onload_req = self.backend.get_onload_request(tags=tags)
        self._run_request_on_all_servers(onload_req)

    async def aget_metrics(
        self, total_timeout: float = 1.0
    ) -> list[dict[str, float] | None]:
        """Query ``/metrics`` (Prometheus) on every sglang address in parallel.

        Pure async: must be called from an already-running event loop (e.g.
        the FastAPI request handler loop inside ``Manager.get_availability``).
        Do **not** reuse ``_run_request_on_all_servers`` — that helper wraps
        ``uvloop.run(...)`` which cannot be nested in an existing loop.

        Returns a list parallel to ``self.addresses``. Each entry is either a
        dict ``{metric_name: float}`` of the metrics we care about (parsed
        from Prometheus text), or ``None`` if the request to that server
        failed. The dict contains at most these keys:

            - ``sglang:num_queue_reqs``   (waiting queue depth)
            - ``sglang:num_running_reqs`` (running batch size)
            - ``sglang:token_usage``      (KV cache fraction, 0.0-1.0)

        Uses a plain ``session.get`` rather than ``arequest_with_retry`` since
        ``/metrics`` returns text (not JSON) and retries on a read-only
        telemetry probe are not useful — on failure we fall back in the
        caller.
        """
        if not self.addresses:
            return []
        # Raises AttributeError for backends without Prometheus support
        # (e.g. vLLM); the manager catches and marks the engine as None.
        req = self.backend.get_metrics_request()
        wanted = _METRICS_WANTED

        async def _one(session, addr):
            url = f"http://{addr}{req.endpoint}"
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "aget_metrics: %s returned status %d",
                            addr, resp.status,
                        )
                        return None
                    text = await resp.text()
            except Exception as exc:
                # Use repr() because many aiohttp/asyncio exceptions
                # have empty str() (e.g. CancelledError, TimeoutError).
                logger.warning(
                    "aget_metrics: failed on %s: %s",
                    addr, repr(exc),
                )
                return None
            return _parse_prometheus_metrics(text, wanted)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=total_timeout),
            connector=get_default_connector(),
        ) as session:
            return await asyncio.gather(
                *[_one(session, addr) for addr in self.addresses]
            )

    def get_metrics_sync(
        self, total_timeout: float = 1.0
    ) -> list[dict[str, float] | None]:
        """Synchronous version of ``aget_metrics`` for use in a thread.

        Uses ``requests.get`` (blocking I/O that releases the GIL) so the
        call is not affected by event-loop congestion.  Intended to be
        called via ``asyncio.to_thread`` or ``run_in_executor``.
        """
        if not self.addresses:
            return []
        req = self.backend.get_metrics_request()
        wanted = _METRICS_WANTED

        def _one(addr: str) -> dict[str, float] | None:
            url = f"http://{addr}{req.endpoint}"
            try:
                resp = requests.get(url, timeout=total_timeout)
                if resp.status_code != 200:
                    logger.warning(
                        "get_metrics_sync: %s returned status %d",
                        addr, resp.status_code,
                    )
                    return None
                return _parse_prometheus_metrics(resp.text, wanted)
            except Exception as exc:
                logger.warning(
                    "get_metrics_sync: failed on %s: %s", addr, repr(exc),
                )
                return None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.addresses)) as pool:
            return list(pool.map(_one, self.addresses))

    def _run_request_on_all_servers(self, req: HttpRequest):
        async def _per_server(session, addr, idx):
            import time as _time

            _t0 = _time.monotonic()
            logger.info(
                "[server %d/%d] %s %s starting ...",
                idx + 1, len(self.addresses), req.endpoint, addr,
            )
            try:
                result = await arequest_with_retry(
                    session=session,
                    addr=addr,
                    endpoint=req.endpoint,
                    payload=req.payload,
                    method=req.method,
                    max_retries=self.config.request_retries,
                    timeout=self.config.request_timeout,
                )
                _elapsed = _time.monotonic() - _t0
                logger.info(
                    "[server %d/%d] %s %s done in %.2fs",
                    idx + 1, len(self.addresses), req.endpoint, addr, _elapsed,
                )
                return result
            except Exception as exc:
                _elapsed = _time.monotonic() - _t0
                logger.error(
                    "[server %d/%d] %s %s failed after %.2fs: %s",
                    idx + 1, len(self.addresses), req.endpoint, addr,
                    _elapsed, exc,
                )
                raise

        async def _fn():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                jobs = [
                    _per_server(session, addr, i)
                    for i, addr in enumerate(self.addresses)
                ]
                await asyncio.gather(*jobs)

        uvloop.run(_fn())

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server."""
        if "host" not in server_args:
            server_args["host"] = gethostip()
        if "port" not in server_args:
            server_args["port"] = find_free_ports(1)[0]
        process = self.backend.launch_server(server_args)
        address = f"{server_args['host']}:{server_args['port']}"
        server_info = LocalInfServerInfo(
            host=server_args["host"],
            port=server_args["port"],
            process=process,
        )
        try:
            self._wait_for_server(address)
            self.local_server_processes.append(server_info)
            # Keep self.addresses in sync: add new address if not present.
            # initialize() sets addresses from local_server_processes, but
            # launch_server() may be called after initialize() for late-added
            # servers (e.g. on RaaS re-bootstrap).
            if address not in self.addresses:
                self.addresses.append(address)
                logger.info(
                    "launch_server: added %s to engine addresses (total=%d)",
                    address,
                    len(self.addresses),
                )
            return server_info
        except TimeoutError:
            logger.warning(
                f"Launch local server timeouted at {address} after {self.config.setup_timeout}s."
            )
            self._shutdown_one_server(server_info)
            raise

    def _shutdown_one_server(self, server_info: LocalInfServerInfo):
        addr = f"{server_info.host}:{server_info.port}"
        if addr in self.addresses:
            self.addresses.remove(addr)
        if server_info.process.poll() is not None:
            return
        kill_process_tree(server_info.process.pid, graceful=True)

    def teardown_server(self):
        """Teardown all locally launched servers."""
        for server_info in self.local_server_processes:
            self._shutdown_one_server(server_info)
        self.local_server_processes.clear()
