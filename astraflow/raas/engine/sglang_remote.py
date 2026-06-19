import os
import shlex
import subprocess
import sys
import uuid
from typing import Any

from astraflow.raas.api.cli_args import InferenceEngineConfig, SGLangConfig
from astraflow.raas.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    ModelRequest,
)
from astraflow.raas.engine import RemoteInfEngine
from astraflow.raas.utils import logging, stats_tracker
from astraflow.raas.utils.launcher import TRITON_CACHE_PATH

logger = logging.getLogger(__name__)


class SGLangBackend:
    """Backend that translates engine operations into SGLang HTTP API calls."""

    def build_generation_request(
        self, req: ModelRequest, lora_name: str | None
    ) -> HttpRequest:
        """Convert a ModelRequest into an SGLang /generate HTTP request.

        ``lora_name`` is the currently-active versioned adapter name (e.g.
        ``lora_v3``) or ``None`` when no adapter is loaded.
        """
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Currently Beam search is not supported in SGLang backend."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop

        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        if lora_name:
            payload["lora_path"] = lora_name

        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Extract tokens, logprobs, and stop reason from an SGLang response."""
        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")
        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
            )

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
        )

    def get_pause_request(self) -> HttpRequest:
        """Build request to pause generation on the server."""
        return HttpRequest(endpoint="/pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        """Build request to resume generation on the server."""
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_health_check_request(self) -> HttpRequest:
        """Build request to check server health."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_metrics_request(self) -> HttpRequest:
        """Build request to fetch Prometheus metrics from sglang.

        sglang serves ``/metrics`` directly from its multiprocess Prometheus
        registry, which is backed by mmap files in
        ``$PROMETHEUS_MULTIPROC_DIR``. The HTTP handler reads those files
        in-process without any ZMQ round-trip to the scheduler, so this
        endpoint stays responsive even when the scheduler is saturated.

        Replaces the old ``/get_load`` path which required a scheduler RPC
        and would time out under heavy generation load. Requires sglang to
        be launched with ``--enable-metrics`` (the default in this repo via
        ``SGLangConfig.enable_metrics=True``).
        """
        return HttpRequest(endpoint="/metrics", payload={}, method="GET")

    def get_offload_request(self) -> HttpRequest:
        """Build request to offload model from GPU to CPU."""
        return HttpRequest(endpoint="/release_memory_occupation", payload={})

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Build request to reload model from CPU to GPU."""
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Spawn an SGLang server subprocess and return its Popen handle."""
        launch_env = server_args.pop("__launch_env__", None)
        autopatch = (launch_env or {}).get("ASTRAFLOW_AUTOPATCH", "false").lower() in ("true", "1")
        if autopatch:
            from astraflow.raas.api.cli_args import get_py_cmd
            cmd = get_py_cmd("astraflow.raas.entrypoint", server_args)
        else:
            cmd = SGLangConfig.build_cmd_from_args(server_args)
        logger.info(
            "Launching SGLang server command: %s", shlex.join([str(c) for c in cmd])
        )

        _env = os.environ.copy()
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))
        if launch_env:
            _env.update({str(k): str(v) for k, v in launch_env.items()})

        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


class SGLangEngine:
    """Inference engine backed by remote SGLang servers.

    All methods are delegated to the underlying RemoteInfEngine via __getattr__.
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._engine = RemoteInfEngine(config, SGLangBackend())

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def export_stats(self) -> dict[str, float]:
        """Export workflow execution statistics without distributed reduction."""
        return stats_tracker.export_all(reduce_group=None)
