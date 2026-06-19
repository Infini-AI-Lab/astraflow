import os
import shlex
import subprocess
import sys
import uuid
from typing import Any

from astraflow.raas.api.cli_args import InferenceEngineConfig, vLLMConfig
from astraflow.raas.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    ModelRequest,
)
from astraflow.raas.engine import RemoteInfEngine
from astraflow.raas.utils import logging, stats_tracker
from astraflow.raas.utils.launcher import TRITON_CACHE_PATH

logger = logging.getLogger(__name__)


class VLLMBackend:
    """Backend that translates engine operations into vLLM HTTP API calls.

    Supports two operating modes selected at launch time:
    - TCP weight transfer mode: uses native vLLM APIs (requires VLLM_SERVER_DEV_MODE=1).
      Weight reload via /collective_rpc, pause/resume via /pause and /resume.
    - NCCL mode (removed): previously used custom vllm_server endpoints for weight updates.
    """

    def __init__(self):
        pass

    def build_generation_request(
        self, req: ModelRequest, lora_name: str | None
    ) -> HttpRequest:
        """Convert a ModelRequest into a vLLM completions or chat HTTP request.

        ``lora_name`` is a truthy marker that a LoRA is active; vLLM selects
        the adapter via ``gconfig.lora_name`` (its own naming), so the marker's
        value is unused here.
        """
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        payload = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
            "use_beam_search": gconfig.use_beam_search,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        if lora_name and len(gconfig.lora_name) > 0:
            payload["model"] = gconfig.lora_name

        if req.vision_msg_vllm:
            images = iter(req.image_data)
            parsed_input = req.vision_msg_vllm[0]
            for msg in parsed_input:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content.get("type") == "image_url":
                            try:
                                base64_img = next(images)
                            except StopIteration:
                                raise ValueError(
                                    "Not enough images in req.image_data to match image_url entries."
                                )
                            content["image_url"] = {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
            payload["messages"] = parsed_input.copy()
            payload["logprobs"] = True
            return HttpRequest(endpoint="/v1/chat/completions", payload=payload)
        else:
            payload["prompt"] = req.input_ids.copy()
            return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        """Extract tokens, logprobs, and stop reason from a vLLM response."""
        meta_info = response["choices"][0]
        stop_reason = meta_info["finish_reason"]

        if "tokens" in meta_info["logprobs"]:
            output_tokens = meta_info["logprobs"]["tokens"]
            output_tokens = [int(t.split(":")[1]) for t in output_tokens]
            output_logprobs = meta_info["logprobs"]["token_logprobs"]
        elif "content" in meta_info["logprobs"]:
            outputs = meta_info["logprobs"]["content"]
            output_tokens = [int(t["token"].split(":")[1]) for t in outputs]
            output_logprobs = [t["logprob"] for t in outputs]
        else:
            raise ValueError("Unexpected vLLM response format.")

        if stop_reason == "abort" and len(output_tokens) == 0:
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
            )
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

    def get_offload_request(self) -> HttpRequest:
        """Build request to offload model from GPU to CPU via /sleep."""
        return HttpRequest(endpoint="/sleep", payload={}, method="POST")

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        """Build request to reload model from CPU to GPU via /wake_up."""
        if tags is not None:
            tags_query = "&".join([f"tags={tag}" for tag in tags])
            endpoint = f"/wake_up?{tags_query}"
        else:
            endpoint = "/wake_up"
        return HttpRequest(endpoint=endpoint, payload={}, method="POST")

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        """Spawn a vLLM server subprocess and return its Popen handle."""
        launch_env = server_args.pop("__launch_env__", None)

        server_args.pop("rollout_manager_address", None)

        if launch_env:
            launch_env.pop("ASTRAFLOW_AUTOPATCH", None)

        cmd = vLLMConfig.build_cmd_from_args(server_args)

        logger.info(
            "Launching vLLM server command (tcp_mode=%s): %s",
            tcp_mode,
            shlex.join([str(c) for c in cmd]),
        )

        _env = os.environ.copy()
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))

        vllm_cache_path = _env.get("VLLM_CACHE_ROOT")
        if vllm_cache_path:
            _env["VLLM_CACHE_ROOT"] = os.path.join(vllm_cache_path, str(uuid.uuid4()))

        if launch_env:
            _env.update({str(k): str(v) for k, v in launch_env.items()})

        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


class VLLMEngine:
    """Inference engine backed by remote vLLM servers.

    All methods are delegated to the underlying RemoteInfEngine via __getattr__.
    """

    def __init__(self, config: InferenceEngineConfig):
        self.config = config
        self._engine = RemoteInfEngine(config, VLLMBackend())
        self._engine.lora_initialized = config.use_lora
        # vLLM selects the adapter via gconfig.lora_name; this just marks LoRA
        # active so the shared generation-request builder passes a truthy flag.
        if config.use_lora:
            self._engine._current_lora_name = "vllm_lora"

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def export_stats(self) -> dict[str, float]:
        """Export workflow execution statistics without distributed reduction."""
        return stats_tracker.export_all(reduce_group=None)
