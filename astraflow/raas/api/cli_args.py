import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import uvloop
from omegaconf import DictConfig, OmegaConf

from astraflow.raas.utils import logging, name_resolve, pkg_version
from astraflow.raas.utils.pkg_version import is_version_less

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

uvloop.install()

logger = logging.getLogger("CLI args")

ConfigT = TypeVar("ConfigT")


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for rollout."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    max_tokens: int = field(
        default=65536,
        metadata={
            "help": "Maximum number of tokens including prompt and generated tokens."
        },
    )
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding (max probability)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )
    top_k: int = field(
        default=int(1e8),
        metadata={"help": "Number of highest probability tokens to consider."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )
    stop_token_ids: list[int] = field(
        default_factory=list,
        metadata={"help": "Stop generation when encountering these token IDs."},
    )
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "Do not stop generation when EOS is encountered."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Skip special tokens when decoding/displaying outputs."},
    )
    include_pad_in_stop_tokens: bool = field(
        default=True,
        metadata={
            "help": "Whether to include PAD token in stop_token_ids (EOS is always included)."
        },
    )
    stop: list[str] | None = field(
        default=None,
        metadata={
            "help": "One or multiple stop words. Generation will stop if one of these words is sampled."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Penalizes tokens based on their frequency in generation so far. "
                "Must be between -2 and 2 where negative numbers encourage repetition."
            )
        },
    )
    lora_name: str = field(
        default="",
        metadata={"help": "Lora name to be used for this generation."},
    )
    use_beam_search: bool = field(
        default=False,
        metadata={
            "help": "Enable beam search in the vLLM engine. When enabled, sampling parameters like temperature, top-p, and top-k are auto ignored."
        },
    )
    # NOTE: to add new parameters, please correctly handle them in the `to_openai_args_dict` method.

    def new(self, **kwargs):
        args = asdict(self)
        args.update(kwargs)
        return GenerationHyperparameters(**args)

    def new_with_stop_and_pad_token_ids(self, tokenizer: "PreTrainedTokenizerFast"):
        """Create a new generation hyperparameters with stop and pad token ids added."""
        new_stop_token_ids = self.stop_token_ids.copy()
        if (
            self.include_pad_in_stop_tokens
            and tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id not in new_stop_token_ids
        ):
            new_stop_token_ids.append(tokenizer.pad_token_id)
            print(f"Appended pad token id: {tokenizer.pad_token_id} to stop_token_ids")
        if (
            tokenizer.eos_token_id is not None
            and tokenizer.eos_token_id not in new_stop_token_ids
        ):
            new_stop_token_ids.append(tokenizer.eos_token_id)
            print(f"Appended eos token id: {tokenizer.eos_token_id} to stop_token_ids")
        return self.new(stop_token_ids=new_stop_token_ids)

    def to_openai_completions_args_dict(
        self, exclude_args: list[str] | None = None
    ) -> dict[str, Any]:
        return self.to_openai_args_dict(
            exclude_args=exclude_args, api_format="completions"
        )

    def to_openai_responses_args_dict(
        self, exclude_args: list[str] | None = None
    ) -> dict[str, Any]:
        return self.to_openai_args_dict(
            exclude_args=exclude_args, api_format="responses"
        )

    def to_openai_agents_model_settings_dict(
        self, exclude_args: list[str] | None = None
    ) -> dict[str, Any]:
        return self.to_openai_args_dict(
            exclude_args=exclude_args, api_format="openai-agents"
        )

    _OPENAI_UNSUPPORTED_ARGS: ClassVar[set[str]] = {
        "min_new_tokens",  # Not supported by OpenAI
        "greedy",  # Not directly supported by OpenAI
        "top_k",  # Not supported by OpenAI
        "stop_token_ids",  # Not supported by OpenAI
        "ignore_eos",  # Not supported by OpenAI
        "skip_special_tokens",  # Not supported by OpenAI
        "lora_name",  # Not supported by OpenAI
        "use_beam_search",  # Not supported by OpenAI
        "max_tokens",  # deprecated by "completions", not used in "responses", should be `max_new_tokens` in "openai-agents"
    }

    def to_openai_args_dict(
        self, exclude_args: list[str] | None = None, api_format: str = "completions"
    ) -> dict[str, Any]:
        """Convert the generation hyperparameters to a dictionary of arguments for OpenAI client."""
        from dataclasses import MISSING as dataclass_missing
        from dataclasses import fields

        final_exclude_args = set(exclude_args) if exclude_args is not None else set()
        final_exclude_args.update(self._OPENAI_UNSUPPORTED_ARGS)
        # TODO: move the excluded args into extra body, so they can be passed through the client request

        mapping = {"n_samples": "n"}
        if api_format == "completions":
            mapping["max_new_tokens"] = "max_completion_tokens"
        elif api_format == "responses":
            mapping["max_new_tokens"] = "max_output_tokens"
        elif api_format == "openai-agents":
            # NOTE: max_tokens in openai-agents means `max_new_tokens` in sglang/vllm. This is not a bug
            mapping["max_new_tokens"] = "max_tokens"
        else:
            raise ValueError(f"Unsupported API format: {api_format}")

        res = {}
        for k, v in asdict(self).items():
            if k in final_exclude_args:
                should_warn = False

                current_value = getattr(self, k)
                f = next(_field for _field in fields(self) if _field.name == k)

                # Check if equal to the default value
                if f.default is not dataclass_missing:
                    if current_value != f.default:
                        should_warn = True
                elif f.default_factory is not dataclass_missing:
                    if current_value != f.default_factory():
                        should_warn = True

                if should_warn:
                    logger.warning(
                        f"Unsupported arg for openai format: `{k}` with value {current_value}"
                    )
                continue
            key = mapping.get(k, k)
            if key in res:
                logger.warning(f"Overriding key: {key} from {k} with value: {v}")
            res[key] = v

        return res


def get_py_cmd(module: str, args: dict[str, Any]):
    # convert to flags
    cmd = ["python3", "-m", module]
    for k, v in args.items():
        if v is None or v is False or v == "" or (isinstance(v, list) and not v):
            continue
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cmd.append(flag)
        elif isinstance(v, list):
            cmd.append(flag)
            cmd.extend(map(str, v))
        else:
            cmd.append(flag)
            cmd.append(str(v))
    return cmd


@dataclass
class vLLMConfig:
    """Configuration for vLLM runtime. Refer to:
    https://docs.vllm.ai/en/stable/api/index.html for detailed documentation.
    """

    model: str = ""
    seed: int = 1
    skip_tokenizer_init: bool = False
    enforce_eager: bool = False
    dtype: str = "bfloat16"
    distributed_executor_backend: str = "mp"
    # original
    max_num_seqs: int = 256
    # kv_cache_type: str = "auto"
    block_size: int = 16
    swap_space: int = 4
    cpu_offload_gb: float = 0
    disable_sliding_window: bool = True
    # NOTE: Defaults max_model_len to 32k because a larger value
    # will enable chunked prefill in vLLM, which will cause
    # evalution performance degeneration.
    max_model_len: int | None = 32768
    enable_chunked_prefill: bool = False
    # NOTE: Setting enable_prefix_caching to False
    # because it will reuse the block after
    # model weights are updated. Using v0.7.2 reset_prefix_cache
    # will fix this issue.
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.9
    worker_extension_cls: str = (
        "astraflow.train_worker.launcher.vllm.vllm_worker_extension.VLLMWorkerExtension"
    )
    enable_sleep_mode: bool = False
    uvicorn_log_level: str = "warning"
    enable_lora: bool = False
    lora_modules: str = ""

    @staticmethod
    def build_args(
        vllm_config: "vLLMConfig",
        tp_size: int,
        pp_size: int = 1,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
    ):
        args: dict = conf_as_dict(vllm_config)
        args = dict(
            # Model and tokenizer
            tokenizer=vllm_config.model,
            load_format="auto",
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            **args,
        )
        if port is not None:
            args["port"] = port
        if host is not None:
            args["host"] = host
        # handle lora modules separately
        lm = args.get("lora_modules")
        if lm:
            if isinstance(lm, str):
                lm = [lm]
            if isinstance(lm, (list, tuple)):
                try:
                    args["lora_modules"] = [
                        json.dumps(json.loads(s), separators=(",", ":")) for s in lm
                    ]
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string in lora_modules: {e}") from e
        return args

    @staticmethod
    def build_cmd_from_args(args: dict[str, Any]):
        return get_py_cmd("astraflow.train_worker.launcher.vllm.vllm_server", args)

    @staticmethod
    def build_cmd(
        vllm_config: "vLLMConfig",
        tp_size: int,
        pp_size: int,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
    ):
        args = vLLMConfig.build_args(
            vllm_config=vllm_config,
            tp_size=tp_size,
            pp_size=pp_size,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
        )
        return vLLMConfig.build_cmd_from_args(args)


@dataclass
class SGLangConfig:
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    model_path: str = ""
    random_seed: int = 1
    skip_tokenizer_init: bool = False
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = True
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: int | None = None
    cuda_graph_bs: list[int] | None = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    attention_backend: str | None = "fa3"
    enable_multimodal: bool = False
    sampling_backend: str | None = None
    context_length: int | None = 32768
    mem_fraction_static: float | None = 0.9
    max_running_requests: int | None = None
    # NOTE: chunked_prefill_size is by default 8192 on GPUs with 80GB mem in SGLang,
    # but we disable it to avoid precision issues
    chunked_prefill_size: int | None = -1
    max_prefill_tokens: int = 32768
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    dp_size: int = 1  # only used for dp attention
    ep_size: int = 1
    # lora
    enable_lora: bool | None = None
    max_lora_rank: int | None = None
    lora_target_modules: list[str] | None = None
    lora_paths: list[str] | None = None
    max_loaded_loras: int = 1
    max_loras_per_batch: int = 1
    lora_backend: str = "triton"
    # logging
    log_level: str = "warning"
    log_level_http: str | None = "warning"
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = True  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: int = 1
    # Extra loader arguments
    # NOTE: These arguments will be parsed into a dict json-string
    # and passed as `model_loader_extra_config` to SGLang.
    enable_multithread_load: bool = False
    enable_fast_load: bool = False

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host: str | None = None,
        port: int | None = None,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args = SGLangConfig.build_args(
            sglang_config=sglang_config,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
            n_nodes=n_nodes,
            node_rank=node_rank,
        )

        return SGLangConfig.build_cmd_from_args(args)

    @staticmethod
    def build_cmd_from_args(args: dict[str, Any]):
        return get_py_cmd("sglang.launch_server", args)

    @staticmethod
    def build_args(
        sglang_config: "SGLangConfig",
        tp_size: int,
        base_gpu_id: int,
        host: str | None = None,
        port: str | None = None,
        dist_init_addr: str | None = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        # Map "all-linear" to "all"
        args: dict = conf_as_dict(sglang_config)
        if sglang_config.enable_multithread_load or sglang_config.enable_fast_load:
            if not pkg_version.is_version_equal("sglang", "0.5.2"):
                raise RuntimeError(
                    "Customized model loading requires exact SGLang version 0.5.2"
                )
            model_loader_extra_config = dict(
                enable_multithread_load=sglang_config.enable_multithread_load,
                enable_fast_load=sglang_config.enable_fast_load,
            )
            args["model_loader_extra_config"] = json.dumps(
                model_loader_extra_config, separators=(",", ":")
            )
        args.pop("enable_multithread_load", None)
        args.pop("enable_fast_load", None)
        # Map "all-linear" to "all"
        if "lora_target_modules" in args and args["lora_target_modules"]:
            args["lora_target_modules"] = [
                x.replace("-linear", "") for x in args["lora_target_modules"]
            ]
        from astraflow.raas.platforms import current_platform

        args = dict(
            # Model and tokenizer
            tokenizer_path=sglang_config.model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            device=current_platform.device_type,
            is_embedding=False,
            # Other runtime options
            tp_size=tp_size,
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=base_gpu_id,
            nnodes=n_nodes,
            node_rank=node_rank,
            # initialization addresses and ports
            dist_init_addr=dist_init_addr,
            **args,
        )
        if host is not None:
            args["host"] = host
        if port is not None:
            args["port"] = port
        if not pkg_version.is_version_greater_or_equal("sglang", "0.4.9.post2"):
            raise RuntimeError("Needs sglang>=0.4.9.post2 to run the code.")
        if is_version_less("sglang", "0.4.10.post2"):
            args.pop("max_loaded_loras", None)
        return args


# Scheduling (used by InferenceEngineConfig)


@dataclass
class SchedulingStrategy:
    type: str = field(
        default="separation", metadata={"choices": ["separation", "colocation"]}
    )
    target: str | None = field(
        default=None, metadata={"help": "The target role to be colocated with"}
    )


@dataclass
class InferenceEngineConfig:
    """Configuration for inference servers, including offpolicyness control."""

    experiment_name: str | None = None
    trial_name: str | None = None
    max_concurrent_rollouts: None | int = field(
        default=None,
        metadata={
            "help": "Maximum number of concurrent rollouts to "
            "the inference engine. Defaults to consumer_batch_size."
        },
    )
    max_concurrent_evals: int = field(
        default=128,
        metadata={
            "help": "Cap on concurrent eval submissions to the inference engine. "
            "Independent from rollout concurrency. OOM safety mainly depends on "
            "sglang.max_prefill_tokens; this knob controls eval throughput vs "
            "queueing fairness. Lower (e.g. 32) for very large vocab models or "
            "high mem_fraction_static; higher for small-vocab models."
        },
    )
    # ------------------------------------------------------------------
    # Adaptive availability control (sglang /get_load driven)
    # ------------------------------------------------------------------
    enable_adaptive_availability: bool = field(
        default=True,
        metadata={
            "help": "Drive /availability off live sglang /get_load queue depth "
            "instead of only the static semaphore. When off, behavior reverts "
            "to the legacy static semaphore. When on (default), keeps the "
            "max_concurrent_rollouts semaphore as a hard safety ceiling."
        },
    )
    target_waiting_queue_per_dp: int = field(
        default=4,
        metadata={
            "help": "Per-sglang-DP waiting-queue target used by adaptive "
            "availability. Effective threshold = this value x total DP count "
            "across all sglang servers in this RaaS. When sum(num_waiting_reqs) "
            "reaches the effective threshold, RaaS reports available=0 until "
            "the queue drains."
        },
    )
    adaptive_step_size: int = field(
        default=4,
        metadata={
            "help": "Rollouts returned per /availability tick when sglang is "
            "below target. At 10 Hz producer polling, peak submission rate is "
            "step_size * 10 rollouts/sec per RaaS."
        },
    )
    load_cache_ttl_ms: int = field(
        default=100,
        metadata={
            "help": "TTL in ms for cached /get_load responses. Matches "
            "AstraFlow's ~10 Hz producer polling so each tick gets a fresh "
            "sglang snapshot without hammering sglang."
        },
    )
    queue_size: None | int = field(
        default=None,
        metadata={"help": "Input/Output queue size for async rollout."},
    )
    consumer_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for consuming rollouts from the queue."},
    )
    max_head_offpolicyness: int = field(
        default=0,
        metadata={
            "help": "Maximum off-policyness for the head. "
            "If the current version is more than this many versions behind, "
            "the request will not be accepted.",
        },
    )
    enable_rollout_tracing: bool = field(
        default=False,
        metadata={
            "help": "Whether to output verbose tracing messages for each generation request."
        },
    )
    check_trajectory_format: bool = field(
        default=False,
        metadata={
            "help": "Whether to check the format of produced trajectories of a customized workflow. Useful when debugging the workflow in isolation. Should be False during RL training."
        },
    )
    schedule_policy: str = field(
        default="round_robin",
        metadata={"help": "Request scheduling policy", "choices": ["round_robin"]},
    )
    setup_timeout: float = field(
        default=600.0,
        metadata={
            "help": "Timeout in seconds of connecting to remote servers or launching local servers."
        },
    )
    request_timeout: float = field(
        default=3600, metadata={"help": "Timeout for HTTP requests."}
    )
    request_retries: int = field(
        default=3, metadata={"help": "Number of retries for failed requests."}
    )
    pause_grace_period: float = field(
        default=0.0,
        metadata={
            "help": "The grace period after calling /pause_generation. Wait until all requests have been dropped."
        },
    )
    reset_training_on_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, the AstraFlow service wipes every RaaS training "
                "engine before each eval window: cancels in-flight training "
                "rollouts, drains SGLang via /pause_generation, clears task "
                "dicts.  Guarantees eval runs on a quiescent server with "
                "no contention from pre-eval training rollouts.  Costs "
                "~max_concurrency * 0.5 * avg_rollout_time GPU-seconds of "
                "discarded in-flight work per eval round (~0.5% of total "
                "compute at eval_freq=50).  Set to false if running eval "
                "every 1-2 steps, where the wasted fraction rises to 10-30%."
            )
        },
    )
    scheduling_strategy: SchedulingStrategy = field(
        default_factory=SchedulingStrategy,
        metadata={
            "help": "The scheduling strategy of this TrainEngine, either separation or colocation. "
            "Currently only used by the RolloutController."
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA. Should be same as actors LORA option."},
    )


# Perf tracing (used by RaaS engine)


@dataclass
class SessionTracerConfig:
    """Configuration for per-session lifecycle tracing."""

    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable per-session lifecycle tracing alongside perf events. "
                "When true, session metadata is captured to sessions.jsonl."
            )
        },
    )
    flush_threshold: int = field(
        default=256,
        metadata={
            "help": (
                "Flush session trace records once this many entries are ready. "
                "Values <= 0 fall back to 1."
            )
        },
    )


@dataclass
class PerfTracerConfig:
    """Configuration for perf tracer emission."""

    experiment_name: str = ""
    trial_name: str = ""
    fileroot: str = ""
    enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Explicitly enable or disable perf tracing. Set to true to capture perf traces."
            )
        },
    )
    save_interval: int = field(
        default=1,
        metadata={
            "help": (
                "Flush trace events to disk every N calls to save(step=...). "
                "A value of 1 writes on every step; values <= 0 fall back to 1."
            )
        },
    )
    profile_steps: list[int] | None = field(
        default=None,
        metadata={
            "help": (
                "List of step numbers at which to capture detailed profiling traces. "
                "If None, no detailed profiling traces are captured."
            )
        },
    )
    session_tracer: SessionTracerConfig | None = field(
        default=None,
        metadata={"help": "Session tracing configuration."},
    )


# Cluster and name resolution


@dataclass
class NameResolveConfig:
    """Configuration for distributed name resolution and service discovery."""

    type: str = field(
        default="nfs",
        metadata={
            "help": "Type of the distributed KV store for name resolving.",
            "choices": ["nfs", "etcd3", "ray"],
        },
    )
    nfs_record_root: str = field(
        default="/tmp/astraflow/name_resolve",
        metadata={
            "help": "Record root for NFS name resolving. Should be available on all nodes."
        },
    )
    etcd3_addr: str = field(
        default="localhost:2379", metadata={"help": "Address of the ETCD3 server."}
    )
    ray_actor_name: str = field(
        default="ray_kv_store",
        metadata={"help": "Name of the distributed Ray KV store."},
    )


@dataclass
class ClusterSpecConfig:
    """Configuration for cluster specification and distributed computing setup."""

    name_resolve: NameResolveConfig = field(
        default_factory=NameResolveConfig,
        metadata={"help": "Name resolving configuration."},
    )
    cluster_name: str = field(
        default="local",
        metadata={"help": "Name of the cluster. Used to set specific environs."},
    )
    fileroot: str = field(
        default="/tmp/astraflow/",
        metadata={
            "help": "Root for logs and checkpoints. Should be available on all nodes."
        },
    )
    n_nodes: int = field(
        default=32,
        metadata={
            "help": "The size of the cluster. Used to decide slurm hostname suffix."
        },
    )
    n_gpus_per_node: int = field(
        default=8,
        metadata={"help": "Number of GPUs per node (physical)."},
    )


# Per-model spec (multi-model RaaS)


@dataclass
class ModelSpec:
    """Per-model configuration within a multi-model RaaS.

    Each entry in ``RaaSConfig.models`` describes one inference engine
    that will be launched and managed by the RaaS process.
    """

    backend: str = field(
        default="sglang",
        metadata={
            "help": "Inference backend for this model.",
            "choices": ["sglang", "vllm"],
        },
    )
    model_path: str = field(
        default="", metadata={"help": "HF model name or local path."}
    )
    tokenizer_path: str = field(
        default="",
        metadata={"help": "Path to tokenizer. Defaults to model_path if empty."},
    )
    sglang: SGLangConfig = field(
        default_factory=SGLangConfig,
        metadata={"help": "SGLang overrides for this model."},
    )
    vllm: vLLMConfig = field(
        default_factory=vLLMConfig,
        metadata={"help": "vLLM overrides for this model."},
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters,
        metadata={"help": "Generation hyperparameters for this model."},
    )
    rollout: InferenceEngineConfig = field(
        default_factory=InferenceEngineConfig,
        metadata={"help": "Inference engine config overrides for this model."},
    )


# Top-level RaaS config


@dataclass
class RaaSConfig:
    """Minimal configuration for the RaaS inference serving component.

    Unlike PPOConfig / BaseExperimentConfig, this contains *only* the fields
    that RaaS actually reads — no trainer-specific settings like
    experiment_name, saver, recover, stats_logger, etc.
    """

    tokenizer_path: str = field(
        default="", metadata={"help": "Path to the HF tokenizer."}
    )
    seed: int = field(default=1, metadata={"help": "Random seed."})
    allocation_mode: Any = field(
        default="",
        metadata={"help": "Engine allocation config. Dict (engine section) or string (legacy)."},
    )
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster specification (RaaS only uses n_gpus_per_node)."},
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters,
        metadata={"help": "Generation hyperparameters for rollout."},
    )
    rollout: InferenceEngineConfig = field(
        default_factory=InferenceEngineConfig,
        metadata={"help": "Inference engine configuration."},
    )
    sglang: SGLangConfig = field(
        default_factory=SGLangConfig,
        metadata={"help": "SGLang server configuration."},
    )
    vllm: vLLMConfig = field(
        default_factory=vLLMConfig,
        metadata={"help": "vLLM server configuration."},
    )
    weight_transfer_mode: str = field(
        default="tcp",
        metadata={
            "help": (
                "Weight transfer mode. "
                "'tcp' (default): non-blocking TCP transfer via sender_agent. "
                "NCCL mode has been removed."
            )
        },
    )
    models: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Multi-model configuration.  Each key is a model id "
                "(e.g. 'model0') mapped to a ModelSpec.  When empty, "
                "RaaS runs in single-model mode using the top-level "
                "sglang/vllm/tokenizer_path fields."
            )
        },
    )
    delta_full_sync_interval: int = field(
        default=10,
        metadata={
            "help": (
                "When delta weight transfer is enabled, force a full "
                "transfer every N steps for resync.  0 = never force "
                "full (delta used whenever available)."
            )
        },
    )


def parse_cli_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        help="Path to config file(s). Can be specified multiple times.",
        required=True,
    )
    # The first argument might be the path to a training script,
    # which should be ignored by the argument parser.
    if argv and argv[0].endswith(".py"):
        argv = argv[1:]
    args, overrides = parser.parse_known_args(argv)

    config_paths = args.config  # list of paths due to action="append"

    from astraflow.config.loader import load_and_merge_configs, load_raas_config

    raw = load_and_merge_configs(config_paths)
    raas_dict = load_raas_config(raw)

    # Apply CLI dotted-key overrides
    import yaml as _yaml

    for override in overrides:
        if "=" in override:
            key, val = override.split("=", 1)
            parts = key.split(".")
            d = raas_dict
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            try:
                parsed = _yaml.safe_load(val)
            except Exception:
                parsed = val
            d[parts[-1]] = parsed

    cfg = OmegaConf.create(raas_dict)
    config_file = Path(config_paths[0]).absolute()
    return cfg, config_file


def to_structured_cfg(cfg, config_cls):
    # Merge with the default configuration.
    # The yaml and commandline can omit some default values defined in python dataclasses.
    default_cfg = OmegaConf.structured(config_cls)
    # Strip top-level keys from the YAML that don't exist on the dataclass
    # (e.g. shared scalars like experiment_name, model_path injected from shared).
    known_keys = set(default_cfg.keys())
    unknown = [k for k in cfg if k not in known_keys]
    if unknown:
        OmegaConf.set_struct(cfg, False)
        for k in unknown:
            del cfg[k]
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


def load_expr_config(argv: list[str], config_cls: type[ConfigT]) -> tuple[ConfigT, str]:
    cfg, config_file = parse_cli_args(argv)
    cfg = to_structured_cfg(cfg, config_cls=config_cls)
    cfg = OmegaConf.to_object(cfg)
    assert isinstance(cfg, config_cls)
    # Setup environment

    name_resolve.reconfigure(cfg.cluster.name_resolve)

    return cfg, str(config_file)


def conf_as_dict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)
