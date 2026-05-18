import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizerFast

from astraflow.train_worker.api.alloc_mode import AllocationMode
from astraflow.train_worker.api.cli_args import GenerationHyperparameters

if TYPE_CHECKING:
    from transformers import AutoProcessor


@dataclass
class ModelRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    image_data: list[str] | None = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # vlm+vllm:
    vision_msg_vllm: list | None = None

    def copy(self):
        return ModelRequest(
            rid=self.rid,
            input_ids=self.input_ids.copy(),
            gconfig=self.gconfig.new(),
            metadata=self.metadata.copy(),
            tokenizer=self.tokenizer,
            image_data=self.image_data.copy() if self.image_data is not None else None,
            processor=self.processor,
            vision_msg_vllm=self.vision_msg_vllm.copy()
            if self.vision_msg_vllm is not None
            else None,
        )


@dataclass
class ModelResponse:
    # outputs
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "interrupt"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    input_images: list[ImageObject | str] = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


@dataclass
class FinetuneSpec:
    total_train_epochs: int
    dataset_size: int
    train_batch_size: int
    _total_train_steps_override: int | None = None

    @property
    def total_train_steps(self):
        # assuming drop_last
        computed = self.total_train_epochs * (self.dataset_size // self.train_batch_size)
        if self._total_train_steps_override is not None:
            return min(computed, self._total_train_steps_override)
        return computed

    @property
    def steps_per_epoch(self):
        return self.dataset_size // self.train_batch_size


@dataclass
class ParamSpec:
    name: str
    shape: tuple
    dtype: str

    @property
    def size(self) -> int:
        """Param bytes"""
        return getattr(torch, self.dtype).itemsize * np.prod(self.shape)


@dataclass
class WeightUpdateMeta:
    type: Literal["disk"]
    path: str | None = None
    alloc_mode: AllocationMode | None = None

    use_lora: bool = False
    lora_name: str = ""
    lora_int_id: int = 0
    base_model_name: str = ""
    peft_config: dict = field(default_factory=dict)

    clear_checkpoint_after_load: bool = True


@dataclass
class HttpRequest:
    """Represents an HTTP request to be sent to a remote inference server."""

    endpoint: str
    payload: dict[str, Any]
    method: str = "POST"


@dataclass
class HttpGenerationResult:
    """Parsed result from a generation response."""

    output_tokens: list[int]
    output_logprobs: list[float]
    stop_reason: str


@dataclass
class WeightUpdateRequests:
    """Collection of HTTP requests needed for a weight update operation."""

    requests: list[HttpRequest]


@dataclass
class SaveLoadMeta:
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None = None
    processor: Optional["AutoProcessor"] = None
    base_model_path: str | None = None
    naive_distributed: bool = False


@dataclass
class RolloutStat:
    accepted: int = 0
    enqueued: int = 0
    rejected: int = 0
    running: int = 0


@dataclass
class StepInfo:
    epoch: int
    epoch_step: int
    global_step: int
    steps_per_epoch: int

    def next(self):
        return StepInfo(
            epoch=self.epoch + (self.epoch_step == self.steps_per_epoch - 1),
            epoch_step=(
                0
                if self.epoch_step == self.steps_per_epoch - 1
                else self.epoch_step + 1
            ),
            global_step=self.global_step + 1,
            steps_per_epoch=self.steps_per_epoch,
        )


@dataclass
class LocalInfServerInfo:
    """Information about a locally launched inference server."""

    host: str
    port: int
    process: subprocess.Popen
