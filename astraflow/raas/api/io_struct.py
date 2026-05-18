# NOTE: This file is an intentional copy of train_worker/api/io_struct.py for
# package independence. Keep both copies in sync or coordinate migration.

import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizerFast

from astraflow.raas.api.cli_args import GenerationHyperparameters

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
class LocalInfServerInfo:
    """Information about a locally launched inference server."""

    host: str
    port: int
    process: subprocess.Popen
