"""GenerationHyperparameters for workflow package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


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
