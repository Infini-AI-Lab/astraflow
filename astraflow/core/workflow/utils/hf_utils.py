from functools import lru_cache

import transformers

from astraflow.core.workflow.utils import logging

logger = logging.getLogger("HF Utility")


@lru_cache(maxsize=8)
def load_hf_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: str | None = None,
) -> transformers.PreTrainedTokenizerFast:
    kwargs = {}
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        fast_tokenizer=fast_tokenizer,
        trust_remote_code=True,
        **kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def apply_chat_template_to_ids(
    tokenizer,
    messages,
    *,
    enable_thinking: bool | None = None,
    **kwargs,
) -> list[int]:
    """Apply a chat template and return a flat ``list[int]`` of token ids.

    Normalizes across transformers versions. transformers>=5 makes
    ``apply_chat_template(tokenize=True)`` return a ``BatchEncoding`` (a
    Mapping) instead of a flat ``list[int]``; calling ``list(...)`` on that
    yields the dict keys (``["input_ids", "attention_mask"]``) rather than
    tokens, which then get sent to the inference engine and rejected. We
    extract ``input_ids`` whenever a mapping is returned.

    ``enable_thinking`` is forwarded only when the tokenizer's chat template
    accepts it (older templates raise ``TypeError``), matching the prior
    per-workflow ``_apply_chat_template`` behavior.
    """
    kwargs.setdefault("tokenize", True)
    try:
        out = tokenizer.apply_chat_template(
            messages,
            **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
            **kwargs,
        )
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(out, "keys"):  # BatchEncoding (transformers>=5) -> token-id list
        out = out["input_ids"]
    return list(out)


@lru_cache(maxsize=8)
def load_hf_processor_and_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: str | None = None,
) -> tuple["transformers.ProcessorMixin | None", transformers.PreTrainedTokenizerFast]:
    """Load a tokenizer and processor from Hugging Face."""
    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer, padding_side)
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        processor = None
        logger.warning(
            f"Failed to load processor for {model_name_or_path}. "
            "Using tokenizer only. This may cause issues with some models."
        )
    return processor, tokenizer
