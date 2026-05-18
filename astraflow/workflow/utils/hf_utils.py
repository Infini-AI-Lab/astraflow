from functools import lru_cache

import transformers

from astraflow.workflow.utils import logging

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
