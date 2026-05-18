import re
from logging import getLogger
from pathlib import Path

from datasets import load_dataset, load_from_disk

from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)


_PROMPT_SUFFIX_RE = re.compile(
    r"(\r?\n|\\n)*\s*Let'?s\s+think\s+step\s+by\s+step\s+and\s+output\s+"
    r"the\s+final\s+answer\s+within\s*\\boxed\{\}\.\s*$",
    re.IGNORECASE,
)


def get_dapo_filter_sft_dataset(
    tokenizer=None,
    max_length: int | None = None,
):
    raise NotImplementedError("dapo_filter dataset not supported for SFT training.")


def download_dataset(
    offline_dir: str,
    dataset_path: str = "aaabiao/dapo_filter",
    split: str = "train",
):
    dataset = load_dataset(path=dataset_path, split=split)
    dataset.save_to_disk(offline_dir)
    return offline_dir


def get_dapo_filter_rl_dataset(
    tokenizer=None,
    max_length: int | None = None,
    offline_dir: str | None = None,
    max_samples: int | None = None,
    dataset_name: str = "dapo_filter",
    strip_cot_suffix: bool = True,
):
    """Load aaabiao/dapo_filter either from an offline dir or from HF online.

    The source rows have the verl/DAPO schema:
      - prompt: [{"role": "user", "content": str}]
      - reward_model: {"ground_truth": str, ...}
      - ability, extra_info, data_source

    By default the trailing
    "Let's think step by step and output the final answer within \\boxed{}."
    suffix is stripped from each prompt — DrMAS does the same so the workflow
    can own response formatting instead of inheriting it from the data.
    """
    if offline_dir is not None:
        if not Path(offline_dir).exists():
            raise FileNotFoundError(
                f"offline_dir does not exist: {offline_dir}. "
                "Run download_dataset(...) first."
            )
        logger.info("Loading dapo_filter dataset from offline path: %s", offline_dir)
        dataset = load_from_disk(offline_dir)
    else:
        logger.info("Loading dapo_filter dataset from online source.")
        dataset = load_dataset(path="aaabiao/dapo_filter", split="train")

    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample):
        prompt = sample["prompt"]
        assert len(prompt) == 1, f"expected single-turn prompt, got {len(prompt)}"
        content = prompt[0]["content"]
        if strip_cot_suffix:
            content = _PROMPT_SUFFIX_RE.sub("", content)
        ground_truth = sample["reward_model"]["ground_truth"]
        return {
            "messages": [{"role": "user", "content": content}],
            "answer": "\\boxed{" + str(ground_truth) + "}",
            "source": "dapo_filter",
        }

    drop_cols = [
        c
        for c in ("prompt", "reward_model", "ability", "extra_info", "data_source")
        if c in dataset.column_names
    ]
    dataset = dataset.map(process).remove_columns(drop_cols)

    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when max_length is set")

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    if max_samples is not None:
        n = min(int(max_samples), len(dataset))
        dataset = dataset.select(range(n))
        logger.info("dapo_filter truncated to %d samples (max_samples)", n)

    return dataset
