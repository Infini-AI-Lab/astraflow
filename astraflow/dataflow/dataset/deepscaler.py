from logging import getLogger
from pathlib import Path

from datasets import load_dataset, load_from_disk

from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)

def get_deepscaler_sft_dataset(
    tokenizer=None,
    max_length: int | None = None,
):
    raise NotImplementedError("DeepScaler dataset not supported for SFT training.")


def download_dataset(
    offline_dir: str,
    dataset_path: str = "agentica-org/DeepScaleR-Preview-Dataset",
    split: str = "train",
):
    """
    Download the dataset from HF and persist it locally for offline use.
    """
    dataset = load_dataset(path=dataset_path, split=split)
    dataset.save_to_disk(offline_dir)
    return offline_dir


def get_deepscaler_rl_dataset(
    tokenizer=None,
    max_length: int | None = None,
    offline_dir: str | None = None,
    max_samples: int | None = None,
    dataset_name: str = "deepscaler",
):
    """
    Load DeepScaleR either from an offline dir (if provided) or from HF online.
    """
    if offline_dir is not None:
        if not Path(offline_dir).exists():
            raise FileNotFoundError(
                f"offline_dir does not exist: {offline_dir}. "
                "Run download_dataset(...) first."
            )
        logger.info("Loading DeepScaleR dataset from offline path: %s", offline_dir)
        dataset = load_from_disk(offline_dir)
    else:
        logger.info("Loading DeepScaleR dataset from online source.")
        dataset = load_dataset(
            path="agentica-org/DeepScaleR-Preview-Dataset",
            split="train",
        )

    # Stamp query_id BEFORE any map/filter so ids reflect the source-row
    # position and stay stable across max_length filtering and max_samples
    # truncation.
    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["problem"],
            }
        ]
        answer = "\\boxed{" + sample['answer'] + "}"
        return {
            "messages": messages,
            "answer": answer,
            "source": "deepscaler",
        }

    dataset = dataset.map(process).remove_columns(["problem", "solution"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when max_length is set")

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    # Optional truncation for quick tests / curriculum experiments.
    if max_samples is not None:
        n = min(int(max_samples), len(dataset))
        dataset = dataset.select(range(n))
        logger.info("DeepScaleR truncated to %d samples (max_samples)", n)

    return dataset
