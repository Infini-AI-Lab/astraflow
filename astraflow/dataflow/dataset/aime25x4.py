from logging import getLogger
from pathlib import Path

from datasets import load_dataset, load_from_disk

from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)

def get_aime_2025_sft_dataset(
    tokenizer=None,
    max_length: int | None = None,
):
    """
    AIME-2025 is not intended for SFT training in this setup.
    """
    raise NotImplementedError("AIME-2025 dataset not supported for SFT training.")


def download_dataset(
    offline_dir: str,
    dataset_path: str = "math-ai/aime25",
    split: str = "test",
):
    """
    Download the dataset from HF and persist it locally for offline use.
    """
    dataset = load_dataset(path=dataset_path, split=split)
    dataset.save_to_disk(offline_dir)
    return offline_dir


def get_aime_2025x4_test_dataset(
    tokenizer=None,
    max_length: int | None = None,
    offline_dir: str | None = None,
    dataset_name: str = "aime_2025",
):
    """
    Load the math-ai/aime25 dataset and adapt it for RL-style evaluation.

    Args:
        tokenizer: Tokenizer with an .encode() method (e.g. HF tokenizer).
        max_length: If provided, filter out samples whose user prompt exceeds
            this token length.
    """
    if offline_dir is not None:
        if not Path(offline_dir).exists():
            raise FileNotFoundError(
                f"offline_dir does not exist: {offline_dir}. "
                "Run download_dataset(...) first."
            )
        logger.info("Loading AIME-2025 dataset from offline path: %s", offline_dir)
        dataset = load_from_disk(offline_dir)
    else:
        logger.info("Loading AIME-2025 dataset from online source.")
        dataset = load_dataset(path="math-ai/aime25", split="test")

    # Stamp query_id BEFORE any map/filter so ids reflect source-row position.
    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["problem"],
            }
        ]
        answer = "\\boxed{" + sample["answer"] + "}"
        # answer = sample["answer"]
        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process)

    # Drop raw fields we don't need if they exist
    cols_to_drop = [
        c
        for c in [
            "id",
        ]
        if c in dataset.column_names
    ]
    if cols_to_drop:
        dataset = dataset.remove_columns(cols_to_drop)

    # Filter by prompt length if requested
    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when max_length is set")

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    from datasets import concatenate_datasets
    dataset = concatenate_datasets([dataset] * 4)

    return dataset


if __name__ == "__main__":
    # Quick test script
    from transformers import AutoTokenizer

    tokenizer_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Example tokenizer
    max_length = 2048

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading & processing dataset...")
    dataset = get_aime_2025x4_test_dataset(tokenizer=tokenizer, max_length=max_length)

    print(len(dataset))

    print("Total samples loaded:", len(dataset))
    print("\nExample sample:")
    sample = dataset[0]
    print(sample)
    print("messages:", sample["messages"])
    print("answer:", sample["answer"])
    print(
        "content token length:",
        len(tokenizer.encode(sample["messages"][0]["content"])),
    )
