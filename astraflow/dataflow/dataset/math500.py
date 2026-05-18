from logging import getLogger
from pathlib import Path

from datasets import load_dataset, load_from_disk

from astraflow.dataflow.dataset.utils import attach_query_ids

logger = getLogger(__name__)

def get_math500_sft_dataset(
    tokenizer=None,
    max_length: int | None = None,
):
    """
    MATH-500 is not intended for SFT training in this setup.
    """
    raise NotImplementedError("MATH-500 dataset not supported for SFT training.")


def download_dataset(
    offline_dir: str,
    dataset_path: str = "HuggingFaceH4/MATH-500",
    split: str = "test",
):
    """
    Download the dataset from HF and persist it locally for offline use.
    """
    dataset = load_dataset(path=dataset_path, split=split)
    dataset.save_to_disk(offline_dir)
    return offline_dir


def get_math500_test_dataset(
    tokenizer=None,
    max_length: int | None = None,
    offline_dir: str | None = None,
    dataset_name: str = "math500",
):
    """
    Load the HuggingFaceH4/MATH-500 dataset and adapt it for RL training.

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
        logger.info("Loading MATH-500 dataset from offline path: %s", offline_dir)
        dataset = load_from_disk(offline_dir)
    else:
        logger.info("Loading MATH-500 dataset from online source.")
        dataset = load_dataset(path="HuggingFaceH4/MATH-500", split="test")

    # Stamp query_id BEFORE any map/filter so ids reflect the source-row
    # position and stay stable across max_length filtering.
    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample):
        # MATH-500 has 'problem' and 'solution' fields.
        # We append a boxed-answer instruction like in the deepscaler example.
        messages = [
            {
                "role": "user",
                "content": sample["problem"],
            }
        ]
        # The solution in MATH-500 is usually a full solution; if you only
        # want the final numeric answer, you may need a parser here.
        # For now, we wrap the entire solution in \\boxed{} for consistency.
        answer = "\\boxed{" + sample["answer"] + "}"
        # answer = sample["answer"]
        return {
            "messages": messages,
            "answer": answer,
            "source": "math500",
        }

    dataset = dataset.map(process)

    # Drop raw fields we don't need if they exist
    cols_to_drop = [c for c in ["problem", "solution"] if c in dataset.column_names]
    if len(cols_to_drop) > 0:
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

    return dataset


if __name__ == "__main__":
    # Quick test script
    from transformers import AutoTokenizer

    tokenizer_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Example tokenizer
    max_length = 2048

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading & processing dataset...")
    dataset = get_math500_test_dataset(tokenizer=tokenizer, max_length=max_length)

    print("Total samples loaded:", len(dataset))
    print("\nExample sample:")
    sample = dataset[0]
    print("messages:", sample["messages"])
    print("answer:", sample["answer"])
    print("content token length:", len(tokenizer.encode(sample["messages"][0]["content"])))
