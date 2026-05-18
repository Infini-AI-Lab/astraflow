"""
AlfWorld Dataset Loader

This dataset loader provides task indices for the AlfWorld task,
which runs via an external task server (AgentBench).

The dataset format is simple JSONL with task indices:
    {"index": 0, "messages": []}

The actual task descriptions and execution are handled by the task server.
This keeps the task environment separate from the RL training framework.
"""

from datasets import load_dataset

from astraflow.dataflow.dataset.utils import attach_query_ids


def get_alfworld_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """SFT training is not supported for AlfWorld tasks."""
    raise NotImplementedError(
        "AlfWorld dataset is designed for RL training only. "
        "Tasks require interactive execution in a household environment, "
        "which cannot be reduced to static SFT examples."
    )


def get_alfworld_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    max_samples: int | None = None,
    dataset_name: str = "alfworld",
):
    """
    Load AlfWorld dataset for RL training.

    Args:
        path: Path to the JSONL file containing task indices
              (e.g., "examples/alfworld/data/train_indices.jsonl")
        split: Dataset split (ignored for JSONL files, but required by API)
        tokenizer: Tokenizer instance (unused for this dataset)
        max_length: Maximum length (unused for this dataset)
        max_samples: Optional truncation. Useful to shrink epoch-1 so GRESO
                     arming fires within a few steps for fast iteration.

    Returns:
        Dataset with fields:
            - index: Task identifier (e.g., "std-00000", "dev-00001")
            - messages: Empty list (actual task data comes from server)

    The workflow will use the 'index' field to request specific tasks
    from the task server via HTTP API.
    """
    # Load JSONL file using HuggingFace datasets
    # For JSONL files, we always use split="train" when loading
    dataset = load_dataset("json", data_files=path, split="train")
    # Stamp query_id BEFORE any map/filter so ids reflect source-row
    # position and stay stable across max_samples truncation.
    dataset = attach_query_ids(dataset, dataset_name)

    def process(sample):
        """
        Process each sample to ensure consistent format.

        The sample already has 'index' and 'messages' fields from JSONL.
        We keep both for compatibility:
        - 'index': Used by TaskServerWorkflow to identify the task
        - 'messages': Empty list, maintained for standard RL dataset interface
        """
        return {
            "index": sample["index"],
            "messages": sample.get("messages", []),
        }

    # Process the dataset
    dataset = dataset.map(process)

    if max_samples is not None:
        n = min(int(max_samples), len(dataset))
        dataset = dataset.select(range(n))

    # No filtering needed - all AlfWorld tasks are valid
    # max_length is not applicable since we don't have actual text content yet

    return dataset
