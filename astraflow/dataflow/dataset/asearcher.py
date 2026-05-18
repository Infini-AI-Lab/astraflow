import json

from datasets import Dataset, load_dataset

from astraflow.dataflow.dataset.utils import attach_query_ids


def _load_json_records(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)!r}")
    return data


def _normalize_record(sample: dict, idx: int | None = None) -> dict:
    messages = sample.get("messages")
    if not messages:
        question = sample.get("question", "")
        messages = [{"role": "user", "content": question}]

    data = {
        "messages": messages,
        "question": sample.get("question", messages[0]["content"]),
        "source": "asearcher",
    }
    if "answer" in sample:
        data["answer"] = sample["answer"]
    return data


def get_asearcher_rl_dataset(
    path: str,
    tokenizer=None,
    max_length: int | None = None,
    dataset_name: str = "asearcher",
):
    """Load ASearcher rollout prompts from a local JSONL file."""
    # ASearcher datasets are commonly stored as JSONL, and some eval sets
    # such as HotpotQA contain nested list fields with mixed element types
    # that PyArrow JSON inference rejects. Load JSONL directly to avoid
    # noisy parse failures from the Hugging Face JSON builder.
    if path.endswith(".jsonl"):
        dataset = Dataset.from_list(
            [
                _normalize_record(sample, idx=i)
                for i, sample in enumerate(_load_json_records(path))
            ]
        )
    else:
        try:
            dataset = load_dataset("json", data_files=path, split="train")
        except Exception:
            dataset = Dataset.from_list(
                [
                    _normalize_record(sample, idx=i)
                    for i, sample in enumerate(_load_json_records(path))
                ]
            )
        else:
            dataset = dataset.map(
                _normalize_record,
                with_indices=True,
                remove_columns=dataset.column_names,
            )

    # Stamp query_id AFTER normalization (since some branches build the
    # dataset row-by-row). add_column always uses the current row order;
    # a later filter() preserves per-row values, so no reordering risk.
    dataset = attach_query_ids(dataset, dataset_name)

    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when max_length is set")

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            return len(tokenizer.encode(content)) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
