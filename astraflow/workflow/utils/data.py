"""Standalone data utilities for workflow tensor building."""

from typing import Any

import torch


# Canonical priority order for extracting a prompt id from a dataset sample.
# The curator gate (pre-rollout) and every workflow's result stamp
# (post-rollout) MUST resolve through this same list so the byte-identical
# id flows through both sides of GRESO's per-prompt streak table.
PROMPT_ID_KEYS: tuple[str, ...] = (
    "query_id",
    "prompt_id",
    "task_id",
    "id",
    "qid",
    "idx",
    "index",
)


def resolve_prompt_id(data: dict[str, Any]) -> str | None:
    """Return the stable prompt id for a dataset sample, or None.

    Tries ``PROMPT_ID_KEYS`` in priority order and stringifies the first
    non-None value. Workflows should pass the result through to
    ``results_to_structured(prompt_id=...)`` (or the manual ``"prompt_id"``
    key on a custom result dict) so curator updates key the same prompt
    the gate keyed.
    """
    for k in PROMPT_ID_KEYS:
        v = data.get(k)
        if v is not None:
            return str(v)
    return None


def get_batch_size(data: dict[str, Any]) -> int:
    if not data:
        return 0

    am = data.get("attention_mask")
    if torch.is_tensor(am) and am.ndim >= 1:
        return int(am.shape[0])

    cu = data.get("cu_seqlens")
    if torch.is_tensor(cu) and cu.ndim >= 1 and cu.numel() >= 1:
        return max(int(cu.shape[0]) - 1, 0)

    mmi = data.get("multi_modal_input")
    if isinstance(mmi, list):
        return len(mmi)

    for v in data.values():
        if torch.is_tensor(v) and v.ndim >= 1:
            return int(v.shape[0])

    return 0


def is_multi_modal_key(key: str) -> bool:
    return key.startswith("multi_modal_input")


def concat_padded_tensors(
    tensor_dicts: list[dict[str, Any]], pad_value: float = 0.0
) -> dict[str, Any]:
    """Concatenate and pad tensors from multiple dictionaries of padded tensors."""
    if not tensor_dicts:
        return {}

    # Find max sequence length across all dictionaries
    assert all("attention_mask" in td for td in tensor_dicts)
    max_length = max([x["attention_mask"].shape[1] for x in tensor_dicts])
    result = {}

    multimodal_keys = {
        key for td in tensor_dicts for key in td if is_multi_modal_key(key)
    }
    # Merge multimodal keys
    for mm_key in multimodal_keys:
        merged_multi_modal = []
        for td in tensor_dicts:
            bs = get_batch_size(td)
            merged_multi_modal.extend(td.get(mm_key, [{} for _ in range(bs)]))
        result[mm_key] = merged_multi_modal

    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []
        if is_multi_modal_key(key):
            continue
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
            # Skip 1D tensors like rewards
            if len(tensor.shape) == 1:
                tensors_to_concat.append(tensor)
                continue
            current_length = tensor.shape[1]
            if current_length < max_length:
                # Pad tensor to max_length
                pad_width = max_length - current_length
                if key == "attention_mask":
                    # Pad attention mask with 0s
                    padding = torch.zeros(
                        (tensor.shape[0], pad_width),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                else:
                    # Pad feature tensors with pad_value
                    padding = torch.full(
                        (tensor.shape[0], pad_width),
                        pad_value,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )

                tensor = torch.cat([tensor, padding], dim=1)
            tensors_to_concat.append(tensor)

        result[key] = torch.cat(tensors_to_concat, dim=0)
    return result


def results_to_structured(
    results: list[dict[str, Any]],
    prompt_id: str | None = None,
) -> dict[str, Any]:
    """Convert a list of per-sequence tensor dicts into structured format.

    Each element in *results* is a single-sequence dict with ``[1, seq_len]``
    tensors (including ``"rewards"``).  This wraps them into the structured
    format expected by ``AstraDataAcquisition._ingest_structured_result``:

        {"n_trajs": N, "rewards": Tensor[N], "trajectories": [...]}

    In the flat→structured mapping each sequence becomes its own trajectory.
    """
    if not results:
        return {
            "n_trajs": 0,
            "rewards": torch.tensor([], dtype=torch.float32),
            "trajectories": [],
        }
    rewards = torch.tensor(
        [float(r["rewards"].flatten()[0].item()) for r in results],
        dtype=torch.float32,
    )
    out: dict[str, Any] = {
        "n_trajs": len(results),
        "rewards": rewards,
        "trajectories": [{"sequences": [r]} for r in results],
    }
    if prompt_id is not None:
        out["prompt_id"] = prompt_id
    return out
