"""Shared dataset helpers.

``attach_query_ids`` stamps a stable ``query_id`` of the form
``f"{dataset_name}-{src_idx:08d}"`` on every row of a Hugging Face dataset.
Each training-dataset loader MUST call this on the freshly-loaded dataset,
**before** any ``map``/``filter``/``select``/``shuffle`` so that the id
reflects the source-row position and stays stable across:

- ``max_samples`` truncation
- per-row ``map`` transforms
- ``max_length`` filters

Eval datasets MAY skip this; the curator gracefully treats absent ids as
"do not profile" (see ``astraflow.dataflow.prompt_curators.GRESOCurator``).

The dataset-name prefix gives every multi-dataset training mix a flat
namespace — two datasets cannot collide on a numeric id because each
prepends its own name.
"""

from __future__ import annotations

from typing import Any


def attach_query_ids(dataset: Any, dataset_name: str) -> Any:
    """Stamp a ``query_id`` column on every row.

    The id is ``f"{dataset_name}-{src_idx:08d}"`` where ``src_idx`` is the
    row's position in the freshly-loaded dataset. ``08d`` zero-pads to 8
    digits so lexicographic sort matches numeric sort.

    Parameters
    ----------
    dataset : ``datasets.Dataset``
        The freshly-loaded HF dataset (or anything with ``__len__`` and
        ``add_column``). Must not have been ``map``-ed / ``filter``-ed yet.
    dataset_name : str
        Namespace prefix. Pick something stable per loader (``"gsm8k"``,
        ``"alfworld"``, ...). The recipe YAML can override per loader.

    Returns
    -------
    ``datasets.Dataset`` with a new (or overwritten) ``query_id`` column.
    """
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(
            f"dataset_name must be a non-empty string, got {dataset_name!r}"
        )
    n = len(dataset)
    ids = [f"{dataset_name}-{i:08d}" for i in range(n)]
    # ``add_column`` raises if the column already exists; if a loader
    # is wired through twice, drop the old column first to make the
    # operation idempotent.
    if "query_id" in getattr(dataset, "column_names", ()):
        dataset = dataset.remove_columns(["query_id"])
    return dataset.add_column("query_id", ids)


__all__ = ["attach_query_ids"]
