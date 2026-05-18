"""Standalone utilities for AstraFlow.

Contains copied utilities that were previously imported from ``train_worker.utils.data``
and ``train_worker.api.cli_args``, allowing AstraFlow to operate independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


@dataclass
class NormConfig:
    """Configuration for reward/advantage normalization."""

    mean_level: str | None = field(
        default="batch",
        metadata={
            "help": "Mean level for normalization. None for no mean normalization.",
            "choices": ["batch", "group", None],
        },
    )
    mean_leave1out: bool = field(
        default=False,
        metadata={"help": "Whether to use leave-one-out average."},
    )
    std_level: str | None = field(
        default="batch",
        metadata={
            "help": "Standard deviation level for normalization. None for no std normalization.",
            "choices": ["batch", "group", None],
        },
    )
    std_unbiased: bool = field(
        default=True,
        metadata={
            "help": "Whether to use unbiased standard deviation computation. Defaults to True (changed from False in v0.3.4)."
        },
    )
    eps: float = field(
        default=1e-5,
        metadata={
            "help": "The eps when dividing by standard deviation to avoid numerical issues."
        },
    )
    group_size: int = field(
        default=1, metadata={"help": "Group size for group-level normalization"}
    )


def is_multi_modal_key(key: str) -> bool:
    # Any key matching: multi_modal_input*
    return key.startswith("multi_modal_input")


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
        # Collect non-tensor values (e.g. prompt_id strings) as a list
        if not torch.is_tensor(tensor_dicts[0][key]):
            result[key] = [td[key] for td in tensor_dicts]
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


def cycle_dataloader(dataloader: StatefulDataLoader):
    """Cycle through a dataloader indefinitely."""
    epoch = 0
    while True:
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1


class Normalization:
    """
    Adaptive normalization with different levels.

    Supports independent specification of normalization level for mean and std:
    - "batch": normalize across entire batch (with optional all_reduce in distributed setting)
    - "group": normalize within fixed-size groups
    - None: no centering or no std scaling
    """

    def __init__(self, config: NormConfig):
        if config.mean_level not in {"batch", "group", None}:
            raise ValueError(
                f"mean_level must be 'batch', 'group' or None, got {config.mean_level}"
            )
        if config.std_level not in {"batch", "group", None}:
            raise ValueError(
                f"std_level must be 'batch', 'group', or None, got {config.std_level}"
            )
        if (
            config.mean_level == "group" or config.std_level == "group"
        ) and config.group_size is None:
            raise ValueError("group_size must be provided if using group normalization")

        self.mean_level = config.mean_level
        self.mean_leave1out = config.mean_leave1out
        self.std_level = config.std_level
        self.std_unbiased = config.std_unbiased
        self.group_size = config.group_size
        self.eps = config.eps

    def _iter_groups(
        self,
        bs: int,
        group_ids: torch.Tensor | None,
    ):
        """Yield (indices,) tuples for each group.

        If ``group_ids`` is provided, groups are defined dynamically by unique
        group_id values.  Otherwise, groups are contiguous slices of size
        ``self.group_size``.
        """
        if group_ids is not None:
            unique_ids = group_ids.unique()
            for gid in unique_ids:
                mask = (group_ids == gid).nonzero(as_tuple=True)[0]
                yield mask
        else:
            for i in range(0, bs // self.group_size):
                yield torch.arange(
                    i * self.group_size,
                    (i + 1) * self.group_size,
                    device=group_ids.device if group_ids is not None else "cpu",
                )

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        high_precision: bool = True,
        reduce_group=None,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize ``x`` at the configured level.

        Parameters
        ----------
        group_ids : Tensor | None
            Optional 1-D tensor of group labels (one per batch element).
            When provided and ``mean_level`` or ``std_level`` is ``"group"``,
            normalization is done within each unique group_id instead of
            using fixed-size contiguous slices.
        """
        bs = x.size(0)
        eps = self.eps

        # Early return if no elements are active (all masked out)
        if loss_mask is not None and loss_mask.sum().item() == 0:
            return x.float()

        # Step 1: Compute mean
        if self.mean_level == "batch":
            mean = self._compute_mean(
                x,
                loss_mask,
                high_precision=high_precision,
                leave_one_out=self.mean_leave1out,
                all_reduce=True,
                reduce_group=reduce_group,
            )
            mean = mean.expand_as(x)
        elif self.mean_level == "group":
            mean = torch.zeros_like(x)
            for idx in self._iter_groups(bs, group_ids):
                xx = x[idx]
                m = loss_mask[idx] if loss_mask is not None else None
                group_size = len(idx)

                # Special case: with group_size=1 and leave_one_out=True, mean should be 0
                if group_size == 1 and self.mean_leave1out:
                    dtype = torch.float64 if high_precision else torch.float32
                    group_mean = torch.zeros(
                        (1, xx.shape[1]), dtype=dtype, device=xx.device
                    )
                else:
                    group_mean = self._compute_mean(
                        xx,
                        m,
                        high_precision=high_precision,
                        leave_one_out=self.mean_leave1out,
                        all_reduce=False,
                        reduce_group=None,
                    )
                mean[idx] = group_mean.expand_as(xx)
        else:  # mean_level == "none"
            mean = torch.zeros_like(x)

        # Subtract mean
        x_centered = x - mean
        # mask unrelevant elements as 0
        if loss_mask is not None:
            x_centered = x_centered * loss_mask

        # Step 2: Compute std
        if self.std_level == "batch":
            std = self._compute_std(
                x,
                loss_mask,
                mean,
                unbiased=self.std_unbiased,
                high_precision=high_precision,
                all_reduce=True,
                reduce_group=reduce_group,
            )
            std = std.expand_as(x)
        elif self.std_level == "group":
            std = torch.zeros_like(x)
            for idx in self._iter_groups(bs, group_ids):
                xx = x[idx]
                m = loss_mask[idx] if loss_mask is not None else None
                group_mean_slice = mean[idx]  # already computed and expanded
                group_size = len(idx)

                # Special case: with group_size=1 and std_unbiased=True, std should be 1 for numerical stability
                if group_size == 1 and self.std_unbiased:
                    dtype = torch.float64 if high_precision else torch.float32
                    group_std = torch.ones(
                        (1, xx.shape[1]), dtype=dtype, device=xx.device
                    )
                else:
                    group_std = self._compute_std(
                        xx,
                        m,
                        group_mean_slice,
                        unbiased=self.std_unbiased,
                        high_precision=high_precision,
                        all_reduce=False,
                        reduce_group=reduce_group,
                    )
                std[idx] = group_std.expand_as(xx)
        else:
            std = torch.ones_like(x)
            eps = 0.0

        # Normalize
        return (x_centered / (std + eps)).float()

    @staticmethod
    def _compute_mean(
        x: torch.Tensor,
        mask: torch.Tensor | None,
        high_precision: bool,
        leave_one_out: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute mean only, using masked_normalization internals."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        dim = tuple(range(len(x.shape)))

        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_masked = x
            x_sum = x.sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_sum = x_masked.sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)

        if leave_one_out:
            if factor.item() <= 1:
                return torch.zeros_like(x_sum)
            # For leave-one-out, we need to compute mean excluding each element individually
            # This requires broadcasting: (total_sum - each_element) / (count - 1)
            if mask is None:
                # Broadcast x_sum to original shape and subtract each element
                x_sum_broadcast = x_sum.expand_as(x)
                leave_one_out_sum = x_sum_broadcast - x
                return leave_one_out_sum / (factor - 1)
            else:
                # For masked case, only subtract where mask is 1
                x_sum_broadcast = x_sum.expand_as(x)
                leave_one_out_sum = x_sum_broadcast - x_masked
                # Only compute leave-one-out where mask is 1, elsewhere return global mean
                regular_mean = x_sum / factor
                leave_one_out_mean = leave_one_out_sum / torch.clamp(
                    factor - mask, min=1.0
                )
                return torch.where(
                    mask > 0, leave_one_out_mean, regular_mean.expand_as(x)
                )

        if factor.item() == 0:
            return torch.zeros_like(x_sum)
        return x_sum / factor

    @staticmethod
    def _compute_std(
        x: torch.Tensor,
        mask: torch.Tensor | None,
        mean: torch.Tensor,
        unbiased: bool,
        high_precision: bool,
        all_reduce: bool,
        reduce_group,
    ) -> torch.Tensor:
        """Compute std only, given precomputed mean."""
        dtype = torch.float64 if high_precision else torch.float32
        x = x.to(dtype)
        mean = mean.to(dtype)
        dim = tuple(range(len(x.shape)))

        if mask is None:
            factor = torch.tensor(
                np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
            )
            x_centered = x - mean
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)
        else:
            mask = mask.to(dtype)
            x_masked = x * mask
            factor = mask.sum(dim, keepdim=True)
            x_centered = x_masked - mean * mask  # only apply mean where mask is 1
            x_sum_sq = (x_centered**2).sum(dim=dim, keepdim=True)

        if dist.is_initialized() and all_reduce:
            dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM, group=reduce_group)

        if unbiased:
            if factor.item() <= 1:
                return torch.ones_like(x_sum_sq)
            return (x_sum_sq / (factor - 1)).sqrt()

        if factor.item() == 0:
            return torch.ones_like(x_sum_sq)
        return (x_sum_sq / factor).sqrt()
