"""Streaming Megatron -> HuggingFace weight export for online weight sync.

Reconstructs the *global* model from Megatron's sharded layout
(TP / PP / EP / ETP / VPP) and yields HuggingFace-named, HF-layout CPU
tensors **one bucket at a time**, so a large or MoE model is never
materialized in full.

This is the single source of truth for "Megatron weights -> HF" used by the
online weight-sync path (``WeightManager.offload``). See
``docs/en/architecture/megatron-weight-sync.md`` for the design and the
HF-space delta invariant.

Implementation note
-------------------
The heavy lifting (PP ``all_gather_object`` + broadcast, EP/ETP/TP
all-gather, local->global expert-id rewrite, and mcore->HF name/layout
conversion) is delegated to ``mbridge``'s ``Bridge.export_weights`` — the
same bridge the engine already uses to load (``_load_model_from_hf``) and
save (``_save_model_to_hf``). It is a battle-tested ``per_tensor_generator``
(equivalent to verl's ``per_tensor_generator`` and slime's
``HfWeightIteratorDirect``) that yields ``(hf_name, full_gpu_tensor)``.

We add the AstraFlow-specific consumer concerns on top: move each tensor to
CPU (the transfer buffer is CPU shared memory), group into byte-bounded
buckets, and a metadata-only mode for buffer sizing.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from astraflow.train_worker.utils import logging

logger = logging.getLogger(__name__)

# Default gather-bucket size, in bytes, measured on the *converted HF*
# tensors. mbridge gathers one source param at a time internally; this only
# bounds how many converted tensors we batch before handing them to the
# consumer (so the consumer can copy a run of tensors without per-tensor
# Python overhead). One bucket is alive at a time.
DEFAULT_BUCKET_BYTES = 512 << 20  # 512 MiB


def export_hf_named_params(
    bridge,
    models: list,
    *,
    to_cpu: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(hf_name, full_unsharded_tensor)`` for every model parameter.

    Reconstructs the global model from Megatron's TP/PP/EP/ETP/VPP layout
    via ``bridge.export_weights`` and yields HF-named tensors. Only one
    gathered tensor is resident at a time (plus transient collective
    buffers), so this is OOM-safe for large / MoE models.

    Parameters
    ----------
    bridge :
        The ``mbridge`` bridge for this model (``engine.bridge``). Already
        configured with the model's ``TransformerConfig`` and dtype.
    models :
        The engine's model chunk list (``_MegatronModelList``): VPP chunks,
        each typically ``DistributedDataParallel``-wrapped. ``mbridge``
        unwraps them internally.
    to_cpu :
        Move each yielded tensor to CPU (default). The transfer buffer is
        CPU shared memory, so this is the normal path. Set False only for
        callers that consume on-GPU.

    Yields
    ------
    tuple[str, torch.Tensor]
        HF parameter name (e.g. ``model.layers.0.self_attn.q_proj.weight``)
        and the full (unsharded) tensor, contiguous, on CPU when
        ``to_cpu``.

    Notes
    -----
    Every rank must call this in lockstep: ``export_weights`` runs
    collectives (PP all_gather_object + broadcast, TP/EP/ETP all_gather)
    across all model-parallel ranks. The yielded values are identical on
    every rank in a model-parallel group, so the caller decides which rank
    actually writes them to the buffer (the DP/PP/TP head).
    """
    for hf_name, param in bridge.export_weights(models):
        tensor = param.detach()
        if to_cpu:
            # bf16/contiguous on CPU — pinned-buffer copy happens in the
            # consumer; .contiguous() guards against non-contiguous views
            # produced by QKV/gate-up splits in the converter.
            tensor = tensor.to("cpu", copy=False).contiguous()
        else:
            tensor = tensor.contiguous()
        yield hf_name, tensor


def iter_param_buckets(
    named_params: Iterator[tuple[str, torch.Tensor]],
    bucket_bytes: int = DEFAULT_BUCKET_BYTES,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Group a ``(name, tensor)`` stream into byte-bounded buckets.

    Yields lists whose cumulative tensor bytes stay under ``bucket_bytes``
    (a single tensor larger than the cap forms its own bucket). Lets the
    consumer amortize per-tensor overhead while keeping only one bucket of
    tensors alive at a time.
    """
    bucket: list[tuple[str, torch.Tensor]] = []
    cur = 0
    for name, tensor in named_params:
        nbytes = tensor.numel() * tensor.element_size()
        if bucket and cur + nbytes > bucket_bytes:
            yield bucket
            bucket = []
            cur = 0
        bucket.append((name, tensor))
        cur += nbytes
    if bucket:
        yield bucket


def hf_weight_metadata(
    bridge,
    models: list,
) -> list[tuple[str, tuple[list[int], str]]]:
    """Return the ordered HF weight layout: ``[(name, (shape, dtype_str)), ...]``.

    Drives the same ``export_weights`` generator but keeps only shape/dtype
    (dropping tensor storage as it goes), so the full model is never
    resident. Consumed by ``WeightManager`` to size the transfer buffer and
    by the RaaS receiver (as ``tensors_meta``) to pre-allocate — both ends
    then agree on layout and order.

    Must be called in lockstep on every rank (it runs the same collectives
    as ``export_hf_named_params``).
    """
    meta: list[tuple[str, tuple[list[int], str]]] = []
    for hf_name, param in bridge.export_weights(models):
        dtype = str(param.dtype).split(".")[-1]
        meta.append((hf_name, (list(param.shape), dtype)))
        del param
    logger.info(
        "[weight_export] HF metadata: %d tensors, first=%s last=%s",
        len(meta),
        meta[0][0] if meta else "?",
        meta[-1][0] if meta else "?",
    )
    return meta
