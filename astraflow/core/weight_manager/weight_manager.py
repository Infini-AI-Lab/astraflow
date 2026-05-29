"""WeightManager — single owner of all weight transfer state and logic.

Replaces FSDPInterface. The trainer calls ``offload()`` after each
training step; everything else (buffer management, GPU→CPU copy,
sender agent lifecycle) is handled internally.

Usage::

    wm = WeightManager(config)
    wm.initialize(model.named_parameters(), local_rank, global_rank)

    # Per training step:
    wm.offload(model.named_parameters(), version, rank, world_size)
"""

from __future__ import annotations

import logging
import os
import queue
import time as _time
from typing import Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._tensor import DTensor

from astraflow.core.weight_manager.config import WeightManagerConfig
from astraflow.core.weight_manager.transfer.sender_agent import (
    create_tensor_from_shared_memory,
    start_transfer_agent,
)

logger = logging.getLogger(__name__)

_DTYPE_SIZES = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
}


def _nbytes(shape: List[int], dtype: str) -> int:
    from math import prod

    return int(prod(shape)) * _DTYPE_SIZES.get(dtype, 2)


class WeightManager:
    """Single owner of all weight transfer state and logic.

    Replaces FSDPInterface.  Handles:

    - Shared-memory double buffer allocation and broadcast
    - GPU→CPU weight copy (shard-direct and all-gather paths)
    - Sender agent subprocess lifecycle
    - Delta computation is done by the sender agent on CPU
    """

    def __init__(self, config: WeightManagerConfig):
        self.config = config

        # Buffer state
        self._buffer: torch.Tensor | None = None
        self._single_buffer_length: int = 0
        self._inactive_buf_idx: int = 0
        self._shm_path: str | None = None

        # Sender agent subprocess
        self._sender_process: mp.Process | None = None
        self._input_queue: mp.Queue | None = None
        self._output_queue: mp.Queue | None = None

        # Async delta: event shared with sender agent. Set when delta
        # compute finishes (or when no delta is pending). The guard in
        # offload() waits on this before writing to the inactive half.
        self._delta_done_event: mp.Event | None = None
        # Last delta metrics stashed by sender agent (retrieved after wait)
        self._last_delta_metrics: dict | None = None

        # Initialization state
        self._local_rank: int = 0
        self._global_rank: int = 0
        self._cross_node: bool = False  # True if some ranks lack shm buffer
        self._hsdp_replica_rank: int = 0  # 0 = primary replica (offloads), >0 = skip
        self._http_bind_port: int = config.sender_config.http_bind_port

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        named_params: Iterator[Tuple[str, torch.nn.Parameter]],
        local_rank: int,
        global_rank: int,
        lora_config: dict | None = None,
        megatron_metadata: Optional[dict] = None,
        megatron_hf_meta: Optional[list] = None,
        dp_replicate_rank: int = 0,
    ) -> None:
        """Initialize buffers and start sender agent.

        Parameters
        ----------
        named_params :
            Model's named parameters (consumed once to compute layout).
            For LoRA, pass only adapter params from ``get_peft_model_state_dict``.
        local_rank :
            Local rank on this node.
        global_rank :
            Global rank across all training nodes.
        lora_config :
            Optional dict with LoRA metadata (``r``, ``lora_alpha``,
            ``target_modules``).  Forwarded to the sender agent so the
            RaaS receiver can save weights in PEFT adapter format.
        megatron_metadata : dict, optional
            (Legacy TP-only shard-direct mode — superseded by
            ``megatron_hf_meta``.) If provided, the sender reassembles raw
            TP shards into HF format on CPU.  Only correct for PP=1/EP=1 and
            cannot compute deltas in HF space; kept for backward compat.
        megatron_hf_meta : list, optional
            Megatron **HF-export** mode (preferred).  The ordered HF weight
            layout ``[(hf_name, (shape, dtype_str)), ...]`` from
            ``hf_weight_metadata``.  The buffer is sized for the full HF
            model and ``offload`` writes already-converted HF tensors (from
            ``export_hf_named_params``) on the DP-head rank.  Because the
            buffer holds HF bytes, the sender's standard full/delta path is
            correct under any TP/PP/EP/VPP combination — see
            ``docs/en/architecture/megatron-weight-sync.md``.
        dp_replicate_rank : int
            HSDP replica group index. 0 = primary replica (owns the shm
            buffer and offloads weights). >0 = secondary replica (skips
            buffer mapping and offload writes).
        """
        self._local_rank = local_rank
        self._global_rank = global_rank
        self._hsdp_replica_rank = dp_replicate_rank
        self._megatron_metadata = megatron_metadata
        self._megatron_hf_meta = megatron_hf_meta

        if megatron_hf_meta is not None:
            # HF-export mode: buffer holds the full HF model in HF layout.
            # The sender treats it exactly like FSDP (no reassembly, delta
            # in HF space), so megatron_metadata stays None for the sender.
            meta_size = [
                (name, _nbytes(shape, dtype))
                for name, (shape, dtype) in megatron_hf_meta
            ]
            tensors_meta = list(megatron_hf_meta)
        elif megatron_metadata is not None:
            meta_size, tensors_meta = self._compute_megatron_buffer_layout(
                megatron_metadata["shard_specs"]
            )
        else:
            params = dict(named_params)
            meta_size, tensors_meta = self._compute_buffer_layout(params)

        # Only the primary HSDP replica (replica_rank=0) runs the sender
        # agent and owns the shm buffer. Secondary replicas skip entirely.
        # In HF-export mode the buffer already holds HF bytes, so the sender
        # runs the plain (FSDP) path — pass megatron_metadata=None.
        if local_rank == 0 and dp_replicate_rank == 0:
            self._start_sender_agent(
                meta_size, tensors_meta,
                lora_config=lora_config,
                megatron_metadata=(
                    None if megatron_hf_meta is not None else megatron_metadata
                ),
            )

        self._broadcast_shm_buffer()

        # Synchronize cross-node flag across all ranks so offload
        # strategy is consistent (avoids deadlock from mixed paths).
        # For HSDP, secondary replicas intentionally skip the buffer —
        # this is not a cross-node issue, so exclude them from the check.
        if dist.is_initialized():
            is_cross_node_issue = self._cross_node and dp_replicate_rank == 0
            flag = torch.tensor(
                [1 if is_cross_node_issue else 0],
                dtype=torch.int64,
                device=f"cuda:{local_rank}",
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            self._cross_node = flag.item() > 0
            if self._cross_node and global_rank == 0:
                logger.info(
                    "[WeightManager] Cross-node FSDP detected — "
                    "offload will use all-gather fallback"
                )
            if dp_replicate_rank == 0 and not self._cross_node and global_rank == 0:
                logger.info(
                    "[WeightManager] HSDP primary replica — "
                    "offload will use shard_copy"
                )

    def _compute_buffer_layout(
        self, state_dict: dict
    ) -> Tuple[
        List[Tuple[str, int]], List[Tuple[str, Tuple[List[int], str]]]
    ]:
        """Compute per-parameter byte sizes and metadata from state_dict."""
        meta_size: List[Tuple[str, int]] = []
        tensors_meta: List[Tuple[str, Tuple[List[int], str]]] = []
        for name, param in state_dict.items():
            assert torch.is_tensor(param) or isinstance(
                param, DTensor
            ), f"Expected tensor for param {name}, got {type(param)}"
            shape = list(param.shape)
            size_in_bytes = param.numel() * param.element_size()
            dtype = str(param.dtype).split(".")[-1]
            meta_size.append((name, size_in_bytes))
            tensors_meta.append((name, (shape, dtype)))
        total_bytes = sum(s for _, s in meta_size)
        logger.info(
            "[WeightManager] buffer layout: %d params, %.1f MB, first=%s, last=%s",
            len(meta_size),
            total_bytes / 1024 / 1024,
            meta_size[0][0] if meta_size else "?",
            meta_size[-1][0] if meta_size else "?",
        )
        return meta_size, tensors_meta

    def _compute_megatron_buffer_layout(
        self, shard_specs: List[dict],
    ) -> Tuple[
        List[Tuple[str, int]], List[Tuple[str, Tuple[List[int], str]]]
    ]:
        """Compute buffer layout from Megatron shard specs using full (gathered) sizes.

        Buffer stores raw TP shards laid out as [shard0 | shard1 | ...] per
        param.  Total size per param = full param size (= shard_size * tp_size).
        """
        from math import prod

        meta_size: List[Tuple[str, int]] = []
        tensors_meta: List[Tuple[str, Tuple[List[int], str]]] = []
        _dtype_sizes = {
            "float32": 4, "float16": 2, "bfloat16": 2,
            "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
        }
        for spec in shard_specs:
            name = spec["mcore_name"]
            full_shape = spec["full_shape"]
            dtype = spec["dtype"]
            elem_size = _dtype_sizes.get(dtype, 2)
            full_numel = prod(full_shape)
            size_in_bytes = full_numel * elem_size
            meta_size.append((name, size_in_bytes))
            tensors_meta.append((name, (full_shape, dtype)))

        total_bytes = sum(s for _, s in meta_size)
        logger.info(
            "[WeightManager] Megatron buffer layout: %d params, %.1f MB, "
            "first=%s, last=%s",
            len(meta_size),
            total_bytes / 1024 / 1024,
            meta_size[0][0] if meta_size else "?",
            meta_size[-1][0] if meta_size else "?",
        )
        return meta_size, tensors_meta

    def _start_sender_agent(
        self,
        meta_size: List[Tuple[str, int]],
        tensors_meta: List[Tuple[str, Tuple[List[int], str]]],
        lora_config: dict | None = None,
        megatron_metadata: Optional[dict] = None,
    ) -> None:
        """Spawn sender agent subprocess and map shared-memory buffer."""
        spawn_ctx = mp.get_context("spawn")
        self._input_queue = spawn_ctx.Queue()
        self._output_queue = spawn_ctx.Queue()
        self._input_queue.put(meta_size)
        self._input_queue.put(tensors_meta)
        self._input_queue.put(self.config.strategies)
        # Megatron metadata for CPU-side shard reassembly (None for FSDP).
        # Must be put before lora_config to match sender_agent read order.
        self._input_queue.put(megatron_metadata)
        self._input_queue.put(lora_config)  # LoRA metadata (or None)

        # Shared event for async delta: sender sets it after delta finishes.
        self._delta_done_event = spawn_ctx.Event()
        self._delta_done_event.set()  # no delta pending initially

        self._sender_process = start_transfer_agent(
            self.config.sender_config,
            self._input_queue,
            self._output_queue,
            ctx=spawn_ctx,
            delta_done_event=self._delta_done_event,
        )

        result = self._output_queue.get(timeout=120)
        if isinstance(result, tuple):
            # Accept both 2-tuple and 3-tuple (with delta shm path)
            if len(result) == 3:
                shm_path, buffer_length, _delta_shm = result
            else:
                shm_path, buffer_length = result
            self._shm_path = shm_path
            logger.info(
                "[WeightManager] mapping shared memory double-buffer from %s "
                "(%d bytes total, %d bytes per half)",
                shm_path,
                buffer_length,
                buffer_length // 2,
            )
            self._buffer = create_tensor_from_shared_memory(
                shm_path, buffer_length
            )
        else:
            assert torch.is_tensor(result) and result.is_cpu
            self._buffer = result
        self._single_buffer_length = self._buffer.numel() // 2

    def _broadcast_shm_buffer(self) -> None:
        """Broadcast shm path from local_rank 0 so all ranks can mmap the buffer."""
        if not dist.is_initialized():
            return

        if self._local_rank == 0 and self._buffer is not None:
            buf_info = [self._shm_path or "", self._buffer.numel()]
        else:
            buf_info = ["", 0]

        dist.broadcast_object_list(buf_info, src=0)
        shm_path, buffer_length = buf_info

        if not shm_path or buffer_length == 0:
            return

        if self._local_rank != 0 and self._buffer is None:
            if not os.path.exists(shm_path):
                # Cross-node FSDP: shm file lives on rank 0's node only.
                # This rank will participate in offload via all-gather
                # fallback instead of direct shard copy.
                self._cross_node = True
                logger.info(
                    "[WeightManager] rank %d: shm buffer %s not found "
                    "(cross-node FSDP), skipping buffer mapping — "
                    "will use all-gather offload",
                    self._global_rank,
                    shm_path,
                )
                return
            logger.info(
                "[WeightManager] rank %d: mapping shared memory buffer from %s "
                "(%d bytes)",
                self._global_rank,
                shm_path,
                buffer_length,
            )
            self._buffer = create_tensor_from_shared_memory(
                shm_path, buffer_length
            )
            self._single_buffer_length = buffer_length // 2

    # ------------------------------------------------------------------
    # Offload
    # ------------------------------------------------------------------

    def offload(
        self,
        named_params: Iterator[Tuple[str, torch.nn.Parameter]],
        version: int,
        rank: int,
        world_size: int,
    ) -> dict:
        """Copy weights from GPU to transfer buffer.

        Called by all ranks.  Delta computation runs asynchronously in
        the sender agent — the trainer is NOT blocked by it.  A guard
        at the top ensures the previous delta has finished before we
        overwrite the inactive half.

        Returns
        -------
        dict
            Weight transfer metrics for wandb logging. Empty on non-rank-0.
        """
        # Megatron HF-export mode: ``named_params`` is a fresh generator that
        # yields gathered HF tensors. It must be streamed (not list()-ed) and
        # iterated in lockstep on every rank (it runs TP/PP/EP collectives),
        # but only the writer rank copies into the buffer.
        if self._megatron_hf_meta is not None:
            return self._offload_megatron_hf(named_params, version, rank, world_size)

        params_list = list(named_params)

        # Guard: wait if previous delta is still reading the inactive half.
        # Only rank 0 has the event; all ranks sync via barrier after.
        # Normally instant — only blocks if training step < delta compute time.
        t_guard_start = _time.perf_counter()
        self._wait_previous_delta()
        # Barrier ensures rank 1+ don't write to the inactive half while
        # the sender agent (rank 0's subprocess) is still reading it.
        if dist.is_initialized():
            dist.barrier()
        t0 = _time.perf_counter()
        guard_time = t0 - t_guard_start

        # Offload strategy selection:
        # - HSDP secondary replicas: skip buffer writes, just barrier
        # - Cross-node flat FSDP: all-gather fallback (remote ranks lack shm)
        # - Single-node / HSDP primary: shard_copy (fast, direct shm write)
        is_secondary_replica = self._hsdp_replica_rank > 0

        if is_secondary_replica:
            # Secondary HSDP replicas have identical weights — only the
            # primary replica needs to offload. Just sync via barrier.
            t1 = _time.perf_counter()
            dist.barrier()
            t2 = _time.perf_counter()
            copy_mode = "hsdp_replica_skip"
        elif self._cross_node:
            # Cross-node flat FSDP: some ranks lack shm, fall back to
            # all-gather (only ranks with the buffer write to it).
            self._copy_all_gather(params_list)
            t1 = _time.perf_counter()
            dist.barrier()
            t2 = _time.perf_counter()
            copy_mode = "all_gather_fallback"
        elif self._megatron_metadata is not None:
            self._copy_megatron_shards(params_list)
            t1 = _time.perf_counter()
            dist.barrier()
            t2 = _time.perf_counter()
            copy_mode = "megatron_shard_copy"
        elif self._all_params_shard0(params_list):
            self._copy_shards(params_list, rank, world_size)
            t1 = _time.perf_counter()
            dist.barrier()
            t2 = _time.perf_counter()
            copy_mode = "shard_copy"
        else:
            self._copy_all_gather(params_list)
            t1 = _time.perf_counter()
            dist.barrier()
            t2 = _time.perf_counter()
            copy_mode = "all_gather_fallback"

        if not is_secondary_replica:
            ack = self._notify_buffer_ready(version)
        t3 = _time.perf_counter()

        metrics: dict = {}
        if rank == 0:
            metrics = {
                "weight_transfer/offload_guard_time": guard_time,
                "weight_transfer/offload_copy_time": t1 - t0,
                "weight_transfer/offload_barrier_time": t2 - t1,
                "weight_transfer/offload_notify_time": t3 - t2,
                "weight_transfer/offload_total_time": t3 - t_guard_start,
            }
            # Include delta metrics from the PREVIOUS step's async compute
            # (stashed by _wait_previous_delta / wait_delta_ready).
            if self._last_delta_metrics:
                metrics.update(self._last_delta_metrics)
                self._last_delta_metrics = None

            print(
                f"[WeightManager] offload mode={copy_mode}, "
                f"guard={guard_time:.3f}s, "
                f"copy={t1 - t0:.3f}s, barrier={t2 - t1:.3f}s, "
                f"notify={t3 - t2:.3f}s, total={t3 - t_guard_start:.3f}s",
                flush=True,
            )

        return metrics

    def _offload_megatron_hf(
        self,
        hf_named_params: Iterator[Tuple[str, torch.Tensor]],
        version: int,
        rank: int,
        world_size: int,
    ) -> dict:
        """Stream gathered HF tensors into the buffer (Megatron HF-export mode).

        Every rank iterates ``hf_named_params`` in lockstep — it drives the
        TP/PP/EP collectives inside ``export_hf_named_params`` — but only the
        writer rank (global rank 0, which owns the shm buffer) copies the
        yielded tensors into the inactive half, in the fixed order that
        matches ``megatron_hf_meta``.  Because the bytes are HF-layout, the
        sender's standard full/delta path is correct (delta in HF space).
        """
        t_guard_start = _time.perf_counter()
        self._wait_previous_delta()
        if dist.is_initialized():
            dist.barrier()
        t0 = _time.perf_counter()
        guard_time = t0 - t_guard_start

        is_writer = self._buffer is not None and self._local_rank == 0
        half_base = self._inactive_buf_idx * self._single_buffer_length
        offset = 0
        n_written = 0
        for _name, tensor in hf_named_params:
            nbytes = tensor.numel() * tensor.element_size()
            if is_writer:
                # Direct device->host DMA straight into the inactive half.
                # self._buffer is cudaHostRegister'd (pinned), so copying a
                # CUDA tensor into a view of it hits the fast PCIe DMA path
                # (~tens of GB/s) instead of the pageable .to("cpu") bounce
                # (~1 GB/s) the generator would otherwise do.
                #
                # Copy through a uint8 view of the *source* (the GPU tensor)
                # into the uint8 destination slice: both sides are uint8 so
                # there is no dtype-alignment requirement on the buffer offset
                # (robust to mixed-dtype models), and the bytes are identical
                # row-major since the source is contiguous.
                src_u8 = tensor.reshape(-1).view(torch.uint8)
                self._buffer[
                    half_base + offset: half_base + offset + nbytes
                ].copy_(src_u8, non_blocking=True)
                n_written += 1
            offset += nbytes
        if is_writer:
            # Fence the async D2H copies before the barrier so the sender
            # agent never reads a half-written half. (non_blocking=True above
            # matches the other copy paths; this synchronize is the fence.)
            torch.cuda.synchronize()
        t1 = _time.perf_counter()

        if dist.is_initialized():
            dist.barrier()
        t2 = _time.perf_counter()

        ack = self._notify_buffer_ready(version)
        t3 = _time.perf_counter()

        metrics: dict = {}
        if rank == 0:
            metrics = {
                "weight_transfer/offload_guard_time": guard_time,
                "weight_transfer/offload_copy_time": t1 - t0,
                "weight_transfer/offload_barrier_time": t2 - t1,
                "weight_transfer/offload_notify_time": t3 - t2,
                "weight_transfer/offload_total_time": t3 - t_guard_start,
            }
            if self._last_delta_metrics:
                metrics.update(self._last_delta_metrics)
                self._last_delta_metrics = None
            print(
                f"[WeightManager] offload mode=megatron_hf_export, "
                f"wrote={n_written} tensors, total_bytes={offset}, "
                f"guard={guard_time:.3f}s, copy={t1 - t0:.3f}s, "
                f"barrier={t2 - t1:.3f}s, notify={t3 - t2:.3f}s, "
                f"total={t3 - t_guard_start:.3f}s",
                flush=True,
            )
        return metrics

    # ------------------------------------------------------------------
    # Copy strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _all_params_shard0(
        params_list: List[Tuple[str, torch.nn.Parameter]],
    ) -> bool:
        """Check if all DTensor params use only Shard(0) or Replicate placement.

        HSDP produces (Replicate, Shard(0)) placements — Replicate is fine
        since all replicas have the same shards.
        """
        from torch.distributed.tensor import Replicate, Shard

        for _, param in params_list:
            if isinstance(param.data, DTensor):
                for p in param.data.placements:
                    if isinstance(p, Replicate):
                        continue
                    if not isinstance(p, Shard) or p.dim != 0:
                        return False
        return True

    def _copy_megatron_shards(
        self,
        params_list: List[Tuple[str, torch.nn.Parameter]],
    ) -> None:
        """Copy Megatron TP shards directly into the buffer — no GPU all-gather.

        Only dp_rank == 0 TP ranks write.  Each TP rank writes its local shard
        at ``param_offset + tp_rank * shard_size_bytes``.  Non-TP-sharded
        params are written only by tp_rank == 0.

        The sender agent handles CPU-side reassembly into HF format.
        """
        assert self._buffer is not None, "Shared memory buffer not mapped."
        meta = self._megatron_metadata
        tp_rank = meta["tp_rank"]
        dp_rank = meta["dp_rank"]
        shard_specs = meta["shard_specs"]

        half_base = self._inactive_buf_idx * self._single_buffer_length
        offset = 0

        for (name, param), spec in zip(params_list, shard_specs):
            tensor = param.data
            shard_size = tensor.numel() * tensor.element_size()
            # full_size = size of full (gathered) param in the buffer
            from math import prod
            full_numel = prod(spec["full_shape"])
            full_size = full_numel * tensor.element_size()

            if dp_rank == 0:
                if spec["is_sharded"]:
                    # Each TP rank writes its shard at the correct offset.
                    write_offset = half_base + offset + tp_rank * shard_size
                    t_cpu = tensor.contiguous().cpu()
                    t_u8 = t_cpu.view(-1).view(torch.uint8)
                    self._buffer[
                        write_offset: write_offset + shard_size
                    ].copy_(t_u8, non_blocking=True)
                elif tp_rank == 0:
                    # Non-sharded (duplicated) param: only tp_rank 0 writes.
                    write_offset = half_base + offset
                    t_cpu = tensor.contiguous().cpu()
                    t_u8 = t_cpu.view(-1).view(torch.uint8)
                    self._buffer[
                        write_offset: write_offset + full_size
                    ].copy_(t_u8, non_blocking=True)

            offset += full_size

        torch.cuda.synchronize()

    def _copy_shards(
        self,
        params_list: List[Tuple[str, torch.nn.Parameter]],
        rank: int,
        world_size: int,
    ) -> None:
        """Copy local FSDP shards directly into the buffer — no all-gather."""
        assert self._buffer is not None, (
            "Shared memory buffer not mapped on this rank."
        )

        half_base = self._inactive_buf_idx * self._single_buffer_length
        offset = 0
        total_shard_bytes = 0

        for name, param in params_list:
            tensor = param.data
            if isinstance(tensor, DTensor):
                local_shard = tensor.to_local()
                full_size = tensor.numel() * tensor.element_size()
                shard_size = local_shard.numel() * local_shard.element_size()

                write_offset = half_base + offset + rank * shard_size
                shard_cpu = local_shard.contiguous().cpu()
                shard_u8 = shard_cpu.view(-1).view(torch.uint8)
                self._buffer[
                    write_offset: write_offset + shard_size
                ].copy_(shard_u8, non_blocking=True)

                total_shard_bytes += shard_size
                offset += full_size
            else:
                size = tensor.numel() * tensor.element_size()
                if rank == 0:
                    t_cpu = tensor.contiguous().cpu()
                    t_u8 = t_cpu.view(-1).view(torch.uint8)
                    self._buffer[
                        half_base + offset: half_base + offset + size
                    ].copy_(t_u8, non_blocking=True)
                    total_shard_bytes += size
                offset += size

        torch.cuda.synchronize()

    def _copy_all_gather(
        self,
        params_list: List[Tuple[str, torch.nn.Parameter]],
    ) -> None:
        """Fallback: all-gather full state dict and copy to buffer."""
        half_base = self._inactive_buf_idx * self._single_buffer_length
        offset = 0

        for name, param in params_list:
            tensor = param.data
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            numel = tensor.numel()
            size_in_bytes = numel * tensor.element_size()

            if self._buffer is not None:
                param_data_cpu = tensor.contiguous()
                param_u8 = param_data_cpu.view(-1).view(torch.uint8)
                if offset + size_in_bytes > self._single_buffer_length:
                    raise RuntimeError(
                        f"Buffer overflow: name={name}, offset={offset}, "
                        f"size={size_in_bytes}, buffer={self._single_buffer_length}"
                    )
                buf_start = half_base + offset
                self._buffer[
                    buf_start: buf_start + size_in_bytes
                ].copy_(param_u8, non_blocking=True)
                offset += size_in_bytes

        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Buffer notification
    # ------------------------------------------------------------------

    def _notify_buffer_ready(self, version: int) -> Optional[dict]:
        """Notify sender agent that new weights are in the buffer.

        The sender agent acks immediately (swaps active index), then
        computes delta asynchronously.  The trainer is NOT blocked by
        delta computation.

        Must be called by ALL ranks. Only local_rank 0 communicates
        with the sender agent.

        Returns the ack dict from the sender (rank 0 only), or None.
        """
        buf_idx = self._inactive_buf_idx
        ack = None
        if self._local_rank == 0:
            assert self._sender_process is not None, (
                "Sender agent not initialized"
            )
            # Clear event BEFORE sending message — delta will be pending.
            if self._delta_done_event is not None:
                self._delta_done_event.clear()
            self._input_queue.put(f"buffer_ready:{version}:{buf_idx}")
            try:
                # Sender acks immediately after swap (no delta wait)
                ack = self._output_queue.get(timeout=60.0)
                if isinstance(ack, str) and ack.startswith("error:"):
                    logger.error(
                        "[WeightManager] Sender agent error: %s", ack[6:],
                    )
                    ack = None
            except queue.Empty:
                logger.warning(
                    "[WeightManager] Sender agent did not acknowledge "
                    "buffer_ready within 60s"
                )
        # ALL ranks flip so the next write targets the other half.
        self._inactive_buf_idx = 1 - buf_idx
        return ack

    def _wait_previous_delta(self) -> None:
        """Wait for the previous step's async delta compute to finish.

        Guards the inactive buffer half from being overwritten while the
        sender agent is still reading it.  Normally instant — only blocks
        if the training step was faster than delta compute (~12s for 8B).
        """
        if self._delta_done_event is None:
            return
        if self._delta_done_event.is_set():
            return
        if self._local_rank == 0:
            logger.info(
                "[WeightManager] Waiting for previous delta to finish..."
            )
        self._delta_done_event.wait(timeout=120.0)
        if not self._delta_done_event.is_set():
            logger.warning(
                "[WeightManager] Previous delta did not complete within 120s"
            )

    def wait_delta_ready(self, timeout: float = 60.0) -> None:
        """Wait for async delta compute to finish and stash metrics.

        Called by the trainer before ``notify_version`` to ensure delta
        is ready when RaaS pulls.  Also retrieves delta metrics from the
        sender agent's output queue for wandb logging.
        """
        if self._delta_done_event is None:
            return
        self._delta_done_event.wait(timeout=timeout)
        # Read the delta message the sender put on the queue before setting
        # the event.  Use a blocking get() instead of empty() + get_nowait()
        # because mp.Queue.empty() is unreliable across processes.
        try:
            msg = self._output_queue.get(timeout=5.0)
            if isinstance(msg, dict) and msg.get("type") == "delta_metrics":
                self._last_delta_metrics = {
                    "weight_transfer/delta_sparsity": msg.get("delta_sparsity", 0.0),
                    "weight_transfer/delta_size_mb": msg.get("delta_size", 0) / (1024 * 1024),
                    "weight_transfer/delta_num_nonzero": float(msg.get("delta_num_nonzero", 0)),
                    "weight_transfer/delta_compute_time": msg.get("delta_compute_time", 0.0),
                }
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_sender_endpoint(self) -> str:
        """Return ``'host:port'`` for the sender agent HTTP server."""
        import socket

        return f"{socket.gethostname()}:{self._http_bind_port}"

    def shutdown(self) -> None:
        """Clean up sender agent subprocess."""
        if self._sender_process is not None and self._sender_process.is_alive():
            self._sender_process.terminate()
            self._sender_process.join(timeout=5)
            logger.info("[WeightManager] Sender agent terminated.")
