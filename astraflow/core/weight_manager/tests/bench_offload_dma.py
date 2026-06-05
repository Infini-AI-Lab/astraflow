"""Perf benchmark: direct-DMA vs pageable offload throughput.

Measures wall-clock to copy the full gathered HF model from GPU into a host
buffer via the new direct-DMA path vs the old pageable .to("cpu") path, on
the same export pass. Reports GB/s for each. Evidence for optimization #1.

Run:
    torchrun --nproc_per_node=<N> \
        astraflow/core/weight_manager/tests/bench_offload_dma.py \
        --model /shared/models/Qwen3-0.6B --tp 2 --iters 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    from astraflow.train_worker.api.alloc_mode import ParallelStrategy
    from astraflow.train_worker.api.cli_args import TrainEngineConfig
    from astraflow.train_worker.api.io_struct import FinetuneSpec
    from astraflow.train_worker.engine.megatron_engine import MegatronEngine

    world = int(os.environ["WORLD_SIZE"])
    dp = world // (args.tp * args.pp * args.ep)
    engine = MegatronEngine(TrainEngineConfig(path=args.model, dtype="bfloat16"))
    engine.create_process_group(
        parallel_strategy=ParallelStrategy(
            data_parallel_size=dp,
            tensor_parallel_size=args.tp,
            pipeline_parallel_size=args.pp,
            expert_parallel_size=args.ep,
        )
    )
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1),
    )
    rank = dist.get_rank()
    is_writer = rank == 0

    from math import prod

    hf_meta = engine.get_hf_weight_metadata()
    total = sum(prod(sh) * (2 if dt == "bfloat16" else 4) for _, (sh, dt) in hf_meta)
    buf = torch.empty(total, dtype=torch.uint8, pin_memory=True) if is_writer else None
    pageable = torch.empty(total, dtype=torch.uint8) if is_writer else None

    def run(mode: str) -> float:
        offset = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _name, tensor in engine.export_hf_named_params():
            nbytes = tensor.numel() * tensor.element_size()
            if is_writer:
                src = tensor.reshape(-1).view(torch.uint8)
                if mode == "dma":
                    buf[offset : offset + nbytes].copy_(src)
                else:  # pageable: host materialize first, then copy
                    host = tensor.to("cpu").contiguous().reshape(-1).view(torch.uint8)
                    pageable[offset : offset + nbytes].copy_(host)
            offset += nbytes
        if is_writer:
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    if rank == 0:
        gb = total / 1e9
        print(
            f"\n=== offload bench: model={os.path.basename(args.model)} "
            f"tp={args.tp} pp={args.pp} ep={args.ep} total={gb:.2f} GB ===",
            flush=True,
        )
    for mode in ("dma", "pageable"):
        ts = [run(mode) for _ in range(args.iters)]
        if rank == 0:
            best = min(ts)
            print(
                f"  {mode:9s}: best={best:.3f}s  ({total / 1e9 / best:.1f} GB/s)  "
                f"all={[round(x, 3) for x in ts]}",
                flush=True,
            )
        dist.barrier()

    engine.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
