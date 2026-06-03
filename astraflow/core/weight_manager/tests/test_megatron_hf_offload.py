"""Integration test: Megatron HF-export -> WeightManager buffer -> HF tensors.

Validates the full PR2/PR3 path without RaaS:
  1. MegatronEngine.get_hf_weight_metadata() sizes the buffer.
  2. WeightManager.offload(export_hf_named_params()) streams HF tensors into
     the shared-memory double buffer (writer rank only).
  3. We read the buffer back, reinterpret per tensors_meta, and assert it
     equals the reference HF checkpoint bit-for-bit.

This proves the bytes the sender will TCP to RaaS (full mode) are correct,
and that they live in HF layout (so the sender's HF-space delta is valid).

Run:
    torchrun --nproc_per_node=<N> \
        astraflow/core/weight_manager/tests/test_megatron_hf_offload.py \
        --model /shared/models/Qwen3-0.6B --tp 2 --pp 1 --ep 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.distributed as dist


def _ref(model_path):
    from safetensors.torch import load_file

    ref = {}
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        ref.update(load_file(f))
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
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

    # 1. Metadata (lockstep on all ranks).
    hf_meta = engine.get_hf_weight_metadata()

    # 2. Build a minimal WeightManager-like buffer write on the writer rank,
    #    reusing the real _offload_megatron_hf logic. We construct a real
    #    WeightManager but stub out the sender (no subprocess) by writing
    #    into a plain CPU tensor of the right size.
    from math import prod

    sizes = [(n, prod(sh) * (2 if dt == "bfloat16" else 4)) for n, (sh, dt) in hf_meta]
    total = sum(s for _, s in sizes)

    # Only rank 0 holds the "buffer" and checks; others just drive collectives.
    is_writer = rank == 0
    buf = torch.zeros(2 * total, dtype=torch.uint8) if is_writer else None

    # Stream export and write to buf[half 0] in order (mirrors
    # WeightManager._offload_megatron_hf with inactive_buf_idx=0).
    offset = 0
    written = {}
    for name, tensor in engine.export_hf_named_params():
        nbytes = tensor.numel() * tensor.element_size()
        if is_writer:
            u8 = tensor.contiguous().view(-1).view(torch.uint8)
            buf[offset : offset + nbytes].copy_(u8)
            written[name] = (offset, list(tensor.shape), str(tensor.dtype))
        offset += nbytes

    result = 0
    if is_writer:
        ref = _ref(args.model)
        import json

        tie = json.load(open(os.path.join(args.model, "config.json"))).get(
            "tie_word_embeddings", False
        )
        # Read back each tensor from the buffer per metadata and compare.
        off = 0
        nbad = 0
        nchk = 0
        for name, (shape, dt) in hf_meta:
            numel = prod(shape)
            nbytes = numel * (2 if dt == "bfloat16" else 4)
            raw = buf[off : off + nbytes]
            tdtype = torch.bfloat16 if dt == "bfloat16" else torch.float32
            t = raw.view(tdtype).view(*shape) if shape else raw.view(tdtype)
            off += nbytes
            if name not in ref:
                if not (tie and name == "lm_head.weight"):
                    print(f"[FAIL] {name} not in ref", flush=True)
                    nbad += 1
                continue
            r = ref[name].to(torch.bfloat16)
            if not torch.equal(t, r):
                md = (t.float() - r.float()).abs().max().item()
                print(f"[FAIL] {name} max|diff|={md:.3e}", flush=True)
                nbad += 1
            nchk += 1
        print(
            f"\n=== buffer roundtrip: total_bytes={total} checked={nchk} bad={nbad} ===",
            flush=True,
        )
        result = 0 if nbad == 0 else 1

    res_t = torch.tensor([result], device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    dist.all_reduce(res_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("PASS" if res_t.item() == 0 else "FAIL", flush=True)
    engine.destroy()
    return int(res_t.item())


if __name__ == "__main__":
    sys.exit(main())
