"""Correctness test for the direct-DMA offload optimization (#1).

Verifies that copying gathered HF tensors *directly* from GPU into a pinned
host buffer (the new path) produces byte-identical buffer contents to the
old path (materialize each tensor in pageable host memory via .to("cpu"),
then copy), AND that both match the HF reference checkpoint.

Three buffers are filled from the SAME export pass (so any nondeterminism in
the gather is shared, not a false diff):
  - buf_new  : new path — self._buffer pinned, src GPU tensor uint8 view, D2H copy
  - buf_old  : old path — tensor.to("cpu").contiguous() then uint8 copy
  - ref      : original HF safetensors bytes (ground truth, writer rank only)

PASS iff buf_new == buf_old (byte-exact) AND buf_new == ref (byte-exact).

Run:
    torchrun --nproc_per_node=<N> \
        astraflow/core/weight_manager/tests/test_direct_dma_offload.py \
        --model /shared/models/Qwen3-0.6B --tp 2 --pp 1 --ep 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.distributed as dist


def _make_pinned_buffer(nbytes: int) -> torch.Tensor:
    """A host uint8 tensor, page-locked + cudaHostRegister'd like the real shm buffer."""
    buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    # The production buffer is cudaHostRegister'd; pin_memory already gives a
    # page-locked allocation that the D2H DMA engine can use, which is the
    # property under test.
    return buf


def _ref_bytes(model_path: str, hf_meta: list) -> bytes:
    from safetensors.torch import load_file

    ref = {}
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        ref.update(load_file(f))
    # Lay out in the exact hf_meta order, as bytes.
    chunks = []
    import json as _json

    tie = _json.load(open(os.path.join(model_path, "config.json"))).get(
        "tie_word_embeddings", False
    )
    for name, (shape, dt) in hf_meta:
        if name not in ref:
            if tie and name == "lm_head.weight" and "model.embed_tokens.weight" in ref:
                t = ref["model.embed_tokens.weight"]
            else:
                raise KeyError(f"ref missing {name}")
        else:
            t = ref[name]
        t = t.to(torch.bfloat16 if dt == "bfloat16" else torch.float32).contiguous()
        chunks.append(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return b"".join(chunks)


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
    is_writer = rank == 0

    # Metadata (lockstep) → total byte size.
    hf_meta = engine.get_hf_weight_metadata()
    from math import prod

    total = sum(prod(sh) * (2 if dt == "bfloat16" else 4) for _, (sh, dt) in hf_meta)

    buf_new = _make_pinned_buffer(total) if is_writer else None
    buf_old = _make_pinned_buffer(total) if is_writer else None

    # Single export pass; fill both buffers from the SAME yielded tensors.
    offset = 0
    for _name, tensor in engine.export_hf_named_params():  # to_cpu=False (GPU)
        nbytes = tensor.numel() * tensor.element_size()
        if is_writer:
            assert tensor.is_cuda, "export must yield GPU tensors for the DMA path"
            src_u8 = tensor.reshape(-1).view(torch.uint8)
            # NEW path: direct D2H into pinned buffer (matches production:
            # non_blocking=True + a single synchronize() after the loop).
            buf_new[offset : offset + nbytes].copy_(src_u8, non_blocking=True)
            # OLD path: pageable host materialization first, then copy.
            host = tensor.to("cpu").contiguous().reshape(-1).view(torch.uint8)
            buf_old[offset : offset + nbytes].copy_(host)
        offset += nbytes
    if is_writer:
        torch.cuda.synchronize()

    result = 0
    if is_writer:
        nb_new = bytes(buf_new.numpy().tobytes())
        nb_old = bytes(buf_old.numpy().tobytes())
        eq_paths = nb_new == nb_old
        ref = _ref_bytes(args.model, hf_meta)
        eq_ref = nb_new == ref
        # Locate first differing byte for diagnostics.
        first_diff_paths = (
            -1
            if eq_paths
            else next(
                (
                    i
                    for i in range(min(len(nb_new), len(nb_old)))
                    if nb_new[i] != nb_old[i]
                ),
                -1,
            )
        )
        first_diff_ref = (
            -1
            if eq_ref
            else next(
                (i for i in range(min(len(nb_new), len(ref))) if nb_new[i] != ref[i]),
                -1,
            )
        )
        print(
            f"\n=== direct-DMA offload: total_bytes={total} "
            f"new==old:{eq_paths} (first_diff={first_diff_paths}) "
            f"new==ref:{eq_ref} (first_diff={first_diff_ref}) "
            f"len(ref)={len(ref)} ===",
            flush=True,
        )
        result = 0 if (eq_paths and eq_ref) else 1

    res_t = torch.tensor([result], device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    dist.all_reduce(res_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("PASS" if res_t.item() == 0 else "FAIL", flush=True)
    engine.destroy()
    return int(res_t.item())


if __name__ == "__main__":
    sys.exit(main())
