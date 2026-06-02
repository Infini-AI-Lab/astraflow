"""Equivalence test for Megatron -> HF weight export.

Loads an HF checkpoint into a Megatron GPTModel under a chosen parallel
strategy, exports it back to HF via ``export_hf_named_params``, and asserts
the reconstructed tensors match the original HF safetensors bit-for-bit
(bf16). This is the PR1 acceptance gate.

Run (torchrun, multi-GPU):
    torchrun --nproc_per_node=<tp*pp*ep*dp> \
        astraflow/train_worker/models/mcore/tests/test_hf_export_equiv.py \
        --model /shared/models/Qwen3-0.6B --tp 2 --pp 1 --ep 1

Exit code 0 = all tensors match. Non-zero = mismatch (details on rank 0).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist


def _load_reference_hf(model_path: str) -> dict[str, torch.Tensor]:
    """Load the original HF checkpoint tensors (bf16) from safetensors."""
    import glob

    from safetensors.torch import load_file

    ref: dict[str, torch.Tensor] = {}
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {model_path}")
    for f in files:
        ref.update(load_file(f))
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--atol", type=float, default=0.0, help="0 = exact bf16 match")
    args = ap.parse_args()

    from astraflow.train_worker.api.alloc_mode import ParallelStrategy
    from astraflow.train_worker.api.cli_args import TrainEngineConfig
    from astraflow.train_worker.engine.megatron_engine import MegatronEngine
    from astraflow.train_worker.models.mcore.weight_export import (
        export_hf_named_params,
    )

    world = int(os.environ["WORLD_SIZE"])
    dp = world // (args.tp * args.pp * args.ep)
    assert dp >= 1, (
        f"world={world} too small for tp*pp*ep={args.tp * args.pp * args.ep}"
    )

    cfg = TrainEngineConfig(path=args.model, dtype="bfloat16")
    # No optimizer -> inference-only engine, faster init.
    engine = MegatronEngine(cfg)
    strategy = ParallelStrategy(
        data_parallel_size=dp,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        expert_parallel_size=args.ep,
    )
    engine.create_process_group(parallel_strategy=strategy)

    from astraflow.train_worker.api.io_struct import FinetuneSpec

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1)
    engine.initialize(addr=None, ft_spec=ft_spec)

    rank = dist.get_rank()
    is_writer = rank == 0

    ref = _load_reference_hf(args.model) if is_writer else None

    n_checked = 0
    n_mismatch = 0
    seen: set[str] = set()
    for name, tensor in export_hf_named_params(engine.bridge, engine.model):
        if not is_writer:
            continue
        seen.add(name)
        if name not in ref:
            print(f"[FAIL] exported tensor not in reference: {name}", flush=True)
            n_mismatch += 1
            continue
        r = ref[name].to(torch.bfloat16)
        t = tensor.to(torch.bfloat16)
        if list(t.shape) != list(r.shape):
            print(
                f"[FAIL] shape {name}: export {list(t.shape)} vs ref {list(r.shape)}",
                flush=True,
            )
            n_mismatch += 1
            continue
        if args.atol == 0.0:
            ok = torch.equal(t, r)
        else:
            ok = torch.allclose(t.float(), r.float(), atol=args.atol, rtol=0)
        if not ok:
            md = (t.float() - r.float()).abs().max().item()
            print(f"[FAIL] values {name}: max|diff|={md:.3e}", flush=True)
            n_mismatch += 1
        n_checked += 1

    if is_writer:
        import json

        with open(os.path.join(args.model, "config.json")) as f:
            tie = json.load(f).get("tie_word_embeddings", False)
        missing = set(ref.keys()) - seen
        # Benign non-exports:
        #  - rotary/inv_freq buffers (not weights);
        #  - lm_head.weight when embeddings are tied (mbridge emits only
        #    embed_tokens; the inference engine ties internally).
        benign = {k for k in missing if "rotary" in k or "inv_freq" in k}
        if tie and "lm_head.weight" in missing:
            benign.add("lm_head.weight")
        hard_missing = missing - benign
        print(
            f"\n=== export equivalence: checked={n_checked} "
            f"mismatch={n_mismatch} missing={len(hard_missing)} "
            f"benign_missing={len(benign)} ===",
            flush=True,
        )
        if hard_missing:
            print(
                f"[FAIL] reference keys never exported: {sorted(hard_missing)[:10]}",
                flush=True,
            )
        result = 0 if (n_mismatch == 0 and not hard_missing) else 1
    else:
        result = 0

    res_t = torch.tensor([result], device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    dist.all_reduce(res_t, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print("PASS" if res_t.item() == 0 else "FAIL", flush=True)
    engine.destroy()
    return int(res_t.item())


if __name__ == "__main__":
    sys.exit(main())
