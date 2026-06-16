# AstraFlow cross-platform verification: AMD (ROCm) ↔ NVIDIA (H100)

Bringing the `examples/math/qwen3-8b-m2po-full` recipe (Qwen3-8B math RL, M2PO,
ctx 16k, TCP weight transfer) up on **AMD MI300/MI325 (ROCm / gfx942)** and
verifying its training dynamics match the known-good **NVIDIA H100** run.

**TL;DR** — AstraFlow runs correctly on ROCm after two code adaptations plus a
ROCm Docker image. The one subtle, high-impact bug was an attention-backend
fallback (`sdpa`) that silently corrupted training under sequence packing;
switching back to `flash_attention_2` (with a Triton-AMD flash-attn) brings the
AMD run onto the same curve as H100.

---

## 1. Environments

| | AMD (this work) | NVIDIA H100 (reference) |
|---|---|---|
| GPU | MI325X, gfx942 (CDNA3), 8/node | H100, 8/node |
| Base | `lmsysorg/sglang:v0.5.12.post1-rocm720-mi30x` (Docker) | conda env |
| torch | 2.9.1+rocm7.2.0 | 2.8.0+cu128 |
| sglang | 0.5.12.post1 | 0.5.5.post1 |
| transformers | 5.6.1 | 4.57.1 |
| flash-attn | 2.8.3 (Triton-AMD backend) | 2.8.3 (CUDA) |
| astraflow | 0.1.1 (`316756b`) + this patch | 0.1.0 (`93517ce`) |
| layout | RaaS/SGLang dp=4 (GPU 0-3) + FSDP trainer dp=4 (GPU 4-7) | same |

Despite the version gap, the **training hyperparameters are identical** (batch
256, n_samples 8, lr 5e-6 constant, M2PO `m2_threshold=0.01`, `eps_clip=100`,
`kl_ctl=0`, `kl_penalty_coef=0.001`, `ppo_n_minibatches=4`, max_new_tokens 14000,
temperature 1.0, seed 1, reward_norm group / adv_norm batch). A config diff of the
two W&B runs shows 382/390 keys identical; the 8 differences are version metadata
plus the attention/ckpt/step-count choices documented here.

## 2. ROCm port (what it took to run at all)

The recipe's inference (SGLang) runs on ROCm out of the box via the official
SGLang ROCm image (uses **aiter** as the attention backend). The work was to
layer AstraFlow on top without disturbing the ROCm GPU stack, plus two code fixes:

1. **Docker image** (`docker/Dockerfile.rocm`): `FROM` the version-matched SGLang
   ROCm image; install AstraFlow's pure-python deps under a constraints file that
   pins the base image's `torch`/`sglang`/`torchvision`/etc. so pip cannot pull
   CUDA wheels over them. `megatron-core`/`mbridge`/`torchdata` are installed
   `--no-deps` (megatron-core 0.13.1 declares `numpy<2.0.0`, conflicting with the
   base's numpy 2.2.6; it runs fine on numpy 2.x). Helper scripts live in
   `docker/rocm/`. No-docker clusters can build the equivalent `.sqsh` with
   `examples/_common/build_astraflow_rocm.sh` (enroot/pyxis).
2. **Platform detection** (`astraflow/train_worker/platforms/__init__.py`): the
   detector only recognized `"NVIDIA"` in the device name and fell back to
   `UnknownPlatform` on AMD. Patched to return `CudaPlatform` when
   `torch.version.hip` is set (ROCm exposes the AMD GPU through the `torch.cuda`
   API; nccl→rccl). The RaaS-side detector already handled this.
3. **torch.compile on ROCm** (`astraflow/train_worker/utils/functional/vocab_parallel.py`):
   `_gather_logprobs`/`_gather_logprobs_entropy` are `torch.compile`d; inductor
   codegen of these reductions fails on gfx942 (torch 2.9), crashing the loop with
   a masked `InductorError: PicklingError`. Falls back to eager on ROCm
   (`ASTRAFLOW_FORCE_TORCH_COMPILE=1/0` overrides).

Launch wrapper: `examples/math/qwen3-8b-m2po-full/scripts/run_qwen3-8b-m2po-full_amd.sh`
(docker-run; flash_attention_2 + Triton-AMD flash-attn; checkpoint off by default).

## 3. The bug: `sdpa` silently corrupts packed-sequence training

During the port, the trainer had no `flash_attn` package on ROCm, so `attn_impl`
was overridden to `sdpa` — treated as a harmless perf fallback. It was not.

AstraFlow's FSDP engine **packs multiple sequences into one microbatch** and
passes `cu_seq_lens_q/k` (`fsdp_engine.py:1203`) with `attention_mask=None`. Only
transformers' `flash_attention_2` path honors `cu_seqlens` to keep attention
inside each sub-sequence. Under `sdpa`, those kwargs are ignored and a single
causal mask spans the whole packed buffer, so packed sub-sequences attend **across
boundaries** → the trainer's recomputed logits (and logprobs) are systematically
wrong.

### Diagnosis (W&B, step 1 — same weights as the rollout, so `old_logp` should ≈ `new_logp`)

| metric | AMD broken (`sdpa`) | AMD fixed (`flash_attention_2`) | H100 ref |
|---|---|---|---|
| `misc/old_logp/avg` vs `new_logp/avg` | −0.30 vs **−3.2** | −0.219 vs **−0.222** | ≈ equal |
| `importance_sampling/importance_weight/avg` | 0.41 | **0.9996** | 1.0000 |
| `importance_sampling/importance_weight/max` | up to **1.5e5** | 6.6 | 2.8 |
| `kl/approx_kl/avg` | −3.0 | **−0.0035** | −0.0006 |
| `clip/m2po_mean_m2` | ~17 | **0.0083** | 0.0022 |

The reward looked superficially fine (it comes from the rollout itself), but the
importance ratios fed into M2PO were garbage, so the **policy gradient was
effectively broken** — eval failed to improve and response length never grew.

## 4. The fix

Install a **Triton-AMD flash-attn** for the trainer and use `flash_attention_2`:

```dockerfile
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
RUN FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    pip install flash-attn==2.8.3 --no-build-isolation
```

The Triton-AMD backend provides `flash_attn_varlen_func` on gfx942 with **no CK/CUDA
compile** (kernels JIT via the base image's Triton; installs in ~8 s). Runtime sets
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. The recipe's default `attn_impl=flash_attention_2`
is restored (no `sdpa` override on ROCm).

## 5. Result: AMD now tracks H100

50-step side-by-side, identical config (`eval freq 10`, `recover off`). AMD reached
step 39/50 before a 4 h node allocation expired (5 eval bursts are expensive on AMD);
the trend is already conclusive.

**eval avg@4 (math500 | amc | aime24 | minerva)**

| step | AMD-fixed | H100 |
|---|---|---|
| 10 | 85.5 / 51.4 / 26.7 / 40.6 | 85.7 / 51.7 / 29.8 / 40.9 |
| 20 | 87.1 / 55.7 / 32.1 / 39.4 | 86.6 / 54.7 / 32.5 / 41.4 |
| 30 | 87.9 / 60.7 / 36.5 / 42.2 | 86.7 / 55.6 / 28.5 / 41.4 |
| 40 / 50 | (node timeout) | 87.6 / 54.9 · 86.9 / 57.8 |

**per-step means (steps 1–30)**

| run | reward_mean | seq_len (start→end) | IW_avg |
|---|---|---|---|
| AMD-fixed | 0.577 | 2142 → 1930 | **1.0000** |
| H100 | 0.534 | 2168 → 2152 | 1.0000 |
| AMD-broken (old) | 0.641 | **1364 → 1364** | **0.626** |

AMD-fixed and H100 sit on essentially the same eval curve (both climbing), with
matching importance weights (≈1.0) and response lengths (~2000). The old broken AMD
run is clearly distinguishable: IW 0.63, length stuck at 1364, eval flat.

W&B runs (project `liquid-ai/astraflow-math`):
- AMD fixed — `qwen3-8b-m2po-full-model0_373f28d2`
- H100 ref — `qwen3-8b-m2po-full-model0_b2a99e8f`
- AMD broken (for contrast) — `qwen3-8b-m2po-full-model0_9038f5e4`

## 6. Takeaways

- On ROCm, `flash_attention_2` is a **correctness requirement**, not a perf option,
  whenever the trainer packs sequences (varlen/`cu_seqlens`). The Triton-AMD
  flash-attn backend supplies it without a CK build.
- `old_logp` vs `new_logp` (equivalently `importance_weight ≈ 1` at version 0) is the
  fastest sanity check that an inference engine and a trainer agree — watch it first
  when porting RL across backends.
- Inference (SGLang/aiter) and the FSDP trainer (transformers + flash-attn) use
  independent attention paths; both must be varlen-correct for importance sampling
  to hold.
