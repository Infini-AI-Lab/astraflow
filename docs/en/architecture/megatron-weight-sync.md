# Megatron Weight Synchronization

This page describes how the Megatron-LM training backend exports its
weights to RaaS, and the invariants that keep the **sparse / delta**
weight-update path correct under tensor (TP), pipeline (PP), expert
(EP), expert-tensor (ETP), virtual-pipeline (VPP), and context (CP)
parallelism.

It complements [WeightManager](weight-manager.md) and
[Delta Weight Transfer](delta-weight-transfer.md), which describe the
backend-agnostic transport. **Read those first.**

## The problem

Megatron stores each parameter sharded across TP/PP/EP ranks, fused
(QKV in one `linear_qkv`, gate+up in one `linear_fc1`), and vocab-padded
— a layout that bears no resemblance to the HuggingFace checkpoint names
and byte layout that SGLang / vLLM expect (`model.layers.N.self_attn.q_proj.weight`,
…). RaaS only understands HF layout.

The transport layer (`WeightManager` + sender agent + RaaS receiver) is
deliberately **backend-agnostic**: it moves an opaque, fixed-order CPU
byte buffer and, in delta mode, ships only the bytes that changed
between two versions of that buffer. For this to be correct, the bytes
in the buffer **must be in the same layout that RaaS applies them in**.

For FSDP this is automatic — the buffer already holds HF-layout tensors.
For Megatron it is the central design constraint.

## Design invariant

> **The trainer always writes HF-named, HF-layout, full-model tensors into
> the transfer buffer. Sparsity / delta is always computed in HF byte
> space, over a double buffer. The RaaS receive path never sees a
> backend-specific layout.**

Concretely, the Megatron backend reconstructs the global model from its
sharded layout and converts it to HF on the GPU, **before** anything
reaches the transfer buffer. The sender agent and RaaS receiver then
treat Megatron exactly like FSDP.

This makes the delta correct **by construction**: both the old and new
buffer halves hold HF bytes, so a bytewise diff produces indices that
the receiver can scatter directly into its HF buffer.

> ⚠️ **Historical bug (fixed by this design).** An earlier Megatron path
> wrote *raw mcore-layout shards* into the buffer and reassembled to HF in
> a separate, single-buffered region in the sender — but computed the
> delta over the *mcore-layout* buffer. mcore byte offsets ≠ HF byte
> offsets (fused QKV vs split, fused gate/up vs split, vocab padding), so
> applying an mcore-space delta to RaaS's HF-space base silently corrupted
> weights. Always diff in HF space.

## The per-tensor generator

The reconstruction is a **streaming generator** that yields
`(hf_name, full_unsharded_cpu_tensor)` one bucket at a time. Only one
bucket of fully-gathered tensors is alive at any moment, so a 100B / MoE
model never materializes in full (no OOM).

```python
# astraflow/train_worker/models/mcore/weight_export.py
def export_hf_named_params(
    models,            # _MegatronModelList: VPP chunks, DDP-wrapped
    tf_config,
    hf_config,
    bucket_bytes,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (hf_name, full HF-layout CPU tensor) bucket by bucket."""
```

Per parameter, in order, it performs the minimal collectives:

1. **Naming + PP/EP offsets** — `utils.megatron.get_named_parameters`
   already maps local mcore names to *global* names (adds the PP layer
   offset and EP expert offset), iterating VPP chunks.
2. **PP gather** — `all_gather_object` of metadata across the pipeline
   group, then broadcast each owner stage's tensor so the DP-head rank
   set collectively holds every global parameter. Embeddings live on the
   first stage, `output_layer` / final norm on the last.
3. **TP / ETP gather** — `utils.megatron.all_gather_param` all-gathers
   the shards along `partition_dim` and concatenates, handling the GLU
   `linear_fc1` stride-2 rechunk and the grouped-MoE `linear_fc2`
   `partition_dim` 0→1 quirk.
4. **EP gather** — for `.experts.` params, all-gather across the expert
   group and rewrite local→global expert id.
5. **mcore → HF convert** — `utils.megatron.convert_to_hf` splits QKV
   (GQA-aware), splits gate/up, renames, and drops vocab padding.
6. **Bucket + stream** — group converted tensors until `bucket_bytes`
   (measured post-gather), `yield`, then free before the next bucket.

This is the same abstraction verl (`per_tensor_generator`) and slime
(`HfWeightIteratorDirect`) converged on; the difference is the consumer.

## How it plugs into WeightManager

verl / slime push the generator's tensors GPU→GPU (NCCL / CUDA-IPC) into
the inference engine. AstraFlow instead **writes them into the CPU
double buffer** that the sender agent TCP-pulls:

```
optimizer.step()
   │
   ▼
WeightManager.offload(export_hf_named_params(...), version, ...)
   │   DP-head ranks write each (hf_name, tensor) into the INACTIVE
   │   half of the HF double buffer, in fixed order; non-heads barrier.
   ▼
notify_buffer_ready  ──►  sender swaps active/inactive
   │
   ▼                       sender._compute_delta() diffs HF-inactive vs
   │                       HF-active  → indices in HF space  ✓
   ▼
RaaS pulls full or delta (unchanged from FSDP)
```

Buffer sizing comes from `MegatronEngine.get_hf_weight_metadata()` — a
metadata-only dry run of the generator that returns the ordered
`[(hf_name, shape, dtype), …]` list. This is the same `tensors_meta`
the RaaS receiver uses to pre-allocate, so both ends agree on layout and
order.

## Rank participation

Only **data-parallel head** ranks write the buffer (one writer per
model-parallel group), mirroring FSDP's primary-replica rule. The
TP/PP/EP gathers happen via collectives *before* the write, so every
DP-head holds the full HF model and writes it once. Other ranks only
participate in the gathers and the post-write barrier.

## Configuration

```yaml
trainer:
  engine:
    backend: megatron
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    expert_parallel_size: 1
  actor:
    megatron:
      weight_export_bucket_bytes: 536870912   # 512 MiB gather bucket
```

`backend: megatron` is auto-selected when `pipeline_parallel_size > 1`
or `expert_parallel_size > 1`.

## Invariants checklist (for reviewers)

1. Trainer hands WeightManager **HF-named, HF-layout, full-model**
   tensors. Backend differences end at `export_hf_named_params`.
2. Sparsity / delta is computed in **HF byte space**, on a double buffer.
3. **One bucket** of gathered tensors alive at a time — never
   `full_tensor()` the whole model.
4. Only **DP-head** ranks write the buffer.
5. The **RaaS receive path is unchanged** between FSDP and Megatron.
