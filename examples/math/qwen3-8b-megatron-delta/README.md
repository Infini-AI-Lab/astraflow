# Qwen3-8B Math RL — Megatron backend, delta TCP weight transfer

Same math RL recipe as [`qwen3-8b-m2po-delta`](../qwen3-8b-m2po-delta) (M2PO,
DeepScaleR data, ctx 16k, lr 5e-6, sparse delta weight sync) but the trainer
uses the **Megatron-LM backend** instead of FSDP. The only difference is the
`trainer_base.engine` block:

```yaml
engine:
  backend: megatron
  data_parallel_size: 1
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
```

This makes it a clean FSDP-vs-Megatron A/B: identical data, algorithm, and
weight-transfer path, so reward curves should track each other.

## How weight sync works (Megatron)

The trainer reconstructs the global model from Megatron's TP/PP/EP/VPP
layout into HuggingFace-named tensors (via `export_hf_named_params`,
backed by mbridge) and streams them into the CPU transfer buffer. Because
the buffer holds HF-layout bytes, the sparse **delta** is computed in HF
space and the RaaS receive path is identical to FSDP. See
[`docs/en/architecture/megatron-weight-sync.md`](../../../docs/en/architecture/megatron-weight-sync.md).

## GPU layout (8 GPUs, single node)

| Component | GPUs | Parallelism |
|-----------|------|-------------|
| RaaS (SGLang, model0) | 0,1,2,3 | DP=4 |
| Trainer model0 (Megatron) | 4,5,6,7 | TP=4 |

## Docker

This recipe uses the Megatron backend, so it needs the Megatron image (Transformer
Engine + apex), **not** the default FSDP one:

```bash
docker pull astraflowai/astraflow:v0.1.1.megatron
```

The plain `astraflowai/astraflow:v0.1.1` image is FSDP-only and lacks these deps. See
[`docker/README.md`](../../../docker/README.md).

## Run

```bash
bash examples/math/qwen3-8b-megatron-delta/scripts/run_qwen3-8b-megatron-delta.sh
```

Or launch the three components separately (terminals 1/2/3):

```bash
bash examples/math/qwen3-8b-megatron-delta/scripts/1_astraflow.sh
bash examples/math/qwen3-8b-megatron-delta/scripts/2_raas.sh
bash examples/math/qwen3-8b-megatron-delta/scripts/3_trainer_model0.sh
```

## Scaling to PP / MoE

For pipeline or expert parallelism (and MoE models), set the corresponding
sizes in the `engine` block, e.g. `pipeline_parallel_size: 2` or
`expert_parallel_size: 2`. The backend auto-selects Megatron when `pp>1` or
`ep>1`. Ensure `data_parallel_size * tensor_parallel_size *
pipeline_parallel_size` equals the number of trainer GPUs.
