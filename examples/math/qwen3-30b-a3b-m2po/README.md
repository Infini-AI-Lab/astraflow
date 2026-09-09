# Qwen3-30B-A3B Math RL — Megatron MoE backend + Rollout Routing Replay (R3)

Math RL recipe for **Qwen3-30B-A3B** (MoE: 48 layers, 128 experts, top-8)
on a single 8xB200 node, cloned from
[`qwen3-8b-megatron-delta`](../qwen3-8b-megatron-delta) with these
deliberate differences:

1. **MoE parallelism** — Megatron backend with TP=1, PP=1, DP=4, EP=4,
   ETP=1 on 4 trainer GPUs; SGLang DP=2 on 2 rollout GPUs.
1. **R3 (Rollout Routing Replay)** — the SGLang server records which
   experts each token was routed to during rollout, and the trainer
   replays exactly that routing in its forwards, closing the
   rollout/training MoE mismatch that destabilises RL on MoE models.
1. **Closed-loop rollout buffer** — `dataflow.buffer.max_buffered_samples`
   pauses prompt submission while two training batches are already
   buffered, so sample staleness stays at 2–5 model versions instead of
   climbing to ~22 (two open-loop runs plateaued or collapsed with
   staleness ~22).
1. **GRPO objective** (clip 0.2/0.28, lr 1e-6, 256 samples per step, no
   KL term) — the recipe both the R3 paper and miles validated on this
   model. The directory keeps its historical `m2po` name; M2PO is off.
1. **Full (not delta) TCP weight transfer** — MoE per-step delta density
   is unmeasured; measure before switching to delta.

This configuration trained for 800 steps: MATH-500 avg@4 rose from 86.95
(step 0) to 92.35 (step 800, pass@4 97.0), still rising at the final eval.

## The three R3 flags

R3 must be enabled on all three sides at once (all set in this recipe):

| Flag                                                     | File                             | What it does                                                                                                                          |
| -------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `sglang.enable_return_routed_experts: true`              | `yaml/raas_b200_1node.yaml`      | SGLang (>=0.5.13) allocates its routed-experts capturer; requests may ask for per-token expert indices.                               |
| `raas.models.model0.gconfig.return_routed_experts: true` | `yaml/experiment_b200_1node.yaml` | Every rollout request asks for the recorded expert indices; they ride the rollout record into the training batch as `routed_experts`. |
| `trainer_base.actor.megatron.moe_router_replay: true`    | `yaml/experiment_b200_1node.yaml` | The trainer's MoE router replays the recorded top-8 expert indices in every training forward instead of re-deciding routing.          |

Supporting settings that are **not optional** with R3:

- `sglang.chunked_prefill_size: 32768` — sizes the capturer's device
  buffer (`max(chunked_prefill_size, max_running_requests)` rows per DP
  rank) and must be > 0 (chunking bounds each prefill batch); the
  AstraFlow default `-1` is refused.
- `sglang.moe_runner_backend: triton` — an explicit capturing backend is
  required. The capture hook lives in `select_experts`, which the
  `flashinfer_trtllm` backend that `auto` picks on Blackwell bypasses:
  the server starts, reports healthy, and records nothing.
- `actor.megatron.use_deterministic_algorithms: true` — the replayed
  forward must be reproducible.

Speculative decoding, hierarchical cache and PD disaggregation are not
exposed by `SGLangConfig` (there is no passthrough for extra server args). The
radix cache is compatible with R3 capture (the capturer is indexed by KV
slot and every weight update flushes the cache; verified by probe), so
`disable_radix_cache` may be left at its AstraFlow default (`true`, as
here) or set to `false`.

## GPU layout (one node, 6 of 8 B200s)

| Component                 | GPUs       | Parallelism                   |
| ------------------------- | ---------- | ----------------------------- |
| AstraFlow HTTP service    | none (CPU) | —                             |
| Trainer model0 (Megatron) | 0-3        | TP=1, PP=1, DP=4, EP=4, ETP=1 |
| RaaS (SGLang, model0)     | 4-5        | DP=2, TP=1                    |

World size = tp x pp x dp = 4; EP=4 nests inside the DP domain (32 experts
per rank). Expert parameters do not shard (dp/ep = 1), so the optimizer
state only fits with the precision-aware settings in the experiment YAML
(measured per GPU, steady state):

| Buffer              | all fp32 (OOMs) | recipe        |
| ------------------- | --------------- | ------------- |
| bf16 params         | 16.4 GiB        | 16.4 GiB      |
| DDP grad buffer     | 32.7 GiB        | 16.4 GiB bf16 |
| fp32 master (shard) | 28.4 GiB        | 28.4 GiB      |
| exp_avg (shard)     | 28.4 GiB        | 14.2 GiB bf16 |
| exp_avg_sq (shard)  | 28.4 GiB        | 14.2 GiB bf16 |
| total               | 134.3 GiB       | 89.6 GiB      |

Weights are 56.9 GiB bf16; a full sync is a loopback TCP transfer of
~50 s plus ~15 s of SGLang load, against a 55–110 s training step, so
queued weight updates are coalesced to the newest version.

## Environment prerequisites

- **Transformer Engine** must be importable: mbridge builds every Megatron
  model with the Transformer Engine layer spec (`llm_bridge.py` passes
  `use_transformer_engine=True`); build from source into the env if
  absent — see `docs/en/get-started/installation.md`.
- **mbridge / transformers 5.x**: transformers 5 moved `rope_theta` into
  `rope_parameters`; mbridge 0.1.0 still reads `hf_config.rope_theta`.
  The Megatron engine backfills it right after `AutoBridge.from_pretrained`
  (`astraflow/train_worker/engine/megatron_engine.py`).
- **/dev/shm**: the sender double-buffers the full HF byte layout —
  budget >=2x57 GiB.
- **sglang >= 0.5.13** (native `--enable-return-routed-experts`).
- **Disk**: an HF save is 56.9 GiB and the recipe saves every 100 steps;
  `experiment.fileroot` must point at a filesystem with room for them.

## Run

Start the three components in order and wait for each to report ready:

```bash
# Terminal 1 (CPU): data service
bash examples/math/qwen3-30b-a3b-m2po/scripts/1_astraflow.sh

# Terminal 2 (GPUs 4,5): SGLang + R3 capture + TCP receiver
bash examples/math/qwen3-30b-a3b-m2po/scripts/2_raas.sh

# Terminal 3 (GPUs 0-3): Megatron trainer
bash examples/math/qwen3-30b-a3b-m2po/scripts/3_trainer_model0.sh
```

`SERVICE_CUDA_VISIBLE_DEVICES` and `TRAINER_MODEL0_GPUS` override the GPU
sets; `ASTRAFLOW_PORT`, `RAAS_PORT` and `WEIGHT_TRANSFER_HTTP_PORT_MODEL0`
the ports.

## What to watch

- First weight sync: ~2 min (the first pull competes with model loading);
  steady state ~76 s. The sync timeout (`WEIGHT_SYNC_TIMEOUT_SEC`) is 300 s.
- `buffer/evicted` and `buffer/skipped_stale` must stay 0 and
  `sample_staleness_*` at 2–5: the closed loop is working. If the trainer
  logs "Waiting for data" every step, raise `rollout.max_concurrent_rollouts`
  in the RaaS YAML (in-flight generation is the supply bound), not
  `max_staleness`.
- RaaS log lines `notify_version: ... already loaded (local=N) after
  acquiring lock, skipping` (and `requested v=N, sender served v=M`) mean
  weight-update coalescing is working.
- RaaS: a `ValueError: cannot reshape array` in `agenerate` means the
  capture path returned a payload that does not match the forwarded
  positions; the trainer refuses a wrong column count at install.
- Trainer: a `RoutingReplayError` means token/routing misalignment; a
  `KeyError: 'routed_experts'` at step 1 means the rollout was not
  capturing.

## Known limitations

- `recover.mode` is `disabled`: a complete distributed checkpoint fails
  to load (`'Metadata' object has no attribute 'mcore_data'`), so HF
  saves (`saver.freq_steps`) are the resume path. Resuming means
  relaunching the whole stack from the saved HF weights: restarting only
  the trainer resets the version counter to 0, and RaaS silently ignores
  any version <= the one it already holds (`notify_version` answers ok,
  `pulled=false`) — training then continues with NO error against frozen
  rollout weights, so always relaunch the whole stack.
- Context is 4096 (prompt <=1024 + 3000 new tokens). Raise
  `sglang.context_length`, `gconfig.max_new_tokens`,
  `rollout_dataset.max_length` and `mb_spec.max_tokens_per_mb` together.
  R3 adds 1.5 KiB per token of rollout payload (48 layers x top-8 int32).
