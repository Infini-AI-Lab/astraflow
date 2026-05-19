# WeightManager

The WeightManager (`astraflow/weight_manager/`) is an independent component
that handles all weight transfer between Trainer and RaaS.

## Design Principle: Independent Transport Layer

WeightManager is a **shared component** — both Trainer and RaaS import
from it, but neither depends on the other. This keeps the three swappable
components (Trainer, RaaS, WeightManager) cleanly separated:

```
train_worker/ ──imports──► weight_manager/
raas/         ──imports──► weight_manager/transfer/

train_worker/ ✗ raas/      (no cross-dependency)
raas/ ✗ train_worker/      (no cross-dependency)
```

The Trainer's integration surface is a single call:

```python
wm.offload(model.named_parameters(), version, rank, world_size)
```

## Key Components

- **`WeightManager`** — Main class that owns buffer allocation, GPU→CPU
  copy (shard-direct and all-gather paths), sender agent lifecycle, and
  double-buffer swap. The trainer calls `offload()` once per step.
- **Transfer modes** — `POST /request_transfer` accepts a `mode` of
  `full` (sends the entire model) or `delta` (sends only changed
  elements, ~1-2% of the model). See
  [Delta Weight Transfer](delta-weight-transfer.md) for details.
- **`TransferAgent`** — Subprocess on the Trainer side that exposes HTTP
  endpoints and serves TCP weight pulls from RaaS.
- **`TCPTransferEngine`** — TCP engine with 6 parallel streams and
  `sendfile()` zero-copy. Used by both sender (Trainer) and receiver (RaaS).
- **`TransferBuffer`** — CPU byte buffer for receiving weights on the RaaS side.

## Weight Sync Flow

1. Trainer calls `wm.offload()` — GPU weights are copied to the inactive
   half of a shared-memory double buffer, then the buffer index is swapped.
2. Trainer notifies Dataflow via `POST /notify_version`.
3. Dataflow's Python-side version barrier waits for all model_ids
   (multi-model), then fans out one `POST /notify_version` per model per
   RaaS instance. Each call carries `{model_id, version, sender_endpoint}`.
4. On each RaaS, the manager acquires a per-model lock and calls
   `POST /request_transfer` on the sender agent to pull weights via
   6 parallel TCP streams.
5. RaaS saves received bytes as safetensors to `/dev/shm`, pauses
   inference engines, loads weights, and resumes.

```
Trainer                   WeightManager              SenderAgent           RaaS
  │                            │                         │                  │
  │ offload(params, v)         │                         │                  │
  │───────────────────────────►│                         │                  │
  │                            │ GPU→CPU shard copy      │                  │
  │                            │ swap buffer index       │                  │
  │                            │ mp.Queue: buffer_ready  │                  │
  │                            │────────────────────────►│                  │
  │◄───────────────────────────│ (returns)               │                  │
  │ next train_step ...        │                         │                  │
  │                            │                         │◄── /request_transfer
  │                            │                         │ TCP send ───────►│
  │                            │                         │ ZMQ done ───────►│
  │                            │                         │         safetensors
  │                            │                         │         load + resume
```

## Timeline: What Blocks What

The key insight is that **weight transfer never blocks the trainer GPU
directly**. The trainer's GPU is only blocked during the `offload()` call
(GPU→CPU copy). Everything else — TCP transfer, RaaS weight loading —
happens asynchronously. The trainer is gated by **data availability**
(the `get_batch` call), not weight transfer.

### Single Training Step

```
Time ──────────────────────────────────────────────────────────────────────────►

TRAINER (GPU)
  │ train_step N               │offload│save│notify│ get_batch  │ train_step N+1
  │[===========================][=====]│chkp│async │ (blocks    │[===============
  │ forward/backward/optim     │GPU→CPU│    │      │  until     │
  │                            │~2s   │    │      │  service   │
  │                            │      │    │      │  version   │
  │                            │  not blocked ──► │  catches   │
  │                            │  (trainer does   │  up)       │
  │                            │   save, log,     │[==========]│
  │                            │   etc.)          │ blocked by │
  │                            │                  │ data, not  │
  │                            │                  │ by weight  │
  │                            │                  │ transfer   │

SENDER AGENT (CPU subprocess)
  │ · · · · · · · · · · · · · ·│swap│ack│ delta compute (async)  │done│ · · · ·
  │         idle               │idx │   │[======================]│evt │  idle
  │                            │    │   │ compare halves (numpy) │    │
  │                            │    │   │ ~1.8s for 1.7B         │    │

DATAFLOW SERVICE (CPU)
  │ · · · · · · · · · · · · · · · · │version│RaaS weight load  │version  │serve
  │                                 │barrier│[================]│updated! │batch
  │                                 │       │notify_version (per model)   │─────►
  │                                 │       │+ TCP pull + load │         │

RAAS (GPU)
  │ generating rollouts · · · · · · · · · · │pull│pause│load│resume│ generating
  │[rollout][rollout][rollout] · · · · · · ·│TCP │[===]│[==]│      │[rollout]..
  │                                         │~3s │     │~5s │      │
```

**Key observations:**

- **Trainer GPU blocks only during `offload()`** (~0.5s for 1.7B, shard copy).
  After offload, the trainer saves checkpoints, logs stats, waits for
  async delta to finish, then fires `notify_version_async`.
- **Delta computation is async** (~1.8s for 1.7B). Runs in the sender
  agent subprocess, overlapped with trainer checkpoint/logging. The
  trainer calls `wait_delta_ready()` before `notify_version` to ensure
  delta is ready when RaaS pulls.
- **`get_batch` is the real synchronization point.** Dataflow won't serve
  a batch until `service_version >= trainer_version`. The service version
  updates only after the RaaS weight load completes. So the trainer
  blocks on data availability, not on weight transfer directly.
- **RaaS blocks during `pause → load → resume`** (~3.7s). During this
  window it cannot serve rollout requests. The TCP pull itself does NOT
  block inference — RaaS pulls into a separate buffer first.
- **Double buffer eliminates trainer↔TCP contention.** The trainer writes
  to one half while the sender serves the other half. A guard barrier
  at the start of `offload()` ensures the previous async delta has
  finished reading before overwriting.

### Multi-Model Timeline

With two models (e.g. actor + verifier), each trainer independently
offloads weights and fires `notify_version_async`. The version barrier
and weight loading happen on the Dataflow side. Trainers are only
blocked when they request the next batch:

```
Time ──────────────────────────────────────────────────────────────────────────►

TRAINER model0 (actor)
  │ train_step       │offload│save│notify│ get_batch(v=N+1)  │ train_step N+1
  │[=================][=====]│chkp│async │ blocks until      │[===============
  │                          │    │      │ service catches up │
  │                          │ free ───► │[==================]│
  │                          │           │ blocked by data    │

TRAINER model1 (verifier)
  │ train_step (slower)          │offload│save│notify│ get_batch(v=N+1)  │ ..
  │[=============================][=====]│chkp│async │ blocks until      │[==
  │                                      │    │      │ service catches up│
  │                                      │ free ──► │[=================]│

DATAFLOW SERVICE
  │ · · · · · · · · · · · · · · · · · · ·│barrier met!│RaaS load  │version │
  │                                      │(both v=N+1)│[=========]│updated!│
  │                                      │            │           │serve   │
  │                                      │            │           │batches │

RAAS
  │ generating · · · · · · · · · · · · · · · · · · · ·│pull │pause│load│resume
  │                                                    │both │[===]│both│
  │                                                    │mdls │     │mdls│
  │                                                    ◄── batches served ──►
```

**Key observations:**

- **Neither trainer blocks on weight transfer.** Both fire
  `notify_version_async` and continue with checkpoint/logging work.
- **The version barrier is trainer-to-trainer synchronization**, not
  weight transfer synchronization. It ensures all model_ids reach the
  same version before triggering the RaaS weight load.
- **`get_batch` is the gating point.** Each trainer blocks on its next
  batch request until Dataflow's `service_version` catches up (after
  the version barrier + RaaS load completes).
- **Trainer GPU is free** between `offload()` and `get_batch()` — this
  window is used for checkpoint saving, wandb logging, and other I/O.

### Eval Steps (Synchronous Exception)

On eval steps, `notify_version` is called **synchronously** because the
trainer needs eval results before continuing. This is the only case where
the trainer explicitly waits for the weight load + eval to complete:

```
  │offload│save│notify_version(sync, eval)         │ eval results │ next step
  │[=====]│chkp│[==================================]│ returned     │[=========
  │       │    │ barrier + RaaS load + eval run     │              │
```

## GPU→CPU Copy Strategies

WeightManager automatically selects the optimal copy strategy based on
FSDP parameter placement:

**Shard copy (fast path)** — When all parameters use `Shard(0)` placement
(the standard FSDP2 case), each rank copies only its local shard to the
correct offset. No cross-rank communication. All ranks write in parallel.

**All-gather (fallback)** — When any parameter has a different placement,
FSDP all-gathers the full tensors. Only rank 0 copies to the buffer.
Slower but handles any sharding strategy.

| | Shard copy | All-gather |
|---|---|---|
| Network | None | Full model over NCCL |
| PCIe per rank | `model_size / N` | `model_size` (rank 0) |
| Parallelism | All N ranks | Rank 0 only |

## Double Buffer

The shared-memory buffer in `/dev/shm` is 2× model size. The trainer
writes to the inactive half while the sender serves the active half
over TCP — no locking between them:

```
Step N:    trainer writes Half 0,  TCP reads Half 1
Step N+1:  trainer writes Half 1,  TCP reads Half 0
```

The buffer index swap is a single Python int assignment (atomic under GIL).

## Optimizations

- Zero-copy `sendfile()` for TCP transfer
- `mlock` and transparent huge pages for shared-memory buffers
- 6 parallel TCP streams for throughput
- `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)` hints
- CUDA host registration for pinned DMA transfers

## Project Structure

```
astraflow/weight_manager/
  __init__.py              ← exports WeightManager, WeightManagerConfig
  weight_manager.py        ← main class: buffer mgmt, GPU→CPU copy, sender lifecycle
  config.py                ← WeightManagerConfig

  transfer/                ← shared transport layer
    config.py              ← TransferEngineConfig, SenderAgentConfig, ReceiverAgentConfig, TransferStatus, ReceiverInfo
    sender_agent.py        ← sender subprocess (TransferAgent): HTTP, TCP, ZMQ
    transfer_engine.py     ← TCPTransferEngine (6-stream, sendfile)
    receiver_agent.py      ← TransferBuffer
```

## Sender Agent HTTP API

The sender agent runs as a subprocess on the trainer node and exposes
these endpoints (used by RaaS to pull weights):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/get_version` | Current weight version |
| `GET` | `/get_buffer_info` | Buffer size and tensor metadata |
| `GET` | `/get_capabilities` | Supported strategies, delta readiness |
| `POST` | `/register_sglang_instance` | Register a RaaS receiver |
| `POST` | `/request_transfer` | Pull weights via TCP (mode: full or delta) |

## Multi-Model Training

In multi-model training (e.g. actor + verifier), each model has its own
WeightManager with its own sender agent and shared-memory buffer.
Dataflow coordinates them via a version barrier — all models must reach
the same version before any RaaS loads weights. This prevents serving
rollouts with mismatched model versions.

## Multi-RaaS

Weight notifications fan out to all RaaS instances in parallel via
`RaaSPool.notify_version(model_id, version, sender_endpoint)`
(`astraflow/dataflow/raas_pool.py`). Each instance independently
pulls, pauses, loads, and resumes. New instances joining mid-training
catch up to the current weights via `AstraFlowService.catchup_raas()`
before entering the live pool.
