# System Overview

AstraFlow is a fully asynchronous reinforcement learning system where inference, training, and data orchestration run as independent services coordinated via HTTP. The system is designed for distributed GPU clusters and supports elastic scaling of both inference and training.

## Components

The system has four cooperating components:

1. **Dataflow** — Central orchestrator. Manages a global pool of RaaS instances, acquires rollout data, buffers and serves training batches, and coordinates weight synchronization.
2. **RaaS** (Remote Agentic Serving) — Inference service. Launches and manages vLLM/SGLang engines, executes rollout workflows with reward functions, and loads updated weights. Multiple RaaS instances can run in parallel, and instances can join or leave dynamically.
3. **Trainer** — Distributed training engine (FSDP/Megatron). Fetches batches from Dataflow, runs RL algorithm updates (GRPO, PPO, M2PO, etc.), and pushes updated weights. Multiple trainers can run in parallel for multi-model training.
4. **WeightManager** — Weight transfer subsystem. Copies FSDP weights to shared-memory buffers and transfers them to RaaS via TCP or NCCL.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       Dataflow                          │
│                (Orchestration + Buffering)              │
│                                                         │
│       ┌──────────────┐          ┌──────────────┐        │
│       │     Data     │          │     Data     │        │
│       │ Acquisition  │          │   Serving    │        │
│       └──────┬───────┘          └──────┬───────┘        │
└──────────────┼─────────────────────────┼────────────────┘
               │                         │
submit / pull  │                         │  get_batch
        ┌──────┴──────┐          ┌───────┴──────┐
        │             │          │              │
┌───────▼───────┐ ┌───▼───────┐ ┌▼───────────┐ ┌▼──────────┐
│   RaaS #1     │ │  RaaS #2  │ │  Trainer   │ │  Trainer  │
│   (SGLang)    │ │  (SGLang) │ │  (model0)  │ │  (model1) │
│    4 GPUs     │ │   4 GPUs  │ │   2 GPUs   │ │   2 GPUs  │
└───────┬───────┘ └───┬───────┘ └─────┬──────┘ └──────┬────┘
        │             │               │               │
        │      weight sync (TCP or NCCL)              │
        ◄─────────────────────────────┘───────────────┘
```

## Inter-Component Call Graph

All HTTP calls between the three components, organized by phase.

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│       Trainer        │     │      Dataflow        │     │        RaaS         │
│  (train_worker/)     │     │    (astraflow/)      │     │      (raas/)        │
└─────────┬───────────┘     └──────────┬──────────┘     └──────────┬──────────┘
          │                            │                           │
    ╔═════╧════════════════════════════╧═══════════════════════════╧═════╗
    ║  STARTUP                                                         ║
    ╚═════╤════════════════════════════╤═══════════════════════════╤═════╝
          │                            │                           │
          │                            │  POST /register_raas   │
          │                            │ <─────────────────────────┤
          │                            │                           │
          │                            │  POST /register_workflow
          │                            ├──────────────────────────>│
          │                            │                           │
          │  POST /ready            │                           │
          ├───────────────────────────>│                           │
          │                            │                           │
    ╔═════╧════════════════════════════╧═══════════════════════════╧═════╗
    ║  TRAINING LOOP (each iteration)                                  ║
    ╚═════╤════════════════════════════╤═══════════════════════════╤═════╝
          │                            │                           │
          │                            │  GET /availability     │
          │                            ├──────────────────────────>│
          │                            │                           │
          │                            │  POST /submit          │
          │                            ├──────────────────────────>│
          │                            │                           │
          │                            │  POST /pull            │
          │                            ├──────────────────────────>│
          │                            │                           │
          │  GET /batch             │                           │
          ├───────────────────────────>│                           │
          │                            │                           │
          │       ... trainer runs RL update ...                   │
          │                            │                           │
          │  POST /notify_version   │                           │
          ├───────────────────────────>│                           │
          │                            │                           │
          │                            │  POST /notify_version     │
          │                            │  (one call per model      │
          │                            │   per RaaS instance)      │
          │                            ├──────────────────────────>│
          │                            │                           │
          │          POST /register_sglang_instance  (first pull)  │
          │ <─────────────────────────────────────────────────────┤
          │                            │                           │
          │          POST /request_transfer   (each pull)          │
          │ <─────────────────────────────────────────────────────┤
          │                            │                           │
    ╔═════╧════════════════════════════╧═══════════════════════════╧═════╗
    ║  EVALUATION (triggered after weight sync)                        ║
    ╚═════╤════════════════════════════╤═══════════════════════════╤═════╝
          │                            │                           │
          │                            │  POST /eval_start      │
          │                            ├──────────────────────────>│
          │                            │                           │
          │                            │  POST /eval_submit     │
          │                            ├──────────────────────────>│
          │                            │                           │
          │                            │  POST /eval_pull       │
          │                            ├──────────────────────────>│
          │                            │                           │
          │                            │  POST /eval_end        │
          │                            ├──────────────────────────>│
          │                            │                           │
    ╔═════╧════════════════════════════╧═══════════════════════════╧═════╗
    ║  LIFECYCLE                                                       ║
    ╚═════╤════════════════════════════╤═══════════════════════════╤═════╝
          │                            │                           │
          │                            │  GET /status           │
          │                            ├──────────────────────────>│
          │                            │                           │
          │                            │  POST /shutdown        │
          │                            ├──────────────────────────>│
          │                            │                           │
```

Arrow direction indicates who initiates the HTTP call. During weight sync, RaaS calls Trainer directly (bypassing Dataflow) for the bulk TCP transfer.

## Dynamic RaaS Pool

RaaS instances are managed by a global `RaaSPool` shared across all agents. Instances register and deregister at runtime via HTTP:

- **Join**: A new RaaS instance calls `POST /register_raas`. The pool initializes the engine, catches it up to the current weight version (so it never serves stale rollouts), then adds it to the live pool.
- **Leave**: An instance can deregister via `POST /deregister_raas`, or the pool automatically detects failures via heartbeat monitoring.
- **Failure detection**: Data-path calls (submit, pull) mark failing instances as *suspect*. A background heartbeat thread confirms via `GET /status` (which reflects actual engine health, not a trivial OK) and deregisters dead instances. This two-phase approach avoids false positives from transient network issues.
- **Capacity-based routing**: `submit_auto` routes each rollout request to the RaaS instance with the most available slots, balancing load across the pool.
- **Parallel collect**: `pull_completed` fans out to all live instances in parallel and merges results.

This means you can scale inference throughput by adding more RaaS instances at any time — no restart required.

## Multi-Model, Multi-Trainer

AstraFlow supports training multiple models simultaneously (e.g., a solver and a verifier in solve-and-verify). Each model has:

- Its own **Trainer** process with dedicated GPUs and a separate weight transfer port.
- Its own **buffer** inside Dataflow with independent staleness filtering.
- Its own **SGLang engine** inside RaaS (e.g., `sglang[model0]:d2+sglang[model1]:d2`).

Coordination uses a **version barrier**: each trainer independently notifies its version after a training step. Weight sync to RaaS is triggered only after all models reach the same version. The last trainer to arrive becomes the "leader" and initiates the coordinated weight load across all RaaS instances. Other trainers block until the leader completes.

## Training Loop

Each iteration proceeds as follows:

1. **Data Acquisition** (background, continuous): Dataflow checks RaaS pool availability, submits prompts to the least-loaded instance, and collects completed rollouts from all instances in parallel. Rollouts are filtered by staleness and reward distribution, then buffered.

2. **Batch Fetch**: Trainer requests a batch from Dataflow via `GET /batch`. Dataflow mixes fresh rollouts and replay data according to `replay_ratio`, applies staleness filtering, and returns a padded batch.

3. **Train Step**: Trainer runs forward/backward pass with the RL algorithm (PPO/GRPO/M2PO), computes loss, and updates model weights.

4. **Weight Sync**:
   - Trainer stages new weights to shared memory (via `WeightManager.offload`) and calls `POST /notify_version` on Dataflow.
   - In multi-model setups, Dataflow's Python-side version barrier waits until every registered `model_id` reports the same version. For non-eval steps, each per-model weight load is fired asynchronously (fire-and-forget) and the staleness filter absorbs briefly-stale rollouts. For eval steps, the weight load is deferred to the barrier leader so every model updates atomically.
   - The leader (or each async firer) fans out one `POST /notify_version` per model to every live RaaS instance via `RaaSPool.notify_version()`. Each RaaS pulls the model's weights over TCP from the sender agent, pauses that model's engine, loads from `/dev/shm`, and resumes.

5. **Repeat** with the next iteration.

## Async Execution Model

Each component runs as a separate process:

| Component | Execution Model | Typical GPUs |
|---|---|---|
| Dataflow | Threaded (Flask + ThreadPoolExecutor) | CPU-only |
| RaaS (×N) | Async (FastAPI + asyncio) | 4+ GPUs each |
| Trainer (×M) | Synchronous (PyTorch DDP/FSDP) | 2–8 GPUs each |

Data acquisition runs two background threads inside Dataflow:
- **Submit thread**: Checks pool availability, gathers prompts, submits in parallel via ThreadPoolExecutor.
- **Collect thread**: Polls all RaaS instances for completed rollouts, applies filtering, ingests into buffer.

This decoupling means inference and training overlap — while the trainer processes one batch, the next batch of rollouts is already being generated across all RaaS instances.

## Version Tracking and Staleness

Every rollout is tagged with the model weight version that produced it. During batch retrieval, samples older than `current_version - max_staleness` are dropped. This prevents training on outdated rollouts while allowing a configurable lag for throughput.

When a new RaaS instance joins mid-training, the pool sends it the current weight version and sender endpoints so it loads the latest weights before serving any traffic.

## Fault Tolerance

- **RaaS heartbeat**: Background thread monitors all instances via `GET /status` every 10s. Two consecutive failures trigger deregistration. The pool continues operating with remaining instances.
- **Suspect-and-confirm**: Data-path failures mark instances as suspect (skipped for routing) but don't immediately deregister — the heartbeat thread confirms via `GET /status` first.
- **Empty pool**: If all RaaS instances fail, submit/collect become no-ops until a new instance registers. Training pauses at the next `get_batch` call (blocks until data is available).
- **Checkpointing**: Both model weights and buffer state can be saved and restored. On recovery, trainers report their recovered version so staleness filtering works correctly.
