# Multi-Agent Weight Transfer

This page describes how weight transfer works in multi-agent (multi-model)
training, where multiple independently-trained models (e.g., a solver and a
verifier) share the same RaaS inference cluster.

## Overview

In a 2-model setup (e.g., `actor_and_verify` workflow):

- **model0 (solver)** and **model1 (verifier)** are separate FSDP trainers,
  each with their own GPU group, WeightManager, and sender agent.
- **RaaS** runs both models' inference engines in a single process, with
  separate SGLang instances per model.
- **AstraFlow** orchestrates the coordination via a **version barrier**.

```
Trainer model0 (GPU 4,5)       Trainer model1 (GPU 6,7)
  WeightManager                  WeightManager
  SenderAgent :19861             SenderAgent :19862
        │                              │
        └──────────┐    ┌──────────────┘
                   ▼    ▼
              AstraFlow Service
              (version barrier)
                   │
                   ▼
              RaaS Manager
        ┌──────────┴──────────┐
  SGLang model0 (GPU 0,1)  SGLang model1 (GPU 2,3)
```

## Sender Side: Trainer → WeightManager → SenderAgent

Each trainer is an independent `torchrun` process group. After each training
step, the trainer offloads weights to its own WeightManager:

1. **GPU → CPU copy** — All FSDP ranks write their local shards into a
   shared-memory double buffer at `offset + rank * shard_size`
   (`WeightManager._copy_shards`). After a `dist.barrier()`, the buffer
   holds the full unsharded model.

2. **Buffer swap** — Rank 0 notifies the sender agent subprocess
   (`_notify_buffer_ready`). The sender swaps active/inactive buffer halves
   so it can serve from the freshly written half while the trainer writes
   the next version to the other half.

3. **Sender agent HTTP** — Each model's sender agent runs its own HTTP
   server on a distinct port (configured via `WEIGHT_TRANSFER_HTTP_PORT`):
   - model0: port 19861
   - model1: port 19862

   The sender exposes `/get_buffer_info`, `/register_sglang_instance`,
   `/request_transfer`, and `/get_capabilities`.

## Coordination: AstraFlow Version Barrier

The trainers are independent, but weight loading is **coordinated** — RaaS
loads both models' weights atomically so the solver and verifier are always
at the same version during rollout generation.

The coordination happens in `AstraFlowService.notify_version()`
(`astraflow/dataflow/service.py:922`). The method has **two phases**:

**Phase 1 — per-model weight load.** For non-eval steps this fires
**immediately** (fire-and-forget) in a daemon thread calling
`_trigger_raas_weight_load_single()` (`service.py:842`). Each model's
weight load is independent — model0 does not wait for model1. Briefly
stale rollouts produced during the overlap are absorbed by the buffer's
staleness filter.

For eval steps, Phase 1 is **deferred** — the weight load is postponed
to the leader in Phase 2 so RaaS is never left in a mixed-version state
while the barrier is still waiting. This avoids a deadlock observed in
multi-model setups where a first-model sync load would starve the
second model of rollouts and prevent it from reaching the barrier.

**Phase 2 — version barrier.** Regardless of eval, every trainer's call
records its version under `_version_barrier_cond`:

1. Each trainer calls `POST /notify_version` with
   `{agent_name, version, run_eval, model_id}`.

2. AstraFlow records the version in `_pending_versions`:
   ```
   _pending_versions = {(default, model0): N}   # model0 arrived first
   ```

3. The first trainer's HTTP handler thread **blocks** on
   `_version_barrier_cond.wait()`.

4. When the second trainer arrives:
   ```
   _pending_versions = {(default, model0): N, (default, model1): N}
   ```
   All registered model_ids have the same version → this thread becomes
   the **leader**.

5. On eval steps, the leader pauses data acquisition, calls
   `reset_training_engine` on the pool (cancels in-flight `arun_episode`
   coroutines so pause/load isn't contended), then iterates the
   registered model_ids sequentially, calling
   `_trigger_raas_weight_load_single()` for each. Non-leaders are woken
   via `_version_barrier_cond.notify_all()`.

```
model0: notify_version(v=5) ──► (async fire-and-forget load) ──► barrier (wait)
                                                                       │
model1: notify_version(v=5) ──► (async fire-and-forget load) ──► barrier (leader)
                                                                       │
                                                                       ▼
                                                       (eval? run reset + sync load per model + eval)
                                                                       │
                                                                       ▼
                                                       notify_all() → both return
```

### Async Notifications

Non-eval steps on the trainer side use `notify_version_async`
(`astraflow/train_worker/trainer/astraflow_client.py:275`), which submits
to a single-threaded background executor. The trainer can run **one step
ahead** of the weight load, but the next `notify_version_async` call
waits for the previous one to complete before submitting.

## Receiver Side: AstraFlow → RaaS → TCP Pull

### One HTTP Request Per Model Per RaaS

AstraFlow sends **one** `POST /notify_version` call per model per live
RaaS instance. There is no batched multi-model endpoint. Each request
body is pickle-serialized:

```python
{"model_id": "model0", "version": 5, "sender_endpoint": "host:19861"}
```

Fan-out happens in `RaaSPool.notify_version()`
(`astraflow/dataflow/raas_pool.py:442`), which submits one
`_notify_one_model()` per live RaaS in a thread pool. Overall latency is
`max(instances)`, not `sum(instances)`.

### Per-Model Weight Update Cycle

Inside each RaaS, `RaaS3Manager.notify_version()`
(`astraflow/raas/server/manager.py:1556`) handles exactly one model.
Calls for different models proceed independently; calls for the same
model are serialized through a per-model `asyncio.Lock` so concurrent
updates cannot race on the same safetensors file.

The handler delegates to `_do_weight_update()` (`manager.py:1612`),
which runs two phases — **pull** then **pause/load/resume** — both
delegated to the thread-pool executor so they do not block the FastAPI
event loop:

```
RaaS Manager (one model_id)                Sender Agent (e.g. port 19861)
     │                                           │
     │  [First pull only — setup]                │
     │── GET  /get_buffer_info ─────────────────>│  Query model metadata
     │<── {tensors_meta, buffer_length} ─────────│  (param names, shapes, dtypes)
     │                                           │
     │  Allocate TransferBuffer                  │
     │  Start TCP listener (auto port)           │
     │  Start ZMQ listener (auto port)           │
     │                                           │
     │── POST /register_sglang_instance ────────>│  Register as receiver
     │   {session_ids, handshake_ports,          │  (TCP/ZMQ addresses)
     │    zmq_endpoint, zmq_port, ...}           │
     │<── {trainer_session_ids, rank} ───────────│
     │                                           │
     │  [Every pull — phase 1: pull to disk]     │
     │── POST /request_transfer ────────────────>│  "Send me the weights"
     │   {instance_id, mode: "full" | "delta"}   │
     │<── {ok: true} ───────────────────────────│
     │                                           │
     │<══════ TCP data push ════════════════════│  Sender reads from shm buffer,
     │   (sendfile, parallel streams)            │  pushes via TCP to receiver's
     │                                           │  listener ports
     │                                           │
     │<── ZMQ [rank, SUCCESS] ──────────────────│  "Transfer complete"
     │                                           │
     │  save_as_safetensors()                    │
     │  → /dev/shm/astraflow_weights/{tag}/{model_id}/model.safetensors
     │                                           │
     │  [Phase 2: pause / load / resume THIS model's engine]
     │  engine.pause_generation()                │
     │  engine.load_weights_from_path(shm_path)  │
     │  engine.continue_generation()             │
     │  self._weight_versions[model_id] = version│
```

Each model_id gets its own:

- `RaaSWeightReceiver` instance, lazily created and cached in
  `self._tcp_receivers[model_id]`.
- TCP session and ZMQ listener.
- Isolated shm directory: `/dev/shm/astraflow_weights/{tag}/{model_id}/`.
- Inference engine handle (`self._engines[model_id]`).

The receiver is created once per `(model_id, sender)` pair and reused
across steps. Only the `/request_transfer` → TCP push → ZMQ signal →
pause/load/resume cycle repeats each step.

### Multi-Model Ordering

Because each model's `notify_version` call is independent, the orchestrator
controls whether updates are effectively parallel or serialized:

- **Non-eval steps** — AstraFlow fires `_trigger_raas_weight_load_single`
  for each model in its own daemon thread (`service.py:963`). Two models'
  pulls and pause/load/resume cycles overlap freely; only same-model
  calls serialize via the per-model lock.
- **Eval steps** — the barrier leader iterates
  `for mid in sorted(expected_model_ids)` and calls
  `_trigger_raas_weight_load_single` sequentially
  (`service.py:1061`). This guarantees both models finish loading
  before eval rollouts start.

## End-to-End Timeline

### Non-Eval Step (async fire-and-forget)

```
Trainer model0          AstraFlow                 RaaS Manager           Trainer model1
     │                      │                         │                       │
 train step N               │                         │                   train step N
 offload v=N+1              │                         │                   offload v=N+1
     │                      │                         │                       │
 notify_version_async ─► phase 1: fire daemon thread  │       ◄─ notify_version_async
     │             POST /notify_version{model0=N+1} ─►│                       │
     │             POST /notify_version{model1=N+1} ─►│                       │
     │                      │                         │                       │
     │                      │          pull model0 from :19861                │
     │                      │          pause/load/resume model0 engine        │
     │                      │                         │                       │
     │                      │          pull model1 from :19862                │
     │                      │          pause/load/resume model1 engine        │
     │                      │                         │                       │
     │             phase 2: both trainers at v=N+1 → leader wakes all
     │                      │                         │                       │
 barrier released ◄───────────────────────────────────────────►  barrier released
     │                      │                         │                       │
 train step N+1             │                         │                   train step N+1
```

Per-model loads on RaaS can overlap — only same-model calls serialize
via the per-model `asyncio.Lock`. Trainers can run one step ahead of
the async load.

### Eval Step (deferred, leader serialises loads)

```
Trainer model0           AstraFlow                    RaaS Manager
     │                      │                              │
 offload v=N+1              │                              │
 notify_version (sync) ─► phase 1 DEFERRED (eval path)     │
                         phase 2: wait for all             │
 Trainer model1 arrives ─────────────► leader triggered    │
                         │                                 │
                         │  leader:                        │
                         │    1. flow.pause() data acq     │
                         │    2. clear suspects            │
                         │    3. reset_training_engine ────►│ cancel arun_episode tasks
                         │    4. /notify_version{model0} ──►│ pull + pause/load/resume m0
                         │    5. /notify_version{model1} ──►│ pull + pause/load/resume m1
                         │    6. run eval                  │
                         │    7. flow.resume()             │
                         │                                 │
                         │  notify_all() → return eval_results
 ◄─── eval_results ──────┘                                 │
 train step N+1                                            │
```

Eval runs synchronously because the trainer needs eval results before
continuing. The sequential load
(`for mid in sorted(expected_model_ids)`) plus `reset_training_engine`
(which cancels all in-flight `arun_episode` tasks) ensures every model
is at `v=N+1` with zero inflight rollouts before eval starts —
preventing the multi-model deadlock that earlier designs suffered from.

## Configuration

In the launch script, each trainer gets a distinct HTTP port for its
sender agent:

```bash
# TCP weight-transfer ports (one per trainer)
export WEIGHT_TRANSFER_HTTP_PORT_MODEL0=19861
export WEIGHT_TRANSFER_HTTP_PORT_MODEL1=19862

# Trainer model0
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0}" \
  torchrun ... --trainer trainer_model0

# Trainer model1
WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL1}" \
  torchrun ... --trainer trainer_model1
```

The experiment YAML defines model_ids that link trainers to RaaS engines:

```yaml
raas:
  models:
    model0:
      backend: sglang
    model1:
      backend: sglang

trainer_model0:
  model_id: model0    # links to raas.models.model0

trainer_model1:
  model_id: model1    # links to raas.models.model1
```

## Key Source Files

| Component | File | Entry Point |
|-----------|------|-------------|
| Version barrier + Phase 1/2 | `astraflow/dataflow/service.py` | `AstraFlowService.notify_version()` (line 922) |
| Per-model RaaS weight load trigger | `astraflow/dataflow/service.py` | `_trigger_raas_weight_load_single()` (line 842) |
| RaaS pool fan-out (one call per model) | `astraflow/dataflow/raas_pool.py` | `RaaSPool.notify_version()` (line 442), `_notify_one_model()` (line 410) |
| RaaS manager pull + load | `astraflow/raas/server/manager.py` | `notify_version()` (line 1556), `_do_weight_update()` (line 1612), `_pull_weights_to_disk()` |
| TCP receiver | `astraflow/raas/server/tcp_receiver.py` | `RaaSWeightReceiver` |
| Sender agent | `astraflow/core/weight_manager/transfer/sender_agent.py` | `SenderAgent` |
| WeightManager offload | `astraflow/core/weight_manager/weight_manager.py` | `offload()` |
| Trainer integration | `astraflow/train_worker/trainer/ppo_trainer.py` | `AstraFlowPPOTrainer` |
