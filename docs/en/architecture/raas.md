# RaaS

RaaS (Remote Agentic Serving, `astraflow/raas/`) manages inference engines and rollout generation.

## Key Components

- **`RaaS3Manager`** — Central async manager that maintains per-model inference engines, handles workflow registration, and manages pause/resume for weight sync.
- **`server/routes.py`** — FastAPI routes exposing the RaaS HTTP API.
- **`server/tcp_receiver.py`** — TCP weight receiver that pulls weights from the Trainer's sender agent.
- **`engine/`** — Backend engine implementations (SGLang, vLLM wrappers).
- **`api/`** — Configuration dataclasses and engine specs.

## Responsibilities

- Launch and manage vLLM/SGLang inference servers.
- Execute rollout workflows with registered reward functions.
- Track weight versions; pull updated weights from Trainer when behind.
- Pause/resume inference during weight synchronization.

## How It Fits

RaaS is the inference side of the loop:

- Receives rollout requests from AstraFlow
- Generates completions using managed inference engines
- Accepts weight updates from Trainer via pull-based TCP transfer

For a full guide on implementing a custom RaaS, see
[Custom RaaS Integration](custom-raas.md).

---

## RaaS HTTP API

All endpoints use binary pickle/cloudpickle serialization
(`Content-Type: application/octet-stream`) **except** `GET /status` and
`GET /availability`, which return JSON. Pickle endpoints wrap responses
as `{"ok": True, "result": ...}` on success or `{"ok": False, "error": ...}`
on failure (HTTP 500). Source: `astraflow/raas/server/routes.py`.

### Health & Status (JSON)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/status` | Readiness check (`"ready"` / `"idle"` / `"starting"` / `"error"`); also polled as heartbeat by the orchestrator |
| `GET` | `/availability` | Capacity for load-balanced routing (`{available, inflight, ...}`) |

### Rollout (pickle)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/register_workflow` | Register a workflow class + reward fn for rollout generation |
| `POST` | `/submit` | Submit one prompt for rollout → `{task_id}` |
| `POST` | `/pull` | Drain completed rollout results → `list[{task_id, result}]` |

### Weight Sync (pickle)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/notify_version` | Per-model weight update. Payload `{model_id, version, sender_endpoint}`. RaaS pulls weights for that one model from the sender and hot-swaps its inference engine. |

### Evaluation (pickle)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/reset_training_engine` | Cancel in-flight training rollouts, drain engines, ready state for eval |
| `POST` | `/eval_start` | Begin eval session (reset tracking state) |
| `POST` | `/eval_submit` | Submit an eval sample |
| `POST` | `/eval_pull` | Collect eval results with progress (`{items, inflight, pending, total_submitted}`) |
| `POST` | `/eval_end` | End eval session (clear tracking state) |

### Lifecycle (pickle)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/shutdown` | Graceful shutdown — destroys engines and exits |

---

## APIs That RaaS Calls (RaaS as Client)

RaaS is not only a server — it also acts as a client to two external services.

### AstraFlow Service

On startup, RaaS self-registers with the AstraFlow orchestrator:

| Method | Target | Purpose |
|--------|--------|---------|
| `POST` | AstraFlow `/register_raas` | Register this RaaS instance into the global pool |

This is triggered at boot when `--astraflow-url` is provided. AstraFlow then knows to route rollout requests to this instance.

### Trainer Sender Agent (Weight Transfer)

When `/notify_version` is called (one call per model) and RaaS detects a
version lag, it initiates a **pull-based TCP weight transfer** by calling
the Trainer's sender agent:

| Method | Target | Purpose |
|--------|--------|---------|
| `GET`  | Trainer `/get_buffer_info` | Query tensor layout (first pull only) |
| `POST` | Trainer `/register_sglang_instance` | Register as a weight receiver (buffer ptr, ZMQ endpoint, handshake ports) |
| `POST` | Trainer `/request_transfer` | Request the actual TCP weight transfer (per pull) |

The flow inside `manager.notify_version(model_id, version, sender_endpoint)`
(`astraflow/raas/server/manager.py:1648`):

```
AstraFlow ──POST /notify_version──> RaaS (for one model_id)
  {model_id, version, sender_endpoint}          │
                                                │ acquire per-model asyncio.Lock
                                                │
                                                ├─ GET  /get_buffer_info ─────────> Trainer SenderAgent (first pull only)
                                                ├─ POST /register_sglang_instance ─> Trainer SenderAgent (first pull only)
                                                ├─ POST /request_transfer ────────> Trainer SenderAgent
                                                ├─ ← TCP bulk transfer + ZMQ signal
                                                │
                                                ├─ save bytes to /dev/shm as safetensors
                                                ├─ engine.pause_generation()
                                                ├─ engine.load_weights_from_path(...)
                                                └─ engine.continue_generation()
```

For multi-model training, AstraFlow's Python-side barrier sends one such
call per model, in sequence or in parallel depending on whether eval is
requested — see [Multi-Agent Weight Transfer](multi-agent-weight-transfer.md).

---

## Call Graph (RaaS-Centric)

All HTTP calls to and from RaaS, organized by phase.

```
                                    ┌─────────────────────┐
                                    │        RaaS         │
                                    │      (raas/)        │
                                    └──────────┬──────────┘
                                               │
    ╔══════════════════════════════════════════════════════════════════╗
    ║  STARTUP                                                       ║
    ╚══════════════════════════════════════════════════════════════════╝
                                               │
      RaaS ──POST /register_raas──────────> AstraFlow     join the pool
                                               │
      AstraFlow ──POST /register_workflow─> RaaS          register rollout workflows
                                               │
    ╔══════════════════════════════════════════════════════════════════╗
    ║  ROLLOUT (continuous, async)                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
                                               │
      AstraFlow ──GET /availability───────> RaaS          check capacity
      AstraFlow ──POST /submit────────────> RaaS          submit prompts
      AstraFlow ──POST /pull──────────────> RaaS          collect results
                                               │
    ╔══════════════════════════════════════════════════════════════════╗
    ║  WEIGHT SYNC (per training step, one call per model)             ║
    ╚══════════════════════════════════════════════════════════════════╝
                                               │
      AstraFlow ──POST /notify_version──────> RaaS        trigger weight load
                  {model_id, version,
                   sender_endpoint}             │
                                               │
        RaaS ──POST /register_sglang_instance──> Trainer      register as receiver (first pull only)
        RaaS ──POST /request_transfer──────────> Trainer      pull weights via TCP
                                               │
    ╔══════════════════════════════════════════════════════════════════╗
    ║  EVALUATION (triggered after weight sync)                      ║
    ╚══════════════════════════════════════════════════════════════════╝
                                               │
      AstraFlow ──POST /eval_start────────> RaaS          begin eval session
      AstraFlow ──POST /eval_submit───────> RaaS          submit eval samples
      AstraFlow ──POST /eval_pull─────────> RaaS          collect eval results
      AstraFlow ──POST /eval_end──────────> RaaS          end eval session
                                               │
    ╔══════════════════════════════════════════════════════════════════╗
    ║  LIFECYCLE                                                     ║
    ╚══════════════════════════════════════════════════════════════════╝
                                               │
      AstraFlow ──GET /status─────────────> RaaS          health check
      AstraFlow ──POST /shutdown──────────> RaaS          graceful shutdown
```

**Inbound (RaaS as server):** AstraFlow calls RaaS for rollout, eval, weight sync, and lifecycle management.
**Outbound (RaaS as client):** RaaS calls AstraFlow once at startup (registration) and calls Trainer directly during weight sync (TCP pull).

### Initial Startup vs Recovery

- **Fresh start (version=0):** Both RaaS and Trainer load the same model checkpoint independently. No weight transfer needed — data acquisition begins immediately.
- **Recovery (version > 0):** Trainer sends `recovered_version` in `POST /ready`. AstraFlow then fans out `POST /notify_version` to all RaaS instances (once per registered model) before starting data acquisition, ensuring every RaaS loads the recovered weights.
