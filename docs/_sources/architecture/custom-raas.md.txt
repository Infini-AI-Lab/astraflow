---
orphan: true
---

# Custom RaaS Integration

AstraFlow's RaaS (Remote Agentic Serving) is a swappable component. You
can replace the built-in implementation with your own inference service
as long as it speaks the correct HTTP protocol. This guide documents the
minimal API your custom RaaS must implement.

This guide assumes **TCP weight transfer mode** (the default and
recommended mode). Weights flow trainer → RaaS via a pull-based TCP
pipeline; see [Weight Manager](weight-manager.md) for the transport
layer used by both sides.

## Architecture

A RaaS has three communication surfaces:

```
                   ┌───────────────────────┐
   Inbound HTTP    │                       │   Outbound HTTP
   (12 endpoints)  │     Custom RaaS       │   (1 + 3 endpoints)
                   │                       │
  AstraFlow ──────►│                       │──────► AstraFlow
  calls RaaS       │  ┌─────────────────┐  │        /register_raas
                   │  │   Inference     │  │         (once, startup)
                   │  │   Engine(s)     │  │
                   │  │ (vLLM / SGLang /│  │──────► Trainer SenderAgent
                   │  │    custom)      │  │        /get_buffer_info
                   │  └─────────────────┘  │        /register_sglang_instance
                   │                       │        /request_transfer
                   │  ┌─────────────────┐  │         (every weight update)
                   │  │  Weight         │  │
                   │  │  Receiver       │  │
                   │  │  (reusable lib) │  │
                   │  └─────────────────┘  │
                   └───────────────────────┘
```

- **Inbound (AstraFlow → RaaS)**: 12 HTTP endpoints covering health,
  rollout, eval, weight-sync notification, and lifecycle.
- **Outbound to AstraFlow**: one `POST /register_raas` call at startup
  to join the pool. That's it — AstraFlow drives the data plane.
- **Outbound to Trainer**: 3 HTTP calls per weight update to the
  trainer's weight sender agent (see
  [Custom Trainer Integration](custom-trainer.md)). The reusable
  `RaaSWeightReceiver` class in
  `astraflow/raas/server/tcp_receiver.py` handles the TCP/ZMQ
  machinery — custom RaaS implementations should reuse it rather than
  reimplement.

## What a RaaS Is

A RaaS is a **weight-versioned rollout service**. Its three jobs:

1. **Host a policy.** Run an inference backend that can generate from
   the latest weights.
2. **Execute workflows, not just generations.** A rollout is a
   workflow (`arun_episode`) that may call `generate` many times,
   compute rewards, and emit a structured trajectory. RaaS owns this
   loop.
3. **Absorb weight updates without downtime.** When the trainer
   publishes version N+1, pull the bytes, hot-swap, keep serving.

### Invariants your implementation MUST preserve

- **Tag every output token with its weight version**
  (`output_versions` in the `ModelResponse`). The orchestrator uses
  this to drop stale rollouts. Break this and training quality
  degrades silently.
- **`GET /status` must answer under heartbeat timeout** (~10s). It is
  polled by the orchestrator; two consecutive failures cause
  deregistration from the pool.
- **Return trajectories exactly as the workflow emits them**, unmodified.
  The workflow is user-defined; RaaS must be transparent.
- **Never block the FastAPI event loop during weight load.** Run
  `pause_generation`, `load_weights_from_path`, and
  `continue_generation` in an executor — not inline in the handler.
  Otherwise `/status` queues behind the weight load and you'll be
  deregistered mid-update.
- **Serialize per-model weight updates.** Two concurrent
  `/notify_version` calls for the same `model_id` must not race on the
  safetensors file.

### What RaaS is NOT

- Not a stateless inference server. It owns a task queue, a version
  counter, and workflow instances.
- Not the loss / optimizer / reward *definition* owner. Reward
  functions are user-provided and passed in at
  `/register_workflow`.
- Not the prompt chooser. AstraFlow submits; RaaS executes.

## API Reference

### Inbound: AstraFlow → RaaS

All endpoints below live on the RaaS HTTP server. Two wire formats
coexist:

- **JSON** (`Content-Type: application/json`) — only `GET /status`
  and `GET /availability`.
- **Pickle / cloudpickle** (`Content-Type: application/octet-stream`)
  — every other endpoint.

For pickle endpoints, the payload is a Python dict serialized via
`cloudpickle.dumps(obj)` (or `pickle.dumps` as fallback). The shape
is JSON-like on the Python side, but the wire bytes are **not** JSON
text. Below, request/response code blocks use the `python` fence to
denote "this is the Python dict *before* pickling."

**Response envelope (pickle endpoints only).** Every pickle endpoint
wraps its return value in an envelope
(`_encode_ok` / `_encode_error` in `routes.py:27-41`):

```python
# Success — HTTP 200
{
    "ok": True,
    "result": <handler-specific value>,
}

# Failure — HTTP 500
{
    "ok": False,
    "error": "<repr(exception)>",
}
```

Clients check `ok` first, then read `result` or `error`. The examples
below show the **full envelope** for every pickle response so you can
see exactly what to emit from your handler.

JSON endpoints (`/status`, `/availability`) return their dict
directly with no envelope.

Reference implementation: `astraflow/raas/server/routes.py`.

The **Inbound: AstraFlow → RaaS** surface is split into two groups:

- **Required** (the 8 endpoints below) — without all of them the
  training loop cannot run.
- **Optional: Eval Support** (5 endpoints, further below) — only
  called when eval windows are enabled. Training-only RaaS
  implementations can omit them.

#### `GET /status` — Readiness (JSON)

Also used as the heartbeat. Must respond quickly; do not block on
engine work.

**Response** — raw JSON text (no envelope):

```json
{
  "status": "ready",
  "message": "optional details"
}
```

`status` values: `"ready"`, `"idle"`, `"starting"`, `"error"`. Only
`"ready"` means the pool can route traffic.

#### `GET /availability` — Capacity (JSON)

Called on **every submit tick** by the orchestrator's data-acquisition
loop. Governs load balancing across the RaaS pool — the orchestrator
routes each request to the instance with the highest `available`.

Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `available` | `int` | Yes | Free slots right now. Return `0` to pause intake. |
| `inflight` | `int` | No | Currently running tasks. |
| `max_concurrency` | `int` | No | Ceiling, for logging. |

**Response** — raw JSON text (no envelope):

```json
{
  "available": 12,
  "inflight": 4,
  "max_concurrency": 16
}
```

Return slowly or never and you throttle the whole pipeline.

#### `POST /register_workflow`

Register a workflow class by string name plus its reward function.

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow_id` | `str` | Yes | Caller-assigned identifier (used later in `/submit`). |
| `workflow_cls` | `str` | Yes | Registry name of a `RolloutWorkflow` subclass. |
| `reward_fn` | `str` | No | Registry name of a reward function. |
| `gconfig_overrides` | `dict` | No | Sampling-param overrides for this workflow. |
| `workflow_kwargs` | `dict` | No | Extra constructor kwargs. |

**Request** (Python dict → `cloudpickle.dumps` → POST as
`application/octet-stream`):

```python
{
    "workflow_id": "rlvr-math",
    "workflow_cls": "RLVRWorkflow",
    "reward_fn": "math_verify",
    "gconfig_overrides": {"temperature": 0.7, "max_new_tokens": 1024},
    "workflow_kwargs": {"tokenizer_path": "/models/qwen3-8b"},
}
```

**Response** (full envelope — pickled):

```python
{
    "ok": True,
    "result": {},   # implementation-defined ack; AstraFlow only checks `ok`
}
```

#### `POST /submit` — Enqueue One Rollout

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | `dict` | Yes | Sample payload — whatever the workflow consumes. |
| `workflow_id` | `str` | No | Defaults to `"default"`. |

**Request** (Python dict, pickled):

```python
{
    "data": {"prompt": "What is 2+2?", "answer": "4"},
    "workflow_id": "rlvr-math",
}
```

**Response** (full envelope):

```python
{
    "ok": True,
    "result": {"task_id": 42},
}
```

Your handler should (1) allocate a task id, (2) create an
`asyncio.Task` wrapping `workflow.arun_episode(engine, data)`, and
(3) return immediately. The task stores its result in a completed-
results dict keyed by task_id when done.

#### `POST /pull` — Drain Completed Rollouts

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_items` | `int` | No | Default `256`. |
| `timeout` | `float` | No | Default `0.0` — non-blocking drain. |

**Request** (Python dict, pickled):

```python
{
    "max_items": 64,
    "timeout": 0.5,
}
```

**Response** (full envelope; `result` is a list):

```python
{
    "ok": True,
    "result": [
        # Successful task — `result` holds whatever arun_episode returned
        {
            "task_id": 42,
            "result": {
                "input_ids": [...],       # list[int] or torch.Tensor
                "output_ids": [...],
                "output_versions": [5, 5, 5, 5],  # per-token weight version
                "rewards": [0.0, 0.0, 0.0, 1.0],
                # ... workflow-defined fields
            },
        },
        # Rejected sample — workflow returned None
        {"task_id": 43, "result": None},
        # Per-task failure — workflow raised; error rides inside `result`
        {
            "task_id": 44,
            "result": {"ok": False, "error": "RuntimeError('timeout')"},
        },
    ],
}
```

The workflow's trajectory dict is passed through unmodified. Per-task
failures are reported inline so one bad sample doesn't fail the whole
drain.

#### `POST /notify_version` — Per-Model Weight Update

Fires when AstraFlow has new weights to load. **One call per
`(RaaS, model_id)`** — multi-model training sends N calls, not a
batched one.

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | `str` | Yes | Which model to update. Use `"default"` for single-model. |
| `version` | `int` | Yes | New weight version. Must be strictly greater than local. |
| `sender_endpoint` | `str` | Yes | `"host:port"` of the trainer's weight sender agent. |

**Request** (Python dict, pickled):

```python
{
    "model_id": "model0",
    "version": 5,
    "sender_endpoint": "10.0.0.1:19861",
}
```

**Response** — note **two `ok` fields**: the outer one is the
transport envelope; the inner one is the manager's semantic status
(e.g., whether any pull actually happened):

```python
# Successful update
{
    "ok": True,                     # envelope — handler ran without raising
    "result": {
        "ok": True,                 # manager — update succeeded
        "model_id": "model0",
        "version": 5,
        "pulled": True,
        "pull_result": {
            "mode": "full",         # or "delta"
            "shm_path": "/dev/shm/astraflow_weights/.../model0/model.safetensors",
        },
        "timing": {
            "pull_s": 1.05,
            "pause_s": 0.12,
            "load_s": 2.81,
            "resume_s": 0.08,
        },
    },
}

# Fast skip (version already loaded)
{
    "ok": True,
    "result": {
        "ok": True,
        "model_id": "model0",
        "pulled": False,
        "reason": "version=5 <= local=5",
    },
}

# Pull/load failed — envelope still ok, inner ok=False
{
    "ok": True,
    "result": {
        "ok": False,
        "model_id": "model0",
        "reason": "TCP pull timed out",
    },
}
```

**Required behavior:**

1. Acquire a per-model lock (`asyncio.Lock`) to serialize concurrent
   updates.
2. Pull weights from `sender_endpoint` via TCP (reuse
   `RaaSWeightReceiver`).
3. Save bytes as safetensors to a per-model shm directory.
4. Pause generation on that model's engine (in an executor thread,
   not inline).
5. Load weights from the shm path.
6. Resume generation.
7. Update local version tracking.

See [Weight Update Lifecycle](#weight-update-lifecycle-end-to-end)
below for the full sequence.

#### `POST /shutdown`

Destroy all engines, kill child processes, exit the uvicorn process.
Respond *before* exiting so the client doesn't see a connection reset.

**Request** — empty dict, pickled:

```python
{}
```

**Response** (full envelope):

```python
{
    "ok": True,
    "result": "shutting down",
}
```

### Inbound: Eval Support (Optional)

The five endpoints below are called **only** when the orchestrator
runs eval windows (any recipe with `eval.freq_steps > 0`). A
training-only RaaS can omit them entirely — AstraFlow will never
call them unless eval is enabled.

If you do want eval support, implement all five; they're co-dependent
(`/reset_training_engine` preconditions `/eval_*`, `/eval_start`
pairs with `/eval_end`, and `/eval_pull` consumes what `/eval_submit`
produces). Half-implementing the eval stack is worse than omitting it.

#### `POST /reset_training_engine`

Called before each eval window to quiesce the server. Cancel all
in-flight training rollout tasks, drain the underlying inference
engine, clear completed-results, and verify zero inflight requests.

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timeout` | `float` | No | Default `5.0`. Cap the drain. |

Response fields (inside `result`):

| Field | Type | Description |
|-------|------|-------------|
| `ready_for_eval` | `bool` | `True` when inflight drained under timeout. |
| `cancelled` | `int` | Tasks cancelled. |
| `stragglers` | `int` | Tasks still running after timeout. |
| `sglang_running` | `int` | Inflight requests at the inference backend. |
| `reset_epoch` | `int` | Monotonically increasing; used to invalidate late results. |

**Request** (Python dict, pickled):

```python
{"timeout": 10.0}
```

**Response** (full envelope):

```python
{
    "ok": True,
    "result": {
        "ready_for_eval": True,
        "cancelled": 32,
        "stragglers": 0,
        "sglang_running": 0,
        "reset_epoch": 7,
    },
}
```

#### `POST /eval_start`, `POST /eval_end`

Mark the boundary of an eval window so you can reset eval-specific
tracking counters (separate from training's).

**Request** — empty dict, pickled:

```python
{}
```

**Response** (full envelope, both endpoints):

```python
{"ok": True, "result": {}}  # ack only; payload is implementation-defined
```

#### `POST /eval_submit` — Enqueue One Eval Rollout

Same request shape as `/submit`. Internally route through a separate
eval task dict so training and eval do not share state.

**Request** (Python dict, pickled):

```python
{
    "data": {"prompt": "evaluate: ...", "ref": "..."},
    "workflow_id": "eval-math",
}
```

**Response** (full envelope):

```python
{
    "ok": True,
    "result": {"task_id": 101},
}
```

#### `POST /eval_pull` — Drain Eval Results (Note: Dict, not List)

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_items` | `int` | No | Default `256`. |
| `timeout` | `float` | No | Default `0.0`. |

**Request** (Python dict, pickled):

```python
{"max_items": 64, "timeout": 0.5}
```

**Response** — asymmetric with `/pull`: `result` is a **dict with
progress counters**, not a list:

```python
{
    "ok": True,
    "result": {
        "items": [
            {"task_id": 101, "result": {"score": 0.85, "tokens": [...]}},
            {"task_id": 102, "result": None},          # rejected
        ],
        "inflight": 3,           # tasks still running on RaaS
        "pending": 5,            # ready to drain but not yet pulled
        "total_submitted": 128,  # cumulative since eval_start
    },
}
```

The extra counters let the client detect stuck or lost tasks. Don't
flatten to a list — the eval manager expects this shape.

### Outbound: RaaS → AstraFlow

One call, once, at startup.

#### `POST /register_raas`

Join the pool. **JSON** (not pickle) — no envelope.

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `uid` | `str` | Yes | Unique id for this instance (used in logs and pool routing). |
| `raas_url` | `str` | Yes | Externally reachable `http://host:port` for this RaaS. |
| `gpu_count` | `int` | Yes | GPUs visible to this instance (used for load reporting). |

**Request** — raw JSON, `Content-Type: application/json`:

```json
{
  "uid": "raas-0-d4a2",
  "raas_url": "http://10.0.0.5:19190",
  "gpu_count": 4
}
```

**Response** — raw JSON, no envelope:

```json
{
  "pool_size": 2
}
```

**Before calling**, wait for your own `GET /status` to return
`"ready"`. Registering before engines are healthy causes the pool to
route traffic to a broken instance. Retry with backoff if AstraFlow
is not yet reachable (components can start in any order).

Reference: `astraflow/raas/server/__main__.py::_self_register`.

### Outbound: RaaS → Trainer (Weight Sender Agent)

Three endpoints on the trainer's weight sender, called during
`/notify_version`. These are fully documented from the trainer side in
[Custom Trainer Integration](custom-trainer.md#raas--trainer-inbound-weight-sender).
Summary:

| Method | Endpoint | Frequency | Purpose |
|--------|----------|-----------|---------|
| `GET` | `/get_buffer_info` | Once per trainer | Query tensor layout |
| `POST` | `/register_sglang_instance` | Once per trainer | Establish TCP session |
| `POST` | `/request_transfer` | Every weight update | Pull bytes via TCP |

**Strong recommendation:** use the reference `RaaSWeightReceiver`
(`astraflow/raas/server/tcp_receiver.py`) instead of rewriting the
TCP/ZMQ client. It handles the double-buffer protocol, parallel TCP
streams, and ZMQ completion signaling. Your RaaS just calls
`receiver.start(sender_endpoint, ...)` and `receiver.pull(version)`.

## Workflow Execution Contract

A workflow is user code that RaaS must execute correctly. Contract:

```python
# astraflow/core/workflow/api/workflow_api.py
class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        ...
```

- Must be `async`.
- Receives an `InferenceEngine` handle and the per-sample `data` dict.
- Returns the trajectory (any JSON/pickle-serializable dict) or
  `None` to reject the sample.
- The reward function (if any) is called **inside** `arun_episode`, not
  by RaaS. RaaS just passes `reward_fn` through at registration so the
  workflow can look it up.

The `InferenceEngine` protocol your RaaS passes to workflows
(`astraflow/core/workflow/api/engine_api.py`):

```python
class InferenceEngine(Protocol):
    async def agenerate(self, req: ModelRequest) -> ModelResponse: ...
    def get_version(self) -> int: ...
    def set_version(self, version: int) -> None: ...

    @asynccontextmanager
    async def managed_session(self): ...
```

For multi-model training, RaaS wraps per-model engines in an
`EngineGroup` so the workflow can call `engine["model0"].agenerate(...)`.

## Weight Update Lifecycle (end-to-end)

What your `/notify_version` handler must do, in order:

```
[client] POST /notify_version {model_id, version, sender_endpoint}
   │
   ▼
[handler] version <= local? → {ok: True, pulled: False} (fast skip)
   │
   ▼
[handler] acquire per-model asyncio.Lock
   │
   ▼
[executor] pull_weights_to_disk(sender_endpoint, model_id)
   │    ├── GET /get_buffer_info            (first pull only)
   │    ├── POST /register_sglang_instance  (first pull only)
   │    ├── POST /request_transfer
   │    ├── ← TCP bulk transfer
   │    ├── ← ZMQ transfer-complete signal
   │    └── write safetensors to /dev/shm/.../{model_id}/
   │
   ▼
[executor] engine.pause_generation()
[executor] engine.load_weights_from_path(shm_path)
[executor] engine.continue_generation()
   │
   ▼
[handler] self._weight_versions[model_id] = version
[handler] engine.set_version(version)
[handler] release lock → return {ok: True, pull_result, timing}
```

Every step involving inference backend I/O (pull, pause, load,
resume) must run in a thread executor via
`loop.run_in_executor(None, fn, ...)`. Running them inline blocks
other async endpoints including `/status`.

Reference: `RaaS3Manager._do_weight_update`
(`astraflow/raas/server/manager.py:1612`).

## Minimum Viable Skeleton

Below is a FastAPI skeleton that wires the required endpoints. Replace
the stubs with your inference backend and workflow runner.

```python
# my_raas/server.py
import asyncio
import pickle
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse

class Manager:
    def __init__(self):
        self._status = "starting"
        self._workflows: dict[str, Any] = {}
        self._inflight: dict[int, asyncio.Task] = {}
        self._done: dict[int, Any] = {}
        self._next_id = 0
        self._weight_versions: dict[str, int] = {}
        self._weight_locks: dict[str, asyncio.Lock] = {}
        # ... your inference engine handles go here

    async def bootstrap(self):
        # launch inference engines, load base model, etc.
        self._status = "ready"

    def get_status(self) -> dict:
        return {"status": self._status}

    async def get_availability(self) -> dict:
        return {
            "available": max(0, MAX_CONCURRENCY - len(self._inflight)),
            "inflight": len(self._inflight),
        }

    def register_workflow(self, *, workflow_id, workflow_cls,
                          reward_fn=None, gconfig_overrides=None, **kw):
        cls = resolve_workflow(workflow_cls)
        rfn = resolve_reward(reward_fn) if reward_fn else None
        self._workflows[workflow_id] = cls(reward_fn=rfn, **kw)
        return {}

    async def submit(self, *, data, workflow_id="default") -> int:
        wf = self._workflows[workflow_id]
        tid = self._next_id; self._next_id += 1
        task = asyncio.create_task(wf.arun_episode(self.engine, data))
        task.add_done_callback(lambda t, i=tid: self._done.update(
            {i: (t.result() if not t.exception() else
                 {"ok": False, "error": repr(t.exception())})}))
        self._inflight[tid] = task
        return tid

    async def pull_completed(self, *, max_items=256, timeout=0.0) -> list:
        if not self._done and timeout > 0:
            await asyncio.sleep(timeout)  # simple implementation
        out = []
        for tid in list(self._done)[:max_items]:
            out.append({"task_id": tid, "result": self._done.pop(tid)})
            self._inflight.pop(tid, None)
        return out

    async def notify_version(self, *, model_id, version, sender_endpoint):
        if version <= self._weight_versions.get(model_id, 0):
            return {"ok": True, "pulled": False}
        lock = self._weight_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            loop = asyncio.get_event_loop()
            shm_path = await loop.run_in_executor(
                None, self.weight_receiver.pull,
                sender_endpoint, model_id, version)
            await loop.run_in_executor(None, self.engine.pause_generation)
            await loop.run_in_executor(
                None, self.engine.load_weights_from_path, shm_path)
            await loop.run_in_executor(None, self.engine.continue_generation)
            self._weight_versions[model_id] = version
            self.engine.set_version(version)
        return {"ok": True, "model_id": model_id, "version": version}

    # ... eval_start / eval_end / eval_submit / eval_pull /
    # reset_training_engine / destroy ...

def _ok(result): return Response(
    pickle.dumps({"ok": True, "result": result}),
    media_type="application/octet-stream")
def _err(exc): return Response(
    pickle.dumps({"ok": False, "error": repr(exc)}),
    media_type="application/octet-stream", status_code=500)

def build_app(mgr: Manager) -> FastAPI:
    app = FastAPI()

    @app.get("/status")
    async def status(): return mgr.get_status()

    @app.get("/availability")
    async def avail(): return await mgr.get_availability()

    @app.post("/register_workflow")
    async def reg_wf(r: Request):
        try: p = pickle.loads(await r.body())
        except Exception as e: return _err(e)
        try: return _ok(mgr.register_workflow(**p))
        except Exception as e: return _err(e)

    # ... /submit, /pull, /notify_version, /shutdown ...
    # To support eval, also wire: /reset_training_engine,
    # /eval_start, /eval_end, /eval_submit, /eval_pull.
    return app
```

A real implementation should additionally: register with AstraFlow on
startup (`POST /register_raas`), wire a `RaaSWeightReceiver` into
`mgr.weight_receiver`, and plug in a real `engine` (SGLang, vLLM, or
custom).

Reference end-to-end: `astraflow/raas/server/__main__.py` +
`astraflow/raas/server/routes.py` + `astraflow/raas/server/manager.py`.

## Done Checklist

**Required** — a training-only RaaS is correct if:

- [ ] It answers `GET /status` in <100 ms even while a weight load is
      in progress.
- [ ] It registers a workflow, accepts `/submit`, runs it end-to-end,
      and returns the trajectory via `/pull`.
- [ ] Every output token in the returned trajectory carries the
      correct `output_versions` entry.
- [ ] After `/notify_version`, it serves the new weights without
      restart and without dropping in-flight requests started before
      the update.
- [ ] Concurrent `/notify_version` calls for the same `model_id` do
      not race on the safetensors file.
- [ ] It survives a trainer restart and continues to serve.
- [ ] It joins the pool via `POST /register_raas` only after its
      engines are healthy.
- [ ] `/shutdown` cleanly stops engines and exits the process.
- [ ] Multi-model: calls for different `model_id`s can proceed in
      parallel; only same-`model_id` calls serialize.

**Optional (only if eval support is implemented):**

- [ ] `/reset_training_engine` cancels in-flight tasks and reports
      zero inflight before returning.
- [ ] `/eval_start` / `/eval_end` reset eval tracking independently
      of training tracking.
- [ ] `/eval_pull` returns a **dict** with `items`, `inflight`,
      `pending`, `total_submitted` — not a list.
- [ ] Training and eval task state do not share queues.

## Reference Files

If you're starting from scratch, read these in order:

1. `astraflow/raas/server/routes.py` — all 12 endpoints, 290 lines.
2. `astraflow/raas/server/manager.py` — reference `RaaS3Manager`,
   especially `notify_version` (line 1556) and `_do_weight_update`
   (line 1612).
3. `astraflow/raas/server/__main__.py` — launcher and self-registration.
4. `astraflow/raas/server/tcp_receiver.py` — `RaaSWeightReceiver`
   (reuse this).
5. `astraflow/core/workflow/api/workflow_api.py` and
   `astraflow/core/workflow/api/engine_api.py` — the contracts your RaaS
   must honor for workflows.
6. `astraflow/dataflow/raas2_engine.py` — the client AstraFlow uses
   to talk to you; matching its method signatures is the
   ground-truth test of compatibility.

## API Summary

### Required (8 inbound + 1 outbound + 3 outbound)

| Direction | Category | Method | Endpoint | Frequency |
|-----------|----------|--------|----------|-----------|
| AstraFlow → RaaS | Health | `GET` | `/status` | Every ~10s (heartbeat) |
| AstraFlow → RaaS | Health | `GET` | `/availability` | Every submit tick |
| AstraFlow → RaaS | Rollout | `POST` | `/register_workflow` | Once per unique workflow |
| AstraFlow → RaaS | Rollout | `POST` | `/submit` | Per sample |
| AstraFlow → RaaS | Rollout | `POST` | `/pull` | Per drain |
| AstraFlow → RaaS | Weights | `POST` | `/notify_version` | Per model per step |
| AstraFlow → RaaS | Lifecycle | `POST` | `/shutdown` | Once at end |
| RaaS → AstraFlow | Lifecycle | `POST` | `/register_raas` | Once at start |
| RaaS → Trainer | Weights | `GET` | `/get_buffer_info` | Once per trainer |
| RaaS → Trainer | Weights | `POST` | `/register_sglang_instance` | Once per trainer |
| RaaS → Trainer | Weights | `POST` | `/request_transfer` | Every weight update |

### Optional — Eval Support (5 inbound)

Implement all five or none.

| Direction | Category | Method | Endpoint | Frequency |
|-----------|----------|--------|----------|-----------|
| AstraFlow → RaaS | Eval | `POST` | `/reset_training_engine` | Per eval window |
| AstraFlow → RaaS | Eval | `POST` | `/eval_start` | Per eval window |
| AstraFlow → RaaS | Eval | `POST` | `/eval_submit` | Per eval sample |
| AstraFlow → RaaS | Eval | `POST` | `/eval_pull` | Per eval drain |
| AstraFlow → RaaS | Eval | `POST` | `/eval_end` | Per eval window |

**Totals:** 11 required + 5 optional = **16 APIs**. The 3 trainer-side
endpoints are served by the reusable sender agent — see
[Custom Trainer Integration](custom-trainer.md) for their full
payload schemas.
