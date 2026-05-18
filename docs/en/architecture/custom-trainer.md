# Custom Trainer Integration

AstraFlow's trainer is a swappable component. You can replace the built-in
PPO trainer with your own training framework as long as it speaks the
correct HTTP protocol. This guide documents the minimal API your custom
trainer must implement.

This guide assumes **TCP weight transfer mode**, which is the recommended
mode for custom trainers because the trainer only needs to talk to
AstraFlow — no direct RaaS communication required for data or coordination.

## Architecture

A trainer has two communication surfaces:

```
                   ┌───────────────────────┐
   Outbound HTTP   │                       │   Inbound HTTP
   (3 endpoints)   │    Custom Trainer     │   (3 endpoints)
                   │                       │
  Trainer ────────►│                       │◄──────── RaaS
  calls AstraFlow  │  ┌─────────────────┐  │  calls trainer's
                   │  │  Sender Agent   │  │  weight sender
                   │  │  (reusable lib) │  │
                   │  └─────────────────┘  │
                   └───────────────────────┘

  Registration ──►  AstraFlow             RaaS  ──► Weight connection
  Data pull    ──►  Service               instances  Weight pull
  Version notify►                         (1..N)
```

- **Outbound (trainer → AstraFlow)**: 3 HTTP calls for registration, data,
  and version notification.
- **Inbound (RaaS → trainer)**: 3 HTTP endpoints served by the weight
  sender agent. RaaS connects to pull updated weights via TCP.

The weight sender agent (`astraflow.train_worker.weight_transfer.sender_agent`)
is provided as a reusable library. Custom trainers do not need to reimplement
the TCP/ZMQ transfer machinery.

## API Reference

### Wire formats

- **Pickle / cloudpickle** (`Content-Type: application/octet-stream`) —
  all three Trainer → AstraFlow endpoints. Payload is a Python dict
  before `cloudpickle.dumps`.
- **JSON** — all three RaaS → Trainer sender endpoints.

Note on envelope shapes: unlike RaaS, AstraFlow's pickle endpoints do
**not** use a nested `{ok, result}` wrapper. Responses are flat dicts
with an `ok` field alongside whatever other keys the endpoint returns.
`GET /batch` returns its payload with no `ok` at all — the HTTP status
is the only success signal. Below, `python` code fences show the
Python dict before pickling; `json` fences show literal JSON text.

### Trainer → AstraFlow (outbound)

These are HTTP calls your trainer makes to the AstraFlow service.

#### Registration: `POST /ready`

Signal that the trainer is ready to receive data. AstraFlow starts data
acquisition only after both RaaS and trainer have signalled readiness.

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `train_batch_size` | `int` | Yes | Number of examples per training batch |
| `model_id` | `str` | No | Model identifier for multi-model setups |
| `sender_endpoint` | `str` | Yes (TCP) | `"host:port"` of the trainer's weight sender HTTP server |
| `recovered_version` | `int` | No | Weight version to resume from after checkpoint recovery |

**Request** (Python dict, pickled):

```python
{
    "train_batch_size": 512,
    "model_id": "model0",              # omit for single-model
    "sender_endpoint": "10.0.0.1:18861",
    "recovered_version": 0,            # omit for fresh start
}
```

**Response** (flat, no envelope — pickled):

```python
{"ok": True}
```

#### Data: `GET /batch`

Pull a training batch. **Blocks** until sufficient data is available in
the buffer.

Query parameters (not a pickle body — real URL query string):

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | `str` | No | Which model's data to pull (multi-model) |
| `version` | `int` | No | Trainer's current version (used for multi-model barrier sync) |

**Request** — a plain HTTP GET:

```
GET /batch?model_id=model0&version=42
```

**Response** — pickled Python dict, **no envelope** (the payload is
the batch directly):

```python
{
    "batch": {
        "input_ids": Tensor,         # [batch, seq_len]
        "rewards": Tensor,           # [batch, seq_len]
        "logprobs": Tensor,          # [batch, seq_len]
        "loss_mask": Tensor,         # [batch, seq_len]
        # ... other fields depending on workflow
    },
    "buffer_stats": {
        "buffer/size": 1024,
        "buffer/staleness_mean": 1.2,
        # ... other buffer/filter metrics for wandb
    },
}
```

#### Weights: `POST /notify_version`

Notify AstraFlow that new weights are available. AstraFlow's Python-side
barrier waits until every registered `model_id` has reported the same
version, then fans out one `POST /notify_version` per model to every
live RaaS instance.

Request fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | `int` | Yes | New model version after training step |
| `run_eval` | `bool` | No | If `true`, AstraFlow runs eval and blocks until complete |
| `model_id` | `str` | No | Model identifier (multi-model) |

**Request** (Python dict, pickled):

```python
{
    "version": 42,
    "run_eval": False,
    "model_id": "model0",    # omit for single-model
}
```

**Response** (flat, no nested envelope — pickled):

```python
# Non-eval step
{
    "ok": True,
    "eval_results": None,
    "weight_transfer_info": {"use_full": 1},   # only present for TCP mode
}

# Eval step
{
    "ok": True,
    "eval_results": {
        "math/acc": 0.512,
        "math/pass@1": 0.47,
        # ... benchmark-specific keys
    },
    "weight_transfer_info": {"use_full": 0},
}
```

### RaaS → Trainer (inbound, weight sender)

These are HTTP endpoints that RaaS calls on the trainer's weight sender
server. The sender agent library handles these automatically — listed here
for completeness.

#### `GET /get_buffer_info`

RaaS queries the trainer's model tensor layout to allocate its receive
buffer. Called **once** when a RaaS instance first connects.

**Response** (JSON):

```json
{
  "single_buffer_length": 3489660928,
  "tensors_meta": [
    ["model.layers.0.self_attn.q_proj.weight", [[2048, 2048], "bfloat16"]],
    ["model.layers.0.self_attn.k_proj.weight", [[512, 2048], "bfloat16"]]
  ]
}
```

#### `POST /register_sglang_instance`

RaaS registers itself as a weight receiver. Both sides exchange TCP session
IDs and buffer pointers to establish a persistent TCP connection. Called
**once** per RaaS instance, after `get_buffer_info`.

**Request** (JSON):

```json
{
    "sglang_http_host": "10.0.0.5",
    "sglang_http_port": 19190,
    "session_ids": ["abc123"],
    "buffer_ptr": 140234567890,
    "buffer_length": 3489660928,
    "zmq_endpoint": "10.0.0.5",
    "zmq_port": 45678,
    "handshake_ports": [21000]
}
```

**Response** (JSON):

```json
{
    "trainer_global_rank": 0,
    "trainer_world_size": 2,
    "trainer_session_ids": [["def456"]],
    "trainer_buffer_ptr": 140111222333,
    "trainer_buffer_length": 6979321856,
    "trainer_hostname": "10.0.0.1",
    "trainer_rpc_port": 21000
}
```

#### `POST /request_transfer`

RaaS requests a weight pull. The sender acquires a buffer lock, performs
TCP bulk copy from the active buffer half to the receiver, and sends a
ZMQ completion signal. Called **every training step** (or when RaaS detects
it is behind the latest version).

**Request** (JSON):

```json
{
    "instance_id": "10.0.0.5:19190"
}
```

**Response** (JSON):

```json
{
    "ok": true,
    "version": 42
}
```

### Connection Lifecycle

The weight transfer connection between a trainer and each RaaS instance
follows this lifecycle:

```
RaaS                                          Trainer Sender
 │                                                │
 │  1. Query tensor layout                        │
 ├──── GET /get_buffer_info ─────────────────────►│
 │◄─── {tensors_meta, buffer_length} ─────────────┤
 │                                                │
 │  2. Allocate receive buffer locally             │
 │  3. Create TCP engine locally                   │
 │  4. Start ZMQ listener locally                  │
 │                                                │
 │  5. Exchange TCP details, establish connection  │
 ├──── POST /register_sglang_instance ───────────►│
 │◄─── {trainer_session_ids, rpc_port, ...} ──────┤
 │                                                │
 │  ═══════ persistent TCP connection open ════════│
 │                                                │
 │  6. Per training step: pull weights             │
 ├──── POST /request_transfer ───────────────────►│
 │◄════ TCP bulk copy (shared-mem → shared-mem) ══┤
 │◄──── ZMQ "transfer complete" ──────────────────┤
 │                                                │
 │  (repeat step 6 each training step)            │
```

Steps 1-5 happen once per RaaS instance. Step 6 repeats every training
step. The TCP connection persists across all transfers.

## Weight Transfer: Double-Buffer Design

The sender agent uses a double-buffer in shared memory (`/dev/shm`) so
that weight copying and weight serving never block each other:

1. **Trainer** writes updated weights to the **inactive** buffer half via
   `copy_weights_to_buffer()`.
2. **Trainer** atomically swaps the active buffer index — new transfers
   now read from the freshly-written half.
3. **RaaS** calls `POST /request_transfer` — sender reads from the
   **active** half and performs TCP bulk copy to the receiver.
4. **RaaS** saves the received bytes as safetensors in `/dev/shm` and
   tells the inference engine to reload.

This means training can continue writing the next set of weights while
RaaS is still pulling the current set.

## Complete Example

```python
from astraflow.train_worker.trainer.astraflow_client import AstraFlowClient

# --- Startup ---

# 1. Start the weight sender agent (reusable library).
#    This launches a subprocess that serves /get_buffer_info,
#    /register_sglang_instance, and /request_transfer.
sender = start_sender_agent(model)

# 2. Connect to AstraFlow and signal readiness.
client = AstraFlowClient(
    service_url="http://astraflow-host:8000",
)
client.initialize()          # wait for AstraFlow service to be ready
client.signal_ready(
    train_batch_size=512,
    sender_endpoint=sender.endpoint,   # e.g. "10.0.0.1:18861"
)

# --- Training Loop ---

version = 0
while training:
    # Pull batch (blocks until data is available)
    batch, buffer_stats = client.get_batch(version=version)

    # Your training logic
    loss = train_step(batch)

    # Copy updated weights to shared-memory buffer
    copy_weights_to_buffer()
    sender.notify_buffer_ready(version + 1)

    # Notify AstraFlow — it broadcasts to all RaaS instances
    version += 1
    should_eval = (version % eval_freq == 0)
    eval_results = client.notify_version(
        version=version,
        run_eval=should_eval,
    )

    if eval_results:
        log_eval(eval_results)

# --- Shutdown ---
client.drain_pending_notifications()
client.shutdown_service()
```

## API Summary

| Direction | Category | Method | Endpoint | Frequency |
|-----------|----------|--------|----------|-----------|
| Trainer → AstraFlow | Registration | `POST` | `/ready` | Once |
| Trainer → AstraFlow | Data | `GET` | `/batch` | Every step |
| Trainer → AstraFlow | Weights | `POST` | `/notify_version` | Every step |
| RaaS → Trainer | Weights | `GET` | `/get_buffer_info` | Once per RaaS |
| RaaS → Trainer | Weights | `POST` | `/register_sglang_instance` | Once per RaaS |
| RaaS → Trainer | Weights | `POST` | `/request_transfer` | Every step |

Total: **6 APIs**. The 3 inbound endpoints are handled by the reusable
sender agent library.
