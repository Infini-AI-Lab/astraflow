# Trainer

The Trainer (`astraflow/train_worker/`) is the distributed training engine
supporting FSDP and Megatron backends.

## Design Principle: Swappable Trainer

The trainer is a **swappable component**. AstraFlow provides a built-in
PPO trainer, but the system is designed so that users can replace it with
their own training framework. The trainer communicates with Dataflow
purely over HTTP — there is no shared library or class hierarchy coupling
the two.

Customizing a trainer is extremely simple — the entire integration surface
is just **6 HTTP APIs** (3 outbound calls to Dataflow + 3 inbound
endpoints for weight transfer). Any training framework (PyTorch, JAX,
Megatron, custom) can integrate with Dataflow as long as it speaks this
protocol:

```
┌──────────────┐     HTTP (3 calls)     ┌──────────────┐
│              │ ──────────────────────► │              │
│   Trainer    │                        │  Dataflow    │
│  (swappable) │                        │  (stable)    │
│              │ ◄────────────────────  │              │
└──────┬───────┘                        └──────────────┘
       │
       │  TCP weight pull (3 endpoints)
       │
┌──────▼───────┐
│    RaaS      │
│  (swappable) │
└──────────────┘
```

The trainer interacts with two components:

- **Dataflow service** (outbound) — 3 HTTP calls for registration, data
  pulling, and version notification.
- **RaaS** (inbound) — 3 HTTP endpoints served by a weight sender agent
  that RaaS connects to for pulling updated weights.

### Trainer → Dataflow API

| Category | Method | Endpoint | Purpose |
|----------|--------|----------|---------|
| Registration | `POST` | `/ready` | Signal readiness, pass batch size and sender endpoint |
| Data | `GET` | `/batch` | Pull a training batch (blocks until available) |
| Weights | `POST` | `/notify_version` | Notify new weight version, trigger RaaS broadcast |

### RaaS → Trainer API (weight sender)

| Category | Method | Endpoint | Frequency |
|----------|--------|----------|-----------|
| Weights | `GET` | `/get_buffer_info` | Once per RaaS (query tensor layout) |
| Weights | `POST` | `/register_sglang_instance` | Once per RaaS (establish TCP link) |
| Weights | `POST` | `/request_transfer` | Every step (pull weights over TCP) |

The weight sender is provided as a reusable library
(`astraflow.weight_manager.transfer.sender_agent`) so custom trainers
don't need to reimplement TCP/ZMQ machinery. See
[WeightManager](weight-manager.md) for details.

For a full guide on implementing a custom trainer, see
[Custom Trainer Integration](custom-trainer.md).
