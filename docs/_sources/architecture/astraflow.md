# AstraFlow

The AstraFlow core (`astraflow/dataflow/`) is the orchestration layer that manages async data flow between rollout generation and training.

## Key Components

- **`AstraFlow`** — Main orchestrator that composes data acquisition and serving. Manages rollout buffers, filtering, staleness tracking, and replay logic.
- **`DataAcquisition`** — Runs threaded producer loops that pull rollouts from RaaS, filter accepted samples, and publish to data serving.
- **`DataServing`** — Owns fresh/replay buffers, manages staleness and normalization, and exposes batch-fetch APIs.
- **`RaaS2InferenceEngine`** — HTTP client that submits rollout requests to RaaS and polls for completions.
- **`RaaSPool`** — Manages a pool of RaaS instances with heartbeating and dynamic registration.
- **`AstraFlowService`** — Flask HTTP service exposing REST endpoints. Trainers register via `POST /ready`, fetch batches via `GET /batch`, and trigger weight sync via `POST /notify_version`. RaaS instances join the pool via `POST /register_raas` and are removed via `POST /deregister_raas`.

## How It Fits

AstraFlow sits between RaaS (upstream) and Trainer (downstream):

- Requests rollouts from RaaS via `RaaS2InferenceEngine`
- Buffers and serves training batches to Trainer via `AstraFlowService`
- Coordinates weight sync timing between Trainer and RaaS
