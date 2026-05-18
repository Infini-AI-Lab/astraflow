# Delta Weight Transfer

Delta weight transfer is an optional mode that sends only changed
bf16 elements instead of the full model. After a single RL optimizer
step, >98% of weights are bit-identical, so the delta is typically
30-40x smaller than the full model.

## How It Works

The trainer config declares supported strategies (`["full", "delta"]`).
RaaS decides which mode to use per pull based on sender capabilities
and its own state.

```
Trainer                SenderAgent (subprocess)           RaaS
  │                         │                              │
  offload()                 │                              │
  (GPU→CPU copy, ~0.5s)    │                              │
  notify_buffer_ready ────► │                              │
                     swap + ack immediately                │
  ◄──── ack ─────────────── │                              │
  save checkpoint           │ _compute_delta() [async]     │
  wait_delta_ready()        │ compare halves (numpy)       │
  ◄──── metrics ────────────│ encode sparse → delta shm    │
                            │ set delta_done_event         │
  notify_version ──────────────────────────────────────► │
                            │         GET /get_capabilities │
                            │ ◄──────── delta_ready? ──────│
                            │         POST /request_transfer│
                            │ ◄──────── mode=delta ────────│
                            │ TCP sendfile(delta) ────────►│
                            │ ZMQ done ───────────────────►│
                            │              mmap patch + load│
```

## Enabling Delta

**Trainer** — set the environment variable:

```bash
export WEIGHT_TRANSFER_STRATEGIES="full,delta"
```

**RaaS** — add to the YAML config under the `raas` section:

```yaml
raas:
  delta_full_sync_interval: 10   # full resync every 10 steps
```

When `delta_full_sync_interval` is 0 (default), RaaS uses delta
whenever available and never forces full. A non-zero value causes
a periodic full transfer for resync (e.g. every 10th step).

## Mode Decision

RaaS decides per-pull in `_choose_transfer_mode()`:

| Condition | Mode | Reason |
|-----------|------|--------|
| First pull (local_v=0) | full | No local weights to patch |
| `delta` not in sender strategies | full | Sender only supports full |
| Delta not ready | full | Async compute not finished |
| `local_v % full_sync_interval == 0` | full | Periodic resync |
| `local_v != delta_base_version` | full | Version mismatch (>1 step behind) |
| All checks pass | delta | Apply sparse patch |

## Sparse Format

```
[header 16 bytes][indices][values]

Header:
  [0:8]   num_nonzero     (uint64)
  [8:10]  element_size    (uint16, 2 for bf16)
  [10:12] flags           (uint16, bit 0 = uint64 indices)
  [12:16] reserved

Indices:  num_nonzero x 4 bytes (uint32) or 8 bytes (uint64)
Values:   num_nonzero x element_size bytes (raw bf16)
```

The indices are element offsets into the flat weight buffer (same
layout as the safetensors data section). Values are the **new** element
values (not deltas) — the receiver overwrites at each index.

## Delta Computation

Runs in the sender agent subprocess (CPU, numpy), asynchronously
after acking the trainer:

1. View both buffer halves as `uint16` arrays (bf16 representation)
2. `diff_mask = new_arr != old_arr` — vectorized comparison
3. `nonzero_indices = np.where(diff_mask)` — find changed elements
4. Encode header + indices + values into the delta shm buffer

Typical performance for a 1.7B model: ~1.8s compute, ~99% sparsity,
~95 MB delta (vs 3.9 GB full = 40x compression).

## Delta Application (RaaS)

RaaS applies the delta via mmap in-place patching:

1. Open the existing safetensors file in `/dev/shm` with `mmap`
2. Create a numpy view of the weight data section
3. Vectorized scatter write: `weight_2d[indices] = values`
4. `mmap.flush()` — data is ready for SGLang to load

No read-copy-write cycle — only the changed elements are touched.
Typical time: ~1.1s for 1.7B.

## Buffer Layout

```
/dev/shm/astraflow_buffer_XXXX  (2x model, same as full transfer)
┌─────────────┬─────────────┐
│   Half 0    │   Half 1    │
└─────────────┴─────────────┘

/dev/shm/astraflow_delta_XXXX   (1x model, for sparse delta)
┌─────────────────────────────┐
│  [header][indices][values]  │
└─────────────────────────────┘
```

The double buffer trick provides both current and previous versions
for the delta comparison. No extra "previous copy" buffer is needed.

## Guard Barrier

The delta computation reads the inactive half asynchronously. A guard
barrier at the start of `offload()` prevents all ranks from writing
to the inactive half until the previous delta finishes:

```python
# In offload():
self._wait_previous_delta()   # rank 0 waits on delta_done_event
dist.barrier()                 # all ranks sync before writing
```

Normally instant (<5ms). Only blocks if the training step is faster
than delta compute — unlikely for models larger than 1.7B.

## Timeline

```
Time ──────────────────────────────────────────────────────────────────────►

TRAINER (GPU)
  │ train_step N               │guard│offload│save│wait │notify│ train_step
  │[===========================][---][======]│chkp│delta│async │[==========
  │ forward/backward/optim     │~3ms│~0.5s  │    │ready│      │
  │                            │    │       │    │     │      │

SENDER AGENT (CPU)
  │ · · · · · · · · · · · · · ·│swap│ack│ delta compute      │done│ · · · ·
  │         idle               │idx │   │[==================]│evt │   idle
  │                            │    │   │ ~1.8s (overlapped) │    │

RAAS (GPU)
  │ generating · · · · · · · · · · · · · · │choose│delta│pause│load│resume
  │                                        │mode  │pull │[===]│[==]│
  │                                        │      │~1.1s│3.0s │0.7s│
```

## Measured Performance (1.7B model, 20 steps)

| Metric | Full | Delta |
|--------|------|-------|
| Trainer offload (blocking) | 0.5s | 0.5s (same) |
| Delta compute (async) | — | 1.8s (overlapped) |
| Guard barrier | — | <5ms |
| TCP transfer | 1.0s | 0.025s |
| Save/Patch on RaaS | 1.4s | 1.1s |
| **Pull total** | **2.4s** | **1.1s** |
| Sparsity | — | 98.4-99.2% |
| Compression ratio | 1x | 21-43x |

## Wandb Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| `weight_transfer/offload_guard_time` | trainer | Guard barrier wait |
| `weight_transfer/offload_copy_time` | trainer | GPU→CPU copy |
| `weight_transfer/offload_total_time` | trainer | End-to-end offload |
| `weight_transfer/delta_sparsity` | sender agent | Fraction unchanged |
| `weight_transfer/delta_size_mb` | sender agent | Sparse delta size |
| `weight_transfer/delta_compute_time` | sender agent | Async compute time |
| `weight_transfer/use_full` | RaaS via AstraFlow | 0=delta, 1=full |
