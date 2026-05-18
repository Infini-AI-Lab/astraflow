# WebShop Concurrency Fix

## Problem

The original WebShop task implementation had a critical concurrency bug that caused race conditions when multiple episodes ran concurrently (e.g., with AReaL's AsyncTaskRunner).

### Root Cause

The task used a **shared environment instance** (`self.env`) across all concurrent episodes:

```python
# BROKEN CODE (original)
class WebShop(Task):
    def __init__(self, **configs):
        self.env = WebAgentTextEnv(...)  # Single shared instance

    async def start_sample(self, index: int, session: Session):
        env = self.env  # All episodes share this!
        env.reset(index)
        ...
```

### Race Conditions

The shared `WebAgentTextEnv` instance has mutable state that gets overwritten by concurrent episodes:

- `self.session` - Current session ID
- `self.prev_obs` - Previous observations
- `self.prev_actions` - Previous actions
- `self.text_to_clickable` - Available actions
- `self.browser` - Browser state (current_url, page_source, session_id)

**Example race condition:**
```
Time 0: Episode A calls env.reset(10) → env.session = "10"
Time 1: Episode B calls env.reset(20) → env.session = "20" (OVERWRITES!)
Time 2: Episode A calls env.step(action) → uses env.session = "20" (WRONG!)
Time 3: Episode A gets observation → returns Episode B's page (CORRUPTED!)
```

This resulted in:
- Mixed-up observations between episodes
- Incorrect reward calculations
- Corrupted training trajectories
- Poor training performance

## Solution

The fix follows the pattern used in the AlfWorld task with three key components:

### 1. Shared Server Infrastructure

Create a single `SimServer` instance in `__init__` that's shared across episodes:

```python
class WebShop(Task):
    def __init__(self, **configs):
        # Shared server with read-only data and per-session isolation
        self.server = SimServer(
            base_url='http://127.0.0.1:3000',
            file_path=DEFAULT_FILE_PATH,
            ...
        )
```

This is safe because:
- Products, goals, and search engine are **read-only** after initialization
- Session state is properly **isolated** via `user_sessions` dict (keyed by session_id)
- Avoids expensive reloading of products/goals for every episode

### 2. Per-Episode Environment Instances

Create a **new** `WebAgentTextEnv` instance for each episode:

```python
async def start_sample(self, index: int, session: Session):
    # NEW environment instance per episode
    env = WebAgentTextEnv(
        observation_mode="text",
        human_goals=True,
        server=self.server  # Reuse shared server
    )
    env.reset(index)
    ...
```

This ensures complete **state isolation** between concurrent episodes.

### 3. Thread-Safe Operations

Add a global lock to protect thread-unsafe operations in `SimServer`:

```python
# Global lock to protect thread-unsafe operations
_webshop_lock = threading.Lock()

# Protect env.reset() and env.step()
with _webshop_lock:
    env.reset(index)

with _webshop_lock:
    observation, reward, done, info = env.step(action)
```

The lock protects:
- **Lucene search engine** (not thread-safe)
- **Timing stats** (search_time, render_time - shared counters)

## Verification

### Run Concurrency Test

```bash
cd AgentBench
python -m src.server.tasks.webshop.test_concurrency --num-episodes 10 --mode concurrent
```

This test:
1. Runs multiple episodes concurrently with different indices
2. Verifies each episode gets isolated environment state
3. Checks for race conditions and errors
4. Compares with sequential execution (baseline)

### Expected Output

```
✅ All episodes completed without errors - no race conditions detected!
```

### Integration Test with AReaL

Test with actual AReaL training:

```bash
# 1. Start WebShop task server
cd AgentBench
python -m src.server.task_server_adapter webshop-dev --port 5000 \
    --config configs/tasks/webshop.yaml

# 2. Run AReaL training with concurrent rollouts
cd AReaL
python -m areal.launcher.local examples/webshop/train.py \
    --config examples/webshop/config.yaml \
    experiment_name=webshop_concurrency_test \
    trial_name=run1
```

Monitor for:
- No errors in task server logs
- Consistent episode completion
- Reasonable reward values
- No observation mixups in trajectories

## Performance Considerations

### Lock Contention

The `_webshop_lock` serializes search and step operations. With high concurrency:

- **Sequential search latency**: Each search operation waits for the lock
- **Typical impact**: ~50-200ms per search (depending on search complexity)
- **Mitigation**: Adjust `max_queue_size` in AsyncTaskRunner to balance concurrency vs. lock contention

### Memory Usage

Creating a new environment per episode increases memory:

- **Previous**: 1 env instance (~100KB)
- **Current**: N env instances for N concurrent episodes (~N × 100KB)
- **Shared server**: Products/goals remain shared (~50MB total, not duplicated)

For most use cases (10-50 concurrent episodes), this is negligible.

## Alternative Approaches (Not Implemented)

1. **Lock-free server**: Make SimServer fully stateless/thread-safe
   - Pros: Better performance with high concurrency
   - Cons: Requires refactoring search engine and reward calculation

2. **Per-episode server**: Create separate SimServer for each episode
   - Pros: No locking needed
   - Cons: Expensive (reloads products/goals ~50MB per episode)

3. **Async-safe environment**: Use async-friendly data structures
   - Pros: Native asyncio support
   - Cons: Major refactoring of WebAgentTextEnv

The current approach balances **correctness**, **performance**, and **minimal code changes**.

## Related Files

- `AgentBench/src/server/tasks/webshop/__init__.py` - Fixed WebShop task class
- `AgentBench/src/server/tasks/webshop/test_concurrency.py` - Concurrency test
- `AgentBench/src/server/tasks/alfworld/task.py` - Reference implementation
- `AReaL/areal/core/async_task_runner.py` - Concurrent episode executor

## References

- AlfWorld task implementation: Similar pattern with per-episode envs and global lock
- AReaL AsyncTaskRunner: Manages concurrent episode execution
- TaskServerWorkflow: Calls `arun_episode()` concurrently via asyncio
