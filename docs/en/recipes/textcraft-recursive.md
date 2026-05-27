# TextCraft (Recursive Agent)

A multi-turn recursive-agent recipe ported from
[platoon](https://github.com/.../platoon)'s TextCraft design. The agent
acts in a stateful crafting environment (Minecraft-style recipes +
inventory) and can recursively spawn up to 4 sub-agents in parallel per
turn — each shares the parent's inventory by reference, so their work
mutates the same state.

**Recipe**: [`examples/textcraft/qwen3-4b-recursive/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/textcraft/qwen3-4b-recursive)

**Workflow class**: [`astraflow/core/workflow/impl/textcraft/workflow.py`](https://github.com/Infini-AI-Lab/astraflow/tree/main/astraflow/core/workflow/impl/textcraft/workflow.py) — registered as `recursive_agent`.

## How it works

### Tool-call protocol

Each turn the model emits **exactly one** action block:

```
<action type="get_info">{"items": ["stick", "oak_planks"]}</action>
<action type="view_inventory">{}</action>
<action type="craft">{"ingredients": {"oak_log": 1}, "target": ["oak_planks", 4]}</action>
<action type="spawn">{"subtasks": [
  {"targets": {"oak_planks": 16}, "max_steps": 8},
  {"targets": {"stick": 8}, "max_steps": 5}
]}</action>
<action type="finish">{"message": "crafted 4 wooden_pickaxe"}</action>
```

XML/JSON instead of platoon's Python-code-as-action because SGLang runs
with `--skip-tokenizer-init` (can't do string-stop) and we want zero
sandbox infrastructure. Base Qwen3 reads the format from the system
prompt with no SFT bootstrap needed.

### Stateful environment

[`TextCraftEnv`](https://github.com/Infini-AI-Lab/astraflow/tree/main/astraflow/core/workflow/impl/textcraft/env.py)
holds a mutable `inventory: dict[str, int]` and a shared read-only
recipe database (~860 Minecraft recipes bundled in
`astraflow/core/workflow/impl/textcraft/recipes/`).

`env.fork(child_task)` returns a child env whose `inventory` is **the
same dict object** as the parent's. When a sub-agent calls `craft`, the
mutation is visible to the parent. Single asyncio loop → no race.

### Spawning

A `<action type="spawn">` block runs all subtasks in parallel via
`asyncio.gather`, each as a full sub-episode with its own forked env
and trajectory. Up to 4 children per spawn, up to depth 3 (root + 2
levels of nesting). Sub-agents share the root's step budget.

### Aggregation — finish-message only (Option A, platoon-faithful)

The parent's view of a spawn is bounded — only each child's
`finish_message`:

```
<spawn_result>
<sub_agent_0 task="craft 16 oak_planks">crafted 16 oak_planks</sub_agent_0>
<sub_agent_1 task="craft 8 stick">crafted 8 stick</sub_agent_1>
</spawn_result>
```

The sub-agent's intermediate turns (other `craft` / `get_info` calls)
are NOT shown to the parent. This forces sub-agents to summarize their
work in `finish` messages and bounds context growth across recursion.

### Training scheme

One trajectory per episode containing **N sequences** (one per agent in
the tree — root + every descendant). Per-sequence layout:

| span | source | `loss_mask` |
|---|---|---|
| chat prompt + all prior turns' env tokens (observations) | env | 0 |
| this agent's own response tokens at each turn | model | **1** |

**Reward broadcast**: all sequences share the root reward
(`env.evaluate()` → 1.0 if every target_item is satisfied else 0.0).
**Depth-level weighting**: per-sequence reward is multiplied by
`1 / (depth + 1)` so deeper agents contribute less per-token weight
(matches platoon's `depth_level_weighting: true`).

## Run

One-time prep (synthesizes 1000 train + 100 val tasks locally from the
bundled recipe DB; no network required):

```bash
# Generated automatically on first launch; or force-regenerate:
python -c "from astraflow.dataflow.dataset.textcraft import download_dataset; download_dataset()"
```

Pre-fetch the model (one-time, ~8 GB):

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
```

Run:

```bash
bash examples/textcraft/qwen3-4b-recursive/scripts/run_qwen3-4b-recursive.sh
```

Smoke run (3 train steps, eval-at-start):

```bash
bash examples/textcraft/qwen3-4b-recursive/scripts/run_qwen3-4b-recursive.sh \
  total_train_steps=3 evaluator.eval_at_start=true evaluator.freq_steps=100
```

## Settings (matches platoon where compatible)

| Setting | Value | matches platoon? |
|---|---|---|
| Model | Qwen/Qwen3-4B-Instruct-2507 | ✅ |
| `enable_thinking` | `false` | ✅ |
| Algorithm | M2PO | ⚠ platoon uses CISPO (same GRPO family) |
| Fine-tuning | Full-FT | ⚠ platoon uses LoRA rank=32 |
| Inference backend | SGLang + RaaS + AstraFlow | ⚠ platoon uses Tinker |
| Tool-call protocol | XML / JSON | ⚠ platoon uses Python sandbox |
| `group_size` (n_samples) | 8 train / 1 eval | ✅ |
| `batch_size` | 32 | ✅ |
| `max_steps_per_episode` | 50 | ✅ |
| `lr` | 3e-5 | ✅ |
| Adam (β₁, β₂) | (0.9, 0.95) | ✅ |
| `grad_clip` | 0 (off) | ✅ (platoon: 1e12) |
| `max_staleness` | 3 | ✅ |
| `total_train_steps` | 100 | ✅ |
| Eval cadence | every 20 steps | ✅ |
| `max_depth` | 3 | platoon: unbounded — AstraFlow safety cap |
| `max_breadth` | 4 | platoon: unbounded — AstraFlow safety cap |
| `max_concurrent_subagents` | 8 | bounds K^N RaaS queue blowup |
| `delegation_reward_cap` | 0.0 | ✅ (= `_TEXTCRAFT_DELEGATION_REWARD_CAP`) |
| `depth_level_weighting` | true | ✅ |
| Dataset | TextCraft 1000 train / 100 val (original Minecraft recipes) | ✅ |
| SGLang context_length | 32768 | bumped from math recipe's 16k for recursion overhead |

## Caveats

- This is a **design reproduction**, not a **results reproduction**.
  The differences in trainer / FT regime / backend / tool-call surface
  (marked ⚠ above) mean wall-clock and final scores can differ. The
  intent is to validate the architecture port and have a path to
  reproduce the qualitative behavior (multi-turn recursion, shared
  inventory, finish-message bottleneck, team reward).
- Tasks are synthesized locally from the bundled `recipes/` JSON
  directory; no network required at training time (unlike the math
  recipes which used HF Hub datasets pre-`offline-math`).
- Each parent's context grows with every observation. The chat history
  for a 50-step root agent can hit several thousand tokens by the end
  even before any spawn injection. We picked `context_length: 32768`
  for headroom; raise it if you increase `max_steps_per_episode` or
  `max_breadth`.
