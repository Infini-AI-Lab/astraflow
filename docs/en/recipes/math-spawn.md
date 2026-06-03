# Math (Spawn sub-agents)

A math RL recipe where the main agent can emit a single `<spawn>` tool call
mid-generation to dispatch up to **four sub-agents in parallel**, then
continues with the aggregated sub-agent outputs spliced back into its
context. All N+1 sequences in the resulting trajectory share the team
reward and contribute gradient — main and sub-agents are the same policy.

**Recipe**: [`examples/math/spawn/qwen3-8b-spawn/`](https://github.com/Infini-AI-Lab/astraflow/tree/main/examples/math/spawn/qwen3-8b-spawn)

**Workflow class**: [`astraflow/core/workflow/impl/spawn.py`](https://github.com/Infini-AI-Lab/astraflow/tree/main/astraflow/core/workflow/impl/spawn.py) — registered as `spawn_rlvr`.

## Protocol

The main agent sees a fixed system prompt describing the tool. To call it,
it emits exactly one block:

```
<spawn>{"tasks": ["<task 1>", "<task 2>", ...]}</spawn>
```

The workflow:

1. Halts the main agent at `</spawn>` via a string-level stop.
2. Parses the JSON; caps `tasks` at 4 (extras dropped + logged).
3. Runs `len(tasks)` sub-agents in parallel via `asyncio.gather`. Each
   sub-agent sees a fixed sub-agent system prompt plus
   `Original problem:\n{problem}\n\nYour sub-task:\n{task}`.
4. Concatenates sub-agent outputs into a `<spawn_result>` block, appends
   it to the main agent's context, and continues generation to the final
   answer.
5. Computes `math_verify` reward on the main agent's final answer.

If the main agent emits no `<spawn>` (or a malformed payload), the
workflow falls back to vanilla single-turn RLVR — the recipe stays valid
across training when the model hasn't yet learned to use the tool.

## Training scheme

Each spawn-using episode returns **one trajectory containing 1 main + N
sub-agent sequences**, all inheriting the trajectory reward. Concretely:

| sequence | input (loss_mask=0) | output (loss_mask=1) | reward |
|---|---|---|---|
| main agent | problem prompt + `<spawn_result>…</spawn_result>` | pre-spawn reasoning + `<spawn>…</spawn>` + post-spawn reasoning + final answer | R |
| sub-agent 0 | fixed system + (problem + task₀) | sub-agent 0's reasoning + answer | R |
| … | … | … | R |
| sub-agent N | … | … | R |

`R = math_verify(main_final_answer, gt_answer)`.

- All sequences route to the same trainer (no `model_ids` tagging).
- GRPO/M2PO advantage normalization runs over `n_samples × (1 + sub-count)`
  sequences per prompt.
- **Credit assignment is noisy by design** (team reward — a sub-agent gets
  +1 even if its output was useless when the main agent still got the
  right answer). Future levers if needed: down-weight sub-agent loss vs
  main, or add per-sub-agent auxiliary reward (e.g. was the sub-agent's
  answer cited / used?). v1 uses equal weight, single shared reward.

## Run

One-time dataset prep (shared with the offline-math recipe):

```bash
python examples/math/offline/download_math_datasets.py --root data-data/math
```

Then:

```bash
bash examples/math/spawn/qwen3-8b-spawn/scripts/run_qwen3-8b-spawn.sh
```

Per-stage overrides (CLI args pass through to the trainer):

```bash
bash examples/math/spawn/qwen3-8b-spawn/scripts/run_qwen3-8b-spawn.sh \
  total_train_steps=3 evaluator.eval_at_start=true
```

## Settings

| Setting | Value |
|---|---|
| Model | Qwen3-8B |
| GPUs | 8 — RaaS ×4 (SGLang, DP=4), Trainer ×4 (FSDP, DP=4) |
| Algorithm | M2PO (`m2_threshold` 0.01) |
| Weight transfer | TCP, full |
| Context length | 16384 |
| Main max_new_tokens | 3000 (so phase-1 + agg + phase-2 fits 16k) |
| Sub-agent max_new_tokens | 1500 (×4 max → 6000 token aggregated injection) |
| Max sub-agents per spawn | 4 |
| Rollouts per prompt | 8 (`temperature` 1.0) |
| Workflow / reward | `spawn_rlvr` / `math_verify` |
| Train dataset | DeepScaleR (offline) |
| Eval datasets | AIME24, AIME25, AMC, Minerva, MATH500 (offline) |

## Caveats

- **Bootstrap problem.** A fresh Qwen3-8B-Base won't naturally emit
  `<spawn>` — the workflow ships a prompt-engineering system prompt that
  describes the tool with examples to seed exploration. If your model
  variant ignores the instructions, you'll see all trajectories degrade
  to vanilla RLVR (no spawn calls, no sub-agent training). Add an SFT
  warm-start with synthetic spawn-using trajectories if needed.
- **Context budget.** Phase-2 input = main_prompt + phase-1 output +
  `</spawn>` + aggregated sub-results. With the defaults
  (main=3000, sub=1500×4), the worst case is roughly
  prompt(~1000) + phase-1(3000) + agg(6000) + phase-2(3000) ≈ 13k,
  comfortably under the 16k SGLang window. If you bump
  `max_new_tokens` or `max_sub_agents`, also bump `context_length` in
  `raas.yaml` to match.
- **Model + tokenizer weights** still pull from HF Hub on first use, as
  with the offline-math recipe. Pre-fetch `Qwen/Qwen3-8B` for an
  air-gapped run.
