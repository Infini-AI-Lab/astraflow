<!-- Operations guide for AI coding agents working on AstraFlow. -->

# AGENTS.md — AstraFlow Agent Operations Guide

## TL;DR for coding agents

- **Runtime**: Distributed GPU clusters (FSDP2 / Megatron). Assume
  containerized, multi-GPU, multi-node execution; do not invent
  standalone local runs.
- **Architecture**: Four cleanly separated components — Dataflow,
  Train Worker, RaaS, Workflow. See
  `docs/en/architecture/overview.md` for the long form.
- **Testing**: Most integration / FSDP / Megatron / RPC tests require
  multi-GPU hardware. Public CI (`.github/workflows/`) runs only lint,
  format, and docs — no GPU tests. State skips explicitly when you
  cannot run a suite.
- **Tooling**: `.pre-commit-config.yaml` runs Ruff (lint + format),
  mdformat, nbstripout, file-hygiene hooks, and the CLI doc generator.
  Install hooks with `pre-commit install` before submitting patches.
- **Formatting**: Ruff (`ruff==0.14.9`) is the single source of truth —
  both pre-commit and the `format-check` CI run `ruff check` and
  `ruff format`. The `[tool.black]` block in `pyproject.toml` is
  legacy and unused.
- **Docs**: Source lives under `docs/` (Sphinx). English pages
  under `docs/en/` with an `index.rst` toctree.
- **Collaboration**: Before non-trivial edits, outline the proposed
  plan and confirm with the user.

When unsure, leave a `TODO(agent)` comment and note the constraint in
your response.

## Repository map

- `astraflow/` — Top-level Python package, split by component:
  - `astraflow/dataflow/` — Async data flow, rollout buffering,
    replay, staleness management, HTTP service layer. Key classes:
    `AstraFlow` (`service.py`), `DataAcquisition`
    (`data_acquisition.py`), `DataServing` (`data_serving.py`),
    `RaaS2InferenceEngine` (`raas2_engine.py`), `RaaSPool`
    (`raas_pool.py`).
  - `astraflow/dataflow/dataset/` — Rollout / eval dataset loaders
    (deepscaler, alfworld, webshop, livecodebench, gsm8k, ...).
  - `astraflow/dataflow/tests/` — Dataflow unit tests.
  - `astraflow/train_worker/` — Training engine (swappable). Owns
    PPO actor/critic, FSDP2/Megatron backends, launchers, model
    adapters, recovery/saving.
    - `api/cli_args.py` — Dataclass configs validated by
      Hydra/OmegaConf. Source of truth for CLI options and YAML
      schemas. `api/io_struct.py` holds runtime structs such as
      `GenerationHyperparameters`.
    - `engine/` — Training/inference engines (FSDP2, Megatron,
      PPO actor/critic, SFT).
    - `launcher/` — `local.py`, `ray.py`, `slurm.py` launchers plus
      `rpc/` and vLLM/SGLang server launchers.
    - `models/` — Megatron-Core (`mcore/`) and HF Transformers
      (`transformers/`) adapters.
    - `trainer/` — `AstraFlowPPOTrainer` (`ppo_trainer.py`) and
      related orchestration.
    - `platforms/` — CPU / CUDA / NPU abstractions.
    - `tools/` — Developer utilities (e.g. validation, profiling).
    - `utils/` — Cross-cutting helpers: logging, stats, data,
      saver, recover, megatron helpers, FSDP helpers.
  - `astraflow/raas/` — RaaS (Remote Agentic Serving). Launches
    vLLM/SGLang servers, exposes HTTP endpoints for rollout
    generation and weight updates.
    - `server/` — Manager, TCP receiver, FastAPI app.
    - `engine/` — Remote inference engine adapters.
    - `api/cli_args.py` — RaaS-side dataclass configs.
  - `astraflow/core/workflow/` — Rollout workflows and reward functions
    (swappable).
    - `api/` — Base interfaces: `RolloutWorkflow`
      (`workflow_api.py`), `AsyncRewardWrapper` (`reward_api.py`).
    - `impl/` — Concrete workflows: `rlvr`, `multi_turn`,
      `vision_rlvr`, `solve_and_verify`, `actor_and_verify`,
      `code_*`, `plan_and_solve`, `sm_lg_router`, etc.
    - `impl/agentbench/`, `impl/asearcher/` — environment-specific
      workflow families.
    - `reward/` — Reward callables (`math_verify`,
      `livecodebench_reward`, `human_eval_reward`, `geometry3k`,
      `clevr_count_70k`).
    - `registry.py` — Decorator-based registries for workflows
      and rewards (`WORKFLOW_REGISTRY`, `REWARD_REGISTRY`).
    - `__init__.py` — Imports every `impl/` and `reward/` module so
      the registration decorators run at import time.
  - `astraflow/core/weight_manager/` — Weight transport (TCP/ZMQ)
    between trainer and RaaS. `transfer/tests/` holds its unit tests.
  - `astraflow/core/config/` — Hydra/OmegaConf config loader and merging.
- `astraEnv/` — Vendored environment code (not part of the package):
  `AgentBench` (alfworld, webshop), `ASearcher` (retrieval-augmented
  search), `human-eval`. Carries upstream licenses; treat as a
  read-only dependency unless coordinating with maintainers.
- `examples/` — Runnable training recipes grouped by task type
  (`math/`, `code/`, `math-multi-agent/`, `code-multi-agent/`,
  `alfworld/`, `webshop/`, `search/`, `math-efficient-data/`), plus
  shared helpers in `_common/` and `launch_trainer.py`. Each recipe
  ships a `yaml/` directory of configs and a `scripts/` directory of
  numbered launch scripts.
- `docs/` — Sphinx sources. English pages under `docs/en/`
  (architecture / recipes / get-started / developer-guide /
  references). CLI reference is generated by
  `docs/generate_cli_docs.py`.
- `docker/` — `Dockerfile.sglang`, the published image (astraflow +
  SGLang + flash-attn). See `docker/README.md` for build details.
- `docs/assets/` — Figures and animations referenced by the README and docs.

## Architecture (four components)

1. **Dataflow** (`astraflow/dataflow/`) — Producer/consumer split
   between rollout generation and training. `AstraFlow` is the
   top-level service. `DataAcquisition` runs producer threads with
   filtering and staleness gating; `DataServing` exposes the buffered
   batch over HTTP. `RaaS2InferenceEngine` is the HTTP client to RaaS.
2. **Train Worker** (`astraflow/train_worker/`) — Swappable training
   engine. `AstraFlowPPOTrainer`
   (`astraflow/train_worker/trainer/ppo_trainer.py`) pulls training
   batches from the Dataflow HTTP service rather than embedding
   rollout orchestration.
3. **RaaS** (`astraflow/raas/`) — Remote Agentic Serving. Launches
   vLLM / SGLang inference servers and exposes `/availability`,
   `/eval_pull`, weight-update, and rollout endpoints. See
   `docs/en/architecture/raas.md` and `custom-raas.md`.
4. **Workflow** (`astraflow/core/workflow/`) — Rollout workflows
   (subclasses of `RolloutWorkflow` with `arun_episode`) and reward
   callables. New workflows/rewards plug in via decorator
   registration in `registry.py`.

## Code style & patterns

Conventions for all code under `astraflow/`:

- **Logging**: `astraflow.train_worker.utils.logging.getLogger(__name__)`
  (workflow code uses `astraflow.core.workflow.utils.logging`) — never
  `print`. Emit metrics through `stats_tracker.get("scope")` and
  `StatsLogger`; the latter pushes to W&B / SwanLab.
- **Async**: Rollout workflows are non-blocking. `await` with
  `aiofiles`; never call synchronous file I/O inside `arun_episode`;
  push CPU-heavy work to executors.
- **Tensor shapes**: Padded batches are `[batch, seq_len, ...]`. Use
  the helpers in `astraflow/train_worker/utils/data.py` (and
  `astraflow/dataflow/utils.py` for `concat_padded_tensors`) for
  padding / broadcasting / splitting. The `check_trajectory_format`
  config flag (in `api/cli_args.py`) enables runtime shape validation.
- **Typing**: Explicit type hints. Reuse dataclasses from
  `astraflow/train_worker/api/cli_args.py` (or
  `astraflow/raas/api/cli_args.py` for RaaS-side options) when
  extending configs. Add new options to an existing dataclass when
  backward-compatible; create a new dataclass when the change is
  conceptually distinct.
- **Imports**: No wildcards. Keep third-party and internal groups
  consistent. Place heavy optional deps inside function bodies to
  avoid import-time side effects (Megatron, flash-attn, etc.).
- **Rewards**: Wrap blocking reward code with `AsyncRewardWrapper`
  (`astraflow/core/workflow/api/reward_api.py`). Standard signature is
  `(prompt, completions, prompt_ids, completion_ids, **data)` where
  `**data` carries dataset-specific fields (e.g. `answer`); return a
  `float`.
- **Config overrides**: Use Hydra-style dotted keys on the CLI; do
  not hardcode paths. Expose new options through dataclasses and
  wire via YAML under `examples/**/yaml/`.
- **Docs & comments**: Document non-obvious behavior inline. Default
  to *no* comment; add one only when the WHY is non-obvious.

## Extension points

### Add a rollout workflow

- Create a new module under `astraflow/core/workflow/impl/<name>.py`.
- Subclass `RolloutWorkflow`
  (`astraflow/core/workflow/api/workflow_api.py`) and implement async
  `arun_episode`.
- Thread `GenerationHyperparameters`, tokenizer, reward callable,
  stat scope, and optional `dump_dir` through `__init__`. Wrap the
  reward via `AsyncRewardWrapper`.
- Drive generation through the inference engine's `agenerate`; emit
  padded tensors with `concat_padded_tensors`.
- Persist transcripts to `{dump_dir}/{engine.get_version()}/` when
  debugging.
- Decorate the class with `@register_workflow("<name>")` from
  `astraflow/core/workflow/registry.py`, then add an
  `import astraflow.core.workflow.impl.<name>` line to
  `astraflow/core/workflow/__init__.py` so the decorator runs at import
  time. Reference `"<name>"` from `workflow_spec.workflow_cls` in the
  recipe YAML.

### Add a reward function

- Create `astraflow/core/workflow/reward/<name>.py` with a callable
  matching the reward API contract (see "Rewards" above).
- Decorate it with `@register_reward("<name>")` from
  `astraflow/core/workflow/registry.py`, then add an
  `import astraflow.core.workflow.reward.<name>` line to
  `astraflow/core/workflow/__init__.py`. Reference `"<name>"` from
  `workflow_spec.reward_fn` in the recipe YAML.
- Keep slow or external-service logic in the reward module; let the
  calling workflow wrap it with `AsyncRewardWrapper`.
- Note: `get_custom_reward_fn` / `VALID_REWARD_FN` in
  `reward/__init__.py` is a separate legacy path used only by the
  vision rewards (`clevr_count_70k`, `geometry3k`). Use the
  `@register_reward` decorator for new rewards.

### Add a dataset

- Create `astraflow/dataflow/dataset/<name>.py` with the loader
  function(s). Mirror an existing dataset (`deepscaler.py`,
  `gsm8k.py`, `alfworld.py`, ...).
- Define the sample schema (`messages`, `answer`, image fields,
  metadata) and validate it before returning rows.
- Optionally re-export the loader from
  `astraflow/dataflow/dataset/__init__.py` (`__all__`).
- Datasets are referenced from YAML by a fully-qualified
  `module:function` path, e.g.
  `dataset_fn: "astraflow.dataflow.dataset.<name>:get_<name>_rl_dataset"`
  — there is no central dataset-name registry.
- Expose path / split / tokenizer / max-length knobs through
  `TrainDatasetConfig` and `ValidDatasetConfig` in
  `astraflow/train_worker/api/cli_args.py`, then reference them
  from the relevant `examples/**/yaml/experiment.yaml`.

### Add a config option

- Extend the relevant dataclass in
  `astraflow/train_worker/api/cli_args.py` (or
  `astraflow/raas/api/cli_args.py` for RaaS-side options).
- Reference it from YAML under `examples/**/yaml/`.
- Regenerate CLI docs with `python docs/generate_cli_docs.py`.

### Launch training / evaluation

- Pick an existing recipe under `examples/**` that mirrors your case
  and reuse its launcher pairing (`local.py`, `ray.py`, `slurm.py`).
- Read the recipe README for scheduler requirements, container
  image, env vars, and data prep steps.
- Keep rollout actors and inference engines version-aligned by
  propagating `WeightUpdateMeta`. Note skipped weight updates
  explicitly if clusters are unavailable.
- Record the Hydra/CLI overrides used (`+train_dataset.path=...`,
  `engine.type=...`) in the PR / test plan so runs are reproducible.

### Publish docs

- Add prose under the appropriate section of `docs/en/`.
- Register new pages in the `docs/en/index.rst` toctree.
- Run `mdformat` on edited Markdown.
- If CLI args changed, regenerate with
  `python docs/generate_cli_docs.py`.

## Testing & validation

Test markers (from `pyproject.toml`):

- `slow` — > 30s, skipped in CI unless also marked `ci`.
- `ci` — forces a `slow` test to run in CI.
- `gpu` — requires a single GPU.
- `multi_gpu` — requires multiple GPUs.

Public CI runs only lint, format, and docs (`.github/workflows/`) —
no GPU tests run there. Gate tests by available hardware with
`@pytest.mark.skipif(current_platform.device_count() < N, ...)`.

Common invocations:

```bash
pytest -sv --sw --lf astraflow/                       # all tests
pytest astraflow/dataflow/tests/test_astraflow.py     # single file
pytest -m "not slow or ci" astraflow/                 # CI selection
```

When end-to-end / FSDP / Megatron / RPC tests require hardware you
cannot access, state the skipped suites and the alternative
validation (static checks, unit-level mocking) you performed.

## Collaboration & review

- **Branches**: kebab-case (`feat/multi-turn-metrics`,
  `fix/fsdp-weight-sync`).
- **Commits**: Conventional Commit prefixes (`feat:`, `fix:`,
  `docs:`, `chore:`, `refactor:`). Imperative voice; ~72 char
  subject. Squash WIP noise before opening/updating a PR.
- **Pre-merge**: Run pre-commit (`pre-commit run --all-files`).
  Note any hook you could not run locally and why. For doc-only
  edits, at least `mdformat --check` the touched files.
- **PR description**: Link the issue, summarize acceptance criteria,
  call out risk areas (perf, breaking changes), list the exact test
  commands you ran (and which suites you skipped, with reasons).
- **Resource safety**: When touching workflows/engines, confirm
  async code awaits I/O, preserves weight versioning, and frees
  resources on cancellation. Document GPU/memory expectations.
- **Cleanup**: Keep metrics flowing through `stats_tracker` /
  `StatsLogger`. Remove stray debug prints and commented-out code
  before merging.

## Reference material

- **Architecture deep dives**: `docs/en/architecture/overview.md`,
  `astraflow.md`, `raas.md`, `trainer.md`, `weight-manager.md`,
  `delta-weight-transfer.md`, `multi-agent-weight-transfer.md`.
- **Customization**: `docs/en/architecture/custom-raas.md`,
  `custom-trainer.md`.
- **Recipes**: `docs/en/recipes/{math,code,multi-agent,agentbench,
  search}.md` and the runnable variants under `examples/`.
- **Getting started**: `docs/en/get-started/installation.md`,
  `quickstart.md`.
- **Contributing**: `docs/en/developer-guide/contributing.md`.
- **Roadmap**: `docs/en/references/roadmap.md`.
