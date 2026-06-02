"""Harbor-backed Terminal-Bench workflows."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import uuid
from pathlib import Path
from typing import Any

import torch

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import EngineGroup, InferenceEngine
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured

logger = logging.getLogger("TerminalBenchHarborWorkflow")


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_reward_from_result(result: Any) -> float | None:
    if not isinstance(result, dict):
        return None

    verifier_result = result.get("verifier_result")
    if isinstance(verifier_result, dict):
        rewards = verifier_result.get("rewards")
        if isinstance(rewards, dict):
            return _coerce_float(rewards.get("reward"))
    return None


def _collect_harbor_rewards(job_root: Path) -> list[float]:
    result_path, result = _load_harbor_trial_result(job_root)
    reward = _extract_reward_from_result(result)
    if reward is None:
        raise RuntimeError(
            f"Harbor trial result has no verifier reward: {result_path}"
        )
    return [reward]


def _load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_harbor_trial_result(job_root: Path) -> tuple[Path, dict[str, Any]]:
    timestamp_dirs = [path for path in job_root.iterdir() if path.is_dir()]
    if len(timestamp_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one Harbor run directory under {job_root}, "
            f"found {len(timestamp_dirs)}."
        )

    trial_result_paths = sorted(timestamp_dirs[0].glob("*/result.json"))
    if len(trial_result_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one Harbor trial result under {timestamp_dirs[0]}, "
            f"found {len(trial_result_paths)}."
        )

    result_path = trial_result_paths[0]
    result = _load_json_file(result_path)
    if not isinstance(result, dict):
        raise RuntimeError(f"Could not read Harbor trial result: {result_path}")
    if not isinstance(result.get("agent_result"), dict) or not isinstance(
        result.get("verifier_result"), dict
    ):
        raise RuntimeError(
            f"Harbor trial result missing agent_result/verifier_result: {result_path}"
        )
    return result_path, result


def _extract_rollout_details(result: dict[str, Any]) -> list[dict[str, Any]]:
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        rollout_details = agent_result.get("rollout_details")
        if isinstance(rollout_details, list):
            return [
                detail
                for detail in rollout_details
                if isinstance(detail, dict)
            ]
    return []


def _as_turn_token_lists(value: Any, field_name: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"Harbor rollout detail missing list field {field_name!r}.")
    turns: list[list[int]] = []
    for turn_idx, turn in enumerate(value):
        if not isinstance(turn, list):
            raise ValueError(
                f"Harbor rollout detail field {field_name!r} turn {turn_idx} "
                f"is {type(turn).__name__}, expected list."
            )
        turns.append([int(token) for token in turn])
    return turns


def _as_turn_float_lists(value: Any, field_name: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"Harbor rollout detail missing list field {field_name!r}.")
    turns: list[list[float]] = []
    for turn_idx, turn in enumerate(value):
        if not isinstance(turn, list):
            raise ValueError(
                f"Harbor rollout detail field {field_name!r} turn {turn_idx} "
                f"is {type(turn).__name__}, expected list."
            )
        turns.append([float(logprob) for logprob in turn])
    return turns


def _prompt_suffix_for_accumulated_sequence(
    prompt_token_ids: list[int],
    accumulated_sequence: list[int],
) -> list[int]:
    if not accumulated_sequence:
        return prompt_token_ids
    if prompt_token_ids[: len(accumulated_sequence)] == accumulated_sequence:
        return prompt_token_ids[len(accumulated_sequence) :]
    return prompt_token_ids


def _harbor_result_to_training_sequence(
    result: dict[str, Any],
    reward: float,
    version: int,
    rollout_detail_index: int = 0,
) -> dict[str, torch.Tensor]:
    rollout_details = _extract_rollout_details(result)
    if not rollout_details:
        raise ValueError(
            "Harbor result has no rollout_details. Set Terminus-2 "
            "agent_kwarg collect_rollout_details=true for RL training."
        )
    if rollout_detail_index >= len(rollout_details):
        raise ValueError(
            f"Harbor result has {len(rollout_details)} rollout detail(s), "
            f"but rollout_detail_index={rollout_detail_index}."
        )

    detail = rollout_details[rollout_detail_index]
    prompt_turns = _as_turn_token_lists(
        detail.get("prompt_token_ids"), "prompt_token_ids"
    )
    completion_turns = _as_turn_token_lists(
        detail.get("completion_token_ids"), "completion_token_ids"
    )
    logprob_turns = _as_turn_float_lists(detail.get("logprobs"), "logprobs")

    n_turns = len(completion_turns)
    if len(prompt_turns) != n_turns or len(logprob_turns) != n_turns:
        raise ValueError(
            "Harbor rollout detail has inconsistent turn counts: "
            f"prompt={len(prompt_turns)}, completion={len(completion_turns)}, "
            f"logprobs={len(logprob_turns)}."
        )

    seq: list[int] = []
    logprobs: list[float] = []
    loss_mask: list[int] = []
    versions: list[int] = []
    n_completion_tokens = 0

    for turn_idx, (prompt_tokens, completion_tokens, turn_logprobs) in enumerate(
        zip(prompt_turns, completion_turns, logprob_turns)
    ):
        if len(completion_tokens) != len(turn_logprobs):
            raise ValueError(
                f"Harbor rollout detail turn {turn_idx} has "
                f"{len(completion_tokens)} completion token(s) but "
                f"{len(turn_logprobs)} logprob(s)."
            )
        prompt_delta = _prompt_suffix_for_accumulated_sequence(prompt_tokens, seq)
        seq += prompt_delta + completion_tokens
        logprobs += [0.0] * len(prompt_delta) + turn_logprobs
        loss_mask += [0] * len(prompt_delta) + [1] * len(completion_tokens)
        versions += [-1] * len(prompt_delta) + [version] * len(completion_tokens)
        n_completion_tokens += len(completion_tokens)

    if not seq:
        raise ValueError("Harbor rollout detail produced an empty token sequence.")
    if n_completion_tokens == 0:
        raise ValueError("Harbor rollout detail has no trainable completion tokens.")

    res = {
        "input_ids": torch.tensor(seq, dtype=torch.int32),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
        "logprobs": torch.tensor(logprobs, dtype=torch.float32),
        "versions": torch.tensor(versions, dtype=torch.int32),
        "attention_mask": torch.ones(len(seq), dtype=torch.bool),
        "rewards": torch.tensor(float(reward), dtype=torch.float32),
    }
    return {key: value.unsqueeze(0) for key, value in res.items()}


def _format_agent_kwarg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _engine_addresses(engine: InferenceEngine) -> list[str]:
    engines: list[Any]
    if isinstance(engine, EngineGroup):
        engines = [engine[key] for key in engine.keys()]
    else:
        engines = [engine]

    addresses: list[str] = []
    seen: set[str] = set()
    for one_engine in engines:
        candidates = [
            one_engine,
            getattr(one_engine, "_engine", None),
            getattr(getattr(one_engine, "default", None), "_engine", None),
        ]
        for candidate in candidates:
            for address in getattr(candidate, "addresses", None) or []:
                address = str(address)
                if address not in seen:
                    seen.add(address)
                    addresses.append(address)
    return addresses


def _append_api_base_suffix(api_base: str, suffix: str) -> str:
    base = (
        api_base
        if api_base.startswith(("http://", "https://"))
        else f"http://{api_base}"
    )
    return f"{base.rstrip('/')}{suffix}" if suffix else base.rstrip("/")


def _as_api_base_list(api_base: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if api_base is None:
        return []
    if isinstance(api_base, str):
        values = [api_base]
    else:
        values = list(api_base)
    return [str(value).strip() for value in values if str(value).strip()]


@register_workflow("terminal_bench_harbor")
class TerminalBenchHarborWorkflow(RolloutWorkflow):
    """Run Terminal-Bench through Harbor and return AstraFlow eval rewards."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer,
        dataset: str = "terminal-bench@2.0",
        dataset_path: str | None = None,
        harbor_binary: str = "harbor",
        harbor_command: list[str] | None = None,
        agent_name: str = "terminus-2",
        model_name: str = "openai/local-model",
        api_base: str | list[str] | tuple[str, ...] | None = None,
        api_base_suffix: str = "/v1",
        api_key: str = "EMPTY",
        api_key_env: str = "OPENAI_API_KEY",
        environment: str | None = None,
        jobs_dir: str | None = None,
        timeout: float = 7200.0,
        rollout_stat_scope: str = "eval-rollout",
        n_concurrent_trials: int = 1,
        max_parallel_jobs: int = 1,
        task_name_arg: str = "--include-task-name",
        auto_confirm: bool = True,
        agent_kwargs: dict[str, Any] | None = None,
        agent_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        dump_dir: str | None = None,
    ):
        del tokenizer
        del dump_dir
        self.gconfig = gconfig
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.harbor_binary = harbor_binary
        self.harbor_command = list(harbor_command) if harbor_command else None
        self.agent_name = agent_name
        self.model_name = model_name
        self.api_base = api_base
        self.api_base_suffix = api_base_suffix
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.environment = environment
        self.jobs_dir = jobs_dir
        self.timeout = timeout
        self.rollout_stat_scope = rollout_stat_scope
        self.n_concurrent_trials = n_concurrent_trials
        self.max_parallel_jobs = max(1, int(max_parallel_jobs))
        self._semaphore = asyncio.Semaphore(self.max_parallel_jobs)
        self.task_name_arg = task_name_arg
        self.auto_confirm = auto_confirm
        self.agent_kwargs = dict(agent_kwargs or {})
        self.agent_env = dict(agent_env or {})
        self.extra_args = list(extra_args or [])
        self._api_base_next_idx = 0

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        # EvalManager already implements repeated sampling with the dataset
        # `repeat`/`k` setting. Keep one Harbor subprocess per eval item so
        # expensive Docker-backed jobs do not multiply unexpectedly.
        configured_samples = int(getattr(self.gconfig, "n_samples", 1))
        if configured_samples != 1:
            logger.warning(
                "Ignoring gconfig.n_samples=%s for Harbor eval; use eval "
                "dataset repeat/k for pass@k.",
                configured_samples,
            )
        results = [await self._run_one_harbor_trial(engine, data)]
        rewards = torch.tensor([r["reward"] for r in results], dtype=torch.float32)
        eval_correct = torch.tensor(
            [1.0 if r["reward"] > 0.0 else 0.0 for r in results],
            dtype=torch.float32,
        )
        stats_tracker.get(self.rollout_stat_scope).scalar(
            reward=float(rewards.mean().item()),
            success=float(eval_correct.mean().item()),
        )
        output: dict[str, Any] = {
            "rewards": rewards,
            "eval_correct": eval_correct,
            "n_trajs": 1,
            "harbor": results,
        }
        prompt_id = resolve_prompt_id(data)
        if prompt_id is not None:
            output["prompt_id"] = prompt_id
        return output

    async def _run_one_harbor_trial(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._semaphore:
            return await self._run_one_harbor_trial_unlocked(engine, data)

    async def _run_one_harbor_trial_unlocked(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        task_path = data.get("task_path")
        if task_path is None and isinstance(data.get("prompt"), str):
            prompt_path = Path(data["prompt"]).expanduser()
            if (prompt_path / "instruction.md").is_file():
                task_path = str(prompt_path)
        task_name = data.get("task_name") or data.get("task_id")
        run_id = uuid.uuid4().hex
        root = Path(self.jobs_dir or "./data-harbor-jobs")
        index_value = data.get("index") or task_name
        if index_value is None and task_path is not None:
            index_value = Path(str(task_path)).expanduser().name
        index = str(index_value or "task").replace("/", "_")
        run_jobs_dir = root / f"{index}-{run_id}"
        run_jobs_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(engine, task_name, run_jobs_dir, task_path=task_path)
        env = os.environ.copy()
        if self.api_key_env and self.api_key_env not in env:
            env[self.api_key_env] = self.api_key
        env.update({str(k): str(v) for k, v in self.agent_env.items()})

        command_text = shlex.join(cmd)
        (run_jobs_dir / "harbor.command.txt").write_text(command_text + "\n")
        logger.info("Running Harbor command: %s", command_text)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), self.timeout)
        except asyncio.CancelledError:
            proc.kill()
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(
                f"Harbor timed out after {self.timeout}s for task {task_name!r}"
            )

        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        (run_jobs_dir / "harbor.stdout.log").write_text(stdout_text)
        (run_jobs_dir / "harbor.stderr.log").write_text(stderr_text)
        if proc.returncode != 0:
            logger.error(
                "Harbor command failed for task %r with returncode=%s. "
                "stdout/stderr logs are under %s. stderr tail:\n%s",
                task_name,
                proc.returncode,
                run_jobs_dir,
                _tail(stderr_text, 2000),
            )
            raise RuntimeError(
                "Harbor command failed "
                f"(returncode={proc.returncode}, task={task_name!r}).\n"
                f"logs: {run_jobs_dir}\n"
                f"stdout:\n{_tail(stdout_text)}\n"
                f"stderr:\n{_tail(stderr_text)}"
            )

        rewards = _collect_harbor_rewards(run_jobs_dir)
        if not rewards:
            logger.error(
                "Could not find Harbor reward for task %r under %s. "
                "stdout tail:\n%s\nstderr tail:\n%s",
                task_name,
                run_jobs_dir,
                _tail(stdout_text, 1000),
                _tail(stderr_text, 1000),
            )
            raise RuntimeError(
                f"Could not find Harbor reward under {run_jobs_dir}. "
                f"stdout tail:\n{_tail(stdout_text, 2000)}\n"
                f"stderr tail:\n{_tail(stderr_text, 2000)}"
            )
        reward = float(sum(rewards) / len(rewards))
        return {
            "task_name": task_name,
            "task_path": str(task_path) if task_path is not None else None,
            "reward": reward,
            "num_harbor_rewards": len(rewards),
            "jobs_dir": str(run_jobs_dir),
        }

    def _build_command(
        self,
        engine: InferenceEngine,
        task_name: Any,
        run_jobs_dir: Path,
        task_path: Any | None = None,
    ) -> list[str]:
        cmd = list(self.harbor_command or [self.harbor_binary]) + ["run"]
        if task_path:
            cmd.extend(["--path", str(Path(str(task_path)).expanduser())])
        elif self.dataset_path:
            cmd.extend(["--path", self.dataset_path])
        else:
            cmd.extend(["--dataset", self.dataset])
        cmd.extend(["--agent", self.agent_name, "--model", self.model_name])
        cmd.extend(["--n-concurrent", str(self.n_concurrent_trials)])
        cmd.extend(["--jobs-dir", str(run_jobs_dir)])
        if self.auto_confirm:
            cmd.append("--yes")

        if task_name and not task_path:
            cmd.extend([self.task_name_arg, str(task_name)])
        if self.environment:
            cmd.extend(["--env", self.environment])

        agent_kwargs = dict(self.agent_kwargs)
        api_base = self._allocate_api_base(engine)
        if not api_base:
            raise RuntimeError(
                "TerminalBenchHarborWorkflow could not find a model API base "
                "URL from configured `api_base` or the RaaS eval engine."
            )
        agent_kwargs.setdefault("api_base", api_base)
        for key, value in agent_kwargs.items():
            formatted_value = _format_agent_kwarg_value(value)
            cmd.extend(["--agent-kwarg", f"{key}={formatted_value}"])

        cmd.extend(self.extra_args)
        return cmd

    def _configured_api_bases(self) -> list[str]:
        return _as_api_base_list(self.api_base)

    def _infer_api_base(self, engine: InferenceEngine) -> list[str]:
        suffix = self.api_base_suffix or ""
        return [
            _append_api_base_suffix(address, suffix)
            for address in _engine_addresses(engine)
        ]

    def _available_api_bases(self, engine: InferenceEngine) -> list[str]:
        return self._configured_api_bases() or self._infer_api_base(engine)

    def _allocate_api_base(self, engine: InferenceEngine) -> str | None:
        api_bases = self._available_api_bases(engine)
        if not api_bases:
            return None
        api_base = api_bases[self._api_base_next_idx % len(api_bases)]
        self._api_base_next_idx += 1
        return api_base


@register_workflow("terminal_bench_harbor_rl")
class TerminalBenchHarborRLWorkflow(TerminalBenchHarborWorkflow):
    """Run Terminal-Bench through Harbor and return AstraFlow RL tensors."""

    def __init__(
        self,
        *args,
        rollout_detail_index: int = 0,
        **kwargs,
    ):
        agent_kwargs = dict(kwargs.pop("agent_kwargs", {}) or {})
        agent_kwargs.setdefault("collect_rollout_details", True)
        agent_kwargs.setdefault("enable_summarize", False)
        kwargs["agent_kwargs"] = agent_kwargs
        kwargs.setdefault("rollout_stat_scope", "rollout")

        # AstraFlow samples should be represented as separate trajectories so
        # group-level reward normalization/filtering sees each attempt.
        n_concurrent_trials = int(kwargs.get("n_concurrent_trials", 1))
        if n_concurrent_trials != 1:
            logger.warning(
                "terminal_bench_harbor_rl uses gconfig.n_samples for repeated "
                "training trajectories; overriding n_concurrent_trials=%s to 1.",
                n_concurrent_trials,
            )
        kwargs["n_concurrent_trials"] = 1

        super().__init__(*args, **kwargs)
        self.rollout_detail_index = int(rollout_detail_index)

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        n_samples = max(1, int(getattr(self.gconfig, "n_samples", 1)))
        sample_results = await asyncio.gather(
            *[
                self._run_one_harbor_training_trial(engine, data)
                for _ in range(n_samples)
            ]
        )
        rewards = [
            float(res["rewards"].flatten()[0].item())
            for res in sample_results
        ]
        if rewards:
            successes = [1.0 if reward > 0.0 else 0.0 for reward in rewards]
            stats_tracker.get(self.rollout_stat_scope).scalar(
                reward=float(sum(rewards) / len(rewards)),
                success=float(sum(successes) / len(successes)),
            )
        return results_to_structured(
            sample_results,
            prompt_id=resolve_prompt_id(data),
        )

    async def _run_one_harbor_training_trial(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        harbor_summary = await self._run_one_harbor_trial(engine, data)
        job_root = Path(harbor_summary["jobs_dir"])
        result_path, result = _load_harbor_trial_result(job_root)
        if not _extract_rollout_details(result):
            raise RuntimeError(
                "Could not find Harbor rollout_details in trial result. "
                "Check that agent_kwargs.collect_rollout_details=true and the "
                "OpenAI-compatible backend returns token IDs/logprobs. "
                f"Result: {result_path}"
            )

        reward = _extract_reward_from_result(result)
        if reward is None:
            reward = float(harbor_summary["reward"])
        try:
            version = int(engine.get_version())
        except Exception:
            logger.exception("Could not read engine version; using version=0.")
            version = 0

        try:
            return _harbor_result_to_training_sequence(
                result,
                reward=float(reward),
                version=version,
                rollout_detail_index=self.rollout_detail_index,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Could not convert Harbor result to RL tensors: {result_path}"
            ) from exc
