"""Per-agent Python sandbox env for Oolong.

Each agent gets a stateful exec() namespace pre-populated with:
  - `context: str` -- the (potentially huge) text to process
  - `asyncio` module
  - `finish(answer: Any) -> None` -- raises FinishSignal to end the episode
  - `launch_subagent(goal: str, context: str = "") -> str` -- async callback
    into the workflow that spawns a child agent and returns its finish text.

State persists across turns: variables defined in turn N are visible in
turn N+1. Top-level `await` is supported via `PyCF_ALLOW_TOP_LEVEL_AWAIT`.

Reward is computed at finish() time via the platoon-ported scorers in
`eval_helpers.py`. For oolong-real the scorer is a placeholder (=0.0)
until we add an LLM judge.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from astraflow.core.workflow.impl.oolong.eval_helpers import (
    dnd_process_response,
    synth_process_response,
)
from astraflow.core.workflow.impl.oolong.tasks import Task
from astraEnv.judge import extract_json, judge


# Rubric sent to the LLM judge when grading a sub-agent's finish_message
# against its delegated goal. Sub-agents have no rule-based verifier (their
# goals are free-form strings produced by the parent), so we ask an external
# LLM to score the answer. Keep this short and unambiguous.
_SUBAGENT_RUBRIC_SYSTEM = (
    "You grade a sub-agent's output against its delegated goal.\n"
    'Return ONLY JSON: {"score": float in [0,1], "reason": "<one short sentence>"}\n'
    "1.0 = output fully and correctly satisfies the goal. "
    "0.5 = partially correct. "
    "0.0 = wrong, empty, or a refusal.\n"
    "Do not include any other text — JSON only."
)


# Sentinel exception used to terminate the agent's code when finish() is
# called. Carries the answer payload.
class FinishSignal(Exception):
    def __init__(self, payload: Any):
        super().__init__("FinishSignal")
        self.payload = payload


# Cap how much sandbox stdout we surface back to the model per turn.
# Oolong agents like to `print(context)`, which can be >50k chars. We
# truncate to keep the chat history manageable; agents can chunk explicitly.
DEFAULT_STDOUT_TRUNCATE = 8000


# Signature: launch_subagent(goal, context) -> finish_message_string.
# The workflow injects this when constructing the env.
SpawnCallback = Callable[[str, str], Awaitable[str]]


@dataclass
class ExecResult:
    """One Python code execution result."""
    stdout: str
    stderr: str
    error: str | None  # traceback string if user code raised
    truncated: bool


class OolongEnv:
    """Stateful Python sandbox for one agent."""

    def __init__(
        self,
        task: Task,
        spawn_callback: SpawnCallback | None = None,
        stdout_truncate: int = DEFAULT_STDOUT_TRUNCATE,
        use_llm_judge: bool = False,
        judge_model: str | None = None,
    ):
        self.task = task
        self.stdout_truncate = stdout_truncate
        self._spawn_cb = spawn_callback
        # When True, sub-agent tasks (task_id without the "oolong." prefix)
        # are graded by an LLM judge via astraEnv.judge. When False, they
        # return score=0.0 with a placeholder reason (legacy behavior).
        self.use_llm_judge = bool(use_llm_judge)
        self.judge_model = judge_model

        self.finished: bool = False
        self.finish_payload: Any | None = None

        # Per-agent telemetry, mirroring TextCraft.
        self.subagent_launched: int = 0
        self.subagent_succeeded: float = 0.0

        # The stateful namespace. Pre-populate the helpers the agent needs.
        ctx = str(task.misc.get("context", ""))
        self.globals: dict[str, Any] = {
            "__name__": "__oolong_sandbox__",
            "__builtins__": __builtins__,
            "asyncio": asyncio,
            "context": ctx,
            "finish": self._finish,
            "launch_subagent": self._launch_subagent,
            "json": json,
        }

    # ---------------------------- exposed Python API --------------------------

    def _finish(self, answer: Any) -> None:
        raise FinishSignal(answer)

    async def _launch_subagent(self, goal: str, context: str = "") -> str:
        """Async callback into the workflow to spawn a child agent.

        Returns the child's finish_message string (or an empty string if the
        child never finished cleanly).
        """
        if self._spawn_cb is None:
            raise RuntimeError(
                "launch_subagent called but no spawn_callback was injected by the workflow"
            )
        self.subagent_launched += 1
        result = await self._spawn_cb(str(goal), str(context))
        # Caller's "success" telemetry is tracked at the workflow level (we
        # do not know the child's reward here without a round-trip).
        return result

    # ----------------------------- code execution -----------------------------

    async def run_code(self, code: str) -> ExecResult:
        """Execute one block of Python code in the persistent namespace.

        Supports top-level `await` (e.g. `await launch_subagent(...)`).
        Captures stdout/stderr, truncates if huge.
        """
        # Compile with top-level-await support so the code may use `await`
        # directly (without us wrapping it in an `async def`).
        flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        finish_payload: Any = None

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        err_str: str | None = None

        try:
            compiled = compile(textwrap.dedent(code), "<agent_code>", "exec", flags=flags)
        except SyntaxError as e:
            return ExecResult(stdout="", stderr=str(e), error=f"SyntaxError: {e}", truncated=False)

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # eval(compiled, ...) returns either None or a coroutine,
                # depending on whether the code contained top-level await.
                result = eval(compiled, self.globals)
                if asyncio.iscoroutine(result):
                    await result
        except FinishSignal as fs:
            finish_payload = fs.payload
            self.finished = True
            self.finish_payload = finish_payload
        except Exception:
            err_str = traceback.format_exc(limit=10)

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        truncated = False
        if len(out) > self.stdout_truncate:
            out = out[: self.stdout_truncate] + f"\n[... stdout truncated, total={len(stdout_buf.getvalue())} chars ...]"
            truncated = True
        return ExecResult(stdout=out, stderr=err, error=err_str, truncated=truncated)

    # ----------------------------- evaluation ---------------------------------

    async def evaluate(self) -> tuple[float, dict[str, Any]]:
        """Reward for THIS agent. Called at the agent's finish() time.

        Routes to the right grader based on task_id prefix:
          - oolong.synth.*  -> rule-based synth_process_response
          - oolong.real.*   -> dnd_process_response (placeholder for now)
          - synthetic sub-agent task ids (no "oolong." prefix) ->
              LLM judge via astraEnv.judge if self.use_llm_judge,
              else 0.0 placeholder.
        """
        if not self.finished:
            return 0.0, {"reason": "agent never called finish()"}

        task_id = self.task.id or ""
        output = "" if self.finish_payload is None else str(self.finish_payload)

        # Sub-agent task IDs inherit the parent's prefix and have a "/sub_"
        # marker (assigned in workflow._make_spawn_callback). Route them to
        # the LLM-judge path regardless of the dataset prefix.
        is_subagent = "/sub_" in task_id
        if not is_subagent and task_id.startswith("oolong.synth."):
            r = synth_process_response(self.task.misc, output)
        elif not is_subagent and task_id.startswith("oolong.real."):
            r = dnd_process_response(self.task.misc, output)
        elif self.use_llm_judge:
            r = await self._grade_subagent_with_llm(output)
        else:
            r = {"score": 0.0, "reason": "no node-local verifier (LLM judge disabled)"}

        return float(r.get("score", 0.0)), r

    async def _grade_subagent_with_llm(self, output: str) -> dict[str, Any]:
        """Send (goal, output) to the LLM judge and parse the score.

        On any failure (network, parse, missing field) returns score=0.0
        with the error in `reason` so a flaky judge never crashes a rollout.
        """
        user = f"GOAL: {self.task.goal}\n\nOUTPUT: {output}"
        judge_kwargs: dict[str, Any] = {}
        if self.judge_model:
            judge_kwargs["model"] = self.judge_model
        try:
            raw = await judge(
                system=_SUBAGENT_RUBRIC_SYSTEM, user=user, **judge_kwargs
            )
        except Exception as e:
            return {"score": 0.0, "reason": f"judge call failed: {e}"}
        try:
            parsed = extract_json(raw)
            score = float(parsed["score"])
        except Exception as e:
            return {
                "score": 0.0,
                "reason": f"judge response unparseable: {e}",
                "judge_raw": raw,
            }
        # Clamp defensively in case the model returns out-of-range.
        score = max(0.0, min(1.0, score))
        return {
            "score": score,
            "reason": str(parsed.get("reason", "")),
            "judge_raw": raw,
        }
