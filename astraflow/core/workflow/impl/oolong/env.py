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
    ):
        self.task = task
        self.stdout_truncate = stdout_truncate
        self._spawn_cb = spawn_callback

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

    def evaluate(self) -> tuple[float, dict[str, Any]]:
        """Reward for THIS agent. Called at the agent's finish() time.

        Routes to the right grader based on task_id prefix:
          - oolong.synth.*  -> rule-based synth_process_response
          - oolong.real.*   -> dnd_process_response (placeholder for now)
          - synthetic sub-agent task ids (no "oolong." prefix) -> 0.0
            placeholder; will need an LLM judge per the paper (Appendix A.7).
        """
        if not self.finished:
            return 0.0, {"reason": "agent never called finish()"}

        task_id = self.task.id or ""
        output = "" if self.finish_payload is None else str(self.finish_payload)

        if task_id.startswith("oolong.synth."):
            r = synth_process_response(self.task.misc, output)
        elif task_id.startswith("oolong.real."):
            r = dnd_process_response(self.task.misc, output)
        else:
            # Sub-agent task — no root verifier available without an LLM judge.
            r = {"score": 0.0, "reason": "no node-local verifier (LLM judge not yet wired)"}

        return float(r.get("score", 0.0)), r
