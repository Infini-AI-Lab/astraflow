"""Execution-based reward for single-turn LiveCodeBench-style tasks."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from astraflow.core.workflow.registry import register_reward
from astraflow.core.workflow.utils import logging
from astraflow.core.workflow.utils.code_execution_mraas import verifier_work_dir

logger = logging.getLogger(__name__)

SINGLE_CASE_EXEC_TIMEOUT = 6
GRACE_AFTER_KILL = 2
# Hard wall-clock cap on the entire program verification (all cases in
# one subprocess). On timeout the subprocess is SIGKILLed — bypasses
# Python signal handling and C extensions that ignore SIGALRM/SIGTERM.
PROGRAM_HARD_DEADLINE = 100


def _repo_root() -> Path:
    # __file__ is astraflow/core/workflow/reward/livecodebench_reward.py;
    # parents[4] = repo root (parents[3] = package root since the reorg).
    return Path(__file__).resolve().parents[4]


def _verifier_module() -> str:
    return "astraflow.core.workflow.utils.testing_util"


def _verifier_script_path() -> str:
    """Absolute path to testing_util.py for direct script invocation.

    We run it with ``python3 -P <path>`` (Python 3.11+) which skips the
    automatic script-dir sys.path injection, avoiding the 14 s
    astraflow package import while still loading the standalone script.
    """
    return str(_repo_root() / "astraflow" / "core" / "workflow" / "utils" / "testing_util.py")


def _extract_python_code(text: str, min_length: int = 20) -> str | None:
    code_pattern = r"(?i)```(?:python|py)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = [block.strip() for block in code_blocks if len(block.strip()) >= min_length]
    if valid_blocks:
        return valid_blocks[-1]

    stripped = text.strip()
    if len(stripped) >= min_length:
        return stripped
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _hard_kill(proc: subprocess.Popen) -> None:
    """SIGKILL the subprocess (and its process group) and reap it.

    SIGKILL bypasses Python signal handling — the kernel removes the
    process immediately regardless of what it's doing in C extensions.
    """
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "getpgid") and hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except Exception:
        logger.debug("Failed to SIGKILL verifier process", exc_info=True)
    try:
        proc.wait(timeout=GRACE_AFTER_KILL)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=GRACE_AFTER_KILL)
        except Exception:
            pass


def _call_verify(
    problem: dict[str, Any],
    generation: str,
    timeout: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Run all test cases in a single subprocess. On PROGRAM_HARD_DEADLINE
    the subprocess is SIGKILLed. Returns (results, info) matching the
    legacy single-subprocess shape.
    """
    tmp_id = str(uuid.uuid4())
    input_path = Path("/tmp") / f"{tmp_id}-input.json"
    output_path = Path("/tmp") / f"{tmp_id}-output.json"
    payload = {
        "sample": problem,
        "test": generation,
        "debug": False,
        "timeout": timeout,
    }
    _write_json(input_path, payload)

    try:
        in_outs = json.loads(problem.get("input_output", "{}"))
        n_cases = len(in_outs.get("inputs", []) or [])
    except Exception:
        n_cases = 0

    if n_cases == 0:
        try:
            input_path.unlink()
        except FileNotFoundError:
            pass
        return [False], {"error": "no test cases"}

    child_env = os.environ.copy()
    repo_root = str(_repo_root())
    child_env["PYTHONPATH"] = (
        repo_root
        if not child_env.get("PYTHONPATH")
        else repo_root + os.pathsep + child_env["PYTHONPATH"]
    )

    start = time.time()
    with verifier_work_dir() as workdir:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-P",
                _verifier_script_path(),
                "--tmp_id",
                tmp_id,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=child_env,
            cwd=workdir,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        timed_out = False
        try:
            proc.wait(timeout=PROGRAM_HARD_DEADLINE)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _hard_kill(proc)

    try:
        if timed_out:
            logger.warning(
                "Verifier program hard-killed for query_id=%s after %.0f ms "
                "(limit=%ds)",
                problem.get("query_id"),
                (time.time() - start) * 1000,
                PROGRAM_HARD_DEADLINE,
            )
            return [-1], {
                "error_code": -3,
                "error_message": "Program Time Limit Exceeded (hard kill)",
            }

        try:
            data = _read_json(output_path)
        except FileNotFoundError:
            logger.warning(
                "Verifier produced no output for query_id=%s (returncode=%s)",
                problem.get("query_id"),
                proc.returncode,
            )
            return [-4], {"error": "no output"}
        except json.JSONDecodeError:
            logger.warning(
                "Verifier output invalid JSON for query_id=%s",
                problem.get("query_id"),
            )
            return [-4], {"error": "invalid output JSON"}

        return data.get("result") or [], data.get("info") or {}
    finally:
        for p in (input_path, output_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


@register_reward("livecodebench_reward")
def livecodebench_reward_fn(
    prompt_text: str,
    completion_text: str,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    input_output: str,
    query_id: str | int | None = None,
    idx: str | int | None = None,
    **kwargs,
) -> float:
    code = _extract_python_code(str(completion_text))
    if code is None:
        return 0.0

    problem = {
        "input_output": input_output,
        "query_id": str(
            query_id if query_id is not None else idx if idx is not None else "unknown"
        ),
    }

    try:
        results, _ = _call_verify(
            problem=problem,
            generation=code,
            timeout=SINGLE_CASE_EXEC_TIMEOUT,
        )
        return 1.0 if results and all(item is True for item in results) else 0.0
    except Exception:
        logger.warning("Exception in livecodebench_reward_fn", exc_info=True)
        return 0.0
