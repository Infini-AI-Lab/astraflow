"""Execution helpers for MRAAS-style code workflows."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from astraflow.workflow.utils import logging

logger = logging.getLogger(__name__)

SINGLE_CASE_EXEC_TIMEOUT = 6
GRACE_AFTER_KILL = 2
# Hard wall-clock cap on the entire program verification (all cases in
# one subprocess). On timeout the subprocess is SIGKILLed — bypasses
# Python signal handling and C extensions that ignore SIGALRM/SIGTERM.
PROGRAM_HARD_DEADLINE = 300

# Suppress the HuggingFace tokenizer fork warning in verifier subprocesses.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _verifier_module() -> str:
    return "astraflow.workflow.utils.testing_util_mraas"


def _verifier_script_path() -> str:
    """Absolute path to testing_util_mraas.py for direct invocation via
    ``python -P <path>`` (skips script-dir sys.path injection, avoiding
    the 14 s astraflow package import)."""
    return str(_repo_root() / "astraflow" / "workflow" / "utils" / "testing_util_mraas.py")


def _verifier_work_root() -> Path | None:
    raw = os.environ.get("ASTRAFLOW_VERIFY_WORK_ROOT")
    if not raw:
        return None

    root = Path(raw).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def verifier_work_dir():
    kwargs: dict[str, str] = {"prefix": "verify-"}
    root = _verifier_work_root()
    if root is not None:
        kwargs["dir"] = str(root)

    with tempfile.TemporaryDirectory(**kwargs) as workdir:
        yield workdir


def _strip_chat_tokens(text: str) -> str:
    return re.sub(r"<\|[^>\n]+?\|>", "", text).strip()


def _looks_like_python(text: str) -> bool:
    python_markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "print(",
        "if __name__",
        "for ",
        "while ",
        "return ",
    )
    return any(marker in text for marker in python_markers)


def extract_python_code(text: str, min_length: int = 20) -> str | None:
    text = _strip_chat_tokens(text)
    code_pattern = r"(?i)```(?:python|py)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = [block.strip() for block in code_blocks if len(block.strip()) >= min_length]
    if valid_blocks:
        return valid_blocks[-1]

    stripped = text.strip()
    if len(stripped) >= min_length and _looks_like_python(stripped):
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

    SIGKILL bypasses Python signal handling — kernel removes the process
    immediately regardless of what it's doing in C extensions.
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


def _make_detail(passed: bool, raw_input, raw_output, error_message: str,
                 error_code: int, error: str | None = None) -> dict[str, Any]:
    return {
        "passed": passed,
        "inputs": raw_input,
        "expected": raw_output,
        "output": raw_output if passed else None,
        "error": error,
        "error_code": error_code,
        "error_message": error_message,
    }


def call_verify_collect_all(
    problem: dict[str, Any],
    generation: str,
    timeout: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Run all test cases in a single subprocess. On PROGRAM_HARD_DEADLINE
    the subprocess is SIGKILLed. Returns ``(results, info)`` where
    ``info["details"]`` matches the legacy mraas shape (per-case feedback
    for retry prompting).
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
        io_spec = json.loads(problem.get("input_output", "{}"))
        raw_inputs = io_spec.get("inputs", []) or []
        raw_outputs = io_spec.get("outputs", []) or []
        n_cases = len(raw_inputs)
    except Exception:
        raw_inputs, raw_outputs, n_cases = [], [], 0

    if n_cases == 0:
        try:
            input_path.unlink()
        except FileNotFoundError:
            pass
        return [False], {
            "details": [],
            "error_code": -4,
            "error_message": "no test cases",
        }

    child_env = os.environ.copy()
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
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

    def _tle_fallback(error_message: str) -> tuple[list[Any], dict[str, Any]]:
        results = [False] * n_cases
        details = [
            _make_detail(
                passed=False,
                raw_input=(raw_inputs[i] if i < len(raw_inputs) else None),
                raw_output=(raw_outputs[i] if i < len(raw_outputs) else None),
                error_message=error_message,
                error_code=-3,
            )
            for i in range(n_cases)
        ]
        return results, {
            "details": details,
            "error_code": -3,
            "error_message": error_message,
        }

    try:
        if timed_out:
            logger.warning(
                "Verifier program hard-killed for query_id=%s after %.0f ms "
                "(limit=%ds)",
                problem.get("query_id"),
                (time.time() - start) * 1000,
                PROGRAM_HARD_DEADLINE,
            )
            return _tle_fallback("Program Time Limit Exceeded (hard kill)")

        try:
            data = _read_json(output_path)
        except FileNotFoundError:
            logger.warning(
                "Verifier produced no output for query_id=%s (returncode=%s)",
                problem.get("query_id"),
                proc.returncode,
            )
            return _tle_fallback("Verifier produced no output")
        except json.JSONDecodeError as exc:
            logger.warning(
                "Verifier output invalid JSON for query_id=%s: %s",
                problem.get("query_id"),
                exc,
            )
            return _tle_fallback(f"Verifier output invalid JSON: {exc}")

        results = data.get("result") or []
        info = data.get("info") or {}
        # The child already normalises to per-case details. If it didn't
        # (e.g. exception before building them), synthesise minimal ones.
        if not info.get("details"):
            details = []
            for i in range(n_cases):
                passed = bool(i < len(results) and results[i] is True)
                details.append(
                    _make_detail(
                        passed=passed,
                        raw_input=(raw_inputs[i] if i < len(raw_inputs) else None),
                        raw_output=(raw_outputs[i] if i < len(raw_outputs) else None),
                        error_message="Passed" if passed else info.get(
                            "error_message", "Wrong Answer"
                        ),
                        error_code=0 if passed else info.get("error_code", -2),
                    )
                )
            info["details"] = details
        # Pad/truncate results to n_cases for caller invariants.
        results = list(results[:n_cases]) + [False] * max(0, n_cases - len(results))
        return results, info
    finally:
        for p in (input_path, output_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
