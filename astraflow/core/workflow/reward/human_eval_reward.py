"""Execution-based reward for single-turn HumanEval tasks."""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

from astraflow.core.workflow.registry import register_reward
from astraflow.core.workflow.utils import logging
from astraflow.core.workflow.utils.code_execution_mraas import (
    SINGLE_CASE_EXEC_TIMEOUT,
    extract_python_code,
)

logger = logging.getLogger(__name__)


def _ensure_human_eval_importable() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    human_eval_root = repo_root / "astraEnv" / "human-eval"
    human_eval_path = str(human_eval_root)
    if human_eval_path not in sys.path:
        sys.path.insert(0, human_eval_path)


def _check_correctness_with_queue(
    problem: dict[str, str],
    completion: str,
    timeout: float,
) -> bool:
    _ensure_human_eval_importable()
    from human_eval.execution import (
        TimeoutException,
        create_tempdir,
        reliability_guard,
        swallow_io,
        time_limit,
    )

    def unsafe_execute(problem: dict[str, str], completion: str, timeout: float, queue) -> None:
        with create_tempdir():
            import os
            import shutil

            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            reliability_guard()

            check_program = (
                problem["prompt"]
                + completion
                + "\n"
                + problem["test"]
                + "\n"
                + f"check({problem['entry_point']})"
            )
            try:
                exec_globals: dict[str, object] = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                queue.put("passed")
            except TimeoutException:
                queue.put("timed out")
            except BaseException as exc:
                queue.put(f"failed: {exc}")
            finally:
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=unsafe_execute, args=(problem, completion, timeout, queue))
    proc.start()
    proc.join(timeout=timeout + 1)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1)

    result = "timed out"
    try:
        if not queue.empty():
            result = queue.get_nowait()
    except Exception:
        result = "timed out"
    finally:
        queue.close()
        queue.join_thread()

    return result == "passed"


@register_reward("human_eval_reward")
def human_eval_reward_fn(
    prompt_text: str,
    completion_text: str,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    task_id: str | int | None = None,
    query_id: str | int | None = None,
    idx: str | int | None = None,
    prompt: str | None = None,
    test: str | None = None,
    entry_point: str | None = None,
    timeout: float = SINGLE_CASE_EXEC_TIMEOUT,
    **kwargs,
) -> float:
    del prompt_text, prompt_token_ids, completion_token_ids, kwargs

    code = extract_python_code(str(completion_text))
    if code is None:
        return 0.0
    if not prompt or not test or not entry_point:
        logger.warning(
            "HumanEval reward called without required fields: prompt=%s test=%s entry_point=%s",
            bool(prompt),
            bool(test),
            bool(entry_point),
        )
        return 0.0

    try:
        problem = {
            "task_id": str(
                task_id
                if task_id is not None
                else query_id
                if query_id is not None
                else idx
                if idx is not None
                else "unknown"
            ),
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
        }
        passed = _check_correctness_with_queue(
            problem,
            code,
            timeout,
        )
        return 1.0 if passed else 0.0
    except Exception:
        logger.warning("Exception in human_eval_reward_fn", exc_info=True)
        return 0.0
