import asyncio
import os
import threading
import traceback
import weakref
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial

from astraflow.workflow.utils import logging

logger = logging.getLogger("Reward API")


def _get_device_count_safely() -> int:
    """
    Safely get device count without initializing CUDA context.
    """
    gpu_types = ["nvidia", "davinci"]
    try:
        if os.path.exists("/dev"):
            for gpu_type in gpu_types:
                devices = [
                    f
                    for f in os.listdir("/dev")
                    if f.startswith(gpu_type) and f[len(gpu_type) :].isdigit()
                ]
                if devices:
                    return len(devices)
    except (OSError, ValueError) as e:
        logger.debug(f"Could not read device list from /dev, using fallback: {e}")

    return 8


def reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs,
):
    """Placeholder for the reward function signature."""


class AsyncRewardWrapper:
    """
    Wraps a synchronous reward function to make it async with timeout handling.
    Automatically manages ProcessPoolExecutor lifecycle based on instance count.
    Includes automatic recovery from broken process pools.
    """

    _executors = {}
    _instance_counts = {}
    _lock = threading.Lock()

    def __init__(
        self,
        reward_fn: Callable,
        timeout_seconds: float = 1800,
        max_workers: int | None = None,
        max_retries: int = 3,
    ):
        self.reward_fn = reward_fn
        self.timeout_seconds = timeout_seconds
        if max_workers is None:
            max_workers = 64
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._executor_key = max_workers

        self._register_instance()

    def _register_instance(self) -> None:
        with self._lock:
            if self._executor_key not in self._executors:
                self._executors[self._executor_key] = ProcessPoolExecutor(
                    max_workers=self.max_workers
                )
                self._instance_counts[self._executor_key] = 0
            self._instance_counts[self._executor_key] += 1

        weakref.finalize(self, AsyncRewardWrapper._cleanup_executor, self._executor_key)

    def __getstate__(self):
        return {
            "reward_fn": self.reward_fn,
            "timeout_seconds": self.timeout_seconds,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "_executor_key": self._executor_key,
        }

    def __setstate__(self, state):
        self.reward_fn = state["reward_fn"]
        self.timeout_seconds = state["timeout_seconds"]
        self.max_workers = state["max_workers"]
        self.max_retries = state["max_retries"]
        self._executor_key = state.get("_executor_key", self.max_workers)
        self._register_instance()

    @classmethod
    def _cleanup_executor(cls, executor_key):
        with cls._lock:
            if executor_key in cls._instance_counts:
                cls._instance_counts[executor_key] -= 1
                if cls._instance_counts[executor_key] <= 0:
                    if executor_key in cls._executors:
                        executor = cls._executors.pop(executor_key)
                        executor.shutdown(wait=False, cancel_futures=True)
                        logger.debug(
                            f"ProcessPoolExecutor with {executor_key} workers shut down"
                        )
                    cls._instance_counts.pop(executor_key, None)

    @classmethod
    def _recreate_executor(cls, executor_key, max_workers):
        with cls._lock:
            if executor_key in cls._executors:
                old_executor = cls._executors[executor_key]
                try:
                    old_executor.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"Error shutting down broken executor: {e}")

                cls._executors[executor_key] = ProcessPoolExecutor(
                    max_workers=max_workers
                )
                logger.info(f"Recreated ProcessPoolExecutor with {max_workers} workers")
                return cls._executors[executor_key]
        return None

    async def __call__(self, *args, **kwargs) -> float:
        last_exception = None

        for attempt in range(self.max_retries + 1):
            with self._lock:
                executor = self._executors.get(self._executor_key)

            if executor is None:
                logger.warning(
                    "ProcessPoolExecutor missing for key=%s. Recreating.",
                    self._executor_key,
                )
                executor = self._recreate_executor(self._executor_key, self.max_workers)
                if executor is None:
                    with self._lock:
                        self._executors[self._executor_key] = ProcessPoolExecutor(
                            max_workers=self.max_workers
                        )
                        if self._executor_key not in self._instance_counts:
                            self._instance_counts[self._executor_key] = 1
                        executor = self._executors[self._executor_key]

            loop = asyncio.get_event_loop()
            try:
                future = loop.run_in_executor(
                    executor,
                    partial(self.reward_fn, *args, **kwargs),
                )
                reward = await asyncio.wait_for(
                    future,
                    timeout=self.timeout_seconds,
                )
                return reward
            except asyncio.TimeoutError:
                logger.warning(
                    "Reward computation timed out after %.0fs, returning 0.0",
                    self.timeout_seconds,
                )
                return 0
            except BrokenProcessPool as e:
                last_exception = e
                logger.warning(
                    f"ProcessPoolExecutor broken (attempt {attempt + 1}/{self.max_retries + 1}). "
                    "Attempting to recreate..."
                )
                if attempt < self.max_retries:
                    new_executor = self._recreate_executor(
                        self._executor_key, self.max_workers
                    )
                    if new_executor is None:
                        logger.error("Failed to recreate ProcessPoolExecutor")
                        break
                    continue
                else:
                    logger.error("Max retries exceeded for BrokenProcessPool.")
                    traceback.print_exc()
                    raise e
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error in reward computation: {e}")
                if attempt < self.max_retries:
                    logger.info(
                        f"Retrying... (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    continue
                else:
                    logger.error("Max retries exceeded for unexpected error.")
                    traceback.print_exc()
                    raise e

        if last_exception:
            traceback.print_exc()
            raise last_exception
        else:
            raise RuntimeError("Reward computation failed after all retries.")
