import getpass
import os
import time

from astraflow.raas.utils import logging, name_resolve, names

logger = logging.getLogger("Launcher Utils")

LOCAL_CACHE_DIR = os.getenv("ASTRAFLOW_CACHE_DIR", f"/tmp/astraflow-{getpass.getuser()}")
TRITON_CACHE_PATH = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/triton/"
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)


def wait_llm_server_addrs(
    experiment_name: str,
    trial_name: str,
    n_rollout_servers: int = 1,
    timeout: int | None = 1200,
):
    """Poll name_resolve until all rollout server addresses are registered."""
    name = names.gen_servers(experiment_name, trial_name)
    start = time.perf_counter()
    while True:
        rollout_addrs = name_resolve.get_subtree(name)
        if len(rollout_addrs) >= n_rollout_servers:
            logger.info(
                f"Found {len(rollout_addrs)} rollout servers: {', '.join(rollout_addrs)}"
            )
            break

        time.sleep(1)
        if timeout is not None and time.perf_counter() - start > timeout:
            raise TimeoutError(
                f"Timeout waiting for rollout servers to be ready. "
                f"Expected {n_rollout_servers} servers, found {len(rollout_addrs)}."
            )
    return rollout_addrs
