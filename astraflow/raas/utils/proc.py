import os
import signal
import sys
import threading

import psutil

from astraflow.raas.utils import logging

logger = logging.getLogger(__name__)


def kill_process_tree(
    parent_pid: int | None = None,
    timeout: int = 5,
    include_parent: bool = True,
    skip_pid: int | None = None,
    graceful: bool = True,
) -> None:
    # 1. REMOVED the signal.SIGCHLD lines.
    # Modifying global signals here is dangerous and likely caused the hang.

    current_pid = os.getpid()

    # Handle None parent_pid - defaults to current process
    if parent_pid is None:
        parent_pid = current_pid
        include_parent = False

    # Get process tree
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        logger.info(f"Process {parent_pid} already terminated")
        return

    # Filter skip_pid from children
    if skip_pid is not None:
        children = [c for c in children if c.pid != skip_pid]

    # Terminate based on mode
    if graceful:
        logger.info(
            f"Sending SIGTERM to process {parent_pid} and {len(children)} children"
        )

        # 2. Add handlers to ensure log is flushed before risky operations
        for handler in logger.handlers:
            handler.flush()

        # Send SIGTERM to children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Send SIGTERM to parent if requested (AND if it's not us!)
        if include_parent:
            try:
                # 3. Prevent accidental suicide preventing the wait() logic
                if parent.pid != current_pid:
                    parent.terminate()
                else:
                    logger.info("Skipping SIGTERM on self until children are reaped")
            except psutil.NoSuchProcess:
                pass

        # Wait for graceful shutdown
        # We only wait for parent if it's NOT us. If it is us, we can't wait on ourselves easily.
        procs_to_wait = children + (
            [parent] if include_parent and parent.pid != current_pid else []
        )
        gone, alive = psutil.wait_procs(procs_to_wait, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                f"Force killing {len(alive)} processes that didn't terminate gracefully"
            )
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=1)

        logger.info(f"Successfully cleaned up process tree for PID {parent_pid}")

        # 4. Handle self-termination at the very end if needed
        if include_parent and parent_pid == current_pid:
            logger.info("Cleaning up self now...")

            running_threads = threading.enumerate()
            logger.info(f"running_threads = {len(running_threads)}")
            if len(running_threads) > 1:
                logger.warning(
                    f"Process is hanging because of these threads: {running_threads}"
                )

            # FLUSH ONE LAST TIME or you might lose the log above
            for handler in logger.handlers:
                try:
                    handler.flush()
                    # Don't close handlers - can block if piped to tee/other process
                    # handler.close()
                except:
                    pass

            try:
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout.close()
                sys.stderr.close()
            except:
                pass

            # FORCE KILL: Do not wait for threads, do not run cleanup handlers.
            # logger.info("os exiting...")
            os._exit(0)
            # logger.info("os exited!")

    else:
        # Aggressive mode logic (kept mostly same, but added self-check)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        if include_parent:
            try:
                if parent_pid == current_pid:
                    parent.kill()
                    sys.exit(0)
                parent.kill()
                parent.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                pass
