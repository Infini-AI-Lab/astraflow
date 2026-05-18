"""SGLang launch entrypoint with inference-engine patches pre-applied."""
import os
import sys

from astraflow.raas.patch import apply_patches

# Apply patches at module level so they also run in spawned child processes.
apply_patches()

if __name__ == '__main__':
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
