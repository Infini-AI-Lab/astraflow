"""SGLang monkey patches for AstraFlow RaaS integration.

TCP v2 architecture: the receiver lives in RaaS (not inside SGLang),
so we only need lightweight patches:

1. ServerArgsPatch — add ``--rollout-manager-address`` CLI arg so SGLang
   can register with RaaS at startup.
2. HttpServerPatch — register SGLang instance with the rollout manager
   during ``launch_server``.
3. LoRAAbortReleasePatch — fix missing ``lora_registry.release()`` in the
   abort path so that LoRA weight updates via abort+unload don't hang.
"""

import logging

from astraflow.raas.patch import BasePatch

logger = logging.getLogger(__name__)


class ServerArgsPatch(BasePatch):
    """Add ``--rollout-manager-address`` to SGLang's ServerArgs."""

    def apply(self) -> bool:
        try:
            from sglang.srt import server_args

            ServerArgs = server_args.ServerArgs

            if hasattr(ServerArgs, "rollout_manager_address"):
                return True

            ServerArgs.rollout_manager_address = None

            original_add_cli_args = ServerArgs.add_cli_args

            def patched_add_cli_args(parser):
                original_add_cli_args(parser)
                parser.add_argument(
                    "--rollout-manager-address",
                    type=str,
                    default=None,
                    help="The address of the rollout manager",
                )

            self._mark_as_patched(patched_add_cli_args, "add_cli_args")
            ServerArgs.add_cli_args = staticmethod(patched_add_cli_args)

            original_prepare_server_args = server_args.prepare_server_args

            def patched_prepare_server_args(args_list):
                result = original_prepare_server_args(args_list)

                import argparse

                parser = argparse.ArgumentParser()
                ServerArgs.add_cli_args(parser)
                parsed_args = parser.parse_args(args_list)

                result.rollout_manager_address = (
                    parsed_args.rollout_manager_address
                )
                return result

            server_args.prepare_server_args = patched_prepare_server_args

            return True
        except Exception as e:
            logger.error(f"ServerArgsPatch failed: {e}")
            return False


class HttpServerPatch(BasePatch):
    """Register SGLang instance with RaaS rollout manager at startup."""

    def apply(self) -> bool:
        try:
            import requests
            from sglang.srt.entrypoints import http_server

            original_launch_server = http_server.launch_server

            if self._is_patched(original_launch_server, "launch_server"):
                return True

            def patched_launch_server(server_args, *args, **kwargs):
                return original_launch_server(server_args, *args, **kwargs)

            self._mark_as_patched(patched_launch_server, "launch_server")
            http_server.launch_server = patched_launch_server

            return True
        except Exception as e:
            logger.error(f"HttpServerPatch failed: {e}")
            import traceback

            traceback.print_exc()
            return False


class LoRAAbortReleasePatch(BasePatch):
    """Fix missing ``lora_registry.release()`` in the abort path.

    When a LoRA request is aborted from the waiting queue, SGLang's
    ``_handle_abort_req`` does NOT call ``lora_registry.release()``,
    leaking the ``ConcurrentCounter``.  This causes
    ``wait_for_unload()`` to hang forever when we try to swap LoRA
    adapters via abort + unload.

    The normal completion path (``_handle_batch_output``) and the
    scheduler error path both release correctly — only the waiting-queue
    abort path is missing the call.

    This patch wraps ``_handle_abort_req`` to add the missing release,
    mirroring the pattern at ``tokenizer_manager.py:1679-1680``.
    """

    def apply(self) -> bool:
        try:
            import asyncio

            from sglang.srt.managers.tokenizer_manager import TokenizerManager

            original = TokenizerManager._handle_abort_req

            if self._is_patched(original, "handle_abort_req"):
                return True

            def patched_handle_abort_req(self_tm, recv_obj):
                original(self_tm, recv_obj)

                # Release LoRA counter for aborted requests — mirrors the
                # normal completion path at tokenizer_manager.py:1679-1680.
                if self_tm.server_args.enable_lora:
                    state = self_tm.rid_to_state.get(recv_obj.rid)
                    if (
                        state is not None
                        and getattr(state.obj, "lora_path", None)
                    ):
                        asyncio.create_task(
                            self_tm.lora_registry.release(state.obj.lora_id)
                        )

            self._mark_as_patched(patched_handle_abort_req, "handle_abort_req")
            TokenizerManager._handle_abort_req = patched_handle_abort_req

            return True
        except Exception as e:
            logger.error(f"LoRAAbortReleasePatch failed: {e}")
            return False
