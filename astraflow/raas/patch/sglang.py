"""SGLang monkey patches for AstraFlow RaaS integration.

TCP v2 architecture: the receiver lives in RaaS (not inside SGLang),
so we only need lightweight patches:

1. ServerArgsPatch — add ``--rollout-manager-address`` CLI arg so SGLang
   can register with RaaS at startup.
2. HttpServerPatch — register SGLang instance with the rollout manager
   during ``launch_server``.
3. LoRACounterLeakPatch — guarantee the LoRA adapter usage counter is
   released for every request, including aborted / client-disconnected ones,
   fixing a weight-sync deadlock at its source (see the class docstring).
"""

import logging

from astraflow.raas.patch import BasePatch

logger = logging.getLogger(__name__)


async def release_lora_ref_once(tm, sub_obj) -> None:
    """Release ``sub_obj``'s LoRA usage counter on ``tm`` (TokenizerManager)
    exactly once, if it is still held.

    Idempotency invariant: SGLang's two native release sites both
    ``del rid_to_state[rid]`` immediately before releasing, so ``rid in
    rid_to_state`` iff the request has NOT yet been released. The membership
    check and ``pop`` have no ``await`` between them, so they are atomic on the
    single-threaded event loop — guaranteeing release is awaited at most once
    per request. This matters because ``ConcurrentCounter.decrement`` has no
    floor: a double-release would drive the counter to -1 and make
    ``wait_for_zero`` (hence ``wait_for_unload``) hang forever.
    """
    if not getattr(tm.server_args, "enable_lora", False):
        return
    if not getattr(sub_obj, "lora_path", None):
        return
    rid = getattr(sub_obj, "rid", None)
    if rid is None or rid not in tm.rid_to_state:
        return
    tm.rid_to_state.pop(rid, None)
    lora_id = getattr(sub_obj, "lora_id", None)
    if lora_id is not None:
        try:
            await tm.lora_registry.release(lora_id)
        except Exception:
            logger.exception(
                "release_lora_ref_once: release failed for rid=%s", rid
            )


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


class LoRACounterLeakPatch(BasePatch):
    """Release the LoRA adapter usage counter on EVERY request teardown.

    Root cause of the LoRA weight-sync deadlock: SGLang's ``LoRARegistry`` keeps
    a per-adapter ``ConcurrentCounter`` (``lora/lora_registry.py``). It is
    ``acquire()``-ed for every generate request but ``release()``-ed only on two
    conditional branches in the tokenizer manager — normal completion
    (``_handle_batch_output``) and one scheduler-abort case (``_wait_one_response``,
    status SERVICE_UNAVAILABLE / INTERNAL_SERVER_ERROR). Requests that are aborted
    or whose client disconnects (which the RaaS per-step drain routinely creates)
    exit ``_wait_one_response`` without releasing — via a ``raise`` (client
    disconnect, BAD_REQUEST) or a plain ``break`` (waiting-queue abort). The
    adapter's counter then never returns to zero, so ``LoRARegistry.wait_for_unload``
    blocks forever. That hangs both an explicit ``/unload_lora_adapter`` AND the
    ``load_lora_adapter`` LRU eviction that fires once ``max_loaded_loras`` versioned
    adapters accumulate — while holding ``lora_update_lock``, freezing all further
    LoRA ops. (The RaaS versioned-name scheme merely defers this to ~``max_loaded_loras``
    steps; this patch removes the leak so unload/eviction is always safe.)

    Fix: wrap ``TokenizerManager.generate_request`` — the single outermost
    per-request async generator, where ``acquire`` happens (via
    ``_validate_and_resolve_lora``) — and release in a ``finally`` so it runs on
    every exit (normal return, raise, ``GeneratorExit``, ``CancelledError``).
    Release is idempotent via the invariant that both native release sites
    ``del rid_to_state[rid]`` immediately before releasing: ``rid in rid_to_state``
    iff not yet released. The membership check and ``pop`` have no ``await``
    between them, so they are atomic on the single-threaded event loop — no
    double-release (which would drive the counter to -1 and hang ``wait_for_zero``
    permanently, since ``ConcurrentCounter.decrement`` has no floor).
    """

    def apply(self) -> bool:
        import os

        if os.getenv("ASTRAFLOW_DISABLE_LORA_LEAK_FIX", "0").lower() in ("1", "true"):
            logger.warning(
                "LoRACounterLeakPatch disabled via ASTRAFLOW_DISABLE_LORA_LEAK_FIX; "
                "LoRA weight-sync may deadlock on registry-LRU eviction."
            )
            return True

        try:
            from sglang.srt.managers.tokenizer_manager import TokenizerManager
        except Exception as e:
            logger.error(f"LoRACounterLeakPatch failed: {e}")
            return False

        original_generate_request = TokenizerManager.generate_request
        if self._is_patched(original_generate_request, "generate_request"):
            return True

        async def patched_generate_request(self, obj, request=None):
            try:
                async for response in original_generate_request(self, obj, request):
                    yield response
            finally:
                # Guaranteed release on every exit path (normal, raise,
                # GeneratorExit, CancelledError). ``obj`` has been normalized by
                # ``original_generate_request`` before it reached the scheduler.
                try:
                    if getattr(obj, "is_single", True):
                        await release_lora_ref_once(self, obj)
                    else:
                        # Batch request: release each sub-request that still
                        # holds its counter. (RaaS rollouts are single;
                        # best-effort.)
                        rids = getattr(obj, "rid", None)
                        if isinstance(rids, (list, tuple)):
                            for i in range(len(rids)):
                                await release_lora_ref_once(self, obj[i])
                except Exception:
                    logger.exception("LoRACounterLeakPatch cleanup error")

        self._mark_as_patched(patched_generate_request, "generate_request")
        TokenizerManager.generate_request = patched_generate_request

        return True
