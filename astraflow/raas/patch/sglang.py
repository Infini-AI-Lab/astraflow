"""SGLang monkey patches for AstraFlow RaaS integration.

TCP v2 architecture: the receiver lives in RaaS (not inside SGLang),
so we only need lightweight patches:

1. ServerArgsPatch — add ``--rollout-manager-address`` CLI arg so SGLang
   can register with RaaS at startup.
2. HttpServerPatch — register SGLang instance with the rollout manager
   during ``launch_server``.
3. OpenAIReturnTokenIdsPatch — preserve token IDs in the OpenAI chat response
   when clients request ``return_token_ids``.
"""

import logging

from astraflow.raas.patch import BasePatch

logger = logging.getLogger(__name__)


def _requested_return_token_ids(body) -> bool:
    if not isinstance(body, dict):
        return False
    if body.get("return_token_ids") is True:
        return True
    extra_body = body.get("extra_body")
    return isinstance(extra_body, dict) and extra_body.get("return_token_ids") is True


def _as_token_id_list(value):
    if not isinstance(value, list):
        return None
    if not value:
        return []
    if all(isinstance(token_id, int) for token_id in value):
        return value
    return None


def _first_token_id_list(value):
    token_ids = _as_token_id_list(value)
    if token_ids is not None:
        return token_ids
    if isinstance(value, list) and value:
        return _as_token_id_list(value[0])
    return None


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


class OpenAIReturnTokenIdsPatch(BasePatch):
    """Return token IDs through SGLang's OpenAI chat endpoint when requested.

    Harbor enables rollout-detail collection by sending ``logprobs=true`` and
    ``extra_body.return_token_ids=true`` through LiteLLM. SGLang already has the
    prompt IDs before dispatch and the generated IDs in ``ret[*]["output_ids"]``;
    the OpenAI response builder simply drops both. This patch only changes
    non-streaming chat responses for requests that explicitly ask for token IDs.
    """

    def apply(self) -> bool:
        try:
            from fastapi.responses import ORJSONResponse
            from sglang.srt.entrypoints.openai.serving_chat import (
                OpenAIServingChat,
            )

            original_handle_request = OpenAIServingChat.handle_request
            original_convert = OpenAIServingChat._convert_to_internal_request
            original_build_response = OpenAIServingChat._build_chat_response

            if self._is_patched(
                original_build_response, "openai_return_token_ids"
            ):
                return True

            async def patched_handle_request(
                self_chat, request, raw_request, *args, **kwargs
            ):
                try:
                    body = await raw_request.json()
                except Exception:
                    body = {}

                if _requested_return_token_ids(body):
                    object.__setattr__(
                        request, "_astraflow_return_token_ids", True
                    )

                return await original_handle_request(
                    self_chat, request, raw_request, *args, **kwargs
                )

            def patched_convert_to_internal_request(
                self_chat, request, *args, **kwargs
            ):
                adapted_request, processed_request = original_convert(
                    self_chat, request, *args, **kwargs
                )

                if getattr(request, "_astraflow_return_token_ids", False):
                    prompt_token_ids = _first_token_id_list(
                        getattr(adapted_request, "input_ids", None)
                    )
                    object.__setattr__(
                        processed_request, "_astraflow_return_token_ids", True
                    )
                    if prompt_token_ids is not None:
                        object.__setattr__(
                            processed_request,
                            "_astraflow_prompt_token_ids",
                            prompt_token_ids,
                        )

                return adapted_request, processed_request

            def patched_build_chat_response(
                self_chat, request, ret, created, *args, **kwargs
            ):
                response = original_build_response(
                    self_chat, request, ret, created, *args, **kwargs
                )

                if not getattr(request, "_astraflow_return_token_ids", False):
                    return response
                if not hasattr(response, "model_dump"):
                    return response

                data = response.model_dump()

                prompt_token_ids = getattr(
                    request, "_astraflow_prompt_token_ids", None
                )
                if prompt_token_ids is not None:
                    data["prompt_token_ids"] = prompt_token_ids

                choices = data.get("choices")
                if isinstance(choices, list):
                    for idx, choice in enumerate(choices):
                        if idx >= len(ret) or not isinstance(choice, dict):
                            continue
                        token_ids = _as_token_id_list(
                            ret[idx].get("output_ids")
                        )
                        if token_ids is not None:
                            choice["token_ids"] = token_ids

                return ORJSONResponse(content=data)

            self._mark_as_patched(
                patched_handle_request, "openai_return_token_ids"
            )
            self._mark_as_patched(
                patched_convert_to_internal_request,
                "openai_return_token_ids",
            )
            self._mark_as_patched(
                patched_build_chat_response, "openai_return_token_ids"
            )

            OpenAIServingChat.handle_request = patched_handle_request
            OpenAIServingChat._convert_to_internal_request = (
                patched_convert_to_internal_request
            )
            OpenAIServingChat._build_chat_response = patched_build_chat_response

            return True
        except Exception as e:
            logger.error(f"OpenAIReturnTokenIdsPatch failed: {e}")
            return False

