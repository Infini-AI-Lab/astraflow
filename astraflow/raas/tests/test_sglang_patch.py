import asyncio
import json
import sys
import types
from types import SimpleNamespace

from astraflow.raas.patch.sglang import OpenAIReturnTokenIdsPatch


class _RawRequest:
    async def json(self):
        return {"extra_body": {"return_token_ids": True}}


def _install_fake_sglang(monkeypatch):
    module_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
        "sglang.srt.entrypoints.openai.serving_chat",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    serving_chat = sys.modules["sglang.srt.entrypoints.openai.serving_chat"]

    class OpenAIServingChat:
        async def handle_request(self, request, raw_request):
            adapted_request, processed_request = (
                self._convert_to_internal_request(request, raw_request)
            )
            assert adapted_request.input_ids == [1, 2, 3]
            return self._build_chat_response(
                processed_request,
                [{"output_ids": [4, 5]}],
                123,
            )

        def _convert_to_internal_request(self, request, raw_request):
            assert isinstance(raw_request, _RawRequest)
            return SimpleNamespace(input_ids=[1, 2, 3]), request

        def _build_chat_response(self, request, ret, created):
            return SimpleNamespace(
                model_dump=lambda: {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": created,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }
            )

    serving_chat.OpenAIServingChat = OpenAIServingChat
    return OpenAIServingChat


def test_openai_return_token_ids_patch_adds_requested_token_ids(monkeypatch):
    OpenAIServingChat = _install_fake_sglang(monkeypatch)

    assert OpenAIReturnTokenIdsPatch().apply()

    response = asyncio.run(
        OpenAIServingChat().handle_request(SimpleNamespace(), _RawRequest())
    )
    body = json.loads(response.body)

    assert body["prompt_token_ids"] == [1, 2, 3]
    assert body["choices"][0]["token_ids"] == [4, 5]
