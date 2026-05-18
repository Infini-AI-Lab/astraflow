import logging
from http import HTTPStatus

import uvloop
from fastapi import Depends, Request
from vllm.entrypoints.openai.api_server import (
    create_completion as original_create_completion,
)
from vllm.entrypoints.openai.api_server import (
    router,
    run_server,
    validate_json_request,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.utils import cli_env_setup, load_aware_call, with_cancellation
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger("vllm_server")
logger.setLevel(logging.INFO)


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completions endpoint."""
    response = await original_create_completion(request, raw_request)
    return response


def _use_warning_once_for_eos() -> None:
    """Backport from latest vLLM: use warning_once instead of warning for
    get_eos_token_id / get_bos_token_id when tokenizer is None.

    In vLLM 0.11.0 these use logger.warning which fires on every request.
    Newer vLLM already switched to logger.warning_once.  We patch the methods
    here so the warning is emitted at most once."""
    try:
        from vllm.inputs.preprocess import InputPreprocessor

        _orig_eos = InputPreprocessor.get_eos_token_id
        _orig_bos = InputPreprocessor.get_bos_token_id
        _warned = {"eos": False, "bos": False}

        def _get_eos_once(self):
            if self.tokenizer is None:
                if not _warned["eos"]:
                    logger.warning("Using None for EOS token id because "
                                   "tokenizer is not initialized")
                    _warned["eos"] = True
                return None
            return self.tokenizer.eos_token_id

        def _get_bos_once(self):
            if self.tokenizer is None:
                if not _warned["bos"]:
                    logger.warning("Using None for BOS token id because "
                                   "tokenizer is not initialized")
                    _warned["bos"] = True
                return None
            return self.tokenizer.bos_token_id

        InputPreprocessor.get_eos_token_id = _get_eos_once
        InputPreprocessor.get_bos_token_id = _get_bos_once
    except Exception:
        pass


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.f
    cli_env_setup()
    _use_warning_once_for_eos()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
