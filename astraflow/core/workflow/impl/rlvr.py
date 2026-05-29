import asyncio
import os
import random
import uuid
from collections.abc import Callable
from typing import Any

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest, ModelResponse
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured
from astraflow.core.workflow.utils.dynamic_import import import_from_string
from astraflow.core.workflow.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("RLVR workflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=False,
    )
    return list(input_ids)


SOLVER_PROMPT_SUFFIX = (
    "\nLet's think step by step. Please put your final answer within \\boxed{}."
)


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    messages = data["messages"]
    # Append solve instruction suffix to user messages
    return [
        {**m, "content": m["content"] + SOLVER_PROMPT_SUFFIX}
        if m["role"] == "user" else m
        for m in messages
    ]


@register_workflow("rlvr")
class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        get_input_ids_fn: Callable[
            [Any, PreTrainedTokenizerFast, bool], list[int]
        ] = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[
            [dict[str, Any]], Any
        ] = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.get_input_ids_fn = get_input_ids_fn
        self.data_extract_prompt_fn = data_extract_prompt_fn
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[float, str]:
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward, completions_str

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float, str]:
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward, completions_str = await self._compute_rewards(
            resp, prompt_str, task_data
        )

        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        return resp, reward, completions_str

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        prompt_strs = [prompt_str] * n_samples

        # Generate responses and collect rewards
        sample_results = await asyncio.gather(
            *[
                self._collect_samples(engine, req, prompt_str, data)
                for _ in range(n_samples)
            ]
        )
        if sample_results:
            resps, rewards, completions_strs = map(list, zip(*sample_results))
        else:
            resps, rewards, completions_strs = [], [], []

        # Build result tensors
        results = []
        for resp, reward in zip(resps, rewards):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                "versions": torch.tensor(versions, dtype=torch.int32),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool),
                "rewards": torch.tensor(reward, dtype=torch.float32),
            }
            res = {k: v.unsqueeze(0) for k, v in res.items()}
            results.append(res)

        # save rollout to file with a 1 in 128 random chance
        if self.dump_dir is not None and random.random() < 1 / 128:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            # Get the unique identifier for this prompt
            qid = resolve_prompt_id(data) or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            seqlens = [
                len(resp.input_tokens) + len(resp.output_tokens) for resp in resps
            ]
            async with aiofiles.open(file_path, "a") as f:
                answer = data.get("answer")
                if answer is not None:
                    await f.write(
                        "answer is: "
                        f"{colorama.Fore.YELLOW + colorama.Style.DIM}{answer}{colorama.Style.RESET_ALL}\n"
                    )
                for i, (prompt, completion, reward, seqlen) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info_lines = [
                        f"idx: {i + 1} / {n_samples}, seqlen: {seqlen}, reward is {reward}.",
                        f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                        f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{completion}{colorama.Style.RESET_ALL}",
                    ]
                    info = "\n".join(info_lines)
                    await f.write(info + "\n")

        return results_to_structured(results, prompt_id=resolve_prompt_id(data))
