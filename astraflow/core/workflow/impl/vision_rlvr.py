import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any, cast

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest, ModelResponse
from astraflow.core.workflow.impl.rlvr import RLVRWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured
from astraflow.core.workflow.utils.image import image2base64
from astraflow.core.workflow.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("RLVR workflow")


@register_workflow("vision_rlvr")
class VisionRLVRWorkflow(RLVRWorkflow):
    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        enable_thinking: bool,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        super().__init__(
            reward_fn,
            gconfig,
            tokenizer,
            enable_thinking,
            rollout_stat_scope=rollout_stat_scope,
            dump_dir=dump_dir,
        )
        self.processor = processor

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[float, str]:
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
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
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        processed_input = processor_callable(
            images=data["images"],
            text=data["messages"],
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]

        n_samples = self.gconfig.n_samples

        byte_images = image2base64(data["images"])
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            vision_msg_vllm=[data["messages_chat"]]
            if "messages_chat" in data
            else None,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
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

            multi_modal_input = [
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input:
                multi_modal_input[0]["image_grid_thw"] = processed_input[
                    "image_grid_thw"
                ]

            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "multi_modal_input": multi_modal_input,
                "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            }
            results.append(res)

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            qid = resolve_prompt_id(data) or uuid.uuid4().hex

            file_path = os.path.join(dump_path, f"{qid}.txt")
            seqlens = [
                len(resp.input_tokens) + len(resp.output_tokens) for resp in resps
            ]
            async with aiofiles.open(file_path, "a") as f:
                for i, (prompt, completion, reward, seqlen) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {seqlen}, reward is {reward}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{completion}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        return results_to_structured(results, prompt_id=resolve_prompt_id(data))
