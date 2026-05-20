import asyncio
import os
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
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured

logger = logging.getLogger("Multi-Turn workflow")


@register_workflow("multi_turn")
class MultiTurnWorkflow(RolloutWorkflow):
    """Multi-attempt workflow that retries generation until the reward is positive."""

    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if not (0.0 < turn_discount <= 1.0):
            raise ValueError("turn_discount must be in (0, 1].")

        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.rollout_stat_scope = rollout_stat_scope
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Create tokens that should be amended if the answer is incorrect.
        messages = [{"role": "assistant", "content": "some random message."}]
        s1 = list(self.tokenizer.apply_chat_template(messages, tokenize=True))
        messages += [
            {
                "role": "user",
                "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the previous errors, and try to answer it again.",
            }
        ]
        s2 = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        self.multi_turn_prompt_ids = s2[len(s1) :]

    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], str, str, float, int]:
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        input_ids: list[int] = list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        t = 0
        reward = 0.0
        discount = 1.0
        prompt_str = ""
        completions_str = ""
        while reward == 0.0 and t < self.max_turns:
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            print(f"MultiTurnWorkflow: About to call engine.agenerate(), rid={req.rid}")
            resp = await engine.agenerate(req)
            print(f"MultiTurnWorkflow: engine.agenerate() returned, rid={req.rid}, output_len={len(resp.output_tokens)}")
            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            reward = await self.async_reward_fn(
                prompt_str,
                completions_str,
                resp.input_tokens,
                resp.output_tokens,
                **data,
            )
            input_len = len(resp.input_tokens) - len(seq)
            assert len(seq) == 0 or resp.input_tokens[:-input_len] == seq, (
                seq,
                resp.input_tokens[:-input_len],
                len(seq),
                len(resp.input_tokens[:-input_len]),
            )
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions
            t += 1
            if reward == 0.0 and t < self.max_turns:
                input_ids = input_ids + resp.output_tokens
                if (
                    resp.output_tokens
                    and resp.output_tokens[-1] != self.tokenizer.eos_token_id
                ):
                    input_ids += [self.tokenizer.eos_token_id]
                input_ids += self.multi_turn_prompt_ids
                discount *= self.turn_discount

        reward = float(reward * discount)

        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward, num_turns=t)

        res = dict(
            input_ids=torch.tensor(seq, dtype=torch.int32),
            logprobs=torch.tensor(logprobs, dtype=torch.float32),
            loss_mask=torch.tensor(loss_mask, dtype=torch.int32),
            versions=torch.tensor(versions, dtype=torch.int32),
            rewards=torch.tensor(reward, dtype=torch.float32),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return (
            res,
            prompt_str,
            completions_str,
            reward,
            len(seq),
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        tasks = [
            self._run_one_episode(engine, data) for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        if self.dump_dir is not None:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            qid = resolve_prompt_id(data) or uuid.uuid4().hex

            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (_, p, c, r, sl) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        prompt_id = resolve_prompt_id(data)
        data = [res[0] for res in results]
        return results_to_structured(data, prompt_id=prompt_id)
