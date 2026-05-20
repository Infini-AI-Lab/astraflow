"""Self-contained ASearcher workflow implementation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import uuid
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging
from astraflow.core.workflow.utils.data import resolve_prompt_id

from .agent import SearchAgent
from .prompts import (
    INVALID_PROMPT,
    SEARCH_ACCESS_PROMPT_TEMPLATE,
    SEARCH_ONLY_PROMPT_TEMPLATE,
    VALID_PROMPT,
)
from .reward import correct_format_fn
from .search import SearchToolBox

logger = logging.getLogger(f"ASearcher @ {uuid.uuid4().hex[:4]}")


def _hash_token_list(numbers: list[int]) -> str:
    return hashlib.sha256(json.dumps(numbers, sort_keys=True).encode()).hexdigest()


@register_workflow("asearcher")
class ASearcherWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dataset_path: str,
        dump_dir: str | None = None,
        max_turns: int = 128,
        n_trajs: int = 1,
        search_client_type: str = "async-online-search-access",
        reward_type: str = "F1",
        topk: int = 5,
        valid_inst_ratio: float = 1.0,
        max_tokens: int = 32000,
        search_only: bool = True,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.search_only = search_only
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.reward_type = reward_type
        self.topk = topk
        self.valid_inst_ratio = valid_inst_ratio
        self.search_client_type = search_client_type
        self.toolbox = SearchToolBox(
            dataset_path=dataset_path,
            reward_type=self.reward_type,
            topk=self.topk,
            search_client_type=self.search_client_type,
        )

    async def collect_agent_trajectory(
        self,
        valid_inst: bool,
        qid: str,
        prompt: str,
        prompt_token_ids: list[int],
        engine: Any,
    ) -> tuple[Any, float, Any, dict[str, Any]]:
        agent = SearchAgent(prompt, prompt_token_ids)
        score = 0.0
        ground_truth = None
        traj_rid = uuid.uuid4().hex

        while agent.num_turns < self.max_turns and not agent.is_finished:
            input_ids, sampling_params = agent.prepare_llm_query(self.tokenizer)
            req = ModelRequest(
                rid=traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            if "stop" in sampling_params:
                req.gconfig.stop = sampling_params["stop"]
            if len(input_ids) + self.gconfig.max_new_tokens >= self.max_tokens:
                break

            resp = await engine.agenerate(req)
            completion_str = self.tokenizer.decode(resp.output_tokens)
            tool_calls = agent.consume_llm_response(resp, completion_str)

            if tool_calls:
                tool_call = tool_calls[0]
                result = (await self.toolbox.step((qid, [tool_call])))[0]
                agent.consume_tool_response(result, topk=self.topk)
                if "score" in result:
                    score = float(result["score"])
                if "ground_truth" in result:
                    ground_truth = result["ground_truth"]

            if resp.output_tokens[-1] in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

        llm_gen_records = agent.memory.filter_records("llm_gen")
        format_reward = float(all(correct_format_fn(idx, record.text) for idx, record in enumerate(llm_gen_records)))
        score = score * format_reward

        pred_answer = agent.get_answer()
        judge_q_invalid = False
        if pred_answer is not None:
            judge_q_invalid = any(token in pred_answer for token in ["question", "invalid", "appropriate", "valid"])
        if valid_inst and judge_q_invalid:
            score = -0.5

        stats = agent.memory.logging_stats()
        stats.update(
            dict(
                score=score,
                judge_q_invalid=judge_q_invalid,
                format_reward=format_reward,
            )
        )
        return ground_truth, score, agent.memory, stats

    async def arun_episode(self, engine: Any, data: dict[str, Any]) -> dict[str, Any] | None:
        # Canonical helper — same id the curator gate saw on this prompt.
        prompt_id = resolve_prompt_id(data)
        # qid is used downstream for trajectory tracing and dump filenames;
        # fall back to a uuid when the dataset doesn't carry an id.
        qid = prompt_id if prompt_id is not None else uuid.uuid4().hex

        version = engine.get_version()
        prompt_template = SEARCH_ONLY_PROMPT_TEMPLATE if self.search_only else SEARCH_ACCESS_PROMPT_TEMPLATE
        prompt = prompt_template.format(question=data["question"])
        valid_inst = random.random() <= self.valid_inst_ratio
        if valid_inst:
            prompt = prompt.replace(INVALID_PROMPT, VALID_PROMPT)
        prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        trajs = await asyncio.gather(
            *[
                self.collect_agent_trajectory(valid_inst, qid, prompt, prompt_token_ids, engine)
                for _ in range(self.n_trajs)
            ]
        )

        ground_truth = None
        scores: list[float] = []
        stats: list[dict[str, Any]] = []
        for gt, score, _, traj_stats in trajs:
            if gt is not None:
                ground_truth = gt
            scores.append(score)
            stats.append(traj_stats)

        trajectories = []
        traj_memories = [traj for _, _, traj, _ in trajs]
        for idx, traj_memory in enumerate(traj_memories):
            seqs: list[dict[str, Any]] = []
            for record in traj_memory.memory:
                if record.type != "llm_gen":
                    continue
                success = False
                for seq in seqs:
                    if record.input_len is None or record.input_tokens is None:
                        continue
                    if record.input_len < len(seq["input_ids"]):
                        continue
                    if _hash_token_list(record.input_tokens[: len(seq["input_ids"])]) == _hash_token_list(
                        seq["input_ids"]
                    ):
                        seq_len = len(seq["input_ids"])
                        seq["input_ids"] = list(record.input_tokens) + list(record.output_tokens or [])
                        seq["logprobs"] += [0.0] * (record.input_len - seq_len) + list(record.output_logprobs or [])
                        seq["loss_mask"] += [0] * (record.input_len - seq_len) + [1] * (record.output_len or 0)
                        seq["versions"] += [-1] * (record.input_len - seq_len) + list(record.output_versions or [])
                        success = True
                        break
                if not success:
                    seqs.append(
                        dict(
                            input_ids=list(record.input_tokens or []) + list(record.output_tokens or []),
                            logprobs=[0.0] * (record.input_len or 0) + list(record.output_logprobs or []),
                            loss_mask=[0] * (record.input_len or 0) + [1] * (record.output_len or 0),
                            versions=[-1] * (record.input_len or 0) + list(record.output_versions or []),
                        )
                    )

            first_llm_gen = True
            seq_dicts = []
            for seq in seqs:
                res = dict(
                    input_ids=torch.tensor(seq["input_ids"]).unsqueeze(0),
                    loss_mask=torch.tensor(seq["loss_mask"]).unsqueeze(0),
                    logprobs=torch.tensor(seq["logprobs"]).unsqueeze(0),
                    versions=torch.tensor(seq["versions"]).unsqueeze(0),
                    attention_mask=torch.ones(len(seq["input_ids"]), dtype=torch.bool).unsqueeze(0),
                    begin_of_trajectory=torch.tensor([int(first_llm_gen)]),
                )
                res.update({key: torch.tensor([value]) for key, value in stats[idx].items()})
                first_llm_gen = False
                seq_dicts.append(res)

            trajectories.append({"sequences": seq_dicts, "stats": stats[idx]})

        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.jsonl"),
                "w",
                encoding="utf-8",
            ) as handle:
                for traj_idx, (traj_memory, score) in enumerate(zip(traj_memories, scores)):
                    handle.write(
                        json.dumps(
                            dict(
                                memory=traj_memory.to_dict(),
                                reward=score,
                                ground_truth=ground_truth,
                                traj_idx=traj_idx,
                            )
                        )
                        + "\n"
                    )

        return {
            "prompt_id": prompt_id,
            "n_trajs": self.n_trajs,
            "rewards": torch.tensor(scores, dtype=torch.float32),
            "trajectories": trajectories,
        }
