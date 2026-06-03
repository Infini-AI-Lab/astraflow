"""
Generic Task Server Workflow

This workflow connects to any task server implementing the standard
Task Server API, enabling training on external environments.
"""

import asyncio
import os
import random
import uuid
from typing import Any, Dict, List, Optional

import aiofiles
import aiofiles.os
import aiohttp
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id, results_to_structured

logger = logging.getLogger("TaskServerWorkflow")


@register_workflow("task_server")
class TaskServerWorkflow(RolloutWorkflow):
    """
    Generic workflow that connects to external task servers.

    Task servers must implement the standard Task Server API:
    - POST /episode/start  - Initialize episode, return initial observation
    - POST /episode/step   - Execute action, return (obs, reward, done, info)
    - POST /episode/cancel - Cancel episode
    """

    def __init__(
        self,
        task_server_url: str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 20,
        rollout_stat_scope: str = "rollout",
        turn_discount: float = 1.0,
        dump_dir: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.task_server_url = task_server_url.rstrip("/")
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.rollout_stat_scope = rollout_stat_scope
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        self.timeout = timeout
        self._server_info = None

    async def _init_server_info(self):
        if self._server_info is None:
            try:
                self._server_info = await self._call_server("api/task/info", "GET")
            except Exception as e:
                logger.error(f"Failed to connect to task server {self.task_server_url}: {e}")
                raise

    async def _call_server(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        url = f"{self.task_server_url}/{endpoint}"

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        connector = aiohttp.TCPConnector(force_close=True, limit=10)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        try:
            if method == "POST":
                async with session.post(url, json=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"Server returned {resp.status}: {error_text}"
                        )
                    result = await resp.json()
            elif method == "GET":
                async with session.get(url, params=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"Server returned {resp.status}: {error_text}"
                        )
                    result = await resp.json()
            else:
                raise ValueError(f"Unsupported method: {method}")

            return result

        except asyncio.TimeoutError:
            raise Exception(f"Request to {url} timed out after {self.timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling {url}: {e}")
        finally:
            await session.close()
            await asyncio.sleep(0)

    def format_observation_to_messages(
        self,
        observation: Dict[str, Any],
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        if observation["type"] == "text":
            return history + [{"role": "user", "content": observation["content"]}]
        elif observation["type"] == "messages":
            return observation["content"]
        else:
            raise NotImplementedError(
                f"Observation type '{observation['type']}' not supported. "
                f"Override format_observation_to_messages() to handle it."
            )

    def format_agent_output_to_action(
        self,
        agent_output: str,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"type": "text", "content": agent_output}

    def postprocess_reward(
        self,
        reward: float,
        num_turns: int,
        info: Dict[str, Any],
    ) -> float:
        return reward * (self.turn_discount ** num_turns)

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> Optional[Dict[str, torch.Tensor]]:
        n_samples = self.gconfig.n_samples

        results = await asyncio.gather(
            *[self._run_one_episode(engine, data) for _ in range(n_samples)]
        )

        failed_count = sum(1 for r in results if r is None)

        sample_id = data.get("task_id") or data.get("index")
        if failed_count > 0:
            if failed_count < n_samples:
                logger.warning(
                    f"{failed_count}/{n_samples} episodes failed for sample {sample_id}, "
                    f"rejecting entire batch to maintain GRPO granularity"
                )
            return None

        return results_to_structured(results, prompt_id=resolve_prompt_id(data))

    def _compute_user_message_delta_tokens(self, user_content: str) -> List[int]:
        from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids
        ref_messages = [{"role": "assistant", "content": "X"}]
        ref_tokens = apply_chat_template_to_ids(
            self.tokenizer, ref_messages, tokenize=True, enable_thinking=False
        )

        full_messages = ref_messages + [{"role": "user", "content": user_content}]
        full_tokens = apply_chat_template_to_ids(
            self.tokenizer, full_messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

        eos_idx = ref_tokens.index(self.tokenizer.eos_token_id)
        return full_tokens[eos_idx + 1:]

    async def _run_one_episode(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> Optional[Dict[str, torch.Tensor]]:
        await self._init_server_info()

        sample_id = data.get("task_id") or data.get("index")
        if sample_id is None:
            raise ValueError("Data must contain 'task_id' or 'index' field")

        episode_id = None

        try:
            response = await self._call_server(
                "api/episode/start",
                data={"sample_id": str(sample_id), "config": {}},
            )

            episode_id = response["episode_id"]
            observation = response["observation"]
            info = response.get("info", {})

            messages = []
            total_reward = 0.0
            done = False

            seq, logprobs, loss_mask, versions = [], [], [], []

            messages = self.format_observation_to_messages(observation, messages)

            from astraflow.core.workflow.utils.hf_utils import apply_chat_template_to_ids
            input_ids: List[int] = apply_chat_template_to_ids(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )

            num_turns = 0
            for turn in range(self.max_turns):
                if done:
                    break

                req = ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=input_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )

                resp = await engine.agenerate(req)

                agent_output = self.tokenizer.decode(resp.output_tokens)

                input_len = len(resp.input_tokens) - len(seq)
                assert len(seq) == 0 or list(resp.input_tokens[:-input_len]) == seq, (
                    f"Token mismatch at turn {turn}: "
                    f"seq_len={len(seq)}, input_tokens_len={len(resp.input_tokens)}, "
                    f"input_len={input_len}"
                )

                if input_len > 0:
                    seq += list(resp.input_tokens[-input_len:])
                    logprobs += [0.0] * input_len
                    loss_mask += [0] * input_len
                    versions += [-1] * input_len

                seq += list(resp.output_tokens)
                logprobs += list(resp.output_logprobs)
                loss_mask += [1] * len(resp.output_tokens)
                versions += list(resp.output_versions)

                action = self.format_agent_output_to_action(agent_output, info)

                step_response = await self._call_server(
                    "api/episode/step",
                    data={"episode_id": episode_id, "action": action},
                )

                observation = step_response.get("observation")
                reward = step_response.get("reward", 0.0)
                done = step_response["done"]
                info = step_response.get("info", {})

                total_reward += reward
                num_turns += 1

                if not done and observation:
                    messages.append({"role": "assistant", "content": agent_output})

                    if observation["type"] == "text":
                        input_ids = list(resp.input_tokens) + list(resp.output_tokens)
                        if resp.output_tokens and resp.output_tokens[-1] != self.tokenizer.eos_token_id:
                            input_ids.append(self.tokenizer.eos_token_id)
                        new_user_tokens = self._compute_user_message_delta_tokens(observation["content"])
                        input_ids += new_user_tokens

                        messages = self.format_observation_to_messages(observation, messages)
                    else:
                        raise NotImplementedError

            if not done and episode_id:
                try:
                    await self._call_server(
                        "api/episode/cancel",
                        data={"episode_id": episode_id},
                    )
                except Exception as cancel_error:
                    logger.warning(f"Failed to cancel incomplete episode: {cancel_error}")

            shaped_reward = self.postprocess_reward(total_reward, num_turns, info)

            stats_tracker.get(self.rollout_stat_scope).scalar(
                reward=shaped_reward,
                num_turns=num_turns,
                success=info.get("success", 0.0),
            )

            # Dump trajectory sample for debugging (deterministic per sample_id
            # so all n_samples trajectories for the same prompt are dumped together)
            if self.dump_dir is not None and hash(str(sample_id)) % 32 == 0:
                version = engine.get_version()
                dump_path = os.path.join(self.dump_dir, str(version))
                await aiofiles.os.makedirs(dump_path, exist_ok=True)
                file_path = os.path.join(dump_path, f"{sample_id}.txt")
                async with aiofiles.open(file_path, "a") as f:
                    header = (
                        f"=== Episode (sample_id={sample_id}, "
                        f"reward={shaped_reward}, turns={num_turns}, "
                        f"success={info.get('success', 'N/A')}, "
                        f"done={done}) ===\n\n"
                    )
                    await f.write(header)
                    await f.write(self.tokenizer.decode(seq))
                    await f.write("\n\n")

            result = {
                "input_ids": torch.tensor(seq, dtype=torch.int32),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
                "versions": torch.tensor(versions, dtype=torch.int32),
                "rewards": torch.tensor(shaped_reward, dtype=torch.float32),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            }
            result = {k: v.unsqueeze(0) for k, v in result.items()}
            return result

        except Exception as e:
            logger.error(f"Episode failed for sample {sample_id}: {e}")

            if episode_id:
                try:
                    await self._call_server(
                        "api/episode/cancel",
                        data={"episode_id": episode_id},
                    )
                except Exception as cancel_error:
                    logger.warning(f"Failed to cancel episode: {cancel_error}")

            return None
