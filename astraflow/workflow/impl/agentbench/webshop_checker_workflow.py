"""WebShop Actor + Checker workflow.

Before sending "Buy Now" to the task server, a checker model estimates
the purchase score.  If the score is below a threshold, feedback is
injected and the actor gets one retry.

Two-model setup via EngineGroup:
  - model0 (actor, e.g. Qwen2.5-7B): plays the WebShop episode
  - model1 (checker, e.g. Qwen3-4B): scores the purchase before "Buy Now"

Both models are trained.  Returns the structured format::

    {
        "n_trajs": int,
        "rewards": Tensor[n_trajs],
        "trajectories": [{"sequences": [actor_seq, checker_seq]}, ...],
    }

Each trainer filters by model_id to compute loss on its own tokens only.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import aiofiles
import aiofiles.os
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.workflow.api.cli_args import GenerationHyperparameters
from astraflow.workflow.api.engine_api import InferenceEngine
from astraflow.workflow.api.io_struct import ModelRequest
from astraflow.workflow.impl.agentbench.webshop_task_server import (
    WebshopTaskServerWorkflow,
)
from astraflow.workflow.registry import register_workflow
from astraflow.workflow.utils.data import resolve_prompt_id
from astraflow.workflow.utils import logging, stats_tracker

logger = logging.getLogger("WebShopCheckerWorkflow")

MODEL_ID_PROMPT = -1
MODEL_ID_ACTOR = 0
MODEL_ID_CHECKER = 1

CHECKER_PROMPT = """\
You are a purchase verification assistant for an online shopping task. \
Your job is to score how well a purchase matches the shopping goal using \
a precise scoring formula.

## Shopping Goal
{goal}

## Current Product Page
{observation}

## Scoring Rules

You must evaluate 4 components and compute the final score.

### Step 1: Type Match (r_type)
Is this the correct TYPE of product? Compare the product name/category \
to what the goal asks for.
- r_type = 1.0 → Correct product type (e.g., goal says "desk lamp", product is a desk lamp)
- r_type = 0.5 → Loosely related but not quite right (e.g., goal says "desk lamp", product is a floor lamp)
- r_type = 0.1 → Very different product (e.g., goal says "desk lamp", product is a lampshade)
- r_type = 0.0 → Completely wrong (e.g., goal says "desk lamp", product is a phone case)

### Step 2: Attribute Match (num_attr_matches / num_goal_attributes)
Extract the required attributes from the goal (e.g., "machine washable", \
"stainless steel", "BPA-free"). Check each one against the product page. \
An attribute matches if it appears in the product title, description, \
bullet points, or attributes list (approximate matching is OK — \
e.g., "machine wash" matches "machine washable").

### Step 3: Option Match (num_option_matches / num_goal_options)
Extract the required options from the goal (e.g., "color: gold", \
"size: large", "style: modern"). Check each one against the SELECTED \
options on the product page. Note: similar colors count as matches \
(e.g., "gold" ≈ "yellow gold").
If the goal has no specific options, skip this step (score = N/A).

### Step 4: Price Check (r_price)
If the goal specifies a price limit (e.g., "price lower than $50"), \
check whether the product price is within budget.
- r_price = 1 if price ≤ budget (or no budget specified)
- r_price = 0 if price > budget

### Final Score Formula
score = r_type × (num_attr_matches + num_option_matches + r_price) \
/ (num_goal_attributes + num_goal_options + 1)
If a component is N/A, exclude it from both numerator and denominator.


## Your Task

Now evaluate the current purchase. Think step by step through each \
component, then compute the final score.

Respond in EXACTLY this format:
Your step by step thoughts.

Final: <show calculation> = <score>
Missing: <what is missing or wrong, or "nothing">
Suggestion: <corrective action, or "none">

"""


def _parse_checker_output(text: str) -> tuple[float, str]:
    """Parse checker output into (score, feedback_message).

    Supports the step-by-step format with "Final: <calc> = <score>" line.
    Extracts the LAST number on the Final: line (the final score after all
    intermediate calculations).  Falls back to fraction parsing and Score:.
    Returns (1.0, "") if parsing fails so the purchase goes through.
    """
    score = None

    # Find all "Final:" lines (skip "Final Score Formula:" from the prompt echo)
    for final_line_match in re.finditer(r"^[Ff]inal:\s*(.+)", text, re.MULTILINE):
        line = final_line_match.group(1)
        # Skip pure template echoes like "<show calculation> = <score>"
        # but NOT lines where model wrote actual numbers after the template
        if "<score>" in line and not re.search(r"\d+\.\d+|\d+/\d+", line):
            continue
        # Extract the LAST decimal number on the line (the final result)
        numbers = re.findall(r"(\d+\.\d+|\d+/\d+)", line)
        if numbers:
            last = numbers[-1]
            try:
                if "/" in last:
                    num, den = last.split("/")
                    score = float(num) / float(den)
                else:
                    score = float(last)
            except (ValueError, ZeroDivisionError):
                pass

    # Fallback: try "score = <number>" or "Score: <number>"
    if score is None:
        for score_match in re.finditer(r"[Ss]core[=:]\s*([\d.]+)", text):
            try:
                score = float(score_match.group(1))
            except ValueError:
                pass

    if score is None:
        logger.warning("Checker output has no score, defaulting to APPROVE")
        return 1.0, ""

    score = max(0.0, min(1.0, score))

    missing_match = re.search(r"[Mm]issing:\s*(.+)", text)
    suggestion_match = re.search(r"[Ss]uggestion:\s*(.+)", text)

    missing = missing_match.group(1).strip() if missing_match else ""
    suggestion = suggestion_match.group(1).strip() if suggestion_match else ""

    feedback_parts = []
    if missing and missing.lower() != "nothing":
        feedback_parts.append(f"Missing: {missing}")
    if suggestion and suggestion.lower() != "none":
        feedback_parts.append(f"Suggestion: {suggestion}")

    feedback = "\n".join(feedback_parts)
    return score, feedback


def _is_buy_action(action_content: str) -> bool:
    """Check if the action is a 'Buy Now' click."""
    return "buy now" in action_content.lower()


def _build_seq_dict(
    input_ids: list[int],
    output_ids: list[int],
    output_logprobs: list[float],
    output_versions: list[int],
    model_id: int,
    reward: float,
    is_first: bool = False,
) -> dict[str, Any]:
    """Build a self-contained sequence tensor dict for one model."""
    full_ids = input_ids + output_ids
    total_len = len(full_ids)
    p_len = len(input_ids)
    o_len = len(output_ids)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * p_len + list(output_logprobs), dtype=torch.float32
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * p_len + [1] * o_len, dtype=torch.int32
        ).unsqueeze(0),
        "model_ids": torch.cat([
            torch.full((p_len,), MODEL_ID_PROMPT, dtype=torch.long),
            torch.full((o_len,), model_id, dtype=torch.long),
        ]).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * p_len + list(output_versions), dtype=torch.int32
        ).unsqueeze(0),
        "attention_mask": torch.ones(total_len, dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([reward], dtype=torch.float32),
        "begin_of_trajectory": torch.tensor([int(is_first)]),
    }


@register_workflow("webshop_checker")
class WebShopCheckerWorkflow(WebshopTaskServerWorkflow):
    """WebShop workflow with a score-based checker before purchase.

    In two-model mode (EngineGroup with model0 + model1):
      - model0 = actor (trained, model_id=0)
      - model1 = checker (trained, model_id=1)

    Returns the structured format so the buffer can split sequences by
    model and filter independently.
    """

    def __init__(
        self,
        *args,
        checker_threshold: float = 0.8,
        tokenizers: dict[str, PreTrainedTokenizerFast] | None = None,
        gconfigs: dict[str, GenerationHyperparameters] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.checker_threshold = checker_threshold
        self._tokenizers = tokenizers or {}
        self._gconfigs = gconfigs or {}

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> Optional[Dict[str, Any]]:
        """Override to return structured format with per-model sequences."""
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

        # Build structured result.
        trajectories = []
        rewards = []
        agent_metrics: dict[str, float] = {}
        n_checker_used = 0

        for sequences, reward, checker_used in results:
            trajectories.append({"sequences": sequences})
            rewards.append(reward)
            if checker_used:
                n_checker_used += 1

        agent_metrics["checker_use_rate"] = n_checker_used / len(results)

        # Extract prompt identifier (canonical helper — same id the
        # curator gate saw on this prompt).
        qid = resolve_prompt_id(data)

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "trajectories": trajectories,
            "agent_metrics": agent_metrics,
        }

    async def _run_one_episode(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> Optional[tuple[list[dict], float, bool]]:
        """Run one episode. Returns (sequences, reward, checker_used) or None."""
        await self._init_server_info()

        sample_id = data.get("task_id") or data.get("index")
        if sample_id is None:
            raise ValueError("Data must contain 'task_id' or 'index' field")

        # Resolve actor/checker engines and tokenizers.
        from astraflow.workflow.api.engine_api import EngineGroup

        multi_model = isinstance(engine, EngineGroup) and "model1" in engine
        if multi_model:
            actor_engine = engine["model0"]
            checker_engine = engine["model1"]
            checker_tokenizer = self._tokenizers.get("model1", self.tokenizer)
            checker_gconfig = self._gconfigs.get("model1", self.gconfig)
        else:
            actor_engine = engine
            checker_engine = engine
            checker_tokenizer = self.tokenizer
            checker_gconfig = self.gconfig

        episode_id = None

        try:
            response = await self._call_server(
                "api/episode/start",
                data={"sample_id": str(sample_id), "config": {}},
            )

            episode_id = response["episode_id"]
            observation = response["observation"]
            info = response.get("info", {})

            messages: List[Dict[str, str]] = []
            total_reward = 0.0
            done = False

            # Actor sequence tokens
            seq: List[int] = []
            logprobs: List[float] = []
            loss_mask: List[int] = []
            versions: List[int] = []

            # Checker response (if called)
            checker_resp_saved = None

            messages = self.format_observation_to_messages(observation, messages)

            input_ids: List[int] = list(self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            ))

            # Extract goal instruction for the checker.
            goal_instruction = self._extract_goal(messages)

            checker_used = False
            last_observation_text: str = ""

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

                resp = await actor_engine.agenerate(req)

                agent_output = self.tokenizer.decode(resp.output_tokens)

                # --- Accumulate actor tokens ---
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

                # --- Checker interception ---
                if (
                    _is_buy_action(action.get("content", ""))
                    and not checker_used
                ):
                    checker_resp, score, feedback = await self._run_checker(
                        checker_engine,
                        checker_tokenizer,
                        checker_gconfig,
                        goal_instruction,
                        last_observation_text,
                    )

                    # Save checker response for building its sequence later.
                    checker_resp_saved = checker_resp


                    if score < self.checker_threshold and feedback:
                        checker_used = True
                        checker_msg = (
                            f"Your estimated purchase score is {score:.1f}/1.0. "
                            f"{feedback}\n"
                            f"Please fix this before buying."
                        )

                        messages.append({"role": "assistant", "content": agent_output})

                        input_ids = list(resp.input_tokens) + list(resp.output_tokens)
                        if resp.output_tokens and resp.output_tokens[-1] != self.tokenizer.eos_token_id:
                            input_ids.append(self.tokenizer.eos_token_id)
                        new_user_tokens = self._compute_user_message_delta_tokens(checker_msg)
                        input_ids += new_user_tokens

                        messages.append({"role": "user", "content": checker_msg})

                        num_turns += 1
                        continue

                # --- Normal flow: send action to task server ---
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
                        last_observation_text = observation["content"]
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
                checker_used=float(checker_used),
            )

            # --- Build per-model sequences ---
            # Split actor prompt vs output tokens using loss_mask.
            # Actor: full multi-turn conversation as one sequence.
            actor_prompt_ids = []
            actor_output_ids = []
            actor_output_logprobs = []
            actor_output_versions = []
            for i, m in enumerate(loss_mask):
                if m == 0:
                    actor_prompt_ids.append(seq[i])
                else:
                    actor_output_ids.append(seq[i])
                    actor_output_logprobs.append(logprobs[i])
                    actor_output_versions.append(versions[i])

            actor_seq_dict = _build_seq_dict(
                input_ids=actor_prompt_ids,
                output_ids=actor_output_ids,
                output_logprobs=actor_output_logprobs,
                output_versions=actor_output_versions,
                model_id=MODEL_ID_ACTOR,
                reward=shaped_reward,
                is_first=True,
            )

            sequences = [actor_seq_dict]

            # Checker sequence (if checker was called).
            if checker_resp_saved is not None:
                checker_seq_dict = _build_seq_dict(
                    input_ids=list(checker_resp_saved.input_tokens),
                    output_ids=list(checker_resp_saved.output_tokens),
                    output_logprobs=list(checker_resp_saved.output_logprobs),
                    output_versions=list(checker_resp_saved.output_versions),
                    model_id=MODEL_ID_CHECKER,
                    reward=shaped_reward,
                    is_first=False,
                )
                sequences.append(checker_seq_dict)

            # Dump trajectory
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
                        f"checker_used={checker_used}, "
                        f"done={done}) ===\n\n"
                    )
                    await f.write(header)
                    await f.write("--- Actor ---\n")
                    await f.write(self.tokenizer.decode(seq))
                    if checker_resp_saved is not None:
                        await f.write("\n\n--- Checker ---\n")
                        await f.write(checker_tokenizer.decode(
                            list(checker_resp_saved.input_tokens)
                            + list(checker_resp_saved.output_tokens)
                        ))
                    await f.write("\n\n")

            return sequences, shaped_reward, checker_used

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

    # ------------------------------------------------------------------
    # Checker helpers
    # ------------------------------------------------------------------

    def _extract_goal(self, messages: List[Dict[str, str]]) -> str:
        """Extract the goal instruction from the conversation messages.

        Supports both observation formats:
          - text (simple):  ``Instruction: [SEP] <goal> [SEP] ...``
          - text_rich:      ``Instruction:\\n<goal>\\n[button] Search [button_]``
        """
        for msg in reversed(messages):
            if msg["role"] != "user":
                continue
            content = msg["content"]
            # text (simple) format: [SEP] delimited
            match = re.search(
                r"Instruction:\s*\[SEP\]\s*(.+?)(?:\s*\[SEP\])", content
            )
            if match:
                return match.group(1).strip()
            # text_rich format: newline delimited, ends before [button]
            match = re.search(
                r"Instruction:\s*\n\s*(.+?)(?:\s*\[button\])", content, re.DOTALL
            )
            if match:
                return match.group(1).strip()
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"][:500]
        return ""

    async def _run_checker(
        self,
        checker_engine: InferenceEngine,
        checker_tokenizer: PreTrainedTokenizerFast,
        checker_gconfig: GenerationHyperparameters,
        goal: str,
        observation: str,
    ) -> tuple[Any, float, str]:
        """Run the checker model to estimate purchase score.

        Returns (model_response, score, feedback_message).
        On failure returns (None, 1.0, "") so the purchase goes through.
        """
        prompt = CHECKER_PROMPT.format(goal=goal, observation=observation)

        checker_messages = [
            {"role": "user", "content": prompt},
        ]

        checker_input_ids = list(checker_tokenizer.apply_chat_template(
            checker_messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        ))

        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=checker_input_ids,
            gconfig=checker_gconfig.new(
                n_samples=1,
                max_new_tokens=4096,
                temperature=0.3,
                greedy=False,
            ),
            tokenizer=checker_tokenizer,
        )

        try:
            resp = await checker_engine.agenerate(req)
            checker_output = checker_tokenizer.decode(
                resp.output_tokens, skip_special_tokens=True,
            )
            score, feedback = _parse_checker_output(checker_output)
            return resp, score, feedback
        except Exception as e:
            logger.warning("Checker call failed, defaulting to APPROVE: %s", e)
            return None, 1.0, ""
