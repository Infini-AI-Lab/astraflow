"""Router: three-model cost-efficient routing workflow.

**model0** (router) reads a math problem and decides whether to route it
to the small solver or the large solver.

**model1** (small solver, e.g. Qwen3-1.7B) attempts to solve the problem.

**model2** (large solver, e.g. Qwen3-8B) attempts to solve the problem.

Training vs inference:

- **Training**: All three models generate in parallel.  Both solvers
  attempt the problem so we can compute the router's reward based on
  whether it picked the *cheapest correct* model.
- **Inference** (``eval_mode=True``): Only the router generates first,
  then only the selected solver generates.  Single pass, no comparison.

Router reward logic:
- Router picked a correct model AND no cheaper correct model exists → 1.0
- Router picked wrong or wastefully (small could have done it) → 0.0
- Neither model can solve it → 0.5

Solver rewards are independent: 1.0 if correct, 0.0 if wrong.

Returns the ASearcher-style structured format::

    {
        "n_trajs": int,
        "rewards": Tensor[n_trajs],
        "trajectories": [{"sequences": [seq1, seq2, ...]}, ...],
    }

Usage in YAML config::

    workflow_spec:
      workflow_cls: router
      reward_fn: "math_verify"
      eval_mode: false
"""

import asyncio
import os
import random
import re
import uuid
from collections.abc import Callable
from typing import Any

import aiofiles
import aiofiles.os
import torch
from transformers import PreTrainedTokenizerFast

from astraflow.core.workflow.api.cli_args import GenerationHyperparameters
from astraflow.core.workflow.api.engine_api import EngineGroup, InferenceEngine
from astraflow.core.workflow.api.io_struct import ModelRequest
from astraflow.core.workflow.api.reward_api import AsyncRewardWrapper
from astraflow.core.workflow.api.workflow_api import RolloutWorkflow
from astraflow.core.workflow.registry import register_workflow
from astraflow.core.workflow.utils import logging, stats_tracker
from astraflow.core.workflow.utils.data import resolve_prompt_id

logger = logging.getLogger(__name__)

MODEL_ID_PROMPT = -1
MODEL_ID_ROUTER = 0    # model0
MODEL_ID_SMALL = 1     # model1
MODEL_ID_LARGE = 2     # model2

ROUTER_SYSTEM_PROMPT = (
    "You are a routing assistant. Given a math problem, decide whether it "
    "should be solved by the SMALL model (fast, cheap) or the LARGE model "
    "(slower, more capable).\n\n"
    "Think step by step about the problem's difficulty, then make your decision. "
    "End your response with your final answer on a new line: SMALL or LARGE."
)

SOLVER_PROMPT_SUFFIX = (
    "\nLet's think step by step. Please put your final answer within \\boxed{}."
)


def _parse_route_decision(text: str) -> str | None:
    """Parse the router's output into 'small' or 'large'.

    Takes the **last** occurrence of SMALL or LARGE in the text,
    since the router may mention both during chain-of-thought reasoning
    before giving its final answer.

    Returns None if the output cannot be parsed.
    """
    # Find all occurrences of SMALL or LARGE
    matches = list(re.finditer(r"\b(SMALL|LARGE)\b", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).lower()
    return None


def _build_seq_dict(
    input_ids: list[int],
    output_ids: list[int],
    output_logprobs: list[float],
    output_versions: list[int],
    model_id: int,
    reward: float,
    is_first: bool,
) -> dict[str, Any]:
    """Build a self-contained sequence tensor dict with per-sequence reward."""
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


def _compute_router_reward(
    decision: str,
    small_correct: bool,
    large_correct: bool,
) -> float:
    """Compute router reward: prefer the cheapest correct model.

    - Picked correct model, no cheaper correct model exists → 1.0
    - Picked wrong or wastefully → 0.0
    - Neither model can solve it → 0.5
    """
    if not small_correct and not large_correct:
        return 0.5

    if decision == "small":
        return 1.0 if small_correct else 0.0
    else:  # large
        # If small could also solve it, routing to large is wasteful
        if small_correct:
            return 0.0
        return 1.0 if large_correct else 0.0


def import_from_string(dotted_path: str) -> Any:
    """Import a callable from a dotted module path."""
    module_path, _, attr = dotted_path.rpartition(".")
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


@register_workflow("sm_lg_router")
class SmLgRouterWorkflow(RolloutWorkflow):
    """Three-model cost-efficient routing workflow.

    Parameters
    ----------
    reward_fn : callable or str
        Reward function for checking solver correctness.
    gconfig : GenerationHyperparameters
        Default generation config.
    tokenizer : str or PreTrainedTokenizerFast
        Tokenizer (shared by all models).
    eval_mode : bool
        If True, only the selected solver generates (inference mode).
        If False, both solvers generate (training mode).
    enable_thinking : bool
        Whether to enable thinking tokens in chat template.
    rollout_stat_scope : str
        Scope name for stats tracking.
    dump_dir : str | None
        If set, dump trajectories for debugging.
    gconfigs : dict[str, GenerationHyperparameters] | None
        Per-model generation configs from RaaS.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        eval_mode: bool = False,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        gconfigs: dict[str, GenerationHyperparameters] | None = None,
    ):
        self.reward_fn = reward_fn
        if isinstance(tokenizer, str):
            from astraflow.core.workflow.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        # Per-model gconfigs (model0=router, model1=small, model2=large)
        if gconfigs is not None:
            self.router_gconfig = gconfigs.get(
                "model0", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
            self.small_gconfig = gconfigs.get(
                "model1", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
            self.large_gconfig = gconfigs.get(
                "model2", gconfig
            ).new_with_stop_and_pad_token_ids(self.tokenizer)
        else:
            self.router_gconfig = self.gconfig
            self.small_gconfig = self.gconfig
            self.large_gconfig = self.gconfig

        self.eval_mode = eval_mode
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    def _apply_chat_template(self, messages, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        try:
            return list(self.tokenizer.apply_chat_template(
                messages, **kwargs, enable_thinking=self.enable_thinking,
            ))
        except TypeError:
            return list(self.tokenizer.apply_chat_template(messages, **kwargs))

    async def _generate(
        self,
        engine: InferenceEngine,
        input_ids: list[int],
        gconfig: GenerationHyperparameters,
    ) -> tuple[str, list[int], list[float], list[int]]:
        """Generate from an engine. Returns (text, output_ids, logprobs, versions)."""
        resp = await engine.agenerate(
            ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
        )
        output_ids = list(resp.output_tokens)
        logprobs = list(resp.output_logprobs)
        versions = (
            list(resp.output_versions)
            if hasattr(resp, "output_versions") and resp.output_versions
            else [engine.get_version()] * len(output_ids)
        )
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return text, output_ids, logprobs, versions

    async def _route_and_solve_train(
        self,
        engine_router: InferenceEngine,
        engine_small: InferenceEngine,
        engine_large: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> tuple[list[dict], float, dict] | None:
        """Training mode: all three models generate in parallel."""
        # Build prompts
        problem_text = messages[-1]["content"] if messages else ""

        # Router prompt
        router_messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]
        router_input_ids = self._apply_chat_template(
            router_messages, add_generation_prompt=True,
        )

        # Solver prompt (same for both, with solve instruction suffix)
        solver_messages = [
            {**m, "content": m["content"] + SOLVER_PROMPT_SUFFIX}
            if m["role"] == "user" else m
            for m in messages
        ]
        solver_input_ids = self._apply_chat_template(
            solver_messages, add_generation_prompt=True,
        )

        # All three generate in parallel
        router_coro = self._generate(
            engine_router, router_input_ids, self.router_gconfig,
        )
        small_coro = self._generate(
            engine_small, solver_input_ids, self.small_gconfig,
        )
        large_coro = self._generate(
            engine_large, solver_input_ids, self.large_gconfig,
        )

        (router_text, router_out_ids, router_logprobs, router_versions), \
        (small_text, small_out_ids, small_logprobs, small_versions), \
        (large_text, large_out_ids, large_logprobs, large_versions) = \
            await asyncio.gather(router_coro, small_coro, large_coro)

        # Parse router decision
        decision = _parse_route_decision(router_text)
        parse_failed = decision is None
        if parse_failed:
            decision = "large"  # fallback for solver selection

        # Compute solver rewards
        prompt_str = self.tokenizer.decode(solver_input_ids)
        small_reward_val, large_reward_val = await asyncio.gather(
            self.async_reward_fn(
                prompt_str, small_text,
                solver_input_ids, small_out_ids,
                **task_data,
            ),
            self.async_reward_fn(
                prompt_str, large_text,
                solver_input_ids, large_out_ids,
                **task_data,
            ),
        )
        small_reward_val = float(small_reward_val)
        large_reward_val = float(large_reward_val)

        small_correct = small_reward_val > 0.5
        large_correct = large_reward_val > 0.5

        # Compute router reward
        if parse_failed:
            router_reward = -0.5
        else:
            router_reward = _compute_router_reward(
                decision, small_correct, large_correct,
            )

        # Build sequences for all three models
        sequences = []

        # Router sequence
        sequences.append(_build_seq_dict(
            input_ids=router_input_ids,
            output_ids=router_out_ids,
            output_logprobs=router_logprobs,
            output_versions=router_versions,
            model_id=MODEL_ID_ROUTER,
            reward=router_reward,
            is_first=True,
        ))

        # Small solver sequence
        sequences.append(_build_seq_dict(
            input_ids=solver_input_ids,
            output_ids=small_out_ids,
            output_logprobs=small_logprobs,
            output_versions=small_versions,
            model_id=MODEL_ID_SMALL,
            reward=small_reward_val,
            is_first=False,
        ))

        # Large solver sequence
        sequences.append(_build_seq_dict(
            input_ids=solver_input_ids,
            output_ids=large_out_ids,
            output_logprobs=large_logprobs,
            output_versions=large_versions,
            model_id=MODEL_ID_LARGE,
            reward=large_reward_val,
            is_first=False,
        ))

        # Trajectory reward = whether the selected model solved the problem.
        # This is what filter_zero_adv sees and what rollout/pre_filter_reward_mean reports.
        if decision == "small":
            final_reward = small_reward_val
        else:
            final_reward = large_reward_val

        debug_info = {
            "problem": problem_text,
            "decision": decision,
            "router_prompt": self.tokenizer.decode(router_input_ids),
            "router_text": router_text,
            "router_reward": router_reward,
            "solver_prompt": prompt_str,
            "small_text": small_text,
            "small_reward": small_reward_val,
            "large_text": large_text,
            "large_reward": large_reward_val,
            "small_correct": small_correct,
            "large_correct": large_correct,
        }

        stats_tracker.get(self.rollout_stat_scope).scalar(
            router_reward=router_reward,
            small_reward=small_reward_val,
            large_reward=large_reward_val,
            routed_to_small=float(decision == "small"),
        )

        return sequences, final_reward, debug_info

    async def _route_and_solve_eval(
        self,
        engine_router: InferenceEngine,
        engine_small: InferenceEngine,
        engine_large: InferenceEngine,
        messages: list[dict],
        task_data: dict[str, Any],
    ) -> tuple[list[dict], float, dict] | None:
        """Eval mode: router decides, then only the selected solver generates."""
        problem_text = messages[-1]["content"] if messages else ""

        # Router generates first
        router_messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]
        router_input_ids = self._apply_chat_template(
            router_messages, add_generation_prompt=True,
        )
        router_text, router_out_ids, router_logprobs, router_versions = \
            await self._generate(
                engine_router, router_input_ids, self.router_gconfig,
            )

        # Parse decision
        decision = _parse_route_decision(router_text)
        parse_failed = decision is None
        if parse_failed:
            decision = "large"  # fallback

        # Only selected solver generates (with solve instruction suffix)
        solver_messages = [
            {**m, "content": m["content"] + SOLVER_PROMPT_SUFFIX}
            if m["role"] == "user" else m
            for m in messages
        ]
        solver_input_ids = self._apply_chat_template(
            solver_messages, add_generation_prompt=True,
        )
        if decision == "small":
            engine_solver = engine_small
            solver_gconfig = self.small_gconfig
            solver_model_id = MODEL_ID_SMALL
        else:
            engine_solver = engine_large
            solver_gconfig = self.large_gconfig
            solver_model_id = MODEL_ID_LARGE

        solver_text, solver_out_ids, solver_logprobs, solver_versions = \
            await self._generate(
                engine_solver, solver_input_ids, solver_gconfig,
            )

        # Compute reward
        prompt_str = self.tokenizer.decode(solver_input_ids)
        solver_reward = await self.async_reward_fn(
            prompt_str, solver_text,
            solver_input_ids, solver_out_ids,
            **task_data,
        )
        solver_reward = float(solver_reward)

        # Build sequences (router + selected solver only)
        sequences = []

        sequences.append(_build_seq_dict(
            input_ids=router_input_ids,
            output_ids=router_out_ids,
            output_logprobs=router_logprobs,
            output_versions=router_versions,
            model_id=MODEL_ID_ROUTER,
            reward=-0.5 if parse_failed else solver_reward,
            is_first=True,
        ))

        sequences.append(_build_seq_dict(
            input_ids=solver_input_ids,
            output_ids=solver_out_ids,
            output_logprobs=solver_logprobs,
            output_versions=solver_versions,
            model_id=solver_model_id,
            reward=solver_reward,
            is_first=False,
        ))

        final_reward = solver_reward

        debug_info = {
            "problem": problem_text,
            "decision": decision,
            "router_text": router_text,
            "solver_text": solver_text[:200],
            "solver_reward": solver_reward,
            "eval_mode": True,
        }

        return sequences, final_reward, debug_info

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        # Resolve reward function if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # Resolve engines
        if isinstance(engine, EngineGroup):
            engine_router = engine["model0"]
            engine_small = engine["model1"]
            engine_large = engine["model2"]
        else:
            engine_router = engine_small = engine_large = engine

        messages = data["messages"]
        n_samples = self.gconfig.n_samples
        version = engine_router.get_version()

        # Pick training vs eval path
        if self.eval_mode:
            solve_fn = self._route_and_solve_eval
        else:
            solve_fn = self._route_and_solve_train

        # Generate n_samples in parallel
        sample_coros = [
            solve_fn(engine_router, engine_small, engine_large, messages, data)
            for _ in range(n_samples)
        ]
        raw_results = await asyncio.gather(*sample_coros)

        # Collect successful results
        trajectories = []
        trajectory_infos = []
        rewards = []
        for r in raw_results:
            if r is not None:
                sequences, reward, info = r
                trajectories.append({"sequences": sequences})
                trajectory_infos.append(info)
                rewards.append(reward)

        # Extract prompt identifier (canonical helper — same id the
        # curator gate saw on this prompt).
        qid = resolve_prompt_id(data)

        # Debug dump
        if self.dump_dir is not None and random.random() < 1 / 32:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)

            dump_qid = qid or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{dump_qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (info, rew) in enumerate(
                    zip(trajectory_infos, rewards)
                ):
                    await f.write(
                        f"=== Sample {i + 1}/{n_samples} "
                        f"(final_reward={rew}) ===\n\n"
                    )

                    # Router
                    await f.write(
                        f"--- Router (decision={info['decision']}, "
                        f"reward={info.get('router_reward', 'N/A')}) ---\n"
                        f"--- Router Prompt ---\n"
                        f"{info.get('router_prompt', '')}\n\n"
                        f"--- Router Output ---\n"
                        f"{info['router_text']}\n\n"
                    )

                    if not info.get("eval_mode"):
                        # Small solver
                        await f.write(
                            f"--- Small Solver (correct={info['small_correct']}, "
                            f"reward={info['small_reward']}) ---\n"
                            f"--- Solver Prompt ---\n"
                            f"{info.get('solver_prompt', '')}\n\n"
                            f"--- Small Output ---\n"
                            f"{info['small_text']}\n\n"
                        )

                        # Large solver
                        await f.write(
                            f"--- Large Solver (correct={info['large_correct']}, "
                            f"reward={info['large_reward']}) ---\n"
                            f"--- Large Output ---\n"
                            f"{info['large_text']}\n\n"
                        )
                    else:
                        # Eval mode: only selected solver
                        await f.write(
                            f"--- Selected Solver "
                            f"(reward={info['solver_reward']}) ---\n"
                            f"--- Solver Output ---\n"
                            f"{info.get('solver_text', '')}\n\n"
                        )

                    await f.write("\n")

        # Compute agent-scoped metrics
        agent_metrics: dict[str, float] = {}
        if trajectory_infos and not self.eval_mode:
            total_small_route = sum(
                1 for info in trajectory_infos if info["decision"] == "small"
            )
            agent_metrics["small_route_rate"] = (
                total_small_route / len(trajectory_infos)
            )
            agent_metrics["router_reward_mean"] = (
                sum(info["router_reward"] for info in trajectory_infos)
                / len(trajectory_infos)
            )
            agent_metrics["small_correct_rate"] = (
                sum(1 for info in trajectory_infos if info["small_correct"])
                / len(trajectory_infos)
            )
            agent_metrics["large_correct_rate"] = (
                sum(1 for info in trajectory_infos if info["large_correct"])
                / len(trajectory_infos)
            )

        return {
            "prompt_id": qid,
            "n_trajs": len(trajectories),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "trajectories": trajectories,
            "agent_metrics": agent_metrics,
        }
