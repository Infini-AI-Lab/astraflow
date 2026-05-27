"""Unit tests for OolongRecursiveWorkflow reward_mode dispatch.

Focuses on the per-agent reward selection logic. The full rollout is
not exercised — that requires a tokenizer, an inference engine, and a
running RaaS, which belongs in integration tests.

Run:
    pytest astraflow/core/workflow/impl/oolong/test_workflow.py -v
"""

from __future__ import annotations

import pytest

from astraflow.core.workflow.impl.oolong.tasks import Task
from astraflow.core.workflow.impl.oolong.workflow import (
    AgentTrajectory,
    OolongRecursiveWorkflow,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _wf(mode: str | None = None) -> OolongRecursiveWorkflow:
    """Build a minimal workflow; pass mode=None to test the default."""
    kwargs: dict = {}
    if mode is not None:
        kwargs["reward_mode"] = mode
    return OolongRecursiveWorkflow(**kwargs)


def _agent(
    traj_id: str = "root",
    parent_id: str | None = None,
    depth: int = 0,
    is_root: bool = True,
    reward: float = 0.0,
    goal: str = "g",
    task_id: str = "oolong.synth.validation.0",
) -> AgentTrajectory:
    return AgentTrajectory(
        traj_id=traj_id,
        parent_id=parent_id,
        depth=depth,
        task=Task(goal=goal, id=task_id),
        is_root=is_root,
        reward=reward,
    )


# --------------------------------------------------------------------------
# reward_mode validation
# --------------------------------------------------------------------------


def test_default_reward_mode_is_team_credit():
    wf = _wf()  # no reward_mode kwarg
    assert wf.reward_mode == "team_credit"
    assert wf.use_llm_judge is False


def test_team_credit_does_not_enable_judge():
    wf = _wf("team_credit")
    assert wf.use_llm_judge is False


def test_per_agent_judge_enables_judge():
    wf = _wf("per_agent_judge")
    assert wf.use_llm_judge is True


def test_unknown_reward_mode_raises():
    with pytest.raises(ValueError, match="reward_mode"):
        _wf("nonsense")


def test_root_only_no_longer_supported():
    """root_only was dropped — must raise to surface stale configs."""
    with pytest.raises(ValueError, match="reward_mode"):
        _wf("root_only")


# --------------------------------------------------------------------------
# _reward_for_agent — the per-agent selection logic
# --------------------------------------------------------------------------


def test_team_credit_uses_root_reward_for_root():
    wf = _wf("team_credit")
    root = _agent(is_root=True, reward=0.9)
    assert wf._reward_for_agent(root, root_reward=0.9) == 0.9


def test_team_credit_broadcasts_root_reward_to_sub_agents():
    """Sub-agent's own reward is ignored; everyone uses root_reward."""
    wf = _wf("team_credit")
    sub = _agent(traj_id="sub1", parent_id="root", depth=1, is_root=False, reward=0.2)
    assert wf._reward_for_agent(sub, root_reward=0.9) == 0.9


def test_per_agent_judge_uses_own_reward_for_root():
    wf = _wf("per_agent_judge")
    root = _agent(is_root=True, reward=1.0)
    assert wf._reward_for_agent(root, root_reward=1.0) == 1.0


def test_per_agent_judge_uses_own_reward_for_sub_agents():
    """Each sub gets its own LLM-judge score, independent of root."""
    wf = _wf("per_agent_judge")
    sub = _agent(traj_id="sub1", is_root=False, reward=0.7)
    # root_reward differs from sub.reward; should pick sub.reward.
    assert wf._reward_for_agent(sub, root_reward=0.0) == 0.7


def test_per_agent_judge_handles_none_reward_defensively():
    """A sub-agent whose reward was never set (None) defaults to 0.0,
    not a crash."""
    wf = _wf("per_agent_judge")
    sub = _agent(traj_id="sub1", is_root=False)
    sub.reward = None  # type: ignore[assignment]
    assert wf._reward_for_agent(sub, root_reward=0.5) == 0.0


# --------------------------------------------------------------------------
# end-to-end matrix on a small tree
# --------------------------------------------------------------------------


def test_team_credit_full_tree():
    """Root (reward=0.8) + 2 subs (rewards=0.1, 0.2). All three should
    receive 0.8."""
    wf = _wf("team_credit")
    root = _agent(traj_id="r", is_root=True, reward=0.8)
    sub_a = _agent(traj_id="a", parent_id="r", depth=1, is_root=False, reward=0.1)
    sub_b = _agent(traj_id="b", parent_id="r", depth=1, is_root=False, reward=0.2)
    rewards = [wf._reward_for_agent(ag, root_reward=0.8) for ag in (root, sub_a, sub_b)]
    assert rewards == [0.8, 0.8, 0.8]


def test_per_agent_judge_full_tree():
    """Same tree under per_agent_judge — each agent keeps its own reward."""
    wf = _wf("per_agent_judge")
    root = _agent(traj_id="r", is_root=True, reward=0.8)
    sub_a = _agent(traj_id="a", parent_id="r", depth=1, is_root=False, reward=0.1)
    sub_b = _agent(traj_id="b", parent_id="r", depth=1, is_root=False, reward=0.2)
    rewards = [wf._reward_for_agent(ag, root_reward=0.8) for ag in (root, sub_a, sub_b)]
    assert rewards == [0.8, 0.1, 0.2]
