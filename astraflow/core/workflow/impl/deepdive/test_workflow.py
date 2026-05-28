"""Unit tests for DeepDiveRecursiveWorkflow reward_mode dispatch.

Run:
    pytest astraflow/core/workflow/impl/deepdive/test_workflow.py -v
"""

from __future__ import annotations

import pytest

from astraflow.core.workflow.impl.deepdive.tasks import Task
from astraflow.core.workflow.impl.deepdive.workflow import (
    AgentTrajectory,
    DeepDiveRecursiveWorkflow,
)


def _wf(mode: str | None = None) -> DeepDiveRecursiveWorkflow:
    kwargs: dict = {}
    if mode is not None:
        kwargs["reward_mode"] = mode
    return DeepDiveRecursiveWorkflow(**kwargs)


def _agent(
    traj_id: str = "root",
    parent_id: str | None = None,
    depth: int = 0,
    is_root: bool = True,
    reward: float = 0.0,
    goal: str = "g",
    task_id: str = "deepdive.qa_rl.0",
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
    wf = _wf()
    assert wf.reward_mode == "team_credit"


def test_per_agent_judge_accepted():
    wf = _wf("per_agent_judge")
    assert wf.reward_mode == "per_agent_judge"


def test_unknown_reward_mode_raises():
    with pytest.raises(ValueError, match="reward_mode"):
        _wf("nonsense")


def test_root_only_no_longer_supported():
    with pytest.raises(ValueError, match="reward_mode"):
        _wf("root_only")


# --------------------------------------------------------------------------
# delegation_lambda default matches platoon (0.0)
# --------------------------------------------------------------------------


def test_default_delegation_lambda_is_zero():
    wf = _wf()
    assert wf.delegation_lambda == 0.0


# --------------------------------------------------------------------------
# _reward_for_agent
# --------------------------------------------------------------------------


def test_team_credit_broadcasts_root_to_subs():
    wf = _wf("team_credit")
    sub = _agent(is_root=False, reward=0.1)
    assert wf._reward_for_agent(sub, root_reward=0.9) == 0.9


def test_per_agent_judge_uses_own_reward():
    wf = _wf("per_agent_judge")
    sub = _agent(is_root=False, reward=0.7)
    assert wf._reward_for_agent(sub, root_reward=0.0) == 0.7


def test_per_agent_judge_handles_none_reward():
    wf = _wf("per_agent_judge")
    sub = _agent(is_root=False)
    sub.reward = None  # type: ignore[assignment]
    assert wf._reward_for_agent(sub, root_reward=0.5) == 0.0


# --------------------------------------------------------------------------
# full-tree matrix
# --------------------------------------------------------------------------


def test_team_credit_full_tree():
    wf = _wf("team_credit")
    root = _agent(traj_id="r", is_root=True, reward=0.8)
    sub_a = _agent(traj_id="a", parent_id="r", depth=1, is_root=False, reward=0.1)
    sub_b = _agent(traj_id="b", parent_id="r", depth=1, is_root=False, reward=0.2)
    rewards = [wf._reward_for_agent(ag, root_reward=0.8) for ag in (root, sub_a, sub_b)]
    assert rewards == [0.8, 0.8, 0.8]


def test_per_agent_judge_full_tree():
    wf = _wf("per_agent_judge")
    root = _agent(traj_id="r", is_root=True, reward=0.8)
    sub_a = _agent(traj_id="a", parent_id="r", depth=1, is_root=False, reward=0.1)
    sub_b = _agent(traj_id="b", parent_id="r", depth=1, is_root=False, reward=0.2)
    rewards = [wf._reward_for_agent(ag, root_reward=0.8) for ag in (root, sub_a, sub_b)]
    assert rewards == [0.8, 0.1, 0.2]
