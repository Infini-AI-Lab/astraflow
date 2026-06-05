"""Unit tests for action-based DeepDiveEnv — judge + cmu_search mocked.

Run:
    pytest astraflow/core/workflow/impl/deepdive/test_env.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from astraflow.core.workflow.impl.deepdive import env as env_module
from astraflow.core.workflow.impl.deepdive.env import DeepDiveEnv
from astraflow.core.workflow.impl.deepdive.tasks import Task


def _root_task(goal: str = "who painted mona lisa", gt: str = "Leonardo da Vinci") -> Task:
    return Task(
        goal=goal,
        id="deepdive.qa_rl.0",
        misc={"ground_truth": gt, "answer": gt},
    )


def _subagent_task(goal: str = "find dates in this email") -> Task:
    return Task(goal=goal, id="deepdive.qa_rl.0/sub_abc123", misc={})


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# evaluate() routing — root vs sub-agent (judge mocked)
# --------------------------------------------------------------------------


def test_evaluate_routes_root_task_to_root_rubric():
    env = DeepDiveEnv(task=_root_task())
    env.finish("Leonardo da Vinci")
    captured = {}

    async def fake_judge(system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return '{"success": true, "reason": "Correct."}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, info = _run(env.evaluate())

    # Root rubric mentions ground truth comparison.
    assert "ground truth" in captured["system"].lower()
    assert "QUESTION:" in captured["user"]
    assert "GROUND TRUTH ANSWER:" in captured["user"]
    assert "Leonardo da Vinci" in captured["user"]
    # Binary success → score 1.0
    assert score == 1.0


def test_evaluate_routes_subagent_through_checklist_grader():
    """Sub-agent grading uses ChecklistGrader (single LLM call, ai-rubric-style)."""
    from astraEnv import checklist as cl_module

    env = DeepDiveEnv(task=_subagent_task())
    env.finish("2024-03-15, 2024-04-02")

    n_calls = {"n": 0}

    # The checklist grader uses astraEnv.judge directly (not env_module.judge).
    async def fake_judge(system, user, **kwargs):
        n_calls["n"] += 1
        # Single-call response: checklist + per-item scores + overall_score
        return (
            '{'
            '"checklist": ["finds dates", "right format", "completeness"],'
            '"checklist_scores": [1.0, 1.0, 0.5],'
            '"reasoning": "Found two dates; partially complete.",'
            '"overall_score": 0.8'
            '}'
        )

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        score, _info = _run(env.evaluate())

    assert n_calls["n"] == 1  # single call, ai-rubric semantics
    assert score == pytest.approx(0.8)


def test_evaluate_root_success_false_returns_zero():
    """Binary: success=false → score 0.0."""
    env = DeepDiveEnv(task=_root_task())
    env.finish("wrong answer")

    async def fake_judge(system, user, **kwargs):
        return '{"success": false, "reason": "Answer is incorrect."}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, _info = _run(env.evaluate())
    assert score == 0.0


def test_evaluate_temperature_passed_as_one():
    """temp=1 is required for platoon parity."""
    env = DeepDiveEnv(task=_root_task())
    env.finish("answer")
    captured = {}

    async def fake_judge(system, user, **kwargs):
        captured.update(kwargs)
        return '{"success": true, "reason": "ok"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        _run(env.evaluate())
    assert captured.get("temperature") == 1.0


def test_evaluate_legacy_continuous_score_still_parses():
    """Backward compat: judges returning {"score": float} still work."""
    env = DeepDiveEnv(task=_root_task())
    env.finish("answer")

    async def fake_judge(system, user, **kwargs):
        return '{"score": 0.7, "reason": "partial"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, _info = _run(env.evaluate())
    assert score == pytest.approx(0.7)


def test_evaluate_unfinished_returns_zero():
    env = DeepDiveEnv(task=_root_task())  # not finished

    async def boom(system, user, **kwargs):
        raise AssertionError("judge must not be called for unfinished agent")

    with patch.object(env_module, "judge", side_effect=boom):
        score, info = _run(env.evaluate())
    assert score == 0.0
    assert "never called finish" in info["reason"]


def test_evaluate_judge_failure_returns_zero():
    env = DeepDiveEnv(task=_root_task())
    env.finish("anything")

    async def boom(system, user, **kwargs):
        raise RuntimeError("fireworks 503")

    with patch.object(env_module, "judge", side_effect=boom):
        score, info = _run(env.evaluate())
    assert score == 0.0
    assert "judge call failed" in info["reason"]


def test_evaluate_clamps_out_of_range_score():
    env = DeepDiveEnv(task=_root_task())
    env.finish("answer")

    async def hi(system, user, **kwargs):
        return '{"score": 1.7, "reason": "over"}'

    with patch.object(env_module, "judge", side_effect=hi):
        score, _info = _run(env.evaluate())
    assert score == 1.0

    async def lo(system, user, **kwargs):
        return '{"score": -0.3, "reason": "below"}'

    env2 = DeepDiveEnv(task=_root_task())
    env2.finish("answer")
    with patch.object(env_module, "judge", side_effect=lo):
        score, _info = _run(env2.evaluate())
    assert score == 0.0


def test_evaluate_unparseable_response_returns_zero():
    env = DeepDiveEnv(task=_root_task())
    env.finish("answer")

    async def fake_judge(system, user, **kwargs):
        return "definitely not json"

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, info = _run(env.evaluate())
    assert score == 0.0
    assert "unparseable" in info["reason"]
    assert info["judge_raw"] == "definitely not json"


def test_evaluate_judge_model_override_forwarded():
    env = DeepDiveEnv(
        task=_root_task(),
        judge_model="accounts/fireworks/models/deepseek-v4-pro",
    )
    env.finish("answer")
    captured = {}

    async def fake_judge(system, user, **kwargs):
        captured.update(kwargs)
        return '{"score": 0.5, "reason": "ok"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        _run(env.evaluate())

    assert captured.get("model") == "accounts/fireworks/models/deepseek-v4-pro"


# --------------------------------------------------------------------------
# search() action (cmu_search mocked)
# --------------------------------------------------------------------------


def test_search_action_calls_cmu_search_and_counts():
    env = DeepDiveEnv(task=_root_task())
    captured = {}

    async def fake_search(query, n_docs=5, **kwargs):
        captured["query"] = query
        captured["n_docs"] = n_docs
        return [
            {"text": "leonardo painted it in 1503", "source": "wiki", "score": 0.9},
            {"text": "louvre", "source": "c4", "score": 0.85},
        ]

    with patch.object(env_module, "cmu_search", side_effect=fake_search):
        obs = _run(env.search("who painted mona lisa", n_docs=2))

    assert captured["query"] == "who painted mona lisa"
    assert captured["n_docs"] == 2
    assert env.search_calls == 1
    # Observation is formatted text, includes query + numbered passages + sources.
    assert "Search results for" in obs
    assert "leonardo painted it in 1503" in obs
    assert "[1]" in obs and "[2]" in obs
    assert "wiki" in obs
    assert "0.90" in obs


def test_search_action_clamps_n_docs():
    env = DeepDiveEnv(task=_root_task())
    captured = {}

    async def fake_search(query, n_docs=5, **kwargs):
        captured["n_docs"] = n_docs
        return []

    with patch.object(env_module, "cmu_search", side_effect=fake_search):
        _run(env.search("q", n_docs=999))
    assert captured["n_docs"] == 20  # MAX_SEARCH_N_DOCS

    with patch.object(env_module, "cmu_search", side_effect=fake_search):
        _run(env.search("q", n_docs=0))
    assert captured["n_docs"] == 1


def test_search_action_empty_query_returns_error_not_raises():
    env = DeepDiveEnv(task=_root_task())

    async def boom(*args, **kwargs):
        raise AssertionError("cmu_search must not be called for empty query")

    with patch.object(env_module, "cmu_search", side_effect=boom):
        obs = _run(env.search("   ", n_docs=5))
    assert "ERROR" in obs
    assert env.search_calls == 0


def test_search_action_network_failure_returns_error_text():
    env = DeepDiveEnv(task=_root_task())

    async def boom(*args, **kwargs):
        raise ConnectionError("network down")

    with patch.object(env_module, "cmu_search", side_effect=boom):
        obs = _run(env.search("anything"))
    assert "ERROR" in obs
    assert "search failed" in obs
    assert env.search_calls == 1  # counter incremented before the call


def test_search_action_no_results_returns_friendly_text():
    env = DeepDiveEnv(task=_root_task())

    async def fake_search(*args, **kwargs):
        return []

    with patch.object(env_module, "cmu_search", side_effect=fake_search):
        obs = _run(env.search("obscure query"))
    assert "no results" in obs


def test_search_action_truncates_long_passages():
    env = DeepDiveEnv(task=_root_task(), passage_truncate=50)
    long_text = "x" * 1000

    async def fake_search(*args, **kwargs):
        return [{"text": long_text, "source": "x", "score": 1.0}]

    with patch.object(env_module, "cmu_search", side_effect=fake_search):
        obs = _run(env.search("q"))
    # Long passage should be truncated; full length noted.
    assert "truncated" in obs
    assert "total 1000 chars" in obs


# --------------------------------------------------------------------------
# spawn() action
# --------------------------------------------------------------------------


def test_spawn_calls_callback_and_counts():
    captured = {}

    async def fake_spawn(goal: str) -> str:
        captured["goal"] = goal
        return "the answer from the sub"

    env = DeepDiveEnv(task=_root_task(), spawn_callback=fake_spawn)
    obs = _run(env.spawn("find the birth year"))
    assert obs == "the answer from the sub"
    assert env.subagent_launched == 1
    assert captured["goal"] == "find the birth year"


def test_spawn_without_callback_returns_error():
    env = DeepDiveEnv(task=_root_task(), spawn_callback=None)
    obs = _run(env.spawn("anything"))
    assert "ERROR" in obs


def test_spawn_with_empty_goal_returns_error_without_launching():
    async def boom(goal):
        raise AssertionError("spawn callback must not be invoked for empty goal")

    env = DeepDiveEnv(task=_root_task(), spawn_callback=boom)
    obs = _run(env.spawn("   "))
    assert "ERROR" in obs
    assert env.subagent_launched == 0


# --------------------------------------------------------------------------
# finish() action (sync)
# --------------------------------------------------------------------------


def test_finish_sets_state_and_is_sync():
    env = DeepDiveEnv(task=_root_task())
    assert not env.finished
    env.finish("the answer")
    assert env.finished
    assert env.finish_payload == "the answer"
