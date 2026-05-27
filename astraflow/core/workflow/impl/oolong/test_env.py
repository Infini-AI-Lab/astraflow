"""Unit tests for OolongEnv.evaluate(), focused on the LLM-judge integration.

The LLM judge call is mocked — these tests do not hit Fireworks.

Run:
    pytest astraflow/core/workflow/impl/oolong/test_env.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from astraflow.core.workflow.impl.oolong import env as env_module
from astraflow.core.workflow.impl.oolong.env import OolongEnv
from astraflow.core.workflow.impl.oolong.tasks import Task


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _subagent_task(goal: str = "find the capital of France") -> Task:
    """Sub-agent task ids do NOT start with 'oolong.' — that's how the env
    routes to the LLM-judge path."""
    return Task(goal=goal, id="parent/sub_abc123", misc={"context": ""})


def _synth_task() -> Task:
    """A real oolong-synth task id — should always go through rule-based grader,
    regardless of the use_llm_judge flag."""
    return Task(
        goal="count items",
        id="oolong.synth.validation.0",
        misc={
            "context": "",
            "answer": "[1]",
            "answer_type": "ANSWER_TYPE.NUMERIC",
        },
    )


def _finished(env: OolongEnv, payload: str) -> OolongEnv:
    env.finished = True
    env.finish_payload = payload
    return env


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# sub-agent path
# --------------------------------------------------------------------------


def test_subagent_judge_disabled_returns_zero():
    """Default (use_llm_judge=False) preserves the legacy placeholder behavior."""
    env = _finished(OolongEnv(task=_subagent_task()), "Paris")
    score, info = _run(env.evaluate())
    assert score == 0.0
    assert "disabled" in info["reason"]


def test_subagent_judge_enabled_parses_score():
    """With the judge enabled and a sensible JSON response, the score flows
    through and reason is preserved."""
    env = _finished(
        OolongEnv(task=_subagent_task(), use_llm_judge=True), "Paris"
    )

    async def fake_judge(system, user, **kwargs):
        # Smoke-check: the prompt actually contains the goal and the output.
        assert "find the capital of France" in user
        assert "Paris" in user
        return '{"score": 0.9, "reason": "Correct capital of France."}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, info = _run(env.evaluate())

    assert score == pytest.approx(0.9)
    assert "Correct" in info["reason"]
    assert "judge_raw" in info


def test_subagent_judge_clamps_out_of_range():
    """Defensive clamping: if the model returns 1.5 or -0.3, we clamp to [0,1]."""
    env = _finished(
        OolongEnv(task=_subagent_task(), use_llm_judge=True), "answer"
    )

    async def fake_judge(system, user, **kwargs):
        return '{"score": 1.5, "reason": "over the top"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, _info = _run(env.evaluate())
    assert score == 1.0

    async def fake_judge_neg(system, user, **kwargs):
        return '{"score": -0.3, "reason": "negative"}'

    with patch.object(env_module, "judge", side_effect=fake_judge_neg):
        score, _info = _run(env.evaluate())
    assert score == 0.0


def test_subagent_judge_network_failure_returns_zero():
    """A flaky judge call must not crash the rollout — return 0.0 with reason."""
    env = _finished(
        OolongEnv(task=_subagent_task(), use_llm_judge=True), "answer"
    )

    async def boom(system, user, **kwargs):
        raise RuntimeError("network down")

    with patch.object(env_module, "judge", side_effect=boom):
        score, info = _run(env.evaluate())

    assert score == 0.0
    assert "judge call failed" in info["reason"]
    assert "network down" in info["reason"]


def test_subagent_judge_unparseable_response_returns_zero():
    """If the model returns garbage, return 0.0 and keep the raw text for audit."""
    env = _finished(
        OolongEnv(task=_subagent_task(), use_llm_judge=True), "answer"
    )

    async def fake_judge(system, user, **kwargs):
        return "this is not json at all"

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, info = _run(env.evaluate())

    assert score == 0.0
    assert "unparseable" in info["reason"]
    assert info["judge_raw"] == "this is not json at all"


def test_subagent_judge_missing_score_field_returns_zero():
    """Malformed JSON (no score key) → 0.0 with a parse-failure reason."""
    env = _finished(
        OolongEnv(task=_subagent_task(), use_llm_judge=True), "answer"
    )

    async def fake_judge(system, user, **kwargs):
        return '{"reason": "I forgot the score field"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        score, info = _run(env.evaluate())

    assert score == 0.0
    assert "unparseable" in info["reason"]


def test_subagent_judge_passes_custom_model():
    """When judge_model is set on the env, it is forwarded to judge() as `model=`."""
    env = _finished(
        OolongEnv(
            task=_subagent_task(),
            use_llm_judge=True,
            judge_model="accounts/fireworks/models/deepseek-v4-pro",
        ),
        "answer",
    )

    captured = {}

    async def fake_judge(system, user, **kwargs):
        captured.update(kwargs)
        return '{"score": 0.5, "reason": "ok"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        _run(env.evaluate())

    assert captured.get("model") == "accounts/fireworks/models/deepseek-v4-pro"


def test_subagent_judge_no_model_override_passes_none():
    """When judge_model is None, no `model=` kwarg is forwarded (judge uses its default)."""
    env = _finished(
        OolongEnv(task=_subagent_task(), use_llm_judge=True), "answer"
    )

    captured = {}

    async def fake_judge(system, user, **kwargs):
        captured.update(kwargs)
        return '{"score": 0.5, "reason": "ok"}'

    with patch.object(env_module, "judge", side_effect=fake_judge):
        _run(env.evaluate())

    assert "model" not in captured


# --------------------------------------------------------------------------
# non-sub-agent paths — judge flag must be irrelevant
# --------------------------------------------------------------------------


def test_synth_task_ignores_judge_flag():
    """A real oolong-synth task must use the rule-based grader even when
    use_llm_judge is True — the judge MUST NOT be called."""
    env = _finished(
        OolongEnv(task=_synth_task(), use_llm_judge=True), "1"
    )

    async def boom(system, user, **kwargs):
        raise AssertionError("judge() must not be called for oolong.synth tasks")

    with patch.object(env_module, "judge", side_effect=boom):
        score, info = _run(env.evaluate())

    # synth_process_response should accept the gold answer "1" and return 1.0
    assert score == 1.0
    assert "judge_raw" not in info


def test_unfinished_env_returns_zero_regardless_of_flag():
    """Unfinished agent — no grading happens, judge MUST NOT be called."""
    env = OolongEnv(task=_subagent_task(), use_llm_judge=True)
    # NOT calling _finished — agent never called finish()

    async def boom(system, user, **kwargs):
        raise AssertionError("judge() must not be called for unfinished agent")

    with patch.object(env_module, "judge", side_effect=boom):
        score, info = _run(env.evaluate())

    assert score == 0.0
    assert "never called finish" in info["reason"]
