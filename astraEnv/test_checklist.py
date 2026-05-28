"""Tests for astraEnv.checklist — judge call mocked, no network.

Matches ai-rubric 0.2.4's RubricChecklistFast single-call semantics.

Run:
    pytest astraEnv/test_checklist.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from astraEnv import checklist as cl_module
from astraEnv.checklist import ChecklistGrader, grade_with_checklist


def _run(coro):
    return asyncio.run(coro)


_OK_RESPONSE = (
    '{'
    '"checklist": ["finds dates", "right format", "completeness"],'
    '"checklist_scores": [1.0, 1.0, 0.5],'
    '"reasoning": "Found dates in correct format; partially complete.",'
    '"overall_score": 0.8'
    '}'
)


# --------------------------------------------------------------------------
# Single-call structure
# --------------------------------------------------------------------------


def test_single_judge_call_returns_overall_score():
    """One LLM call; score from overall_score field."""
    n_calls = {"n": 0}

    async def fake_judge(system, user, **kwargs):
        n_calls["n"] += 1
        return _OK_RESPONSE

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        score, reason = _run(ChecklistGrader(goal="goal").aevaluate(context="ctx"))

    assert n_calls["n"] == 1
    assert score == pytest.approx(0.8)
    assert "Found dates" in reason


def test_system_prompt_is_the_ai_rubric_one():
    """Verify we send ai-rubric's verbatim system prompt."""
    captured = {}

    async def fake_judge(system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return _OK_RESPONSE

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))

    # Hallmark phrases from the verbatim prompt:
    assert "decomposing success criteria for the task into a checklist" in captured["system"]
    assert "checklist_scores" in captured["system"]
    assert "overall_score" in captured["system"]
    # User prompt format:
    assert "# Task\ng" in captured["user"]
    assert "ctx" in captured["user"]
    assert "# Your Evaluation Output" in captured["user"]


# --------------------------------------------------------------------------
# Parsing variants
# --------------------------------------------------------------------------


def test_overall_score_clamped_to_unit_interval():
    """Defensive: if the LLM returns >1 or <0, we clamp instead of raising."""
    async def fake_high(system, user, **kwargs):
        return '{"checklist": [], "checklist_scores": [], "reasoning": "x", "overall_score": 1.7}'

    with patch.object(cl_module, "judge", side_effect=fake_high):
        score, _ = _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))
    assert score == 1.0

    async def fake_neg(system, user, **kwargs):
        return '{"checklist": [], "checklist_scores": [], "reasoning": "x", "overall_score": -0.3}'

    with patch.object(cl_module, "judge", side_effect=fake_neg):
        score, _ = _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))
    assert score == 0.0


def test_missing_overall_score_returns_zero():
    async def fake_judge(system, user, **kwargs):
        return '{"checklist": ["a"], "checklist_scores": [1.0], "reasoning": "no overall"}'

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        score, _ = _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))
    assert score == 0.0  # defaults to 0.0 per .get(..., 0.0)


def test_last_checklist_and_scores_exposed():
    async def fake_judge(system, user, **kwargs):
        return _OK_RESPONSE

    g = ChecklistGrader(goal="g")
    with patch.object(cl_module, "judge", side_effect=fake_judge):
        _run(g.aevaluate(context="ctx"))

    assert g.last_checklist == ["finds dates", "right format", "completeness"]
    assert g.last_checklist_scores == [1.0, 1.0, 0.5]
    assert g.last_overall_score == pytest.approx(0.8)
    assert "Found dates" in g.last_reasoning


# --------------------------------------------------------------------------
# Failure modes — never raise; return (0.0, reason)
# --------------------------------------------------------------------------


def test_network_failure_returns_zero():
    async def boom(system, user, **kwargs):
        raise ConnectionError("network down")

    with patch.object(cl_module, "judge", side_effect=boom):
        score, reason = _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))
    assert score == 0.0
    assert "call failed" in reason
    assert "network down" in reason


def test_unparseable_response_returns_zero():
    async def fake_judge(system, user, **kwargs):
        return "not json at all"

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        score, reason = _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))
    assert score == 0.0
    assert "unparseable" in reason


def test_non_numeric_overall_score_returns_zero():
    async def fake_judge(system, user, **kwargs):
        return '{"overall_score": "not a number", "reasoning": ""}'

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        score, reason = _run(ChecklistGrader(goal="g").aevaluate(context="ctx"))
    assert score == 0.0
    assert "not a number" in reason


# --------------------------------------------------------------------------
# Wrapper + passthrough
# --------------------------------------------------------------------------


def test_grade_with_checklist_helper():
    async def fake_judge(system, user, **kwargs):
        return _OK_RESPONSE

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        score, reason = _run(grade_with_checklist("goal", "ctx"))

    assert score == pytest.approx(0.8)


def test_temperature_and_model_forwarded():
    captured: list[dict] = []

    async def fake_judge(system, user, **kwargs):
        captured.append(dict(kwargs))
        return _OK_RESPONSE

    g = ChecklistGrader(goal="g", judge_model="custom-model", temperature=1.0)
    with patch.object(cl_module, "judge", side_effect=fake_judge):
        _run(g.aevaluate(context="ctx"))

    assert captured[0].get("model") == "custom-model"
    assert captured[0].get("temperature") == 1.0


def test_no_caching_between_evaluations():
    """Each aevaluate() call makes a fresh judge call (no checklist cache)."""
    n_calls = {"n": 0}

    async def fake_judge(system, user, **kwargs):
        n_calls["n"] += 1
        return _OK_RESPONSE

    with patch.object(cl_module, "judge", side_effect=fake_judge):
        _run(grade_with_checklist("same goal", "ctx1"))
        _run(grade_with_checklist("same goal", "ctx2"))

    assert n_calls["n"] == 2  # no caching
