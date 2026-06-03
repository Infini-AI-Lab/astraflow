"""Smoke tests for astraEnv.judge.

Run with:
    pytest astraEnv/test_judge.py -v

The end-to-end Fireworks test is skipped automatically when
FIREWORKS_API_KEY is not set.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from astraEnv.judge import JudgeError, extract_json, judge


# --------------------------------------------------------------------------
# extract_json — pure, no network
# --------------------------------------------------------------------------


def test_extract_json_bare():
    assert extract_json('{"score": 0.8, "reason": "ok"}') == {
        "score": 0.8,
        "reason": "ok",
    }


def test_extract_json_with_whitespace():
    assert extract_json('   \n  {"score": 1}\n  ') == {"score": 1}


def test_extract_json_fenced_json():
    text = 'sure thing!\n```json\n{"score": 0.5, "reason": "partial"}\n```\nend'
    assert extract_json(text) == {"score": 0.5, "reason": "partial"}


def test_extract_json_fenced_plain():
    text = '```\n{"score": 0}\n```'
    assert extract_json(text) == {"score": 0}


def test_extract_json_fenced_uppercase():
    text = '```JSON\n{"x": 1}\n```'
    assert extract_json(text) == {"x": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        extract_json("not json at all")


# --------------------------------------------------------------------------
# judge — network, gated on API key
# --------------------------------------------------------------------------


_HAS_KEY = bool(os.environ.get("FIREWORKS_API_KEY"))


def test_judge_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="FIREWORKS_API_KEY"):
        asyncio.run(judge(system="x", user="y"))


@pytest.mark.skipif(not _HAS_KEY, reason="FIREWORKS_API_KEY not set")
def test_judge_end_to_end_returns_json():
    response = asyncio.run(
        judge(
            system=(
                'You grade outputs. Return ONLY JSON: '
                '{"score": float in [0,1], "reason": "<one short sentence>"}'
            ),
            user="GOAL: name a primary color\n\nOUTPUT: red",
            temperature=0.0,
            max_tokens=100,
        )
    )
    parsed = extract_json(response)
    assert "score" in parsed
    assert 0.0 <= float(parsed["score"]) <= 1.0
    # "red" should score high — sanity check the judge is sensible
    assert float(parsed["score"]) >= 0.5, f"Unexpectedly low score: {parsed}"
