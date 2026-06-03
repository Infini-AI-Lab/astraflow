"""Tests for astraEnv.search.

Run with:
    pytest astraEnv/test_search.py -v

The live smoke test is skipped automatically when CMU_SEARCH_API_KEY is
not set.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from astraEnv.search import SearchError, _flatten_results, search


# --------------------------------------------------------------------------
# _flatten_results — pure, no network
# --------------------------------------------------------------------------


def test_flatten_normal_shape():
    data = {
        "results": {
            "passages": [
                [
                    {"text": "p1", "source": "wiki"},
                    {"text": "p2", "source": "c4"},
                ]
            ],
            "scores": [[1.4, 1.3]],
        },
    }
    out = _flatten_results(data)
    assert len(out) == 2
    assert out[0]["text"] == "p1"
    assert out[0]["source"] == "wiki"
    assert out[0]["score"] == pytest.approx(1.4)
    assert out[1]["score"] == pytest.approx(1.3)


def test_flatten_preserves_existing_score():
    """If server already attached a score on the passage, don't overwrite."""
    data = {
        "results": {
            "passages": [[{"text": "p1", "score": 0.99}]],
            "scores": [[0.5]],
        }
    }
    out = _flatten_results(data)
    assert out[0]["score"] == pytest.approx(0.99)  # passage's own wins


def test_flatten_empty_passages():
    assert _flatten_results({"results": {"passages": []}}) == []


def test_flatten_no_scores_array():
    """Server can omit scores; passages just don't get a score field."""
    data = {"results": {"passages": [[{"text": "p1", "source": "x"}]]}}
    out = _flatten_results(data)
    assert out[0]["text"] == "p1"
    assert "score" not in out[0]


def test_flatten_missing_results_key_raises():
    with pytest.raises(SearchError, match="Missing 'results'"):
        _flatten_results({"query": "foo"})


# --------------------------------------------------------------------------
# search — network, mocked
# --------------------------------------------------------------------------


def _make_ok_response(passages, scores) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"results": {"passages": [passages], "scores": [scores]}},
    )


def test_search_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("CMU_SEARCH_API_KEY", raising=False)
    with pytest.raises(SearchError, match="CMU_SEARCH_API_KEY"):
        asyncio.run(search("anything"))


def test_search_returns_normalized_passages(monkeypatch):
    monkeypatch.setenv("CMU_SEARCH_API_KEY", "fake")
    mock_resp = _make_ok_response(
        passages=[{"text": "hello", "source": "wiki"}],
        scores=[0.9],
    )
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        out = asyncio.run(search("test query", n_docs=1))
    assert out == [{"text": "hello", "source": "wiki", "score": 0.9}]


def test_search_propagates_non_retryable_error(monkeypatch):
    monkeypatch.setenv("CMU_SEARCH_API_KEY", "fake")
    bad = httpx.Response(status_code=403, text="forbidden")
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=bad)):
        with pytest.raises(SearchError, match="403"):
            asyncio.run(search("test"))


def test_search_retries_on_429(monkeypatch):
    """First call returns 429, second succeeds — we should get the success."""
    monkeypatch.setenv("CMU_SEARCH_API_KEY", "fake")
    responses = [
        httpx.Response(status_code=429, text="rate limited"),
        _make_ok_response(passages=[{"text": "ok", "source": "x"}], scores=[1.0]),
    ]
    call_count = {"n": 0}

    async def fake_post(self, *args, **kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        return responses[i]

    with patch("httpx.AsyncClient.post", new=fake_post), \
         patch("asyncio.sleep", new=AsyncMock()):  # skip backoff sleep
        out = asyncio.run(search("retry-me"))
    assert out == [{"text": "ok", "source": "x", "score": 1.0}]
    assert call_count["n"] == 2


def test_search_exhausts_retries_then_raises(monkeypatch):
    monkeypatch.setenv("CMU_SEARCH_API_KEY", "fake")
    bad = httpx.Response(status_code=503, text="unavailable")
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=bad)), \
         patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(SearchError):
            asyncio.run(search("nope"))


# --------------------------------------------------------------------------
# Live smoke — only runs if CMU_SEARCH_API_KEY is set
# --------------------------------------------------------------------------


_HAS_KEY = bool(os.environ.get("CMU_SEARCH_API_KEY"))


@pytest.mark.skipif(not _HAS_KEY, reason="CMU_SEARCH_API_KEY not set")
def test_search_live_smoke():
    out = asyncio.run(search("who painted the mona lisa?", n_docs=3))
    assert len(out) > 0
    first = out[0]
    assert "text" in first
    assert len(first["text"]) > 50  # passages should be substantive
    # Heuristic: a query about the Mona Lisa should return at least one
    # passage that mentions "Leonardo" or "da Vinci".
    joined = " ".join(p["text"] for p in out)
    assert "Leonardo" in joined or "da Vinci" in joined, (
        f"Unexpected results: {joined[:300]}"
    )
