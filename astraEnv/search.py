"""Minimal client for the CMU RAG search server.

One public function: `search(query, n_docs=5, ...) -> list[dict]`.
Each result is `{"text": str, "source": str, "score": float, ...}`.

Backed by a process-global semaphore (default 256 concurrent) calibrated
to the server's measured ceiling — going higher tends to degrade latency
without raising throughput. Retries 3 times with exponential backoff on
transient failures (429 / 5xx / timeouts).

Requires the env var `CMU_SEARCH_API_KEY` (stored in ~/.cmu_search_key).

Usage:
    from astraEnv.search import search

    passages = await search("who painted the mona lisa?", n_docs=5)
    for p in passages:
        print(f"[{p['score']:.2f}] ({p['source']}) {p['text'][:120]}...")
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

_URL = os.environ.get(
    "CMU_SEARCH_URL", "http://catalyst-fleet1.cs.cmu.edu:30888/search"
)
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_MAX_CONCURRENT = 256  # measured ceiling — going higher does not help throughput

# Module-level semaphore + lock for lazy init from the right event loop.
_semaphore: asyncio.Semaphore | None = None
_sem_lock = asyncio.Lock()


class SearchError(RuntimeError):
    """Raised when the search call cannot return a usable response."""


async def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    async with _sem_lock:
        if _semaphore is None:
            _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
        return _semaphore


async def search(
    query: str,
    n_docs: int = 5,
    *,
    backend: str = "faiss",
    nprobe: int = 128,
    min_words: int = 20,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Query the CMU RAG server; return a flat list of passages.

    Each returned dict has at minimum:
      - text:    str         the full passage
      - source:  str         dataset name the passage came from
      - score:   float       similarity score (higher = better)
    plus whatever extra fields the server provides (filename, passage_id, etc.).

    Raises:
      SearchError on persistent failure (after retries) or unparseable response.
    """
    api_key = os.environ.get("CMU_SEARCH_API_KEY")
    if not api_key:
        raise SearchError("CMU_SEARCH_API_KEY environment variable is not set")

    payload = {
        "query": query,
        "n_docs": n_docs,
        "backend": backend,
        "nprobe": nprobe,
        "min_words": min_words,
    }
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    sem = await _get_semaphore()
    last_err: Exception | None = None

    # Retry loop is OUTSIDE the semaphore so that backoff sleeps don't
    # starve other concurrent searches. If the server is degraded, this
    # prevents a thundering-herd where all semaphore slots stall on
    # exponential backoff at once (locking out searches that would
    # otherwise succeed). The semaphore is re-acquired per attempt.
    for attempt in range(_MAX_ATTEMPTS):
        async with sem:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s, connect=5.0)
            ) as client:
                try:
                    resp = await client.post(_URL, json=payload, headers=headers)
                except httpx.RequestError as exc:
                    last_err = exc
                    resp = None

            if resp is not None:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        raise SearchError(
                            f"Server returned non-JSON: {resp.text[:300]}"
                        ) from exc
                    return _flatten_results(data)

                if resp.status_code in _RETRY_STATUSES:
                    last_err = SearchError(
                        f"Server returned {resp.status_code}: {resp.text[:200]}"
                    )
                else:
                    # Non-retryable status (e.g. 401/403/404).
                    raise SearchError(
                        f"Server returned {resp.status_code}: {resp.text[:500]}"
                    )

        # Backoff happens OUTSIDE the semaphore: other searches can run.
        if attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(2**attempt)

    raise SearchError(
        f"search() failed after {_MAX_ATTEMPTS} attempts: {last_err}"
    ) from last_err


def _flatten_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the server's nested response into a flat list of passages.

    Server response shape:
      {
        "results": {
          "passages": [[ {text, source, ...}, ... ]],   # outer list = per-query
          "scores":   [[ float, ... ]],
          ...
        },
        ...
      }

    We send one query at a time, so we take `passages[0]` and zip with
    `scores[0]` to attach the similarity score to each passage.
    """
    results = data.get("results")
    if not isinstance(results, dict):
        raise SearchError(f"Missing 'results' in response: {str(data)[:300]}")

    passages_outer = results.get("passages")
    scores_outer = results.get("scores")
    if not passages_outer:
        return []

    # First (and only) query's passages.
    passages = passages_outer[0] if isinstance(passages_outer[0], list) else []
    scores = (
        scores_outer[0]
        if scores_outer and isinstance(scores_outer[0], list)
        else [None] * len(passages)
    )

    out: list[dict[str, Any]] = []
    for p, s in zip(passages, scores):
        if not isinstance(p, dict):
            continue
        entry = dict(p)
        if "score" not in entry and s is not None:
            entry["score"] = float(s)
        out.append(entry)
    return out
