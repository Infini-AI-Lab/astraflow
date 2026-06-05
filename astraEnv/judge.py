"""Minimal LLM-as-a-judge utility.

Two functions. Both stateless.

- `judge(system, user, ...)` posts a (system, user) pair to Fireworks and
  returns the raw assistant content string.
- `extract_json(text)` parses JSON out of an LLM response, tolerating
  common code-fence wrapping.

Callers write their own rubric prompts and parse what they expect.
See claude-doc/minimal-llm-judge-plan.md for the design rationale.

Usage:
    from astraEnv.judge import judge, extract_json

    response = await judge(
        system='You grade outputs. Return JSON {"score", "reason"}.',
        user=f"Goal: {goal}\\n\\nOutput: {output}",
    )
    parsed = extract_json(response)
    score = float(parsed["score"])

Requires the env var `FIREWORKS_API_KEY`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx

_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
_DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


class JudgeError(RuntimeError):
    """Raised when the judge call cannot return a usable response."""


async def judge(
    system: str,
    user: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_s: float = 60.0,
) -> str:
    """Send (system, user) to Fireworks; return the raw assistant content.

    Retries up to 3 times with exponential backoff on transient failures
    (429, 5xx, network errors). Raises JudgeError on persistent failure.

    Default `max_tokens` is set generously (2048) because reasoning models
    like gpt-oss-120b consume tokens for internal chain-of-thought before
    emitting the final answer; too-tight budgets truncate before content.

    For reasoning models that put their chain-of-thought into a separate
    `reasoning_content` field, this function returns `content` if non-empty,
    otherwise falls back to `reasoning_content`. extract_json() handles
    both shapes.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise JudgeError("FIREWORKS_API_KEY environment variable is not set")

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await client.post(_API_URL, json=payload, headers=headers)
            except httpx.RequestError as exc:
                last_err = exc
                await asyncio.sleep(2**attempt)
                continue

            if resp.status_code == 200:
                try:
                    message = resp.json()["choices"][0]["message"]
                except (KeyError, IndexError, ValueError) as exc:
                    raise JudgeError(
                        f"Unexpected response shape: {resp.text[:500]}"
                    ) from exc
                # Prefer the canonical `content` field. Reasoning models
                # (e.g. gpt-oss-120b) may emit only `reasoning_content`
                # when truncated; fall back to that so extract_json can
                # still find a JSON snippet inside the chain-of-thought.
                content = message.get("content") or message.get("reasoning_content")
                if not content:
                    raise JudgeError(
                        f"Empty assistant content: {resp.text[:500]}"
                    )
                return content

            if resp.status_code in _RETRY_STATUSES:
                last_err = JudgeError(
                    f"Fireworks returned {resp.status_code}: {resp.text[:200]}"
                )
                await asyncio.sleep(2**attempt)
                continue

            raise JudgeError(
                f"Fireworks returned {resp.status_code}: {resp.text[:500]}"
            )

    raise JudgeError(
        f"judge() failed after {_MAX_ATTEMPTS} attempts: {last_err}"
    ) from last_err


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON out of an LLM response, tolerating common fence wrapping.

    Strategy (first success wins):
      1. json.loads on the trimmed text
      2. strip ```json ... ``` fences and retry
      3. strip plain ``` ... ``` fences and retry
      4. re-raise the original JSONDecodeError
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1).strip())

    fenced = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    return json.loads(text)
