"""Oolong reward function (API-parity stub).

The oolong_recursive workflow computes its own reward via the platoon-ported
rule-based grader in `impl/oolong/eval_helpers.py`. This module exists so a
yaml `reward_fn: oolong_success` resolves to *something* — the workflow
won't actually call it.
"""

from typing import Any

from astraflow.core.workflow.registry import register_reward


@register_reward("oolong_success")
def oolong_success_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs: Any,
) -> float:
    """Stub reward — oolong_recursive does not call this. Returns 0.0 if invoked."""
    return 0.0
