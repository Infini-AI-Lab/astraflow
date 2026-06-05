"""DeepDive reward function (API-parity stub).

The deepdive_recursive workflow computes its own reward via the LLM judge
in `impl/deepdive/env.py:evaluate()`. This module exists so a yaml
`reward_fn: deepdive_success` resolves to *something* — the workflow
won't actually call it.
"""

from typing import Any

from astraflow.core.workflow.registry import register_reward


@register_reward("deepdive_success")
def deepdive_success_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs: Any,
) -> float:
    """Stub reward — deepdive_recursive does not call this. Returns 0.0 if invoked."""
    return 0.0
