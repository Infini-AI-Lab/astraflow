"""TextCraft reward function (API-parity stub).

The recursive_agent workflow computes its own reward from
``TextCraftEnv.evaluate()`` (rule-based on final inventory). This module
exists so a yaml ``reward_fn: textcraft_success`` resolves to *something*
— the workflow won't actually call it.

If you ever want a verifier-style reward over the final assistant message,
this is the place to add it.
"""

from typing import Any

from astraflow.core.workflow.registry import register_reward


@register_reward("textcraft_success")
def textcraft_success_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs: Any,
) -> float:
    """Stub reward — recursive_agent does not call this. Returns 0.0 if invoked."""
    return 0.0
