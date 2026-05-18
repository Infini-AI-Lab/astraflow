"""WebShop-specific TaskServer workflow registered for RaaS usage."""

from __future__ import annotations

from typing import Any

from astraflow.workflow.impl.agentbench.task_server import TaskServerWorkflow
from astraflow.workflow.registry import register_workflow


@register_workflow("webshop_task_server")
class WebshopTaskServerWorkflow(TaskServerWorkflow):
    """TaskServer workflow variant for WebShop server protocol quirks."""

    def __init__(
        self,
        *args,
        failure_penalty: float = -0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.failure_penalty = failure_penalty

    async def _call_server(
        self,
        endpoint: str,
        method: str = "POST",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # WebShop expects numeric sample_id; base workflow sends it as str.
        if data and "sample_id" in data:
            try:
                data = dict(data)
                data["sample_id"] = int(data["sample_id"])
            except (TypeError, ValueError):
                pass
        return await super()._call_server(endpoint, method, data)

    def postprocess_reward(
        self,
        reward: float,
        num_turns: int,
        info: dict[str, Any],
    ) -> float:
        del info
        if reward <= 1e-3:
            reward = self.failure_penalty
        return reward * (self.turn_discount**num_turns)
