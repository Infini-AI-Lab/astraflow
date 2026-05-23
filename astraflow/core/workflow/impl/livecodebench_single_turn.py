"""Single-turn LiveCodeBench workflow."""

from __future__ import annotations

from astraflow.core.workflow.impl.rlvr import RLVRWorkflow
from astraflow.core.workflow.registry import register_workflow


def _identity_prompt_extractor(data: dict):
    return data["messages"]


@register_workflow("livecodebench_single_turn")
class LiveCodeBenchSingleTurnWorkflow(RLVRWorkflow):
    """Single-turn code generation workflow for execution-based code tasks."""

    def __init__(self, *args, data_extract_prompt_fn=_identity_prompt_extractor, **kwargs):
        super().__init__(
            *args,
            data_extract_prompt_fn=data_extract_prompt_fn,
            **kwargs,
        )
