"""Oolong recursive-agent workflow for long-context aggregation.

Port of platoon's Oolong setup (see `claude-doc/oolong-plan.md`):
- workflow_cls: oolong_recursive
- Python sandbox env: stateful exec() namespace per agent, pre-populated
  with `context`, `finish`, `launch_subagent`, and `asyncio`.
- Agent emits `<thought>...</thought><python>...</python>` blocks; the
  Python may call `finish(answer)` or `await launch_subagent(goal, context)`.
- Reward (per Gandhi et al. 2026, "Recursive Agent Optimization"):
      R(X) = success(X) + lambda * mean(success(children))
  with lambda=0.4 by default for Oolong (paper's choice for OOLONG-REAL).
"""

# Importing workflow triggers registration of the `oolong_recursive`
# workflow_cls via @register_workflow.
from astraflow.core.workflow.impl.oolong import workflow  # noqa: F401
