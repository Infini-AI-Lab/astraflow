"""Standalone workflow package for rollout workflows and reward functions.

Importing this package triggers auto-registration of all built-in
workflows and reward functions via their @register_workflow / @register_reward
decorators.
"""

# Auto-import implementations to trigger registry decorators
import astraflow.core.workflow.impl.agentbench.alfworld_task_server
import astraflow.core.workflow.impl.agentbench.task_server
import astraflow.core.workflow.impl.agentbench.webshop_task_server
import astraflow.core.workflow.impl.agentbench.webshop_checker_workflow
import astraflow.core.workflow.impl.asearcher
import astraflow.core.workflow.impl.code_actor_and_verify
import astraflow.core.workflow.impl.code_actor_and_verify_v2
import astraflow.core.workflow.impl.code_actor_and_verify_v3
import astraflow.core.workflow.impl.code_solve_and_select
import astraflow.core.workflow.impl.livecodebench_single_turn
import astraflow.core.workflow.impl.multi_turn
import astraflow.core.workflow.impl.plan_and_solve
import astraflow.core.workflow.impl.solve_and_check
import astraflow.core.workflow.impl.sep_solve_and_check
import astraflow.core.workflow.impl.solve_and_verify
import astraflow.core.workflow.impl.actor_and_verify
import astraflow.core.workflow.impl.rlvr
import astraflow.core.workflow.impl.sm_lg_router
import astraflow.core.workflow.impl.spawn
import astraflow.core.workflow.impl.textcraft.workflow  # registers recursive_agent
import astraflow.core.workflow.impl.deepdive.workflow  # registers deepdive_recursive
import astraflow.core.workflow.impl.vision_rlvr
import astraflow.core.workflow.reward.clevr_count_70k
import astraflow.core.workflow.reward.geometry3k
import astraflow.core.workflow.reward.math_verify
import astraflow.core.workflow.reward.human_eval_reward
import astraflow.core.workflow.reward.livecodebench_reward
import astraflow.core.workflow.reward.textcraft_success  # noqa: F401
import astraflow.core.workflow.reward.deepdive_success  # noqa: F401
