"""Standalone workflow package for rollout workflows and reward functions.

Importing this package triggers auto-registration of all built-in
workflows and reward functions via their @register_workflow / @register_reward
decorators.
"""

# Auto-import implementations to trigger registry decorators
import astraflow.workflow.impl.agentbench.alfworld_task_server
import astraflow.workflow.impl.agentbench.task_server
import astraflow.workflow.impl.agentbench.webshop_task_server
import astraflow.workflow.impl.agentbench.webshop_checker_workflow
import astraflow.workflow.impl.asearcher
import astraflow.workflow.impl.code_actor_and_verify
import astraflow.workflow.impl.code_actor_and_verify_v2
import astraflow.workflow.impl.code_actor_and_verify_v3
import astraflow.workflow.impl.code_solve_and_select
import astraflow.workflow.impl.livecodebench_single_turn
import astraflow.workflow.impl.multi_turn
import astraflow.workflow.impl.plan_and_solve
import astraflow.workflow.impl.solve_and_check
import astraflow.workflow.impl.sep_solve_and_check
import astraflow.workflow.impl.solve_and_verify
import astraflow.workflow.impl.actor_and_verify
import astraflow.workflow.impl.rlvr
import astraflow.workflow.impl.sm_lg_router
import astraflow.workflow.impl.vision_rlvr
import astraflow.workflow.reward.clevr_count_70k
import astraflow.workflow.reward.geometry3k
import astraflow.workflow.reward.math_verify
import astraflow.workflow.reward.human_eval_reward
import astraflow.workflow.reward.livecodebench_reward
