from typing import List, Dict, Any
from src.client.task import TaskClient, TaskClientOutput
from src.client.agent import AgentClient
from src.typings import SampleIndex

class RLHFTaskClient(TaskClient):
    
    def run_sample_with_trajectory(
        self, 
        index: SampleIndex, 
        agent: AgentClient
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "index": sample index,
                "trajectory": [
                    {
                        "observation": str,
                        "action": str,
                        "action_log_probs": List[float],
                        "action_ids": List[int],
                        ...
                    },
                    ...
                ],
                "final_status": str,
                "reward": float,
                "success": bool,
                "metadata": {...}
            }
        """
        # Enable logging in agent
        if hasattr(agent, 'log_mode'):
            agent.log_mode = True
            agent.reset_trajectory()
        
        result: TaskClientOutput = self.run_sample(index, agent)
        
        trajectory_steps = []
        if hasattr(agent, 'get_trajectory_data'):
            agent_trajectory = agent.get_trajectory_data()
            
            history = result.output.history if result.output else []
            
            for step_idx, agent_step in enumerate(agent_trajectory):
                obs_idx = step_idx * 2 + 1 
                observation = ""
                if obs_idx < len(history):
                    observation = history[obs_idx]["content"]
                
                trajectory_steps.append({
                    "observation": observation,
                    "action": agent_step["action"],
                    "action_log_probs": agent_step["log_probs"],
                    "action_ids": agent_step["action_ids"],
                    "prompt": agent_step["prompt"],
                })
        
        reward = self._calculate_reward(result)
        
        return {
            "index": index,
            "trajectory": trajectory_steps,
            "history": result.output.history if result.output else [],
            "final_status": result.output.status if result.output else "error",
            "reward": reward,
            "success": result.error is None,
            "error": result.error,
            "metadata": {
                "task_name": self.name,
                "num_steps": len(trajectory_steps),
                "result": result.output.result if result.output else None,
            }
        }
    
    def _calculate_reward(self, result: TaskClientOutput) -> float:
        """
        Calculate reward from task result.
        Customize this based on your reward shaping strategy.
        """
        if result.error:
            return -1.0  # Penalty for errors
        
        if not result.output:
            return -1.0
        
        # Check task completion status
        from src.typings import SampleStatus
        if result.output.status == SampleStatus.COMPLETED:
            # Check if task has explicit reward/score
            if hasattr(result.output, 'result') and result.output.result:
                if isinstance(result.output.result, dict):
                    # Use task-specific success metric
                    if 'succeed' in result.output.result:
                        return 1.0 if result.output.result['succeed'] else 0.0
                    if 'score' in result.output.result:
                        return float(result.output.result['score'])
            return 1.0  # Completed successfully
        
        # Partial rewards for different statuses
        status_rewards = {
            SampleStatus.AGENT_CONTEXT_LIMIT: -0.5,
            SampleStatus.AGENT_INVALID_ACTION: -0.8,
            SampleStatus.TASK_LIMIT_REACHED: 0.0, 
            SampleStatus.TASK_ERROR: -1.0,
        }
        return status_rewards.get(result.output.status, 0.0)