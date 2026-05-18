import torch
from typing import List, Dict, Optional
from ..agent import AgentClient
from src.typings import AgentOutput

class RLHFAgentClient(AgentClient):
    
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        max_length=2048,
        temperature=1.0,
        top_p=0.95,
        log_mode=False,  # Set True during rollout to collect RL data
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.log_mode = log_mode
        
        # Storage for RL training data
        self.trajectory_data = [] if log_mode else None
        
    def inference(self, history: List[dict]) -> str:
        prompt = self._format_history(history)
        
        input_ids = self.tokenizer.encode(
            prompt, 
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,  # For log probs
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        action_ids = outputs.sequences[0][input_ids.shape[1]:]
        action = self.tokenizer.decode(action_ids, skip_special_tokens=True)
        
        # Store trajectory data if in log mode
        if self.log_mode:
            self._log_step(
                prompt=prompt,
                input_ids=input_ids,
                action=action,
                action_ids=action_ids,
                scores=outputs.scores,
                history=history
            )
        
        return action
    
    def _format_history(self, history: List[dict]) -> str:
        prompt = ""
        for msg in history:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            else:
                prompt += f"Agent: {msg['content']}\n\n"
        prompt += "Agent:"
        return prompt
    
    def _log_step(self, prompt, input_ids, action, action_ids, scores, history):
        log_probs = []
        for idx, score in enumerate(scores):
            token_id = action_ids[idx]
            log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
            log_probs.append(log_prob)
        
        step_data = {
            "prompt": prompt,
            "action": action,
            "action_ids": action_ids.cpu().tolist(),
            "log_probs": log_probs,
            "history_length": len(history),
        }
        self.trajectory_data.append(step_data)
    
    def get_trajectory_data(self) -> List[Dict]:
        data = self.trajectory_data
        self.trajectory_data = []
        return data
    
    def reset_trajectory(self):
        self.trajectory_data = []