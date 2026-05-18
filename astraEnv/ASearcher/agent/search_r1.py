import re
from typing import Dict, List, Any, Optional
from tools.search_utils import AsyncSearchBrowserClient


class SearchR1Agent:
    
    def __init__(self,
                 max_turns: int = 10,
                 topk: int = 5):

        self.max_turns = max_turns
        self.topk = topk
        self.stop = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>", "</search>"]
        self.stop_sequences = self.stop      

        self.current_process = None
        self.tokenizer = None
        
        print(f"SearchR1Agent initialized.")

    def get_query_from_text(self, text: str) -> Optional[str]:
        pattern = r'<\|begin_of_query\|>(.*?)<\|end_of_query\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        if '<|begin_of_query|>' in text:
            parts = text.split('<|begin_of_query|>')
            if len(parts) > 1:
                query_part = parts[-1]
                if not query_part.strip().endswith('<|end_of_query|>'):
                    return query_part.strip()
        
        search_pattern = r'<search>(.*?)</search>'
        search_matches = re.findall(search_pattern, text, re.DOTALL)
        if search_matches:
            return search_matches[-1].strip()
        
        if '<search>' in text:
            parts = text.split('<search>')
            if len(parts) > 1:
                query_part = parts[-1]
                if not query_part.strip().endswith('</search>'):
                    return query_part.strip()
        
        return None

    def fix_incomplete_search_tag(self, text: str) -> str:
        if '<|begin_of_query|>' in text and not text.strip().endswith('<|end_of_query|>'):
            return text.strip() + '<|end_of_query|>'
        
        if '<search>' in text and not text.strip().endswith('</search>'):
            return text.strip() + '</search>'
            
        return text

    def all_finished(self, processes: List[Dict]) -> bool:
        finished = []
        for process in processes:
            finished.append(not process.get("running", True))
        return all(finished)

    def initialize_with_prompt(self, prompt):
        """Initialize agent with a specific prompt"""
        self.current_process = {
            "prompt": prompt,
            "history": [dict(type="prompt", text=prompt)],
            "running": True,
            "id": "0"
        }
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for the agent"""
        self.tokenizer = tokenizer
    
    @property
    def num_turns(self):
        """Get current number of turns"""
        if not self.current_process:
            return 0
        return len([h for h in self.current_process["history"] if h["type"] == "act"])
    
    @property
    def is_finished(self):
        """Check if agent is finished"""
        if not self.current_process or not self.current_process.get("running", False):
            return True
        
        # Check if we have an answer
        full_text = "".join([h.get("text", "") for h in self.current_process["history"] if h["type"] != "prompt"])
        has_answer = "<answer>" in full_text and "</answer>" in full_text
        
        # Check max turns
        action_count = len([h for h in self.current_process["history"] if h["type"] == "act"])
        max_turns_exceeded = action_count >= self.max_turns
        
        return has_answer or max_turns_exceeded

    def prepare_llm_query(self):
        """Prepare LLM query for current process"""
        if not self.current_process:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")
        
        process = self.current_process
        
        if not process.get("running", False):
            return "", {"stop": self.stop}
        
        # Check if last text contains a search query
        last_text = process["history"][-1]["text"]
        
        # Handle search query patterns - return empty to trigger tool calling
        if (("<|begin_of_query|>" in last_text and last_text.strip().endswith("<|end_of_query|>")) or
            ("<search>" in last_text and last_text.strip().endswith("</search>"))):
            return "", {"stop": self.stop}
        
        # Normal LLM generation
        input_text = "".join([h["text"] for h in process["history"]])
        query_len = self.tokenizer([input_text], return_length=True)['length'][0]
        
        sampling_params = {"stop": self.stop}
        
        return input_text, sampling_params

    def consume_llm_response(self, resp, completion_text):
        """Consume LLM response and extract tool calls"""
        if not self.current_process:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        process = self.current_process
        
        # Handle different response formats
        if hasattr(resp, 'stop_reason') and hasattr(resp, 'text'):
            stop_reason = resp.stop_reason
            generated_text = resp.text
        elif isinstance(resp, dict):
            stop_reason = resp.get('stop_reason', '')
            generated_text = resp.get('text', str(resp))
        elif resp is None:
            stop_reason = ""
            generated_text = completion_text or ""
        else:
            stop_reason = "</answer>" if "<answer>" in str(resp) else ""
            generated_text = completion_text or str(resp)
        
        # Fix incomplete search tags
        fixed_text = self.fix_incomplete_search_tag(generated_text)
        if fixed_text != generated_text:
            generated_text = fixed_text
        
        # Extract query and check for actions
        extracted_query = self.get_query_from_text(generated_text)
        tool_calls = []
        
        if extracted_query:
            # This is a search action
            process["history"].append(dict(
                type="act", 
                text=generated_text.strip()
            ))
            # Create tool call for search
            if ("<|begin_of_query|>" in generated_text and generated_text.strip().endswith("<|end_of_query|>")):
                query_text = generated_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
                tool_calls.append(f"<search>{query_text.strip()}</search>")
            elif ("<search>" in generated_text and generated_text.strip().endswith("</search>")):
                query_text = generated_text.split("<search>")[-1].split("</search>")[0]
                tool_calls.append(f"<search>{query_text.strip()}</search>")
                
        elif "<answer>" in generated_text and (stop_reason == "</answer>" or "</answer>" in generated_text):
            # This is a final answer
            if not generated_text.strip().endswith("</answer>"):
                generated_text = generated_text.strip() + "</answer>"
            process["history"].append(dict(
                type="act", 
                text=generated_text
            ))
            process["running"] = False
            # Extract answer for tool call
            if "<answer>" in generated_text and "</answer>" in generated_text:
                answer_text = generated_text.split("<answer>")[-1].split("</answer>")[0]
                tool_calls.append(f"<answer>{answer_text.strip()}</answer>")
                
        elif (("<search>" in generated_text and generated_text.strip().endswith("</search>")) or 
              ("<|begin_of_query|>" in generated_text and generated_text.strip().endswith("<|end_of_query|>"))):
            # This is a complete search query
            process["history"].append(dict(
                type="act", 
                text=generated_text.strip() + "\n\n"
            ))
            # Extract query for tool call
            if ("<|begin_of_query|>" in generated_text and generated_text.strip().endswith("<|end_of_query|>")):
                query_text = generated_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
                tool_calls.append(f"<search>{query_text.strip()}</search>")
            elif ("<search>" in generated_text and generated_text.strip().endswith("</search>")):
                query_text = generated_text.split("<search>")[-1].split("</search>")[0]
                tool_calls.append(f"<search>{query_text.strip()}</search>")
        else:
            # Invalid action, add auxiliary message
            process["history"].append(dict(
                type="act", 
                text=generated_text.strip() + "\n\n"
            ))
            process["history"].append(dict(
                type="auxilliary",
                text="\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
            ))
        
        # Check if max turns reached
        action_count = len([h for h in process["history"] if h["type"] == "act"])
        if action_count >= self.max_turns:
            process["running"] = False
        
        return tool_calls

    def consume_tool_response(self, res, topk=5):
        """Consume tool response (search) - Updated for agent v2 compatibility"""
        if not self.current_process:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        process = self.current_process
        
        if res["type"] == "search":
            if isinstance(res, list):
                r = res[0]
            else:
                r = res
                
            if isinstance(r, dict) and 'documents' in r:
                documents = r["documents"]
                urls = r.get("urls", [])
            else:
                documents = []
                urls = []
            
            # Add formatted content for the agent's internal use (LLM consumption)
            if len(documents) > 0:
                doc_content_list = []
                for j, doc in enumerate(documents):
                    if isinstance(doc, str):
                        doc_clean = re.sub(r'^\d+\s+', '', doc.strip())
                        doc_content_list.append(f"{j+1}. {doc_clean}\n")
                doc_content = '\n'.join(doc_content_list)
            else:
                doc_content = ""
            
            if doc_content:
                formatted_content = "\n\n<information>" + doc_content + "</information>\n\n"
            else:
                formatted_content = "\n\n<information>No relevant documents found.</information>\n\n"
                
            # Add formatted content for LLM consumption
            process["history"].append({
                "type": "documents",  # Keep for backward compatibility
                "text": formatted_content
            })

    def get_answer(self):
        """Get final answer from current process"""
        if not self.current_process:
            return None
            
        process = self.current_process
        
        if "pred_answer" not in process:
            full_text = "".join([h["text"] for h in process["history"] if h["type"] != "prompt"])
            
            if "<answer>" in full_text and "</answer>" in full_text:
                answer = full_text.split("<answer>")[-1].split("</answer>")[0].strip()
            else:
                answer = full_text.strip()
            
            process["pred_answer"] = answer
        
        return process["pred_answer"]

    def fix_process_incomplete_tags(self, process: Dict) -> Dict:
        fixed_count = 0
        history = process.get("history", [])
        
        for i, entry in enumerate(history):
            if entry.get("type") == "act":
                original_text = entry["text"]
                fixed_text = self.fix_incomplete_search_tag(original_text)
                
                if fixed_text != original_text:
                    history[i]["text"] = fixed_text
                    fixed_count += 1
        
        return {
            "total_entries": len(history),
            "fixed_entries": fixed_count,
            "process_id": process.get("process_id", "unknown")
        }
