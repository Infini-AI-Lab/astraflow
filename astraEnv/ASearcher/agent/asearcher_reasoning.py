import re
import time
import copy
from typing import Dict, List, Any, Optional
from datetime import datetime

class ASearcherReasoningPrompts:
    THINK_AND_ACT_PROMPT_v1 =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question and the history context, generate the thought as well as the next action (only one action). Tthe completed thought should contain analysis of available information and planning for future steps. Enclose the thought within <thought> </thought> tags. 

The next action could be one of the following three, each with specific tags:
1. Search w. a search engine, e.g. <search> the search query </search>

2. Accessing some url found in prior history, e.g. <access> the url to access </access>

3. Answering the question, e.g. <answer> the answer (usually in less than 10 words) </answer> (WARNING: Answer the question only after you double check the results with sufficient search!)

Guidelines:
1. You should double check previous conclusions and identified facts using search from different perspectives. 
3. You can try different directions to solve the question, such as using different search queries.
3. If you find related entries in the search results, it is usually useful to access the corresponding urls to find more information.
4. You should find the most likely answer.
5. The next action should follow after the thought.
6. Make sure you choose only one action.
7. Carefully select the type of language to conduct your search query (Chinese or English)

Current Time: Today is {current_date} 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""

    THINK_AND_ACT_PROMPT = \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question and the history context, generate the thought as well as the next action (only one action). The completed thought should contain a detailed analysis of current situation and a plan for future steps. The action is either a query to google search or accessing some URL. Enclose the thought within <thought> </thought> tags. 

The next action could be one of the following two, each with specific tags:
1. Search w. a search engine, e.g. <search> the search query </search>

2. Accessing some url found in prior history to find more information, e.g. <access> the url to access </access>

Guidelines:
1. You should double check previous conclusions and identified facts using search from different perspectives. 
3. You can try different directions to solve the question, such as using different search queries.
3. If you find related entries in the search results, it is usually useful to access the corresponding urls to find more information.
4. The next action should follow after the thought.
5. Make sure you should choose only one action.

Current Time: Today is {current_date}

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""

    THINK_AND_ANSWER_PROMPT =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question and the history context, generate the thought as well as the final answer. The completed thought should contain detailed analysis of available information. Enclose the thought within <thought> </thought> tags, and the answer within <answer> </answer> tags.

Guideline:
1. Determine the answer based on the the available information.
2. Try to make your best guess if the found information is not enough.


Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Thought: ... // the thought to be completed

Final Answer: ... // the final answer
"""
    READ_PAGE_PROMPT =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question, the history context, and the current web page, generate a thought after reading the webpage. The completed thought should contain information found related to the question, relevant links from the current webpage, and detailed analysis of available information. Enclose the thought within <thought> </thought> tags. 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Current webpage:
```txt
{content}
```

Thought: ... // the thought to be completed
"""
    READ_SEARCH_RESULTS_PROMPT =  \
"""Given a question, you are an autonomous agent trying to solve the question with web browser. Given the question, the history context, and the search results of the latest query, generate a thought after reading the search results. The completed thought should contain information found related to the question, relevant links from the latest search results that may help solve the question, and detailed analysis of available information. Enclose the thought within <thought> </thought> tags. 

Question:
```txt
{question}
```

Reasoning history:
```txt
{history}
```

Latest search results:
```txt
{content}
```

Thought: ... // the thought to be completed
"""

def process_webpage(content):
    keys = [("title", "title"), ("p", "p"), ("li", "li", lambda c: "\n" not in c)] 
    content_list = []
    init_length = len(content)
    while any([f"<{k[0]}" in content and f"</{k[1]}>" in content for k in keys]):
        klr = []
        for k in keys:
            start = 0
            while True:
                ls = [content[start:].find(f"<{k[0]}{c}") for c in [">", " "]]
                ls = [l for l in ls if l != -1]
                l = -1 if len(ls) == 0 else min(ls)
                if l == -1:
                    break
                l += start
                r = content[l:].find(f"</{k[1]}>")
                if r == -1:
                    break
                if (len(k) <= 2) or (len(k) >= 3 and k[2](content[l:l+r])):
                    klr.append((k, l, l+r))
                    break
                start = l + r

        if len(klr) == 0:
            break
        klr = sorted(klr, key=lambda x:x[1])
        k, l, r = klr[0]
        content_list.append(content[l:r+len(f"</{k[1]}>")])
        if k[0] == "p":
            content_list[-1] += "\n\n"
        elif k[0] == "li":
            content_list[-1] += "\n"
        content = content[r:]
    content = "".join(content_list)
    return content

class AsearcherReasoningAgent:
    
    def __init__(self,
                 max_turns: int = 128,
                 force_turns: int = 32,
                 topk: int = 10,
                 force_valid: bool = True):

        self.max_turns = max_turns
        self.force_turns = force_turns
        self.force_valid = force_valid
        self.topk = topk

        self.stop = ["<|im_end|>", "<|endoftext|>"]
        self.stop_sequences = self.stop
        
        self.current_process = None
        self.tokenizer = None
        
        # Agent initialized

    def get_query_from_text(self, text: str) -> Optional[str]:
        pattern = r'<search>(.*?)</search>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<search>" + matches[-1].strip() + "</search>"
        
        return None
    
    def get_url_from_text(self, text: str) -> Optional[str]:
        pattern = r'<access>(.*?)</access>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<access>" + matches[-1].strip() + "</access>"
        
        return None
        
    def get_thought_from_text(self, text: str) -> Optional[str]:
        pattern = r'<thought>(.*?)</thought>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<think>" + matches[-1].strip() + "</think>"
        
        return None

    def get_answer_from_text(self, text: str) -> Optional[str]:
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "<answer>" + matches[-1].strip() + "</answer>"
        
        return None


    def all_finished(self, processes: List[Dict]) -> bool:
        finished = []
        for process in processes:
            finished.append(not process.get("running", True))
        return all(finished)

    def initialize_with_prompt(self, process):
        """Initialize agent with a specific prompt"""
        if "question" not in process:
            process["question"] = process["prompt"]
        if "prompt" not in process:
            process["prompt"] = process["question"]
        if len(process["history"]) == 0:
            process["history"] = [dict(type="prompt", text=process["prompt"])]
            process["running"] = True
            process["phase"] = "search"
        self.current_process = copy.deepcopy(process)
    
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
        
        # Check action count limits
        action_count = len([h for h in self.current_process["history"] if h["type"] == "act"])
        max_turns_exceeded = action_count >= self.max_turns + 20
        
        # Check failure count
        llm_gen_fail = self.current_process.get("llm_gen_fail", 0)
        too_many_failures = llm_gen_fail > 32
        
        return has_answer or max_turns_exceeded or too_many_failures

    def prepare_llm_query(self):
        """Prepare LLM query for current process"""
        if not self.current_process:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")
        
        process = self.current_process
        
        if not process.get("running", False):
            return "", {"stop": self.stop}
        
        # Handle reading mode - when we have info_str but no text
        if "text" not in process["history"][-1] and "info_str" in process["history"][-1]:
            history = ""
            for idx, h in enumerate(process["history"][:-1]):
                history += h.get("short_info_str", h.get("text", ""))
            if len(history) > 25000:
                history = history[-25000:]
            
            if process["history"][-1]["type"] == "page":
                prompt = ASearcherReasoningPrompts.READ_PAGE_PROMPT.format(
                    question=process.get("question", process["prompt"]), 
                    history=history, 
                    content=process["history"][-1]["info_str"]
                )
            elif process["history"][-1]["type"] == "documents":
                prompt = ASearcherReasoningPrompts.READ_SEARCH_RESULTS_PROMPT.format(
                    question=process.get("question", process["prompt"]), 
                    history=history, 
                    content=process["history"][-1]["info_str"]
                )
            else:
                raise RuntimeError(f"Not supported history type: {process['history'][-1]['type']}")
            
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            query_len = self.tokenizer([input_text], return_length=True)['length'][0]

            if query_len <= 28000:
                print(f"Reading @ Qid {process['id']}", query_len, flush=True)
                sampling_params = {"stop": self.stop, "max_new_tokens": 31000-query_len}
                # sampling_params = {"max_completion_tokens": 31000 - query_len}
                # return messages, sampling_params
                return input_text, sampling_params
            
            if "cache_gen_text" in process:
                process.pop("cache_gen_text")
        
        # Handle normal generation mode - building prompt from history
        history = ""
        for idx, h in enumerate(process["history"]):
            history += h.get("short_info_str", h.get("text", ""))
        if len(history) > 25000:
            history = history[-25000:]
        
        # Determine if we should force answer generation
        action_count = len([h for h in process["history"] if h["type"] == "act"])
        doc_count = len([h for h in process["history"] if h["type"] == "documents"])
        should_answer = any([
            doc_count >= 20,
            action_count >= self.force_turns,
            process.get("phase", "search") == "answer"
        ])
        
        if should_answer:
            process["phase"] = "answer"
            prompt = ASearcherReasoningPrompts.THINK_AND_ACT_PROMPT_v1.format(
                question=process.get("question", process["prompt"]), 
                history=history,
                current_date=datetime.now().strftime("%Y.%m.%d")
            )
        else:
            prompt = ASearcherReasoningPrompts.THINK_AND_ACT_PROMPT.format(
                question=process.get("question", process["prompt"]), 
                history=history,
                current_date=datetime.now().strftime("%Y.%m.%d")
            )
        
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False) + process.get("cache_gen_text", "")
        
        # Apply force_valid logic
        if self.force_valid:
            input_text = input_text.replace(
                '4. If you find information contradicting context of the question, you should point out that the question is invalid and the incorrect information in the question.', 
                "4. You should find the most likely answer even when conflicting information is founded."
            )

        # Check if process should be terminated
        input_len = len(self.tokenizer(input_text, add_special_tokens=False)["input_ids"])
        if input_len > 32000 or self.get_answer_from_text(process["history"][-1].get("text", "")):
            print(f"Process done (input too long or has answer): {process['id']}")
            process["running"] = False
            return "", {"stop": self.stop}
        
        query_len = self.tokenizer([input_text], return_length=True)['length'][0]
        max_new_tokens = max(0, 31000 - query_len)
        
        print(f"Generate {'Answer' if should_answer else 'Act'} @ Qid {process['id']}", 
              input_len, doc_count, action_count, max_new_tokens, flush=True)
        
        sampling_params = {"stop": self.stop, "max_new_tokens": max_new_tokens}
        return input_text, sampling_params

    def consume_llm_response(self, resp, completion_text):
        """Consume LLM response and extract tool calls"""
        if not self.current_process:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        process = self.current_process
        
        # Handle different response formats
        if hasattr(resp, 'stop_reason') and hasattr(resp, 'text'):
            generated_text = resp.text
        elif isinstance(resp, dict):
            generated_text = resp.get('text', str(resp))
        else:
            generated_text = completion_text
        
        if generated_text is None:
            generated_text = ""
        
        raw_generated_text = generated_text
        generated_text = process.get("cache_gen_text", "") + generated_text
        
        # Return tool calls for V2 interface
        tool_calls = []

        # Extract different components
        extracted_thought = self.get_thought_from_text(generated_text)
        extracted_answer = self.get_answer_from_text(generated_text)
        extracted_query = self.get_query_from_text(generated_text)
        extracted_url = self.get_url_from_text(generated_text)

        if process.get("phase", "unknown") != "answer" and extracted_answer is not None:
            print(f"Not time for producing answer for {process['id']}", extracted_answer, flush=True)
            extracted_answer = None
        
        # Build think_and_act text
        think_and_act = ""
        if extracted_thought is not None:
            think_and_act = think_and_act + extracted_thought
        for act in [extracted_query, extracted_url, extracted_answer]:
            if act is not None:
                think_and_act = think_and_act.strip() + "\n\n" + act
                break
        
        # Update process history if we have a thought
        if extracted_thought is not None:
            process["history"].append(dict(
                type="act", 
                full_reasoning_text=generated_text,
                text=think_and_act.strip()
            ))
            if "cache_gen_text" in process:
                process.pop("cache_gen_text")

            # tool calls
            if extracted_query:
                tool_calls.append(extracted_query)
            if extracted_url:
                tool_calls.append(extracted_url)
            if extracted_answer:
                tool_calls.append(extracted_answer)
                    
            # Handle page cache
            if "page_cache" in process and len(process["page_cache"]) > 0:
                page = process["page_cache"].pop(0)
                print(f"{process['id']} pop page cache: {[page[:100]]}")
                info_str = "\n\n<information>" + page + "\n</information>\n\n"
                short_info_str = "\n\n<information>\n" + page[:100] + "...\n\n" + "</information>\n\n"

                process["history"].append(dict(
                    type="page", 
                    info_str=info_str,
                    short_info_str=short_info_str
                ))
        elif len(raw_generated_text) == 0:
            process["cache_gen_text"] = ""
            process["llm_gen_fail"] = process.get("llm_gen_fail", 0) + 1
            if process["llm_gen_fail"] > 16:
                print("process is done (2)", process["id"], process["llm_gen_fail"])
                process["running"] = False
        else:
            if process["history"][-1]["type"] in ["page", "documents"]:
                process["cache_gen_text"] = ""
                process["history"].append(dict(
                    type="act", 
                    full_reasoning_text=generated_text,
                    text="<think>\n\n</think>"
                ))
                process["llm_gen_fail"] = process.get("llm_gen_fail", 0) + 1
                process["page_cache"] = []
            else:
                process["llm_gen_fail"] = process.get("llm_gen_fail", 0) + 1
                process["cache_gen_text"] = generated_text
        
        # Check termination conditions
        action_count = len([h for h in process["history"] if h["type"] == "act"])
        if action_count >= self.max_turns + 20 or "<answer>" in think_and_act:
            print("process is done (3)", process["id"], action_count, self.max_turns, "<answer>" in think_and_act, flush=True)
            process["running"] = False
        
        return tool_calls

    def consume_tool_response(self, res, topk=5):
        """Consume tool response (search or access)"""
        if not self.current_process:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        process = self.current_process
        
        if res["type"] == "search":
            # Handle search results
            documents = res.get("documents", [])[:topk]
            urls = res.get("urls", [])[:topk]
            
            print(f"Count of Search documents: {len(documents)}")

            if len(documents) > 0:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                info_str = "\n\n<information>\n" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + doc for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>\n\n"
                short_info_str = "\n\n<information>" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + doc + "..." for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>\n\n"
            else:
                info_str = "\n\n<information>\n" + "No Results Found." + "\n</information>\n\n"
                short_info_str = info_str

            process["history"].append(dict(
                type="documents", 
                info_str=info_str,
                short_info_str=short_info_str
            ))
            
        elif res["type"] == "access":
            # Handle webpage access results
            page = res.get("page", "")
            
            if page and len(page) > 0:
                page = page[:250000]
                if "page_cache" not in process:
                    process["page_cache"] = []
                process["page_cache"] = []
                
                # Split page into chunks
                while len(page) > 0 and len(process["page_cache"]) < 10:
                    _len = min(10000, len(page))
                    process["page_cache"].append(f">>>> Page {len(process["page_cache"]) + 1} >>>>\n\n" + page[:_len])
                    page = page[_len:]
                
                print("[DEBUG] add page", process["id"], len(res.get("page", "")), len(process["page_cache"]), flush=True)
                
                # Add first page immediately if available
                if "page_cache" in process and len(process["page_cache"]) > 0:
                    page = process["page_cache"].pop(0)
                    info_str = "\n\n<information>" + page + "\n</information>\n\n"
                    short_info_str = "\n\n<information>\n" + page[:100] + "...\n\n" + "</information>\n\n"

                    process["history"].append(dict(
                        type="page", 
                        info_str=info_str,
                        short_info_str=short_info_str
                    ))
            else:
                # Empty or invalid page
                process["page_cache"] = []
                info_str = "\n\n<information>\nNo More Information is Found for this URL.\n</information>\n\n"
                short_info_str = info_str

                process["history"].append(dict(
                    type="page", 
                    info_str=info_str,
                    short_info_str=short_info_str
                ))

    def get_answer(self):
        """Get final answer from current process"""
        if not self.current_process:
            return None
            
        process = self.current_process
        
        if "pred_answer" not in process:
            full_text = "".join(
                [h["text"] for h in process["history"] if h["type"] != "prompt" and "text" in h]
            )
            
            if "<answer>" in full_text and "</answer>" in full_text:
                answer = full_text.split("<answer>")[-1].split("</answer>")[0].strip()
            else:
                reasoning_text = "\n\n".join([h["full_reasoning_text"] for h in process["history"] if "full_reasoning_text" in h] + [process.get("cache_gen_text", "")])
                # find the last line mentioning 'answer'
                lines = reasoning_text.split("\n")
                lines = [l for l in lines if 'answer' in l.lower()]
                if len(lines) > 0:
                    answer = lines[-1]
                else:
                    answer = reasoning_text.strip().split("</think>")[-1].strip()
            
            process["pred_answer"] = answer
        
        return process["pred_answer"]
