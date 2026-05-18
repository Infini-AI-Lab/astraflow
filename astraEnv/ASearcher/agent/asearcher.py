import queue
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

@dataclass
class Record:
    type: str # prompt/llm_gen/search_results/webpage
    text: str
    # for webpage and search results
    short_text: str = ""
    # RL data
    input_len: Optional[int] = None
    input_tokens: Optional[List[int]] = None
    output_len: Optional[int] = None
    output_tokens: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    output_versions: Optional[List[int]] = None

    def to_dict(self):
        return asdict(self)

class AgentMemory:
    def __init__(self, prompt):
        self.memory = [Record(type="prompt", text=prompt)]
    
    def llm_gen_count(self):
        return sum([r.type == "llm_gen" for r in self.memory])
    
    def filter_records(self, record_type):
        return [r for r in self.memory if r.type == record_type]
    
    def prepare_prompt(self):
        prompt = ""
        for r in self.memory:
            if r.type == "prompt":
                prompt = r.text
            elif r.type in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + r.short_text + "\n<think>\n"
            elif r.type == "llm_gen":
                prompt = prompt + r.text
            else:
                raise RuntimeError(f"Unknown record type: {r.type}")
        return prompt
    
    def add_record(self, r: Record):
        self.memory.append(r)
    
    def logging_stats(self) -> Dict:
        llm_gens = self.filter_records(record_type="llm_gen")
        search_results = self.filter_records(record_type="search_results")
        webpages = self.filter_records(record_type="webpage")
        ret = dict(
            num_llm_gens=len(llm_gens),
            num_input_tokens=sum([len(r.input_tokens) for r in llm_gens if r.input_tokens is not None]),
            num_output_tokens=sum([len(r.output_tokens) for r in llm_gens if r.output_tokens is not None]),
            num_search_queries=len(search_results),
            num_success_search_queries=len([r for r in search_results if "No search results are found" not in r.text]),
            num_failed_search_queries=len([r for r in search_results if "No search results are found" in r.text]),
            num_pages=len(webpages),
            num_success_url_accesses=len([r for r in webpages if ">>>> Page 1 >>>>" in r.text]),
            num_failed_url_accesses=len([r for r in webpages if ">>>> Page 1 >>>>" not in r.text]),
        )
        return ret
    
    def to_dict(self):
        return [r.to_dict() for r in self.memory]

class AsearcherAgent:
    def __init__(self, prompt=None):
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt) if prompt else None
        self.job_queue = queue.Queue(128)
        self.max_turns = 64  # Default max turns like other agents
        
    def initialize_with_prompt(self, process):
        """Initialize or reset agent with a specific prompt"""
        prompt = process["prompt"]
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt)
        self.job_queue = queue.Queue(128)
    
    @property
    def num_turns(self):
        return self.memory.llm_gen_count() if self.memory else 0
    
    @property
    def is_finished(self):
        if not self.memory:
            return False
        pattern = r'<answer>(.*?)</answer>'
        return any([len(re.findall(pattern, r.text, re.DOTALL)) > 0 for r in self.memory.filter_records("llm_gen")])
    
    def add_jobs(self, jobs):
        if not isinstance(jobs, list):
            jobs = [jobs]
        for job in jobs:
            assert (job.get("type", "unkown") in ["search_results", "webpage"]), ("Unknown job type: " + job.get("type", "unknown"))
            self.job_queue.put_nowait(job)
    
    def prepare_llm_query(self):
        if not self.memory:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
        
        prompt = self.memory.prepare_prompt()
        sampling_params = dict(stop=["</search>", "</access>", "</answer>"])
        if not self.job_queue.empty():
            job = self.job_queue.get_nowait()
            if job["type"] in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + job["text"] + "\n<think>\n"
                new_record = Record(
                    type=job["type"], 
                    text=job["text"], 
                    short_text=job.get("short_text", job["text"]),
                )
                self.memory.add_record(new_record)
                sampling_params["stop"] = ["</think>"]
        return prompt, sampling_params
    
    def consume_llm_response(self, resp, completion_text):
        if not self.memory:
            raise RuntimeError("Agent not initialized with prompt. Call initialize_with_prompt() first.")
            
        new_record = Record(
            type="llm_gen",
            text=completion_text,
            input_len=resp.input_len,
            output_len=resp.output_len,          
        )
        self.memory.add_record(new_record)

        tool_calls = []
        for pattern in [r'<search>(.*?)</search>', r'<access>(.*?)</access>', r'<answer>(.*?)</answer>']:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                match = matches[-1]
                tool_calls.append(str(pattern.replace('(.*?)', match)))

        return tool_calls

    def consume_tool_response(self, res, topk=5):
        # process the search results
        if res["type"] == "search":
            job = dict(type="search_results")

            # Safely handle potentially None documents and urls
            documents = res.get("documents") or []
            urls = res.get("urls") or []
            
            # Ensure we slice safely
            documents = documents[:topk] if documents else []
            urls = urls[:topk] if urls else []

            if len(documents) > 0:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                text = "<information>\n" + "\n\n".join([doc_id_template.format(doc_id=str(k+1), url=url) + doc[:5000] for k, (doc, url) in enumerate(zip(documents, urls))]) + "\n</information>"
            else:
                text = "<information>\nNo search results are found.\n</information>"

            job["text"] = text 
            self.add_jobs(job)
        
        # process the webpage
        elif res["type"] == "access":
            jobs = []          
            page = res["page"]
            if page is not None and page.strip() != "":
                page = page[:250000]
                while len(page) > 0 and len(jobs) < 10:
                    _len = min(25000, len(page))
                    jobs.append(dict(
                        type="webpage",
                        text=f"<information>\n>>>> Page {len(jobs) + 1} >>>>\n\n" + page[:_len] + "\n</information>",
                        short_text=f"<information>\n>>>> Page {len(jobs) + 1} >>>>\n\n" + page[:100] + "\n</information>",
                    ))
                    page = page[_len:]
            else:
                jobs.append(dict(
                    type="webpage",
                    text="<information>\nNo More Information is Found for this URL.\n</information>",
                ))
            self.add_jobs(jobs)

    def get_answer(self):
        if not self.memory:
            return None
        text, _ = self.prepare_llm_query()
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None