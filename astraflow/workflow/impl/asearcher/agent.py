"""Agent state tracking for the ASearcher workflow."""

from __future__ import annotations

import queue
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Record:
    type: str
    text: str
    token_ids: list[int]
    short_text: str = ""
    input_len: int | None = None
    input_tokens: list[int] | None = None
    output_len: int | None = None
    full_token_ids: list[int] | None = None
    output_tokens: list[int] | None = None
    output_logprobs: list[float] | None = None
    output_versions: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentMemory:
    def __init__(self, prompt: str, prompt_token_ids: list[int]):
        self.memory = [Record(type="prompt", text=prompt, token_ids=prompt_token_ids)]

    def llm_gen_count(self) -> int:
        return sum(record.type == "llm_gen" for record in self.memory)

    def filter_records(self, record_type: str) -> list[Record]:
        return [record for record in self.memory if record.type == record_type]

    def prepare_prompt(self) -> str:
        prompt = ""
        for record in self.memory:
            if record.type == "prompt":
                prompt = record.text
            elif record.type in ["search_results", "webpage"]:
                prompt = prompt + "\n\n" + record.short_text + "\n<think>\n"
            elif record.type == "llm_gen":
                prompt = prompt + record.text
            else:
                raise RuntimeError(f"Unknown record type: {record.type}")
        return prompt

    def prepare_prompt_token_ids(self) -> list[int]:
        prompt_token_ids: list[int] = []
        for record in self.memory:
            prompt_token_ids += record.token_ids
        return prompt_token_ids

    def add_record(self, record: Record) -> None:
        self.memory.append(record)

    def logging_stats(self) -> dict[str, int]:
        llm_gens = self.filter_records("llm_gen")
        search_results = self.filter_records("search_results")
        webpages = self.filter_records("webpage")
        return dict(
            num_llm_gens=len(llm_gens),
            num_input_tokens=sum(len(record.input_tokens or []) for record in llm_gens),
            num_output_tokens=sum(len(record.output_tokens or []) for record in llm_gens),
            num_search_queries=len(search_results),
            num_success_search_queries=len(
                [record for record in search_results if "No search results are found" not in record.text]
            ),
            num_failed_search_queries=len(
                [record for record in search_results if "No search results are found" in record.text]
            ),
            num_pages=len(webpages),
            num_success_url_accesses=len(
                [record for record in webpages if ">>>> Page 1 >>>>" in record.text]
            ),
            num_failed_url_accesses=len(
                [record for record in webpages if ">>>> Page 1 >>>>" not in record.text]
            ),
        )

    def to_dict(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.memory]


class SearchAgent:
    def __init__(self, prompt: str, prompt_token_ids: list[int]):
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt, prompt_token_ids=prompt_token_ids)
        self.summary_job_queue: queue.Queue[dict[str, Any]] = queue.Queue(128)

    @property
    def num_turns(self) -> int:
        return self.memory.llm_gen_count()

    @property
    def is_finished(self) -> bool:
        pattern = r"<answer>(.*?)</answer>"
        return any(
            len(re.findall(pattern, record.text, re.DOTALL)) > 0
            for record in self.memory.filter_records("llm_gen")
        )

    def add_summary_jobs(self, summary_jobs: dict[str, Any] | list[dict[str, Any]]) -> None:
        if not isinstance(summary_jobs, list):
            summary_jobs = [summary_jobs]
        for summary_job in summary_jobs:
            assert summary_job.get("type", "unknown") in ["search_results", "webpage"], (
                "Unknown summary_job type: " + summary_job.get("type", "unknown")
            )
            self.summary_job_queue.put_nowait(summary_job)

    def prepare_llm_query(self, tokenizer: Any) -> tuple[list[int], dict[str, list[str]]]:
        prompt_token_ids = self.memory.prepare_prompt_token_ids()
        sampling_params: dict[str, list[str]] = {"stop": ["</search>", "</access>", "</answer>"]}
        if not self.summary_job_queue.empty():
            summary_job = self.summary_job_queue.get_nowait()
            if summary_job["type"] in ["search_results", "webpage"]:
                full_text = "\n\n" + summary_job["text"] + "\n<think>\n"
                short_text = "\n\n" + summary_job.get("short_text", summary_job["text"]) + "\n<think>\n"
                full_token_ids, short_token_ids = tokenizer(
                    [full_text, short_text], add_special_tokens=False
                )["input_ids"]
                self.memory.add_record(
                    Record(
                        type=summary_job["type"],
                        text=full_text,
                        short_text=short_text,
                        token_ids=short_token_ids,
                        full_token_ids=full_token_ids,
                    )
                )
                prompt_token_ids += full_token_ids
                sampling_params["stop"] = ["</think>"]
        return prompt_token_ids, sampling_params

    def consume_llm_response(self, resp: Any, completion_text: str) -> list[str]:
        self.memory.add_record(
            Record(
                type="llm_gen",
                text=completion_text,
                token_ids=list(resp.output_tokens),
                input_len=resp.input_len,
                input_tokens=list(resp.input_tokens),
                output_len=resp.output_len,
                output_tokens=list(resp.output_tokens),
                output_logprobs=list(resp.output_logprobs),
                output_versions=list(resp.output_versions),
            )
        )

        tool_calls: list[str] = []
        for pattern in [r"<search>(.*?)</search>", r"<access>(.*?)</access>", r"<answer>(.*?)</answer>"]:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                tool_calls.append(str(pattern.replace("(.*?)", matches[-1])))
        return tool_calls

    def consume_tool_response(self, response: dict[str, Any], topk: int = 5) -> None:
        if response["type"] == "search":
            documents = response["documents"][:topk]
            urls = response["urls"][:topk]
            if documents:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                text = (
                    "<information>\n"
                    + "\n\n".join(
                        doc_id_template.format(doc_id=str(idx + 1), url=url) + doc[:5000]
                        for idx, (doc, url) in enumerate(zip(documents, urls))
                    )
                    + "\n</information>"
                )
            else:
                text = "<information>\nNo search results are found.\n</information>"
            self.add_summary_jobs(dict(type="search_results", text=text))
            return

        if response["type"] == "access":
            summary_jobs: list[dict[str, str]] = []
            page = response.get("page")
            if page is not None and page.strip() != "":
                page = page[:250000]
                while page and len(summary_jobs) < 10:
                    chunk_len = min(25000, len(page))
                    summary_jobs.append(
                        dict(
                            type="webpage",
                            text=(
                                f"<information>\n>>>> Page {len(summary_jobs) + 1} >>>>\n\n"
                                + page[:chunk_len]
                                + "\n</information>"
                            ),
                            short_text=(
                                f"<information>\n>>>> Page {len(summary_jobs) + 1} >>>>\n\n"
                                + page[:100]
                                + "\n</information>"
                            ),
                        )
                    )
                    page = page[chunk_len:]
            else:
                summary_jobs.append(
                    dict(
                        type="webpage",
                        text="<information>\nNo More Information is Found for this URL.\n</information>",
                    )
                )
            self.add_summary_jobs(summary_jobs)

    def get_answer(self) -> str | None:
        matches = re.findall(r"<answer>(.*?)</answer>", self.memory.prepare_prompt(), re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
