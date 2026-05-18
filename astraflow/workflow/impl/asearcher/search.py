"""Search backends and toolbox for the ASearcher workflow."""

from __future__ import annotations

import atexit
import asyncio
from collections import OrderedDict
import glob
import hashlib
import json
import os
import random
import threading
import time
from typing import Any

import aiohttp

from astraflow.workflow.utils import logging

from .reward import compute_score_em, compute_score_f1

logger = logging.getLogger("ASearcherSearch")
SERPER_STATS = {"num_requests": 0}


class WebPageCache:
    def __init__(
        self,
        max_size: int = 100000,
        cache_file: str = "./webpage_cache.json",
        save_interval: int = 10,
    ):
        self.max_size = max_size
        self.cache_file = cache_file
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.save_interval = save_interval
        self.operations_since_save = 0

        self.load_from_file()
        atexit.register(self.save_to_file)

    def _generate_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def put(self, url: str, content: str) -> None:
        if not url or not content:
            return
        cache_key = self._generate_cache_key(url)
        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
            self.cache[cache_key] = {"url": url, "content": content, "timestamp": time.time()}
            self.operations_since_save += 1
            if self.operations_since_save >= self.save_interval:
                self.operations_since_save = 0
                threading.Thread(target=self._background_save, daemon=True).start()

    def get(self, url: str) -> str | None:
        cache_key = self._generate_cache_key(url)
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache.pop(cache_key)
                self.cache[cache_key] = entry
                self.stats["hits"] += 1
                return str(entry["content"])
            self.stats["misses"] += 1
            return None

    def has(self, url: str) -> bool:
        cache_key = self._generate_cache_key(url)
        with self.lock:
            return cache_key in self.cache

    def _background_save(self) -> None:
        try:
            self.save_to_file()
        except Exception as exc:
            logger.warning(f"WebPageCache background save failed: {exc}")

    def save_to_file(self) -> None:
        try:
            with self.lock:
                cache_data = {
                    "cache_ordered": list(self.cache.items()),
                    "stats": self.stats,
                    "max_size": self.max_size,
                    "saved_at": time.time(),
                }
            with open(self.cache_file, "w", encoding="utf-8") as handle:
                json.dump(cache_data, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"WebPageCache save failed for {self.cache_file}: {exc}")

    def load_from_file(self) -> None:
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as handle:
                cache_data = json.load(handle)
            with self.lock:
                self.cache = OrderedDict(cache_data.get("cache_ordered", []))
                self.stats = cache_data.get("stats", self.stats)
                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                    self.stats["evictions"] += 1
        except Exception as exc:
            logger.warning(f"WebPageCache load failed for {self.cache_file}: {exc}")
            with self.lock:
                self.cache = OrderedDict()
                self.stats = {"hits": 0, "misses": 0, "evictions": 0}


class AsyncSearchBrowserClient:
    def __init__(self):
        self.server_list = self.get_server_list()
        if not self.server_list:
            rag_dir = os.environ.get("RAG_SERVER_ADDR_DIR", "")
            raise RuntimeError(
                "No RAG servers found for async-search-access. "
                f"RAG_SERVER_ADDR_DIR={rag_dir!r}, cwd={os.getcwd()!r}, "
                f"pattern={rag_dir + '/Host*_IP*.txt'!r}"
            )
        self.server_addr = random.choice(self.server_list)

    def get_server_list(self) -> list[str]:
        rag_server_addr_dir = os.environ.get("RAG_SERVER_ADDR_DIR", "")
        server_list: list[str] = []
        for filename in glob.glob(rag_server_addr_dir + "/Host*_IP*.txt"):
            try:
                with open(filename, encoding="utf-8") as handle:
                    server_list.extend([line.strip() for line in handle.readlines() if line.strip()])
            except OSError:
                continue
        return server_list

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict[str, Any]]:
        last_exception: Exception | None = None
        for _ in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=30, sock_connect=30),
                    ) as response:
                        response.raise_for_status()
                        payload = await response.json()
                        return [
                            dict(
                                documents=[result["contents"] for result in item],
                                urls=[result["url"] for result in item],
                                server_type="async-search-browser",
                            )
                            for item in payload["result"]
                        ]
            except Exception as exc:
                last_exception = exc
                self.server_list = self.get_server_list()
                if not self.server_list:
                    raise RuntimeError(
                        "RAG server list became empty while retrying query_async. "
                        f"RAG_SERVER_ADDR_DIR={os.environ.get('RAG_SERVER_ADDR_DIR', '')!r}"
                    ) from exc
                self.server_addr = random.choice(self.server_list)
                await asyncio.sleep(10)
        raise RuntimeError("Fail to post search query to RAG server") from last_exception

    async def access_async(self, urls: list[str]) -> list[dict[str, Any]]:
        last_exception: Exception | None = None
        for _ in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/access",
                        json={"urls": urls},
                        timeout=aiohttp.ClientTimeout(total=30, sock_connect=30),
                    ) as response:
                        response.raise_for_status()
                        payload = await response.json()
                        return [
                            dict(
                                page=item["contents"] if item is not None else "",
                                type="access",
                                server_type="async-search-browser",
                            )
                            for item in payload["result"]
                        ]
            except Exception as exc:
                last_exception = exc
                self.server_list = self.get_server_list()
                if not self.server_list:
                    raise RuntimeError(
                        "RAG server list became empty while retrying access_async. "
                        f"RAG_SERVER_ADDR_DIR={os.environ.get('RAG_SERVER_ADDR_DIR', '')!r}"
                    ) from exc
                self.server_addr = random.choice(self.server_list)
                await asyncio.sleep(10)
        raise RuntimeError("Fail to post access request to RAG server") from last_exception


class AsyncOnlineSearchClient:
    _search_semaphore: asyncio.Semaphore | None = None
    _access_semaphore: asyncio.Semaphore | None = None

    @classmethod
    def _get_search_semaphore(cls) -> asyncio.Semaphore:
        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore

    @classmethod
    def _get_access_semaphore(cls) -> asyncio.Semaphore:
        if cls._access_semaphore is None:
            cls._access_semaphore = asyncio.Semaphore(10)
        return cls._access_semaphore

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 10000,
        cache_file: str = "../webpage_cache.json",
        use_jina: bool = False,
        jina_api_key: str | None = None,
        wrapper_format: bool = True,
    ):
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        if not self.serper_api_key:
            raise RuntimeError(
                "Serper API key is not set. Please configure it in config.yaml "
                "or set the SERPER_API_KEY environment variable."
            )
        self.serper_headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        self.wrapper_format = wrapper_format
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key or os.environ.get("JINA_API_KEY", "")
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError(
                "Jina is enabled but the API key is not set. Please configure it "
                "in config.yaml or set the JINA_API_KEY environment variable."
            )
        self.webpage_cache = (
            WebPageCache(cache_size, cache_file, save_interval=5) if enable_cache else None
        )

    async def _jina_readpage_async(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "Content-Type": "application/json",
            }
            async with session.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    return await response.text()
                return f"[visit] Failed to read page. Status code: {response.status}"
        except Exception as exc:
            return f"[visit] Failed to read page. Error: {exc}"

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)
        if not queries:
            return []

        async def single_serper_query_async(
            session: aiohttp.ClientSession,
            query: str,
            topk_value: int,
        ) -> dict[str, Any]:
            query = query[:2000]
            async with self._get_search_semaphore():
                payload = {"q": query, "num": topk_value}
                for attempt in range(4):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
                        await asyncio.sleep(0.1)
                        SERPER_STATS["num_requests"] += 1
                        async with session.post(
                            f"{self.serper_server_addr}/search",
                            headers=self.serper_headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as response:
                            if response.status == 200:
                                return {"success": True, "data": await response.json()}
                            response_text = await response.text()
                            if attempt == 3:
                                return {
                                    "success": False,
                                    "error": f"HTTP {response.status}: {response_text[:100]}",
                                }
                    except Exception as exc:
                        if attempt == 3:
                            return {
                                "success": False,
                                "error": f"{type(exc).__name__}: {str(exc)[:100]}",
                            }
                return {"success": False, "error": "Unknown error after all retries"}

        async with aiohttp.ClientSession() as session:
            serper_results = await asyncio.gather(
                *[single_serper_query_async(session, query, topk) for query in queries]
            )

        formatted_results: list[list[dict[str, Any]]] = []
        for query, serper_result in zip(queries, serper_results):
            query_results: list[dict[str, Any]] = []
            if serper_result and serper_result.get("success", False):
                organic_results = serper_result.get("data", {}).get("organic", [])[:topk]
                for result in organic_results:
                    query_results.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "snippet": result.get("snippet", ""),
                            "server_type": "async-online-search",
                        }
                    )
            else:
                logger.warning(
                    "AsyncOnlineSearchClient query failed for %r: %s",
                    query,
                    serper_result.get("error", "Unknown error") if serper_result else "No response",
                )
            formatted_results.append(query_results)

        if self.wrapper_format:
            first_query_results = formatted_results[0] if formatted_results else []
            return [
                {
                    "documents": [
                        result.get("title", "") + " " + result.get("snippet", "")
                        for result in first_query_results
                    ],
                    "urls": [result.get("url", "") for result in first_query_results],
                    "server_type": "async-online-search",
                }
            ]
        if len(queries) == 1:
            return formatted_results[0]
        return formatted_results

    async def access_async(self, urls: list[str]) -> list[dict[str, Any]]:
        if not urls:
            return []

        results: list[dict[str, Any] | None] = []
        urls_to_fetch: list[str] = []
        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access"))
                    continue
            urls_to_fetch.append(url)
            results.append(None)

        if urls_to_fetch:
            if self.use_jina and self.jina_api_key:
                try:
                    async with self._get_access_semaphore():
                        fetched_results = await self._access_urls_jina_async(urls_to_fetch)
                    fetch_index = 0
                    for idx, result in enumerate(results):
                        if result is None:
                            if fetch_index < len(fetched_results):
                                fetched_result = fetched_results[fetch_index]
                                results[idx] = fetched_result
                                if self.webpage_cache and fetched_result.get("page"):
                                    self.webpage_cache.put(urls[idx], str(fetched_result["page"]))
                                fetch_index += 1
                            else:
                                results[idx] = dict(page="", type="access")
                except Exception:
                    for idx, result in enumerate(results):
                        if result is None:
                            results[idx] = dict(page="", type="access")
            else:
                for idx, result in enumerate(results):
                    if result is None:
                        results[idx] = dict(page="", type="access")

        final_results = [result for result in results if result is not None]
        for result in final_results:
            result["server_type"] = "async-online-search"
        return final_results

    async def _access_urls_jina_async(self, urls: list[str]) -> list[dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                results: list[dict[str, Any]] = []
                for url in urls:
                    content = await self._jina_readpage_async(session, url)
                    if content and not content.startswith("[visit] Failed"):
                        results.append(dict(page=content, type="access"))
                    else:
                        results.append(dict(page="", type="access"))
        except Exception:
            results = [dict(page="", type="access") for _ in urls]
        for result in results:
            if len(result["page"]) > 0:
                result["type"] = "jina"
        return results


def make_search_client(
    search_client_type: str,
    use_jina: bool = False,
    jina_api_key: str | None = None,
) -> AsyncSearchBrowserClient | AsyncOnlineSearchClient:
    if search_client_type == "async-search-access":
        return AsyncSearchBrowserClient()
    if search_client_type in ["async-online-search", "async-online-search-access"]:
        return AsyncOnlineSearchClient(
            use_jina=use_jina,
            jina_api_key=jina_api_key,
            wrapper_format=(search_client_type == "async-online-search-access"),
        )
    raise ValueError(f"Unknown search_client_type: {search_client_type}")


def load_metadata(dataset_path: str) -> dict[str, dict[str, Any]]:
    with open(dataset_path, encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]
    result: dict[str, dict[str, Any]] = {}
    # Derive dataset_name from path stem so the AstraFlow-format keys
    # match attach_query_ids() output. attach_query_ids defaults the
    # dataset_name to the loader argument; for asearcher the loader
    # passes "asearcher", which is also the recipe's workflow_cls.
    # Falling back to "asearcher" matches the production layout.
    dataset_name = "asearcher"
    for idx, item in enumerate(data):
        if "idx" in item:
            item["idx"] = str(item["idx"])
        elif "qid" in item:
            item["idx"] = str(item["qid"])
        elif "id" in item:
            item["idx"] = str(item["id"])
        elif "_id" in item:
            item["idx"] = str(item["_id"])
        elif "query_id" in item:
            item["idx"] = str(item["query_id"])
        else:
            item["idx"] = str(idx)
        # Key by the dataset's own id field
        result[item["idx"]] = item
        # Also key by the AstraFlow-format prompt ID (row-position based,
        # matches attach_query_ids: f"{dataset_name}-{i:08d}"). Without
        # this, the workflow's lookup `id2info[qid.split("@")[0]]` raises
        # KeyError for every task because qid is the AstraFlow ID, not
        # the dataset's qid field.
        result[f"{dataset_name}-{idx:08d}"] = item
    return result


class SearchToolBox:
    def __init__(
        self,
        dataset_path: str,
        reward_type: str = "F1",
        topk: int = 10,
        search_client_type: str = "async-online-search-access",
        use_jina: bool = False,
    ):
        self.id2info = load_metadata(dataset_path)
        self.reward_type = reward_type
        self.topk = topk
        self.use_jina = use_jina
        self.search_client_type = search_client_type
        self.search_client = make_search_client(search_client_type, use_jina=self.use_jina)

    async def step(self, qid_actions: tuple[str, list[str]]) -> list[dict[str, Any]]:
        qid, actions = qid_actions
        results: list[dict[str, Any]] = []
        for action in actions:
            result: dict[str, Any] = dict(documents=None, score=None, ground_truth=None, type=None)
            if "<search>" in action and "</search>" in action:
                query = action.split("<search>")[-1].split("</search>")[0].strip()
                response = await self.search_client.query_async(
                    {"queries": [query], "topk": self.topk, "return_scores": False}
                )
                result["documents"] = response[0]["documents"]
                result["urls"] = response[0]["urls"]
                result["type"] = "search"
            elif "<access>" in action and "</access>" in action:
                url = action.split("<access>")[-1].split("</access>")[0].strip()
                response = await self.search_client.access_async([url])
                page = None
                if self.search_client_type == "async-online-search-access":
                    if self.use_jina:
                        page = response[0].get("page", "")
                    else:
                        page = self.process_webpage(response[0].get("page", ""))
                elif self.search_client_type == "async-search-access":
                    page = response[0].get("page", "")
                result["page"] = page
                result["type"] = "access"

            item = self.id2info[qid.split("@")[0]]
            ground_truth: str | list[str]
            if isinstance(item["answer"], (list, tuple)):
                ground_truth = [str(value) for value in item["answer"]]
            else:
                ground_truth = str(item["answer"])

            ground_truth_aug: str | list[str] | None = None
            if "aug_answer" in item and len(item["aug_answer"]) > 0:
                if isinstance(item["aug_answer"], (list, tuple)):
                    ground_truth_aug = [str(value) for value in item["aug_answer"]]
                else:
                    ground_truth_aug = str(item["aug_answer"])

            if self.reward_type == "F1":
                extracted, score = compute_score_f1(action, ground_truth, method="strict")
            elif self.reward_type == "EM":
                extracted, score = compute_score_em(action, ground_truth, method="strict")
            else:
                raise ValueError(f"Unknown reward_type: {self.reward_type}")

            if ground_truth_aug is not None:
                if self.reward_type == "F1":
                    _, score_aug = compute_score_f1(action, ground_truth_aug, method="strict")
                else:
                    _, score_aug = compute_score_em(action, ground_truth_aug, method="strict")
                result["score"] = score * 0.7 + max(score_aug, score) * 0.3
                result["ground_truth_aug"] = ground_truth_aug
            else:
                result["score"] = score

            result["extracted"] = extracted
            result["ground_truth"] = item["answer"]
            results.append(result)
        return results

    def process_webpage(self, content: str) -> str:
        keys = [
            ("title", "title"),
            ("p", "p"),
            ("li", "li", lambda chunk: "\n" not in chunk),
            ("td", "td"),
            ("tr", "tr"),
        ]
        content_list: list[str] = []
        init_length = len(content)
        while any(f"<{key[0]}" in content and f"</{key[1]}>" in content for key in keys):
            candidates = []
            for key in keys:
                start = 0
                while True:
                    offsets = [content[start:].find(f"<{key[0]}{suffix}") for suffix in [">", " "]]
                    offsets = [offset for offset in offsets if offset != -1]
                    left = -1 if not offsets else min(offsets)
                    if left == -1:
                        break
                    left += start
                    right = content[left:].find(f"</{key[1]}>")
                    if right == -1:
                        break
                    chunk = content[left : left + right]
                    if len(key) <= 2 or key[2](chunk):
                        candidates.append((key, left, left + right))
                        break
                    start = left + right
            if not candidates:
                break
            key, left, right = sorted(candidates, key=lambda item: item[1])[0]
            content_list.append(content[left : right + len(f"</{key[1]}>")])
            if key[0] == "p":
                content_list[-1] += "\n\n"
            elif key[0] == "li":
                content_list[-1] += "\n"
            content = content[right:]
        processed = "".join(content_list)
        logger.info("process the webpage: %s -> %s", init_length, len(processed))
        return processed
