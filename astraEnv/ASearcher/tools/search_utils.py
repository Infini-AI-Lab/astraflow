import requests
import random
import time
import json
import asyncio
from pprint import pprint
import os
import sys
from typing import Dict, Any, List
from pathlib import Path

import aiohttp
import asyncio
from typing import Dict, List, Any

# 动态添加项目根目录到 Python 路径，以便正确导入 evaluation 模块
def add_project_root_to_path():
    current_file = Path(__file__).resolve()  # /path/to/project/tools/search_utils.py
    project_root = current_file.parent.parent  # /path/to/project
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

add_project_root_to_path()

try:
    from evaluation.config_loader import get_api_key
except ImportError as e:
    print(f"[WARNING] Could not import config_loader: {e}")
    print("Using environment variables for API keys...")
    def get_api_key(key_name: str) -> str:
        """Fallback function to get API keys from environment variables"""
        env_mapping = {
            'serper_api_key': 'SERPER_API_KEY',
            'openai_api_key': 'OPENAI_API_KEY',
            'jina_api_key': 'JINA_API_KEY'
        }
        env_key = env_mapping.get(key_name, key_name.upper())
        return os.environ.get(env_key, '')

try:
    from .web_browser import WebPageCache
    WEBPAGECACHE_AVAILABLE = True
except ImportError:
    try:
        from web_browser import WebPageCache
        WEBPAGECACHE_AVAILABLE = True
    except ImportError as e:
        print(f"[WARNING] import web browser error: {e}")
        WEBPAGECACHE_AVAILABLE = False
        WebPageCache = None


class AsyncSearchBrowserClient:
    def __init__(self, address, port, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None
        self.server_addr = f"http://{address}:{port}"
        # print(self.server_list)     
        
    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                documents=[r["contents"] for r in result],
                                urls=[r["url"] for r in result],
                                server_type="async-search-browser",
                            ) for result in res["result"]
                        ]
            except Exception as e:
                print("query_async", e.__class__.__name__, e.__cause__)
                last_exception = e
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
        
        raise RuntimeError("Fail to post search query to RAG server") from last_exception
        
    async def access_async(self, urls: List[str]) -> List[Dict]:
        cnt = 0
        last_exception = None        
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_addr}/access",
                        json={"urls": urls},
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                page=result["contents"] if result is not None else "",
                                type="access",
                                server_type="async-search-browser",
                            ) for result in res["result"]
                        ]
            except Exception as e:
                print("access_async", e.__class__.__name__, e.__cause__)
                last_exception = e
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
        
        raise RuntimeError("Fail to post access request to RAG server") from last_exception

class AsyncOnlineSearchClient:
    
    def __init__(self, enable_cache: bool = True, cache_size: int = 10000, cache_file: str = "../webpage_cache.json",
                 use_jina: bool = False, jina_api_key: str = None, wrapper_format: bool = True):
        # Serper API
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = get_api_key('serper_api_key') or os.environ.get('SERPER_API_KEY', '')
        if not self.serper_api_key:
            raise RuntimeError("Serper API key is not set. Please configure it in config.yaml or set the SERPER_API_KEY environment variable.")
        self.serper_headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        self.max_retries = 2
        self.wrapper_format = wrapper_format
        
        self.search_semaphore = asyncio.Semaphore(10)
        self.access_semaphore = asyncio.Semaphore(5)
        
        # Jina API
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key or get_api_key('jina_api_key') or os.environ.get('JINA_API_KEY', '')
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError("Jina is enabled but the API key is not set. Please configure it in config.yaml or set the JINA_API_KEY environment variable.")
        
        # 网页缓存
        if enable_cache and WEBPAGECACHE_AVAILABLE:
            self.webpage_cache = WebPageCache(cache_size, cache_file, save_interval=5)
        else:
            self.webpage_cache = None
    
    async def _jina_readpage_async(self, session, url: str) -> str:
        try:
            headers = {
                'Authorization': f'Bearer {self.jina_api_key}',
                'Content-Type': 'application/json',
            }
            
            async with session.get(f'https://r.jina.ai/{url}', headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    return f"[visit] Failed to read page. Status code: {response.status}"
        
        except Exception:
            return ""
    
    async def query_async(self, req_meta):

        import aiohttp
        
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)
        
        if not queries:
            return []
        
        async def single_serper_query_async(session, query: str, topk: int) -> dict:

            query = query[:2000]
            async with self.search_semaphore:
                payload = {
                    "q": query,
                    "num": topk
                }
                

                for attempt in range(3):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(1.0) 
                        
                        async with session.post(
                            f"{self.serper_server_addr}/search",
                            headers=self.serper_headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=20)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                return {"success": True, "data": data}
                            elif attempt == 2: 
                                return {"success": False, "error": f"HTTP {response.status}"}
                            
                    except Exception as e:
                        if attempt == 2: 
                            return {"success": False, "error": str(e)}
                
                return {"success": False, "error": "Request failed"}
        
        async with aiohttp.ClientSession() as session:
            tasks = [single_serper_query_async(session, query, topk) for query in queries]
            serper_results = await asyncio.gather(*tasks)
        
        formatted_results = []
        for query, serper_result in zip(queries, serper_results):
            query_results = []
            
            if serper_result and serper_result.get("success", False):
                data = serper_result.get("data", {})
                organic_results = data.get("organic", [])[:topk]
                
                for result in organic_results:
                    query_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "server_type": "async-online-search",
                    })
                
                print(f"[DEBUG] AsyncOnlineSearchClient: Found {len(query_results)} results for: {query}")

            
            formatted_results.append(query_results)
        
        if self.wrapper_format:
            first_query_results = formatted_results[0] if formatted_results else []
            return [{
                "documents": [result.get("title", "") + " " + result.get("snippet", "") for result in first_query_results],
                "urls": [result.get("url", "") for result in first_query_results],
                "server_type": "async-online-search"
            }]
        else:
            if len(queries) == 1:
                return formatted_results[0]  # return [{...}, {...}] rather than [[{...}, {...}]]
            else:
                return formatted_results  # return [[...], [...]]

    async def access_async(self, urls):

        if not urls:
            return []
        
        results = []
        urls_to_fetch = []
        
        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access"))
                else:
                    urls_to_fetch.append(url)
                    results.append(None)
            else:
                urls_to_fetch.append(url)
                results.append(None)
        
        if urls_to_fetch:
            if self.use_jina and self.jina_api_key:
                try:
                    async with self.access_semaphore:
                        fetched_results = await self._access_urls_jina_async(urls_to_fetch)
                    
                    fetch_index = 0
                    for i, result in enumerate(results):
                        if result is None:
                            if fetch_index < len(fetched_results):
                                fetched_result = fetched_results[fetch_index]
                                results[i] = fetched_result
                                
                                if self.webpage_cache and fetched_result.get("page"):
                                    self.webpage_cache.put(urls[i], fetched_result["page"])
                                
                                fetch_index += 1
                            else:
                                results[i] = dict(page="", type="access")
                                
                except Exception:
                    pass
            
            for i, result in enumerate(results):
                if result is None:
                    results[i] = dict(page="", type="access")
        
        for result in results:
            if result is not None:
                result["server_type"] = "async-online-search"
        return results

    async def _access_urls_jina_async(self, urls):

        results = []
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    content = await self._jina_readpage_async(session, url)
                    results.append(dict(page=content, type="access"))
                        
        except Exception:
            results = [dict(page="", type="access") for _ in urls]
        
        return results



    def get_cache_stats(self):
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}
    
    def clear_cache(self):
        if self.webpage_cache:
            self.webpage_cache.clear()
    
    def force_save_cache(self):
        if self.webpage_cache:
            self.webpage_cache.force_save()


SEARCH_CLIENTS = {
    "async-search-access": AsyncSearchBrowserClient,
    "async-web-search-access": AsyncOnlineSearchClient,
}

def make_search_client(search_client_type: str, use_jina: bool = False, jina_api_key: str = None):    
    if search_client_type == "async-web-search-access":
        jina_status = "Enabled" if use_jina else "Disabled"
        print(f"Online Search Client: Serper + Jina ({jina_status})")
        
        return SEARCH_CLIENTS[search_client_type](use_jina=use_jina, jina_api_key=jina_api_key, wrapper_format=True)
    else:
        print("Initializing Local Search Client:")
        
        # Get local server configuration from config file
        try:
            from evaluation.config_loader import get_local_server_config
            server_config = get_local_server_config()
            address = server_config['address']
            port = server_config['port']
            
            print(f"Local Server: http://{address}:{port}")
            print(f"Address: {address}")
            print(f"Port: {port}")
            
        except Exception as e:
            print(f"Failed to load server config: {e}")
            raise RuntimeError(f"Failed to load server config: {e}")
        
        return SEARCH_CLIENTS[search_client_type](address=address, port=port)


if __name__ == "__main__":
    # Use config-based initialization for testing
    try:
        from evaluation.config_loader import get_local_server_config
        server_config = get_local_server_config()
        search_client = AsyncSearchBrowserClient(address=server_config['address'], port=server_config['port'])
    except:
        # Fallback for testing
        search_client = AsyncSearchBrowserClient(address="xxxx", port="5201")

    req = {
            "queries": ['"useless spin" True Detective Season 1 episode'],
            "topk": 3,
            "return_scores": False
        }
    response = asyncio.run(search_client.query_async(req))
    pprint(response)

    urls = response[0]['urls']
    url_response = asyncio.run(search_client.access_async(urls))

    # print(len(url_response[0]['page']))
    print(url_response[0]['page'][:1000])
