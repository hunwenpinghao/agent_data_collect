#!/usr/bin/env python3
"""
火山方舟并发客户端
支持批量请求、流式响应、连接池等功能
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArkClientConfig:
    """火山方舟客户端配置"""
    api_key: str
    endpoint_url: str
    timeout: int = 60
    max_connections: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0

class ArkAsyncClient:
    """异步火山方舟客户端"""
    
    def __init__(self, config: ArkClientConfig):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for async client")
        
        self.config = config
        self.session = None
        self.connector = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ArkAsyncClient/1.0"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def generate_single(self, prompt: str, max_length: int = 512, 
                             temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """单个文本生成请求"""
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    f"{self.config.endpoint_url}/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Request failed: {response.status} - {error_text}")
                        if attempt == self.config.max_retries - 1:
                            raise Exception(f"HTTP {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def concurrent_generate(self, prompts: List[str], max_concurrency: int = 10) -> List[Dict]:
        """并发生成多个文本"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def generate_with_semaphore(prompt: str) -> Dict:
            async with semaphore:
                return await self.generate_single(prompt)
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                processed_results.append({
                    "error": str(result),
                    "prompt": prompts[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results

class ArkSyncClient:
    """同步火山方舟客户端"""
    
    def __init__(self, config: ArkClientConfig):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is required for sync client")
        
        self.config = config
        self.session = None
        
    def __enter__(self):
        """上下文管理器入口"""
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ArkSyncClient/1.0"
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self.session:
            self.session.close()
    
    def generate_single(self, prompt: str, max_length: int = 512, 
                       temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """单个文本生成请求"""
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.config.endpoint_url}/generate",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Request failed: {response.status_code} - {response.text}")
                    if attempt == self.config.max_retries - 1:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay * (2 ** attempt))
    
    def concurrent_generate(self, prompts: List[str], max_workers: int = 10) -> List[Dict]:
        """并发生成多个文本"""
        def generate_wrapper(prompt: str) -> Dict:
            try:
                return self.generate_single(prompt)
            except Exception as e:
                logger.error(f"Generate failed for prompt: {prompt[:50]}... - {e}")
                return {"error": str(e), "prompt": prompt}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(generate_wrapper, prompt): prompt 
                               for prompt in prompts}
            
            results = []
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Future failed for prompt: {prompt[:50]}... - {e}")
                    results.append({"error": str(e), "prompt": prompt})
            
            return results

def main():
    """示例用法"""
    config = ArkClientConfig(
        api_key="your_api_key",
        endpoint_url="https://ark.volcengine.com/api/endpoints/your_endpoint",
        timeout=60,
        max_connections=100
    )
    
    # 同步客户端示例
    with ArkSyncClient(config) as sync_client:
        result = sync_client.generate_single("你好")
        print(f"Sync request result: {result}")
        
        # 并发请求
        prompts = [
            "请推荐一家好的餐厅",
            "今天天气怎么样？",
            "请介绍一下火山方舟"
        ]
        
        results = sync_client.concurrent_generate(prompts, max_workers=5)
        print(f"Sync concurrent requests completed: {len(results)}")

if __name__ == "__main__":
    main()
