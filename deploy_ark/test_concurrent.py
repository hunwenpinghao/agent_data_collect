#!/usr/bin/env python3
"""
火山方舟并发测试脚本
测试API服务器的并发性能和稳定性
"""

import asyncio
import time
import json
import argparse
import statistics
from typing import List, Dict

class SimpleTester:
    """简单的并发测试器"""
    
    def __init__(self, endpoint_url: str, api_key: str = "test_key"):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        
    def get_test_prompts(self) -> List[str]:
        """获取测试提示词"""
        return [
            "你好，请介绍一下自己",
            "推荐一家位于正弘城的好餐厅",
            "今天天气怎么样？有什么活动推荐吗？",
            "请介绍一下小红书风格的文案写作技巧",
            "给我推荐一些正弘城的购物店铺"
        ]
    
    def sync_test(self, concurrent_requests: int = 5, total_requests: int = 20) -> Dict:
        """同步测试"""
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        print(f"Starting test: {concurrent_requests} concurrent, {total_requests} total")
        
        prompts = self.get_test_prompts()
        results = []
        results_lock = threading.Lock()
        start_time = time.time()
        
        def single_request(prompt_index: int) -> Dict:
            prompt = prompts[prompt_index % len(prompts)]
            request_start = time.time()
            
            try:
                response = requests.post(
                    f"{self.endpoint_url}/generate",
                    json={
                        "prompt": prompt,
                        "max_length": 512,
                        "temperature": 0.7
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=60
                )
                
                request_end = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "success",
                        "response_time": request_end - request_start,
                        "prompt_index": prompt_index,
                        "response_length": len(result.get("response", ""))
                    }
                else:
                    return {
                        "status": "error",
                        "response_time": request_end - request_start,
                        "prompt_index": prompt_index,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
            except Exception as e:
                request_end = time.time()
                return {
                    "status": "error",
                    "response_time": request_end - request_start,
                    "prompt_index": prompt_index,
                    "error": str(e)
                }
        
        # 执行所有请求
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            future_to_index = {executor.submit(single_request, i): i 
                              for i in range(total_requests)}
            
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                except Exception as e:
                    with results_lock:
                        results.append({
                            "status": "error",
                            "error": str(e),
                            "response_time": 0
                        })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算统计信息
        success_results = [r for r in results if r["status"] == "success"]
        error_results = [r for r in results if r["status"] == "error"]
        
        if success_results:
            response_times = [r["response_time"] for r in success_results]
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        summary = {
            "test_config": {
                "concurrent_requests": concurrent_requests,
                "total_requests": total_requests,
                "test_duration": total_time
            },
            "results": {
                "total_requests": len(results),
                "success_count": len(success_results),
                "error_count": len(error_results),
                "success_rate": len(success_results) / len(results) if results else 0,
                "requests_per_second": len(results) / total_time if total_time > 0 else 0,
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            },
            "errors": [r["error"] for r in error_results]
        }
        
        return summary

def print_results(results: Dict):
    """打印测试结果"""
    config = results["test_config"]
    metrics = results["results"]
    
    print("\n=== 测试结果 ===")
    print(f"并发数: {config['concurrent_requests']}")
    print(f"总请求数: {config['total_requests']}")
    print(f"测试持续时间: {config['test_duration']:.2f}s")
    print()
    
    print("=== 性能指标 ===")
    print(f"总请求数: {metrics['total_requests']}")
    print(f"成功请求数: {metrics['success_count']}")
    print(f"失败请求数: {metrics['error_count']}")
    print(f"成功率: {metrics['success_rate']:.2%}")
    print(f"QPS: {metrics['requests_per_second']:.2f}")
    print()
    
    print("=== 响应时间 ===")
    print(f"平均响应时间: {metrics['avg_response_time']:.3f}s")
    print(f"最小响应时间: {metrics['min_response_time']:.3f}s")
    print(f"最大响应时间: {metrics['max_response_time']:.3f}s")
    
    if results["errors"]:
        print("\n=== 错误信息 ===")
        for error in results["errors"][:3]:  # 只显示前3个错误
            print(f"- {error}")

def main():
    parser = argparse.ArgumentParser(description="火山方舟并发测试工具")
    parser.add_argument("--endpoint", type=str, required=True, help="API端点URL")
    parser.add_argument("--api-key", type=str, default="test_key", help="API密钥")
    parser.add_argument("--concurrent", type=int, default=5, help="并发数")
    parser.add_argument("--total", type=int, default=20, help="总请求数")
    parser.add_argument("--output", type=str, help="结果输出文件")
    
    args = parser.parse_args()
    
    tester = SimpleTester(args.endpoint, args.api_key)
    
    print("🚀 开始并发测试...")
    results = tester.sync_test(args.concurrent, args.total)
    print("✅ 测试完成")
    print_results(results)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
