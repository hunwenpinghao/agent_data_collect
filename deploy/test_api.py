#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API服务测试脚本
用于测试部署后的API服务功能和性能
"""

import requests
import json
import time
import asyncio
import concurrent.futures
import argparse
from typing import List, Dict
import statistics

class APITester:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 健康检查通过: {result}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 健康检查异常: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        try:
            response = requests.get(f"{self.base_url}/model/info", headers=self.headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"📊 模型信息: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"❌ 获取模型信息失败: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ 获取模型信息异常: {e}")
            return {}
    
    def single_inference(self, prompt: str, **kwargs) -> Dict:
        """单次推理测试"""
        payload = {
            "prompt": prompt,
            "max_length": kwargs.get("max_length", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/generate", 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                elapsed_time = time.time() - start_time
                
                print(f"🤖 输入: {prompt}")
                print(f"📝 输出: {result['response']}")
                print(f"⏱️  推理时间: {result['inference_time']:.2f}s")
                print(f"🔄 总时间: {elapsed_time:.2f}s")
                print("-" * 50)
                
                return {
                    "success": True,
                    "response": result["response"],
                    "inference_time": result["inference_time"],
                    "total_time": elapsed_time
                }
            else:
                print(f"❌ 推理失败: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ 推理异常: {e}, 耗时: {elapsed_time:.2f}s")
            return {"success": False, "error": str(e), "total_time": elapsed_time}
    
    def batch_inference_test(self, prompts: List[str], concurrent: int = 5) -> Dict:
        """批量推理性能测试"""
        print(f"🚀 开始批量推理测试，并发数: {concurrent}")
        print(f"📝 测试样本数: {len(prompts)}")
        
        results = []
        start_time = time.time()
        
        def single_request(prompt):
            return self.single_inference(prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            future_to_prompt = {executor.submit(single_request, prompt): prompt for prompt in prompts}
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                result = future.result()
                results.append(result)
        
        total_time = time.time() - start_time
        
        # 统计结果
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", True)]
        
        if successful_results:
            inference_times = [r["inference_time"] for r in successful_results]
            total_times = [r["total_time"] for r in successful_results]
            
            stats = {
                "total_requests": len(prompts),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(prompts) * 100,
                "total_time": total_time,
                "average_inference_time": statistics.mean(inference_times),
                "median_inference_time": statistics.median(inference_times),
                "p95_inference_time": statistics.quantiles(inference_times, n=20)[18] if len(inference_times) > 1 else inference_times[0],
                "average_total_time": statistics.mean(total_times),
                "throughput": len(successful_results) / total_time
            }
        else:
            stats = {
                "total_requests": len(prompts),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0,
                "total_time": total_time
            }
        
        print("\n📊 批量测试统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats
    
    def stress_test(self, prompt: str, duration: int = 60, concurrent: int = 10) -> Dict:
        """压力测试"""
        print(f"💪 开始压力测试")
        print(f"⏱️  测试时长: {duration}秒")
        print(f"🔄 并发数: {concurrent}")
        print(f"📝 测试prompt: {prompt}")
        
        results = []
        start_time = time.time()
        end_time = start_time + duration
        
        def worker():
            worker_results = []
            while time.time() < end_time:
                result = self.single_inference(prompt)
                worker_results.append(result)
            return worker_results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent)]
            
            for future in concurrent.futures.as_completed(futures):
                worker_results = future.result()
                results.extend(worker_results)
        
        actual_duration = time.time() - start_time
        
        # 统计结果
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", True)]
        
        if successful_results:
            inference_times = [r["inference_time"] for r in successful_results]
            
            stats = {
                "duration": actual_duration,
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "average_inference_time": statistics.mean(inference_times),
                "p95_inference_time": statistics.quantiles(inference_times, n=20)[18] if len(inference_times) > 1 else inference_times[0],
                "qps": len(successful_results) / actual_duration
            }
        else:
            stats = {
                "duration": actual_duration,
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0,
                "qps": 0
            }
        
        print("\n💪 压力测试统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="API服务测试工具")
    parser.add_argument("--url", type=str, required=True, help="API服务地址")
    parser.add_argument("--api_key", type=str, help="API密钥")
    parser.add_argument("--test_type", type=str, default="all", 
                       choices=["health", "single", "batch", "stress", "all"],
                       help="测试类型")
    parser.add_argument("--concurrent", type=int, default=5, help="并发数")
    parser.add_argument("--duration", type=int, default=60, help="压力测试持续时间(秒)")
    
    args = parser.parse_args()
    
    tester = APITester(args.url, args.api_key)
    
    # 测试样本
    test_prompts = [
        "请为一家咖啡店写一段小红书风格的文案",
        "如何制作一杯好喝的拿铁咖啡？",
        "推荐几家上海的网红咖啡店",
        "介绍一下人工智能的发展历史",
        "写一首关于春天的诗",
        "解释一下机器学习和深度学习的区别",
        "推荐一些适合新手的编程书籍",
        "介绍一下火山引擎的主要服务",
        "如何做好时间管理？",
        "分析一下当前AI行业的发展趋势"
    ]
    
    print("🧪 开始API服务测试")
    print("=" * 50)
    
    try:
        # 健康检查
        if args.test_type in ["health", "all"]:
            print("\n1. 健康检查测试")
            if not tester.health_check():
                print("❌ 健康检查失败，终止测试")
                return
        
        # 模型信息
        if args.test_type in ["single", "batch", "stress", "all"]:
            print("\n2. 模型信息测试")
            tester.get_model_info()
        
        # 单次推理测试
        if args.test_type in ["single", "all"]:
            print("\n3. 单次推理测试")
            tester.single_inference(test_prompts[0])
        
        # 批量推理测试
        if args.test_type in ["batch", "all"]:
            print("\n4. 批量推理测试")
            tester.batch_inference_test(test_prompts[:5], concurrent=args.concurrent)
        
        # 压力测试
        if args.test_type in ["stress", "all"]:
            print("\n5. 压力测试")
            tester.stress_test(test_prompts[0], duration=args.duration, concurrent=args.concurrent)
        
        print("\n🎉 测试完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")

if __name__ == "__main__":
    main() 