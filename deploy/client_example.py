#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API客户端调用示例
展示如何在业务代码中调用部署后的API服务
"""

import requests
import json
import time
import asyncio
import aiohttp
from typing import Optional, List, Dict

class QwenAPIClient:
    """Qwen模型API客户端"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9) -> Dict:
        """同步生成文本"""
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"API调用异常: {e}")
    
    async def async_generate(self, prompt: str, max_length: int = 512, 
                            temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """异步生成文本"""
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"API调用失败: {response.status} - {text}")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        try:
            response = requests.get(f"{self.base_url}/model/info", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"获取模型信息失败: {response.status_code}")
        except Exception as e:
            raise Exception(f"获取模型信息异常: {e}")
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", headers=self.headers)
            return response.status_code == 200
        except:
            return False

# 使用示例
def basic_usage_example():
    """基础使用示例"""
    print("📝 基础使用示例")
    print("-" * 30)
    
    # 初始化客户端
    client = QwenAPIClient("http://localhost:8000")
    
    # 健康检查
    if not client.health_check():
        print("❌ 服务不可用")
        return
    
    # 获取模型信息
    try:
        model_info = client.get_model_info()
        print(f"🤖 模型信息: {model_info['model_type']}")
    except Exception as e:
        print(f"⚠️ 获取模型信息失败: {e}")
    
    # 文本生成
    prompts = [
        "请为一家咖啡店写一段小红书风格的文案",
        "如何制作一杯美味的拿铁？",
        "推荐几个学习Python的好资源"
    ]
    
    for prompt in prompts:
        try:
            print(f"\n💭 输入: {prompt}")
            result = client.generate(prompt, max_length=256, temperature=0.8)
            print(f"🤖 输出: {result['response']}")
            print(f"⏱️ 推理时间: {result['inference_time']:.2f}s")
        except Exception as e:
            print(f"❌ 生成失败: {e}")

async def async_usage_example():
    """异步使用示例"""
    print("\n🔄 异步使用示例")
    print("-" * 30)
    
    client = QwenAPIClient("http://localhost:8000")
    
    prompts = [
        "介绍一下人工智能的发展历史",
        "写一首关于编程的诗",
        "解释什么是机器学习"
    ]
    
    # 并发处理多个请求
    tasks = []
    for prompt in prompts:
        task = client.async_generate(prompt, max_length=200)
        tasks.append(task)
    
    start_time = time.time()
    try:
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"📊 并发处理{len(prompts)}个请求，总耗时: {total_time:.2f}s")
        
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\n{i+1}. 💭 输入: {prompt}")
            print(f"   🤖 输出: {result['response'][:100]}...")
            print(f"   ⏱️ 推理时间: {result['inference_time']:.2f}s")
            
    except Exception as e:
        print(f"❌ 异步处理失败: {e}")

def business_scenario_example():
    """业务场景示例"""
    print("\n💼 业务场景示例")
    print("-" * 30)
    
    client = QwenAPIClient("http://localhost:8000")
    
    # 场景1: 内容生成
    print("📝 场景1: 内容生成")
    content_prompts = {
        "商品描述": "为一款智能手表写一段产品描述",
        "营销文案": "为一家新开的健身房写一段宣传文案",
        "社交媒体": "为公司新产品发布写一条微博"
    }
    
    for scenario, prompt in content_prompts.items():
        try:
            result = client.generate(prompt, temperature=0.8, max_length=200)
            print(f"\n{scenario}:")
            print(f"  输入: {prompt}")
            print(f"  输出: {result['response']}")
        except Exception as e:
            print(f"  ❌ {scenario}生成失败: {e}")
    
    # 场景2: 问答系统
    print("\n❓ 场景2: 问答系统")
    qa_prompts = [
        "什么是云计算？请简单解释一下",
        "如何提高团队协作效率？",
        "Python和Java有什么区别？"
    ]
    
    for prompt in qa_prompts:
        try:
            result = client.generate(prompt, temperature=0.3, max_length=300)
            print(f"\nQ: {prompt}")
            print(f"A: {result['response']}")
        except Exception as e:
            print(f"❌ 问答失败: {e}")
    
    # 场景3: 代码生成
    print("\n💻 场景3: 代码生成")
    code_prompts = [
        "写一个Python函数来计算斐波那契数列",
        "用JavaScript实现一个简单的计时器",
        "写一个SQL查询来找出销量最高的产品"
    ]
    
    for prompt in code_prompts:
        try:
            result = client.generate(prompt, temperature=0.1, max_length=400)
            print(f"\n需求: {prompt}")
            print(f"代码: {result['response']}")
        except Exception as e:
            print(f"❌ 代码生成失败: {e}")

def performance_monitoring_example():
    """性能监控示例"""
    print("\n📊 性能监控示例")
    print("-" * 30)
    
    client = QwenAPIClient("http://localhost:8000")
    
    # 测试不同参数对性能的影响
    test_prompt = "请介绍一下人工智能技术的应用场景"
    
    test_configs = [
        {"max_length": 100, "temperature": 0.1, "name": "快速模式"},
        {"max_length": 300, "temperature": 0.5, "name": "平衡模式"},
        {"max_length": 500, "temperature": 0.8, "name": "创意模式"}
    ]
    
    print("测试不同配置的性能:")
    for config in test_configs:
        try:
            start_time = time.time()
            result = client.generate(
                test_prompt, 
                max_length=config["max_length"],
                temperature=config["temperature"]
            )
            total_time = time.time() - start_time
            
            print(f"\n{config['name']}:")
            print(f"  参数: max_length={config['max_length']}, temperature={config['temperature']}")
            print(f"  推理时间: {result['inference_time']:.2f}s")
            print(f"  总时间: {total_time:.2f}s")
            print(f"  输出长度: {len(result['response'])} 字符")
            
        except Exception as e:
            print(f"  ❌ {config['name']}测试失败: {e}")

def main():
    """主函数"""
    print("🚀 Qwen API客户端使用示例")
    print("=" * 50)
    
    try:
        # 基础使用
        basic_usage_example()
        
        # 异步使用
        asyncio.run(async_usage_example())
        
        # 业务场景
        business_scenario_example()
        
        # 性能监控
        performance_monitoring_example()
        
        print("\n🎉 所有示例运行完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 示例被用户中断")
    except Exception as e:
        print(f"\n❌ 运行示例时出现异常: {e}")

if __name__ == "__main__":
    main() 