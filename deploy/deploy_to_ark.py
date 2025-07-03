#!/usr/bin/env python3
"""
使用火山方舟API部署模型
"""

import requests
import json
import time
import os
import argparse

class VolcanoArkClient:
    def __init__(self, api_key, api_secret, endpoint):
        self.api_key = api_key
        self.api_secret = api_secret
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_model(self, model_path, model_name):
        """上传模型到火山方舟"""
        print(f"上传模型: {model_name}")
        
        # 压缩模型文件
        import tarfile
        with tarfile.open(f"{model_name}.tar.gz", "w:gz") as tar:
            tar.add(model_path, arcname=model_name)
        
        # 上传到对象存储
        # 这里需要根据火山引擎的具体API调整
        upload_url = f"{self.endpoint}/models/upload"
        
        with open(f"{model_name}.tar.gz", "rb") as f:
            files = {"model": f}
            data = {"name": model_name, "version": "1.0.0"}
            
            response = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files
            )
        
        if response.status_code == 200:
            model_id = response.json()["model_id"]
            print(f"✅ 模型上传成功，ID: {model_id}")
            return model_id
        else:
            raise Exception(f"模型上传失败: {response.text}")
    
    def create_endpoint(self, model_id, endpoint_name, instance_type="ml.g4dn.xlarge"):
        """创建推理端点"""
        print(f"创建推理端点: {endpoint_name}")
        
        endpoint_config = {
            "name": endpoint_name,
            "model_id": model_id,
            "instance_type": instance_type,
            "initial_instance_count": 1,
            "auto_scaling": {
                "enabled": True,
                "min_capacity": 1,
                "max_capacity": 10,
                "target_value": 70
            },
            "environment": {
                "MAX_BATCH_SIZE": "32",
                "MAX_SEQUENCE_LENGTH": "2048"
            }
        }
        
        response = requests.post(
            f"{self.endpoint}/endpoints",
            headers=self.headers,
            json=endpoint_config
        )
        
        if response.status_code == 201:
            endpoint_info = response.json()
            print(f"✅ 端点创建成功: {endpoint_info['endpoint_url']}")
            return endpoint_info
        else:
            raise Exception(f"端点创建失败: {response.text}")
    
    def wait_for_endpoint(self, endpoint_id, timeout=1800):
        """等待端点就绪"""
        print("等待端点部署完成...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.endpoint}/endpoints/{endpoint_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                status = response.json()["status"]
                print(f"端点状态: {status}")
                
                if status == "InService":
                    print("✅ 端点已就绪")
                    return True
                elif status == "Failed":
                    raise Exception("端点部署失败")
            
            time.sleep(30)
        
        raise Exception("端点部署超时")
    
    def test_endpoint(self, endpoint_url, test_prompt="你好，请介绍一下自己"):
        """测试端点"""
        print(f"测试端点: {test_prompt}")
        
        payload = {
            "prompt": test_prompt,
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(
            f"{endpoint_url}/generate",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 测试成功: {result['response']}")
            return result
        else:
            raise Exception(f"测试失败: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="部署模型到火山方舟")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--endpoint_name", type=str, required=True, help="端点名称")
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.2xlarge", help="实例类型")
    
    args = parser.parse_args()
    
    # 配置信息
    api_key = os.getenv("VOLCANO_API_KEY")
    api_secret = os.getenv("VOLCANO_API_SECRET")
    endpoint = os.getenv("VOLCANO_ENDPOINT", "https://ark.volcengine.com/api/v1")
    
    if not api_key or not api_secret:
        print("❌ 请设置环境变量 VOLCANO_API_KEY 和 VOLCANO_API_SECRET")
        return
    
    client = VolcanoArkClient(api_key, api_secret, endpoint)
    
    try:
        # 1. 上传模型
        model_id = client.upload_model(args.model_path, args.model_name)
        
        # 2. 创建端点
        endpoint_info = client.create_endpoint(
            model_id, 
            args.endpoint_name,
            args.instance_type
        )
        
        # 3. 等待端点就绪
        client.wait_for_endpoint(endpoint_info["endpoint_id"])
        
        # 4. 测试端点
        client.test_endpoint(endpoint_info["endpoint_url"])
        
        print("🎉 部署完成！")
        print(f"API端点: {endpoint_info['endpoint_url']}")
        
    except Exception as e:
        print(f"❌ 部署失败: {e}")

if __name__ == "__main__":
    main() 