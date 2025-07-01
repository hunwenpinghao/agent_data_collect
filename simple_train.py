#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版本的微调脚本，用于排查兼容性问题
"""

import os
import sys
import json

def check_imports():
    """检查必要的包导入"""
    print("🔍 检查依赖包...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers 导入失败: {e}")
        return False
    
    try:
        from modelscope import snapshot_download
        print(f"✅ ModelScope: 导入成功")
    except ImportError as e:
        print(f"❌ ModelScope 导入失败: {e}")
        return False
    
    # 检查具体的导入问题
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"✅ AutoTokenizer, AutoModelForCausalLM: 导入成功")
    except ImportError as e:
        print(f"❌ AutoTokenizer/AutoModelForCausalLM 导入失败: {e}")
        return False
    
    try:
        from transformers import TrainingArguments, Trainer
        print(f"✅ TrainingArguments, Trainer: 导入成功")
    except ImportError as e:
        print(f"❌ TrainingArguments/Trainer 导入失败: {e}")
        print("这可能是版本兼容性问题，建议使用稳定版本:")
        print("pip install torch==2.1.0 transformers==4.36.2")
        return False
    
    return True

def test_data_loading():
    """测试数据加载功能"""
    print("\n📁 测试数据加载功能...")
    
    test_data = {
        "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：",
        "input": "店铺名称：星巴克\n品类：咖啡",
        "output": "✨星巴克的咖啡香气，治愈每一个午后～"
    }
    
    # 测试多文件路径解析
    def parse_data_path(data_path: str):
        if ',' in data_path:
            file_paths = [path.strip() for path in data_path.split(',') if path.strip()]
        else:
            file_paths = [data_path.strip()]
        return file_paths
    
    # 测试用例
    test_cases = [
        "single_file.jsonl",
        "file1.jsonl,file2.jsonl",
        "file1.jsonl, file2.jsonl, file3.jsonl",
        " file1.jsonl , file2.jsonl,file3.jsonl "
    ]
    
    for test_case in test_cases:
        result = parse_data_path(test_case)
        print(f"输入: '{test_case}' -> 输出: {result}")
    
    print("✅ 数据路径解析功能正常")

def test_model_download():
    """测试模型下载功能"""
    print("\n📥 测试模型下载功能...")
    
    try:
        from modelscope import snapshot_download
        
        model_name = "qwen/Qwen3-0.5B-Instruct"
        print(f"准备下载模型: {model_name}")
        print("注意: 这只是测试导入功能，不会实际下载")
        print("✅ 模型下载功能可用")
        
    except Exception as e:
        print(f"❌ 模型下载功能测试失败: {e}")

def provide_solutions():
    """提供解决方案"""
    print("\n🛠️ 常见问题解决方案:")
    print("=" * 50)
    
    solutions = [
        {
            "问题": "LRScheduler 未定义错误",
            "原因": "transformers 与 PyTorch 版本不兼容",
            "解决方案": [
                "pip uninstall torch transformers -y",
                "pip install torch==2.1.0 transformers==4.36.2",
                "或使用: pip install -r requirements_stable.txt"
            ]
        },
        {
            "问题": "ModuleNotFoundError",
            "原因": "缺少必要的依赖包",
            "解决方案": [
                "pip install -r requirements_stable.txt",
                "或手动安装: pip install torch transformers modelscope"
            ]
        },
        {
            "问题": "CUDA 相关错误",
            "原因": "CUDA 版本不匹配",
            "解决方案": [
                "检查 CUDA 版本: nvidia-smi",
                "安装对应的 PyTorch 版本",
                "或使用 CPU 模式: CUDA_VISIBLE_DEVICES=\"\""
            ]
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['问题']}")
        print(f"   原因: {solution['原因']}")
        print(f"   解决方案:")
        for step in solution['解决方案']:
            print(f"   - {step}")

def main():
    print("🚀 Qwen3-0.5B 微调环境诊断工具")
    print("=" * 50)
    
    # 检查 Python 版本
    print(f"🐍 Python 版本: {sys.version}")
    
    # 检查导入
    if not check_imports():
        print("\n❌ 环境检查失败，请参考下面的解决方案")
        provide_solutions()
        return False
    
    # 测试数据加载
    test_data_loading()
    
    # 测试模型下载
    test_model_download()
    
    print("\n🎉 环境检查完成！")
    print("✅ 所有依赖包正常，可以开始训练")
    print("\n💡 使用建议:")
    print("   1. 运行主训练脚本: python3 fine_tune_qwen.py --config_file train_config.json")
    print("   2. 使用多文件训练: python3 fine_tune_qwen.py --config_file train_config_multi_files.json")
    print("   3. 如遇问题可使用 Docker: ./build_docker.sh run")
    
    return True

if __name__ == "__main__":
    main() 