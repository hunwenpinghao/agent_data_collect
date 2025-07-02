#!/usr/bin/env python3
"""
比较ModelScope和HuggingFace下载的模型文件差异
"""

import os
import json
import hashlib
from pathlib import Path

def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    if not os.path.exists(file_path):
        return None
    
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return "error"

def compare_json_files(file1, file2):
    """比较两个JSON文件的内容差异"""
    if not os.path.exists(file1) or not os.path.exists(file2):
        return "文件不存在"
    
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        differences = []
        
        # 检查所有键值对
        all_keys = set(data1.keys()) | set(data2.keys())
        
        for key in all_keys:
            if key not in data1:
                differences.append(f"+ {key}: {data2[key]}")
            elif key not in data2:
                differences.append(f"- {key}: {data1[key]}")
            elif data1[key] != data2[key]:
                differences.append(f"~ {key}: {data1[key]} → {data2[key]}")
        
        return differences if differences else "完全相同"
        
    except Exception as e:
        return f"比较失败: {e}"

def compare_model_directories(hf_path, ms_path):
    """比较HuggingFace和ModelScope下载的模型目录"""
    
    print("🔍 模型文件对比分析")
    print("=" * 60)
    
    if not os.path.exists(hf_path):
        print(f"❌ HuggingFace模型路径不存在: {hf_path}")
        return
    
    if not os.path.exists(ms_path):
        print(f"❌ ModelScope模型路径不存在: {ms_path}")
        return
    
    print(f"📁 HuggingFace: {hf_path}")
    print(f"📁 ModelScope:  {ms_path}")
    print()
    
    # 获取所有文件列表
    hf_files = set(os.listdir(hf_path)) if os.path.exists(hf_path) else set()
    ms_files = set(os.listdir(ms_path)) if os.path.exists(ms_path) else set()
    all_files = hf_files | ms_files
    
    print("📊 文件对比结果:")
    print("-" * 60)
    
    important_files = [
        "config.json",
        "tokenizer_config.json", 
        "tokenizer.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin"
    ]
    
    for file in sorted(all_files):
        hf_file = os.path.join(hf_path, file)
        ms_file = os.path.join(ms_path, file)
        
        hf_exists = file in hf_files
        ms_exists = file in ms_files
        
        status = "  "
        if hf_exists and ms_exists:
            # 比较文件哈希
            hf_hash = get_file_hash(hf_file)
            ms_hash = get_file_hash(ms_file)
            
            if hf_hash == ms_hash:
                status = "✅"
            else:
                status = "⚠️ "
        elif hf_exists:
            status = "🔷"  # 只有HF有
        elif ms_exists:
            status = "🔶"  # 只有MS有
        
        print(f"{status} {file}")
        
        # 对重要文件进行详细比较
        if file in important_files and hf_exists and ms_exists:
            if file.endswith('.json'):
                diff = compare_json_files(hf_file, ms_file)
                if diff != "完全相同":
                    print(f"    📝 JSON差异:")
                    if isinstance(diff, list):
                        for d in diff[:3]:  # 只显示前3个差异
                            print(f"       {d}")
                        if len(diff) > 3:
                            print(f"       ... 还有{len(diff)-3}个差异")
                    else:
                        print(f"       {diff}")
    
    print()
    print("图例:")
    print("✅ 文件相同    ⚠️  文件不同")
    print("🔷 仅HF有      🔶 仅MS有")

def test_model_compatibility():
    """测试模型兼容性"""
    print("\n🧪 兼容性测试")
    print("-" * 60)
    
    # 常见的模型路径
    possible_paths = [
        ("./models/Qwen2.5-0.5B-Instruct", "ModelScope本地"),
        ("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct", "HF缓存"),
        ("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct", "HF缓存(root)")
    ]
    
    found_models = []
    for path, source in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"✅ 发现模型: {source} ({expanded_path})")
            found_models.append((expanded_path, source))
        else:
            print(f"❌ 未找到: {source} ({expanded_path})")
    
    if len(found_models) >= 2:
        print(f"\n🔄 对比两个来源的模型:")
        compare_model_directories(found_models[0][0], found_models[1][0])
    elif len(found_models) == 1:
        print(f"\n💡 只找到一个模型，无法对比")
        print(f"   可尝试从另一个源下载进行对比")
    else:
        print(f"\n❌ 未找到任何模型文件")

def main():
    print("🔄 ModelScope vs HuggingFace 模型对比工具")
    print("=" * 60)
    
    # 运行兼容性测试
    test_model_compatibility()
    
    print(f"\n💡 总结:")
    print("1. 核心模型权重文件(model.safetensors)通常完全相同")
    print("2. JSON配置文件可能有路径名称的微小差异")
    print("3. 两种来源的模型在训练中都能正常使用")
    print("4. 建议优先使用ModelScope（国内访问更稳定）")

if __name__ == "__main__":
    main() 