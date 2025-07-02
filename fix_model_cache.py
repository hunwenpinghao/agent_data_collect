#!/usr/bin/env python3
"""
诊断和修复模型缓存问题
"""

import os
import shutil
import json
from pathlib import Path

def check_model_cache():
    """检查模型缓存状态"""
    print("🔍 检查模型缓存状态...")
    
    cache_dirs = [
        "./models",
        "~/.cache/huggingface",
        "/root/.cache/huggingface"
    ]
    
    for cache_dir in cache_dirs:
        expanded_dir = os.path.expanduser(cache_dir)
        if os.path.exists(expanded_dir):
            print(f"📁 发现缓存目录: {expanded_dir}")
            
            # 检查Qwen模型
            for item in os.listdir(expanded_dir):
                if "qwen" in item.lower() or "Qwen" in item:
                    model_dir = os.path.join(expanded_dir, item)
                    print(f"   🔍 检查模型: {model_dir}")
                    
                    # 检查必要文件
                    required_files = [
                        "config.json",
                        "tokenizer.json",
                        "tokenizer_config.json"
                    ]
                    
                    missing_files = []
                    for file in required_files:
                        file_path = os.path.join(model_dir, file)
                        if os.path.exists(file_path):
                            try:
                                if file.endswith('.json'):
                                    with open(file_path, 'r') as f:
                                        json.load(f)
                                print(f"      ✅ {file}")
                            except Exception as e:
                                print(f"      ❌ {file} (损坏: {e})")
                                missing_files.append(file)
                        else:
                            print(f"      ❌ {file} (缺失)")
                            missing_files.append(file)
                    
                    if missing_files:
                        print(f"      ⚠️  模型不完整，缺失: {missing_files}")
                        return False, model_dir
                    else:
                        print(f"      ✅ 模型完整")
                        return True, model_dir
        else:
            print(f"📁 缓存目录不存在: {expanded_dir}")
    
    return False, None

def clean_cache():
    """清理损坏的模型缓存"""
    print("🧹 清理模型缓存...")
    
    cache_dirs = [
        "./models",
        "~/.cache/huggingface/hub",
        "/root/.cache/huggingface/hub"
    ]
    
    for cache_dir in cache_dirs:
        expanded_dir = os.path.expanduser(cache_dir)
        if os.path.exists(expanded_dir):
            try:
                print(f"🗑️  清理: {expanded_dir}")
                
                # 只删除Qwen相关的模型
                for item in os.listdir(expanded_dir):
                    if "qwen" in item.lower():
                        item_path = os.path.join(expanded_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            print(f"   ✅ 已删除: {item}")
                        else:
                            os.remove(item_path)
                            print(f"   ✅ 已删除: {item}")
                
            except Exception as e:
                print(f"   ❌ 清理失败: {e}")

def download_model_with_modelscope():
    """使用ModelScope下载模型"""
    print("📥 尝试使用ModelScope下载模型...")
    
    try:
        from modelscope import snapshot_download
        
        model_name = "qwen/Qwen2.5-0.5B-Instruct"
        cache_dir = "./models"
        
        print(f"下载模型: {model_name}")
        print(f"保存到: {cache_dir}")
        
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        print(f"✅ 下载成功: {model_dir}")
        return model_dir
        
    except ImportError:
        print("❌ ModelScope未安装")
        return None
    except Exception as e:
        print(f"❌ ModelScope下载失败: {e}")
        return None

def main():
    print("🛠️  模型缓存修复工具")
    print("=" * 50)
    
    # 检查现有缓存
    is_valid, model_path = check_model_cache()
    
    if is_valid:
        print(f"\n✅ 模型缓存完整: {model_path}")
        print("可以直接使用现有模型进行训练")
        return model_path
    
    print(f"\n❌ 模型缓存不完整或损坏")
    
    # 询问是否清理
    response = input("\n是否清理损坏的缓存并重新下载? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # 清理缓存
        clean_cache()
        
        # 尝试重新下载
        model_dir = download_model_with_modelscope()
        
        if model_dir:
            print(f"\n🎉 模型修复成功!")
            print(f"模型位置: {model_dir}")
            print("\n现在可以运行训练:")
            print("python fine_tune_qwen.py --config_file configs/train_config_full.json")
            return model_dir
        else:
            print(f"\n❌ 自动下载失败")
            print("请手动下载模型:")
            print("1. ./download_model.sh")
            print("2. 或使用其他下载方式")
            return None
    else:
        print("\n跳过清理，请手动处理模型缓存问题")
        return None

if __name__ == "__main__":
    main() 