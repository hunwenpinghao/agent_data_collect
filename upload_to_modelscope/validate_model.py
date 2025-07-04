#!/usr/bin/env python3
"""
模型文件验证脚本
用于在上传前验证模型文件的完整性
"""

import os
import json
import argparse
from pathlib import Path
import sys

def check_file_size(file_path):
    """检查文件大小"""
    size = os.path.getsize(file_path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"

def validate_json_file(file_path):
    """验证 JSON 文件格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, "✅ JSON 格式正确"
    except json.JSONDecodeError as e:
        return False, f"❌ JSON 格式错误: {e}"
    except Exception as e:
        return False, f"❌ 读取失败: {e}"

def validate_model_directory(model_dir):
    """验证模型目录"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return False
    
    print(f"📁 验证模型目录: {model_dir}")
    print("=" * 60)
    
    # 定义文件检查规则
    file_checks = {
        'config.json': {'required': True, 'type': 'json', 'description': '模型配置文件'},
        'model.safetensors': {'required': True, 'type': 'binary', 'description': '模型权重文件'},
        'tokenizer.json': {'required': False, 'type': 'json', 'description': '分词器配置'},
        'tokenizer_config.json': {'required': False, 'type': 'json', 'description': '分词器配置'},
        'generation_config.json': {'required': False, 'type': 'json', 'description': '生成配置'},
        'vocab.json': {'required': False, 'type': 'json', 'description': '词汇表'},
        'merges.txt': {'required': False, 'type': 'text', 'description': 'BPE合并规则'},
        'README.md': {'required': False, 'type': 'text', 'description': '模型说明文档'},
        'LICENSE': {'required': False, 'type': 'text', 'description': '许可证文件'},
    }
    
    all_valid = True
    total_size = 0
    
    for filename, check_info in file_checks.items():
        file_path = model_path / filename
        
        if file_path.exists():
            size = os.path.getsize(file_path)
            total_size += size
            size_str = check_file_size(file_path)
            
            # 验证文件格式
            if check_info['type'] == 'json':
                is_valid, msg = validate_json_file(file_path)
                status = msg
                if not is_valid:
                    all_valid = False
            else:
                status = "✅ 文件存在"
            
            print(f"✅ {filename:<25} {size_str:>10} - {check_info['description']} - {status}")
            
        else:
            if check_info['required']:
                print(f"❌ {filename:<25} {'缺失':>10} - {check_info['description']} (必需)")
                all_valid = False
            else:
                print(f"⚠️  {filename:<25} {'缺失':>10} - {check_info['description']} (可选)")
    
    print("=" * 60)
    print(f"📊 总文件大小: {check_file_size_total(total_size)}")
    
    # 检查模型配置
    config_path = model_path / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print("\n🔧 模型配置信息:")
            print(f"   模型类型: {config.get('model_type', '未知')}")
            print(f"   架构: {config.get('architectures', ['未知'])[0] if config.get('architectures') else '未知'}")
            print(f"   词汇表大小: {config.get('vocab_size', '未知')}")
            print(f"   隐藏层大小: {config.get('hidden_size', '未知')}")
            print(f"   注意力头数: {config.get('num_attention_heads', '未知')}")
            print(f"   层数: {config.get('num_hidden_layers', '未知')}")
            
        except Exception as e:
            print(f"⚠️  无法读取模型配置: {e}")
    
    return all_valid

def check_file_size_total(total_bytes):
    """格式化总文件大小"""
    if total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{total_bytes / (1024 * 1024 * 1024):.1f} GB"

def main():
    parser = argparse.ArgumentParser(description="验证模型文件完整性")
    parser.add_argument("--model_dir", type=str, 
                       default="../deploy_ark/output_qwen/Qwen2.5-0.5B-Instruct",
                       help="模型文件目录")
    
    args = parser.parse_args()
    
    print("🔍 模型文件验证工具")
    print("==================")
    
    is_valid = validate_model_directory(args.model_dir)
    
    print("\n📋 验证结果:")
    if is_valid:
        print("✅ 模型文件验证通过，可以上传到魔搭社区！")
        print("\n🚀 下一步:")
        print("   1. 运行上传脚本: ./quick_upload.sh")
        print("   2. 或手动上传: python3 upload_to_modelscope.py --token YOUR_TOKEN")
        return 0
    else:
        print("❌ 模型文件验证失败，请检查缺失的必需文件！")
        print("\n💡 建议:")
        print("   1. 确认模型训练/微调已完成")
        print("   2. 检查模型保存路径是否正确")
        print("   3. 重新运行模型保存流程")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 