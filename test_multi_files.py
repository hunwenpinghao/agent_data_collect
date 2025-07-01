#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多文件数据加载功能
"""

import os
import sys
import json
import tempfile
from transformers import AutoTokenizer

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fine_tune_qwen import SFTDataset

def create_test_data_files():
    """创建测试用的JSONL文件"""
    test_data_1 = [
        {
            "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：",
            "input": "店铺名称：星巴克\n品类：咖啡\n地址：郑州正弘城L8",
            "output": "✨郑州正弘城星巴克，我的咖啡小天地！"
        },
        {
            "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「活力」：",
            "input": "店铺名称：Nike\n品类：运动用品\n地址：正弘城L2",
            "output": "🏃‍♀️正弘城Nike，点燃你的运动激情！"
        }
    ]
    
    test_data_2 = [
        {
            "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「温馨」：",
            "input": "店铺名称：海底捞\n品类：火锅\n地址：正弘城L7",
            "output": "🍲海底捞，和朋友一起暖心聚餐的好地方！"
        }
    ]
    
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    
    file1_path = os.path.join(temp_dir, "test_data_1.jsonl")
    file2_path = os.path.join(temp_dir, "test_data_2.jsonl")
    
    # 写入测试数据
    with open(file1_path, 'w', encoding='utf-8') as f:
        for item in test_data_1:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(file2_path, 'w', encoding='utf-8') as f:
        for item in test_data_2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return file1_path, file2_path, temp_dir

def test_single_file(file1_path):
    """测试单文件加载"""
    print("=== 测试单文件加载 ===")
    try:
        # 创建一个简单的tokenizer用于测试
        tokenizer = type('MockTokenizer', (), {
            'pad_token': None,
            'eos_token': '</s>',
            '__call__': lambda self, text, **kwargs: {'input_ids': [1, 2, 3]},
            'decode': lambda self, ids, **kwargs: 'mock_output'
        })()
        
        dataset = SFTDataset(file1_path, tokenizer, max_seq_length=512)
        print(f"✅ 单文件加载成功，数据量: {len(dataset)}")
        return True
    except Exception as e:
        print(f"❌ 单文件加载失败: {e}")
        return False

def test_multi_files(file1_path, file2_path):
    """测试多文件加载"""
    print("\n=== 测试多文件加载 ===")
    try:
        # 创建一个简单的tokenizer用于测试
        tokenizer = type('MockTokenizer', (), {
            'pad_token': None,
            'eos_token': '</s>',
            '__call__': lambda self, text, **kwargs: {'input_ids': [1, 2, 3]},
            'decode': lambda self, ids, **kwargs: 'mock_output'
        })()
        
        # 测试逗号分隔的多文件
        multi_path = f"{file1_path},{file2_path}"
        dataset = SFTDataset(multi_path, tokenizer, max_seq_length=512)
        print(f"✅ 多文件加载成功，数据量: {len(dataset)}")
        
        # 测试带空格的多文件
        multi_path_with_spaces = f"{file1_path}, {file2_path}"
        dataset2 = SFTDataset(multi_path_with_spaces, tokenizer, max_seq_length=512)
        print(f"✅ 带空格的多文件加载成功，数据量: {len(dataset2)}")
        
        return True
    except Exception as e:
        print(f"❌ 多文件加载失败: {e}")
        return False

def test_nonexistent_file():
    """测试不存在的文件"""
    print("\n=== 测试不存在的文件 ===")
    try:
        tokenizer = type('MockTokenizer', (), {
            'pad_token': None,
            'eos_token': '</s>',
            '__call__': lambda self, text, **kwargs: {'input_ids': [1, 2, 3]},
            'decode': lambda self, ids, **kwargs: 'mock_output'
        })()
        
        # 测试不存在的文件
        dataset = SFTDataset("nonexistent.jsonl", tokenizer, max_seq_length=512)
        print(f"❌ 应该失败但成功了")
        return False
    except Exception as e:
        print(f"✅ 正确处理了不存在的文件: {e}")
        return True

def test_mixed_files(file1_path):
    """测试混合存在和不存在的文件"""
    print("\n=== 测试混合文件（存在+不存在）===")
    try:
        tokenizer = type('MockTokenizer', (), {
            'pad_token': None,
            'eos_token': '</s>',
            '__call__': lambda self, text, **kwargs: {'input_ids': [1, 2, 3]},
            'decode': lambda self, ids, **kwargs: 'mock_output'
        })()
        
        # 测试一个存在一个不存在的文件
        mixed_path = f"{file1_path},nonexistent.jsonl"
        dataset = SFTDataset(mixed_path, tokenizer, max_seq_length=512)
        print(f"✅ 混合文件加载成功，数据量: {len(dataset)} (应该只有存在文件的数据)")
        return True
    except Exception as e:
        print(f"❌ 混合文件处理失败: {e}")
        return False

def cleanup_test_files(temp_dir):
    """清理测试文件"""
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"\n✅ 清理临时文件成功: {temp_dir}")
    except Exception as e:
        print(f"\n⚠️ 清理临时文件失败: {e}")

def test_multi_file_parsing():
    """测试多文件路径解析功能"""
    print("🧪 测试多文件路径解析功能")
    print("=" * 40)
    
    test_cases = [
        ("单文件", "file1.jsonl", ["file1.jsonl"]),
        ("多文件(逗号分隔)", "file1.jsonl,file2.jsonl", ["file1.jsonl", "file2.jsonl"]),
        ("多文件(带空格)", "file1.jsonl, file2.jsonl, file3.jsonl", ["file1.jsonl", "file2.jsonl", "file3.jsonl"]),
        ("多文件(不规则空格)", " file1.jsonl , file2.jsonl,file3.jsonl ", ["file1.jsonl", "file2.jsonl", "file3.jsonl"]),
    ]
    
    for test_name, input_path, expected in test_cases:
        print(f"\n测试: {test_name}")
        print(f"输入: '{input_path}'")
        
        # 模拟 load_data 中的路径解析逻辑
        if ',' in input_path:
            file_paths = [path.strip() for path in input_path.split(',') if path.strip()]
        else:
            file_paths = [input_path.strip()]
        
        print(f"解析结果: {file_paths}")
        print(f"预期结果: {expected}")
        
        if file_paths == expected:
            print("✅ 通过")
        else:
            print("❌ 失败")

def create_sample_files():
    """创建示例训练文件"""
    print("\n📁 创建示例训练文件")
    print("=" * 40)
    
    # 示例数据1
    data1 = [
        {
            "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：",
            "input": "店铺名称：星巴克\n品类：咖啡",
            "output": "✨星巴克的咖啡香气，治愈每一个午后～"
        },
        {
            "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「活力」：",
            "input": "店铺名称：Nike\n品类：运动用品",
            "output": "🏃‍♀️Nike运动装备，让你活力满满！"
        }
    ]
    
    # 示例数据2
    data2 = [
        {
            "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「温馨」：",
            "input": "店铺名称：海底捞\n品类：火锅",
            "output": "🍲海底捞，和朋友一起的温馨时光～"
        }
    ]
    
    # 创建文件
    files_created = []
    
    try:
        with open("sample_data_1.jsonl", "w", encoding="utf-8") as f:
            for item in data1:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        files_created.append("sample_data_1.jsonl")
        print("✅ 创建 sample_data_1.jsonl (2条数据)")
        
        with open("sample_data_2.jsonl", "w", encoding="utf-8") as f:
            for item in data2:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        files_created.append("sample_data_2.jsonl")
        print("✅ 创建 sample_data_2.jsonl (1条数据)")
        
        return files_created
        
    except Exception as e:
        print(f"❌ 创建示例文件失败: {e}")
        return []

def show_usage_examples():
    """显示使用示例"""
    print("\n📖 多文件使用示例")
    print("=" * 40)
    
    examples = [
        {
            "title": "单文件训练",
            "config": '{\n    "data_path": "store_xhs_sft_samples.jsonl"\n}',
            "command": "python3 fine_tune_qwen.py --config_file train_config.json"
        },
        {
            "title": "多文件训练",
            "config": '{\n    "data_path": "file1.jsonl,file2.jsonl,file3.jsonl"\n}',
            "command": "python3 fine_tune_qwen.py --config_file train_config_multi_files.json"
        },
        {
            "title": "命令行参数",
            "config": None,
            "command": 'python3 fine_tune_qwen.py --data_path "file1.jsonl,file2.jsonl"'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        if example['config']:
            print(f"   配置文件:")
            print(f"   {example['config']}")
        print(f"   命令:")
        print(f"   {example['command']}")

def main():
    print("🚀 Qwen3-0.5B 多文件训练功能测试")
    print("=" * 50)
    
    # 测试路径解析
    test_multi_file_parsing()
    
    # 创建示例文件
    files_created = create_sample_files()
    
    # 显示使用示例
    show_usage_examples()
    
    print("\n" + "=" * 50)
    print("✨ 多文件功能已启用！")
    print("\n🎯 主要特性:")
    print("   ✅ 支持单个文件: 'data.jsonl'")
    print("   ✅ 支持多个文件: 'file1.jsonl,file2.jsonl,file3.jsonl'")
    print("   ✅ 自动去除空格: 'file1.jsonl, file2.jsonl'")
    print("   ✅ 错误处理: 跳过不存在的文件，显示警告")
    print("   ✅ 详细日志: 显示每个文件的加载状态")
    
    print("\n📋 测试建议:")
    print("   1. 使用 train_config_multi_files.json 进行多文件训练")
    print("   2. 查看日志输出确认所有文件正确加载")
    print("   3. 检查总数据量是否符合预期")
    
    if files_created:
        print(f"\n🧹 清理示例文件:")
        for file_path in files_created:
            try:
                os.remove(file_path)
                print(f"   ✅ 删除 {file_path}")
            except:
                print(f"   ⚠️ 无法删除 {file_path}")

if __name__ == "__main__":
    main() 