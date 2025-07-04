#!/usr/bin/env python3
"""
LoRA模型上传准备脚本
筛选需要上传到魔搭社区的文件，排除训练临时文件
"""

import os
import shutil
import argparse
from pathlib import Path
import json

def create_lora_readme(output_dir, model_name):
    """为LoRA模型创建README.md"""
    readme_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- qwen
- lora
- fine-tuned
- chinese
library_name: peft
---

# {model_name}

## 模型描述

这是基于 Qwen2.5-0.5B-Instruct 微调的 LoRA 适配器模型，专门针对小红书数据进行了优化。

## 模型信息

- **基础模型**: Qwen/Qwen2.5-0.5B-Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **LoRA 参数**: r=64, alpha=16, dropout=0.1
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## 使用方法

### 通过 ModelScope 使用

```python
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型和分词器
base_model_path = "qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载LoRA适配器
model = PeftModel.from_pretrained(model, "hunwenpinghao/{model_name}")

# 生成文本
def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例使用
result = generate_text("小红书种草文案：")
print(result)
```

### 通过 transformers + peft 使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "hunwenpinghao/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 推理
prompt = "今天的心情："
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 训练详情

- **训练数据**: 小红书相关数据集
- **训练框架**: Transformers + PEFT
- **硬件**: GPU 训练
- **优化器**: AdamW

## 许可证

Apache-2.0

## 注意事项

这是一个 LoRA 适配器模型，需要与基础模型 Qwen/Qwen2.5-0.5B-Instruct 一起使用。

## 引用

```bibtex
@misc{{{model_name.replace('-', '_')},
  author = {{hunwenpinghao}},
  title = {{{model_name}}},
  year = {{2025}},
  publisher = {{ModelScope}},
  journal = {{ModelScope Repository}},
  howpublished = {{\\url{{https://modelscope.cn/models/hunwenpinghao/{model_name}}}}}
}}
```
"""
    
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ 已创建 LoRA 模型说明: {readme_path}")

def prepare_lora_files(source_dir, output_dir, model_name="zhc_xhs_qwen2.5_0.5b_instruct"):
    """准备LoRA模型上传文件"""
    
    # 需要上传的文件列表
    required_files = [
        'adapter_model.safetensors',  # LoRA权重
        'adapter_config.json',        # LoRA配置
        'tokenizer.json',             # 分词器
        'tokenizer_config.json',      # 分词器配置
        'vocab.json',                 # 词汇表
        'merges.txt',                 # BPE合并规则
        'special_tokens_map.json',    # 特殊token映射
        'added_tokens.json',          # 添加的token
        'configuration.json',         # 模型配置
    ]
    
    # 可选文件
    optional_files = [
        '.gitattributes',             # Git属性
    ]
    
    # 不需要的文件（训练临时文件）
    skip_files = [
        'optimizer.pt',               # 优化器状态
        'scaler.pt',                  # 梯度缩放器
        'scheduler.pt',               # 调度器状态
        'rng_state.pth',             # 随机数状态
        'training_args.bin',          # 训练参数
        'trainer_state.json',         # 训练器状态
        'README.md',                  # 原始README（会重新生成）
    ]
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
    
    total_size = 0
    copied_files = 0
    skipped_files = 0
    
    # 复制必需文件
    for filename in required_files:
        source_file = source_path / filename
        if source_file.exists():
            dest_file = output_path / filename
            shutil.copy2(source_file, dest_file)
            size = source_file.stat().st_size
            total_size += size
            copied_files += 1
            size_str = format_size(size)
            print(f"✅ {filename:<30} {size_str:>10} - 已复制")
        else:
            print(f"⚠️  {filename:<30} {'缺失':>10} - 必需文件缺失")
    
    # 复制可选文件
    for filename in optional_files:
        source_file = source_path / filename
        if source_file.exists():
            dest_file = output_path / filename
            shutil.copy2(source_file, dest_file)
            size = source_file.stat().st_size
            total_size += size
            copied_files += 1
            size_str = format_size(size)
            print(f"✅ {filename:<30} {size_str:>10} - 已复制（可选）")
    
    # 显示跳过的文件
    for filename in skip_files:
        source_file = source_path / filename
        if source_file.exists():
            size = source_file.stat().st_size
            size_str = format_size(size)
            skipped_files += 1
            print(f"⏭️  {filename:<30} {size_str:>10} - 已跳过（训练文件）")
    
    # 跳过 .git 目录
    git_dir = source_path / '.git'
    if git_dir.exists():
        print(f"⏭️  {'.git/':<30} {'目录':>10} - 已跳过（版本控制）")
        skipped_files += 1
    
    # 创建新的README
    create_lora_readme(output_dir, model_name)
    copied_files += 1
    
    print("=" * 60)
    print(f"📊 总结:")
    print(f"   已复制文件: {copied_files}")
    print(f"   跳过文件: {skipped_files}")
    print(f"   总大小: {format_size(total_size)}")
    
    return output_path

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def main():
    parser = argparse.ArgumentParser(description="准备LoRA模型上传文件")
    parser.add_argument("--source_dir", type=str,
                       default="../output_qwen/zhc_xhs_qwen2.5_0.5b_instruct",
                       help="LoRA模型源目录")
    parser.add_argument("--output_dir", type=str,
                       default="../upload_ready/zhc_xhs_qwen2.5_0.5b_instruct",
                       help="准备上传的目录")
    parser.add_argument("--model_name", type=str,
                       default="zhc_xhs_qwen2.5_0.5b_instruct",
                       help="模型名称")
    
    args = parser.parse_args()
    
    print("🔧 LoRA模型上传准备工具")
    print("=" * 30)
    
    if not Path(args.source_dir).exists():
        print(f"❌ 源目录不存在: {args.source_dir}")
        return 1
    
    try:
        output_path = prepare_lora_files(args.source_dir, args.output_dir, args.model_name)
        
        print("\n🎉 准备完成！")
        print(f"📁 上传目录: {output_path}")
        print("\n🚀 下一步:")
        print(f"   cd upload_to_modelscope")
        print(f"   python3 upload_to_modelscope.py --model_dir {output_path} --token YOUR_TOKEN")
        
        return 0
        
    except Exception as e:
        print(f"❌ 准备失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 