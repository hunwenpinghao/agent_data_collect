#!/usr/bin/env python3
"""
转换模型格式以适配火山方舟
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def convert_lora_to_full_model(base_model_path, lora_path, output_path):
    """将LoRA模型合并为完整模型"""
    print(f"加载基础模型: {base_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    print(f"加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并LoRA权重
    print("合并LoRA权重...")
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    print(f"保存合并后的模型: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("✅ 模型转换完成")

def main():
    parser = argparse.ArgumentParser(description="转换LoRA模型为完整模型")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="LoRA模型路径")
    parser.add_argument("--output_path", type=str, default="./merged_model",
                       help="输出路径")
    
    args = parser.parse_args()
    
    # 检查LoRA路径是否存在
    if not os.path.exists(args.lora_path):
        print(f"❌ LoRA路径不存在: {args.lora_path}")
        exit(1)
    
    # 执行转换
    convert_lora_to_full_model(args.base_model, args.lora_path, args.output_path)

if __name__ == "__main__":
    main() 