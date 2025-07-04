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
    
    try:
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
        
        try:
            model = PeftModel.from_pretrained(base_model, lora_path)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                print(f"❌ PEFT版本兼容性错误: {e}")
                print("💡 请先运行修复脚本: python deploy/fix_lora_config.py")
                return False
            else:
                raise e
        
        # 合并LoRA权重
        print("合并LoRA权重...")
        model = model.merge_and_unload()
        
        # 保存合并后的模型
        print(f"保存合并后的模型: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        print("✅ 模型转换完成")
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="转换LoRA模型为完整模型")
    parser.add_argument("--base_model", type=str, default="models/Qwen/Qwen2.5-0.5B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--lora_path", type=str, 
                       default="output_deepspeed/zhc_xhs_qwen2.5_0.5b_lora",
                       help="LoRA模型路径")
    parser.add_argument("--output_path", type=str, default="./merged_model",
                       help="输出路径")
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.base_model):
        print(f"❌ 基础模型路径不存在: {args.base_model}")
        exit(1)
    
    if not os.path.exists(args.lora_path):
        print(f"❌ LoRA路径不存在: {args.lora_path}")
        exit(1)
    
    # 执行转换
    success = convert_lora_to_full_model(args.base_model, args.lora_path, args.output_path)
    
    if success:
        print(f"\n🎉 转换成功！合并后的模型保存在: {args.output_path}")
    else:
        print(f"\n❌ 转换失败！")
        exit(1)

if __name__ == "__main__":
    main() 