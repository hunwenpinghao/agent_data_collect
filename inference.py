#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用微调后的模型进行推理
生成小红书风格的店铺文案
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json

def load_model_and_tokenizer(model_path: str):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_text(model, tokenizer, instruction: str, input_text: str, max_length: int = 2048):
    """生成文本"""
    # 构建prompt
    if input_text.strip():
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction
    
    # 构建对话格式
    conversation = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant的回复
    assistant_start = generated_text.find("<|im_start|>assistant\n")
    if assistant_start != -1:
        response = generated_text[assistant_start + len("<|im_start|>assistant\n"):].strip()
        if response.endswith("<|im_end|>"):
            response = response[:-len("<|im_end|>")].strip()
        return response
    else:
        return generated_text

def main():
    parser = argparse.ArgumentParser(description="使用微调后的模型生成小红书文案")
    parser.add_argument("--model_path", type=str, default="./output_qwen", 
                       help="微调后的模型路径")
    parser.add_argument("--interactive", action="store_true", 
                       help="交互式模式")
    parser.add_argument("--test_file", type=str, 
                       help="测试文件路径（JSONL格式）")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    if args.interactive:
        print("=== 小红书文案生成器 ===")
        print("请输入店铺信息，生成小红书风格文案")
        print("输入 'quit' 退出程序")
        print()
        
        while True:
            print("请输入指令（例如：根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：）:")
            instruction = input("指令: ").strip()
            
            if instruction.lower() == 'quit':
                break
                
            print("请输入店铺信息:")
            input_text = input("店铺信息: ").strip()
            
            if not instruction:
                instruction = "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」："
            
            print("\n生成中...")
            response = generate_text(model, tokenizer, instruction, input_text)
            
            print(f"\n生成的文案:\n{response}")
            print("=" * 50)
            print()
    
    elif args.test_file:
        print(f"使用测试文件: {args.test_file}")
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                instruction = data.get('instruction', '')
                input_text = data.get('input', '')
                expected_output = data.get('output', '')
                
                print(f"\n=== 测试样例 {i+1} ===")
                print(f"指令: {instruction}")
                print(f"输入: {input_text}")
                print(f"期望输出: {expected_output}")
                
                generated = generate_text(model, tokenizer, instruction, input_text)
                print(f"生成输出: {generated}")
                print("=" * 50)
                
                # 询问是否继续
                if i < 5:  # 只显示前5个样例
                    continue
                else:
                    user_input = input("继续显示更多样例? (y/n): ")
                    if user_input.lower() != 'y':
                        break
    
    else:
        # 默认测试
        test_cases = [
            {
                "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：",
                "input": "店铺名称：星巴克\n品类：咖啡\n地址：郑州正弘城L8\n营业时间：10:00-22:00\n环境风格：现代简约,安静,明亮\n配套设施：有停车场,有WIFI,商场公共卫生间"
            },
            {
                "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：",
                "input": "店铺名称：海底捞\n品类：火锅\n地址：郑州正弘城L7\n营业时间：10:00-22:00\n环境风格：热闹,有包厢\n配套设施：有停车场,有包厢,有WIFI"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n=== 测试样例 {i+1} ===")
            print(f"指令: {test_case['instruction']}")
            print(f"输入: {test_case['input']}")
            
            response = generate_text(model, tokenizer, test_case['instruction'], test_case['input'])
            print(f"生成的文案:\n{response}")
            print("=" * 50)

if __name__ == "__main__":
    main() 