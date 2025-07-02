#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA/QLoRA微调模型推理脚本
支持加载LoRA适配器进行推理
"""

import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRAInference:
    def __init__(self, base_model_path: str, lora_model_path: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
        """
        初始化LoRA推理器
        
        Args:
            base_model_path: 基础模型路径
            lora_model_path: LoRA适配器路径
            load_in_4bit: 是否使用4bit量化加载基础模型
            load_in_8bit: 是否使用8bit量化加载基础模型
        """
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            trust_remote_code=True,
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置量化配置
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 加载基础模型
        logger.info("加载基础模型...")
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            **model_kwargs
        )
        
        # 加载LoRA适配器
        logger.info("加载LoRA适配器...")
        self.model = PeftModel.from_pretrained(self.base_model, lora_model_path)
        logger.info("模型加载完成！")
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                     top_p: float = 0.9, do_sample: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top_p采样参数
            do_sample: 是否使用采样
            
        Returns:
            生成的文本
        """
        # 构建对话格式
        conversation = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入
        inputs = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取assistant的回复
        assistant_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        response = generated_text[assistant_start:].strip()
        
        return response
    
    def chat(self):
        """交互式对话"""
        print("LoRA模型推理系统已启动！输入'quit'退出。")
        print("-" * 50)
        
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            try:
                response = self.generate_text(user_input)
                print(f"助手: {response}")
                print("-" * 50)
            except Exception as e:
                print(f"生成错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="LoRA模型推理")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA适配器路径") 
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化")
    parser.add_argument("--load_in_8bit", action="store_true", help="使用8bit量化")
    parser.add_argument("--prompt", type=str, default="", help="单次推理的提示文本")
    parser.add_argument("--chat", action="store_true", help="启动交互式对话模式")
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = LoRAInference(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    
    if args.chat:
        # 交互式对话
        inference.chat()
    elif args.prompt:
        # 单次推理
        response = inference.generate_text(args.prompt)
        print(f"输入: {args.prompt}")
        print(f"输出: {response}")
    else:
        # 示例推理
        example_prompts = [
            "请为一家咖啡店写一段小红书风格的文案",
            "如何制作一杯好喝的拿铁咖啡？",
            "推荐几家上海的网红咖啡店"
        ]
        
        print("示例推理：")
        for i, prompt in enumerate(example_prompts, 1):
            print(f"\n示例 {i}:")
            print(f"输入: {prompt}")
            response = inference.generate_text(prompt)
            print(f"输出: {response}")
            print("-" * 50)

if __name__ == "__main__":
    main() 