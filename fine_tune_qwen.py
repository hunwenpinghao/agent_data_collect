#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ModelScope 微调 Qwen3-0.5B 模型
用于小红书风格店铺文案生成任务
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

# 设置兼容性导入
try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        set_seed
    )
    from torch.utils.data import Dataset
    from modelscope import snapshot_download
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了正确版本的依赖包:")
    print("pip install torch==2.1.0 transformers==4.36.2 modelscope")
    raise

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="qwen/Qwen3-0.5B-Instruct",
        metadata={"help": "模型名称或路径"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型缓存目录"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="store_xhs_sft_samples.jsonl",
        metadata={"help": "训练数据文件路径，支持单个文件或多个文件（逗号分隔）"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """训练相关参数"""
    output_dir: str = field(default="./output_qwen")
    overwrite_output_dir: bool = field(default=True)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    num_train_epochs: int = field(default=3)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    fp16: bool = field(default=True)
    deepspeed: Optional[str] = field(default=None)

class SFTDataset(Dataset):
    """SFT数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        """加载JSONL格式的数据，支持单个文件或多个文件"""
        data = []
        
        # 解析文件路径，支持逗号分隔的多个文件
        if ',' in data_path:
            file_paths = [path.strip() for path in data_path.split(',') if path.strip()]
        else:
            file_paths = [data_path.strip()]
        
        logger.info(f"准备加载 {len(file_paths)} 个数据文件: {file_paths}")
        
        total_loaded = 0
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在，跳过: {file_path}")
                continue
                
            file_data = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                if all(key in item for key in ['instruction', 'input', 'output']):
                                    file_data.append(item)
                                else:
                                    logger.warning(f"{file_path} 第{line_num}行缺少必要字段，跳过")
                            except json.JSONDecodeError as e:
                                logger.warning(f"{file_path} 第{line_num}行JSON解析失败: {e}")
                                continue
                
                logger.info(f"从 {file_path} 加载了 {len(file_data)} 条数据")
                data.extend(file_data)
                total_loaded += len(file_data)
                
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
                raise
        
        if total_loaded == 0:
            raise ValueError("没有加载到任何有效的训练数据")
            
        logger.info(f"总共成功加载 {total_loaded} 条训练数据")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建对话格式
        instruction = item['instruction']
        input_text = item['input']
        output_text = item['output']
        
        # 组合输入文本
        if input_text.strip():
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
            
        # 构建完整对话
        conversation = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
        # 编码
        model_inputs = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        # 创建标签，只计算assistant部分的损失
        labels = model_inputs["input_ids"].copy()
        
        # 找到assistant开始的位置
        assistant_start = conversation.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        user_part = conversation[:assistant_start]
        user_tokens = self.tokenizer(user_part, add_special_tokens=False)["input_ids"]
        
        # 将用户部分的标签设为-100（不计算损失）
        for i in range(min(len(user_tokens), len(labels))):
            labels[i] = -100
            
        model_inputs["labels"] = labels
        
        return model_inputs

def download_model(model_name: str, cache_dir: str = "./models") -> str:
    """从ModelScope下载模型"""
    try:
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        logger.info(f"模型下载完成: {model_dir}")
        return model_dir
    except Exception as e:
        logger.error(f"模型下载失败: {e}")
        raise

def main():
    # 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # 检查是否有配置文件参数
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == '--config_file':
        config_file = sys.argv[2]
        logger.info(f"使用配置文件: {config_file}")
        
        # 读取JSON配置文件并过滤掉非参数字段
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 过滤掉以下划线开头的字段（通常是注释或文档字段）
        filtered_config = {k: v for k, v in config.items() if not k.startswith('_')}
        logger.info(f"过滤后的配置参数: {list(filtered_config.keys())}")
        
        model_args, data_args, training_args = parser.parse_dict(filtered_config)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed if hasattr(training_args, 'seed') else 42)
    
    # 创建输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 下载模型
    logger.info("正在下载模型...")
    model_path = download_model(model_args.model_name_or_path, model_args.cache_dir)
    
    # 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    logger.info("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 确保模型支持梯度检查点
    model.gradient_checkpointing_enable()
    
    # 创建数据集
    logger.info("创建训练数据集...")
    train_dataset = SFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length
    )
    
    # 创建验证数据集（这里简单地使用训练数据的一部分）
    eval_dataset = train_dataset
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main() 