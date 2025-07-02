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

# 设置国内环境的镜像源
def setup_china_mirror():
    """为大陆环境设置镜像源"""
    try:
        # 设置HuggingFace镜像
        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        print("✅ 已设置HuggingFace镜像源: https://hf-mirror.com")
        return True
    except Exception as e:
        print(f"设置镜像源失败: {e}")
        return False

# 首先尝试设置镜像源
setup_china_mirror()

# 设置兼容性环境变量
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', 'true')

# 设置兼容性导入
try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        set_seed,
        BitsAndBytesConfig
    )
    from torch.utils.data import Dataset
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy
    
    # ModelScope将在需要时延迟导入，避免启动时的兼容性问题
    print("📦 ModelScope: 将在需要时动态导入")
    
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training
    )
    import bitsandbytes as bnb
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了正确版本的依赖包:")
    print("pip install torch==2.1.0 transformers>=4.36.2 peft bitsandbytes")
    print("大陆用户可选择安装: pip install modelscope")
    raise

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "模型名称或路径"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型缓存目录"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA微调"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "是否使用QLoRA微调（需要量化）"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA的rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA的alpha参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA的dropout率"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA目标模块，逗号分隔"}
    )
    quantization_bit: int = field(
        default=4,
        metadata={"help": "量化位数，支持4或8位"}
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
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据文件路径，可选，支持单个文件或多个文件（逗号分隔）"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
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
    evaluation_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    eval_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    save_strategy: SaveStrategy = field(default=SaveStrategy.STEPS)
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

def get_model_path(model_name: str, cache_dir: str = "./models") -> str:
    """获取模型路径，优先使用本地模型，其次使用HuggingFace镜像"""
    logger.info(f"准备加载模型: {model_name}")
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 首先检查是否有本地模型
    local_model_paths = [
        f"{cache_dir}/{model_name}",
        f"{cache_dir}/{model_name.split('/')[-1]}",  # 自定义缓存目录
        f"models/{model_name.split('/')[-1]}",  # models/Qwen2.5-0.5B-Instruct
        model_name  # 如果已经是本地路径
    ]
    
    for local_path in local_model_paths:
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            logger.info(f"✅ 发现本地模型: {local_path}")
            return os.path.abspath(local_path)
    
    # 如果没有本地模型，直接使用模型名称，让transformers处理下载
    # HuggingFace镜像已在启动时设置
    logger.info("📥 将从HuggingFace镜像下载模型...")
    logger.info(f"   镜像源: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    logger.info("   如果下载失败，请运行: ./download_model.sh")
    
    return model_name

def create_quantization_config(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    """创建量化配置"""
    if not model_args.use_qlora:
        return None
    
    if model_args.quantization_bit == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.quantization_bit == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"不支持的量化位数: {model_args.quantization_bit}")
    
    logger.info(f"使用{model_args.quantization_bit}位量化")
    return quantization_config

def create_lora_config(model_args: ModelArguments) -> Optional[LoraConfig]:
    """创建LoRA配置"""
    if not (model_args.use_lora or model_args.use_qlora):
        return None
    
    target_modules = model_args.lora_target_modules.split(',') if model_args.lora_target_modules else None
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    logger.info(f"LoRA配置: r={model_args.lora_r}, alpha={model_args.lora_alpha}, "
                f"dropout={model_args.lora_dropout}, target_modules={target_modules}")
    return lora_config

def find_all_linear_names(model):
    """找到模型中所有的线性层名称"""
    cls = bnb.nn.Linear4bit if hasattr(bnb.nn, 'Linear4bit') else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main():
    # 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
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
    
    # 获取模型路径
    logger.info("准备加载模型...")
    model_path = get_model_path(model_args.model_name_or_path, model_args.cache_dir)
    
    # 加载tokenizer
    logger.info("加载tokenizer...")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"缓存目录: {model_args.cache_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            cache_dir=model_args.cache_dir
        )
        logger.info("✅ Tokenizer加载成功")
    except Exception as e:
        logger.error(f"❌ Tokenizer加载失败: {e}")
        logger.info("尝试解决方案...")
        
        # 尝试使用不同的tokenizer配置
        try:
            logger.info("尝试方案1: 使用fast tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                cache_dir=model_args.cache_dir
            )
            logger.info("✅ Fast tokenizer加载成功")
        except Exception as e2:
            logger.error(f"❌ Fast tokenizer也失败: {e2}")
            
            # 如果是Qwen模型，尝试使用备选tokenizer
            if "Qwen" in model_path:
                try:
                    logger.info("尝试方案2: 使用Qwen2Tokenizer")
                    from transformers import Qwen2Tokenizer
                    tokenizer = Qwen2Tokenizer.from_pretrained(
                        model_path,
                        cache_dir=model_args.cache_dir
                    )
                    logger.info("✅ Qwen2Tokenizer加载成功")
                except Exception as e3:
                    logger.error(f"❌ Qwen2Tokenizer也失败: {e3}")
                    
                    logger.error("所有tokenizer加载方案都失败，请检查模型文件是否完整")
                    logger.info("建议解决方案:")
                    logger.info("1. 清空缓存目录: rm -rf ./models")
                    logger.info("2. 重新下载模型: ./download_model.sh")
                    logger.info("3. 或使用本地模型路径")
                    raise e3
            else:
                raise e2
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建量化配置
    quantization_config = create_quantization_config(model_args)
    
    # 加载模型
    logger.info("加载模型...")
    
    # 基础参数
    base_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
    }
    
    if quantization_config is not None:
        base_kwargs["quantization_config"] = quantization_config
        base_kwargs["torch_dtype"] = torch.float16
    else:
        base_kwargs["torch_dtype"] = torch.float16 if training_args.fp16 else torch.float32
    
    # 尝试不同的加载方案
    loading_strategies = [
        # 方案1: 基础加载，不使用 device_map 和 attn_implementation
        base_kwargs,
        
        # 方案2: 添加 low_cpu_mem_usage
        {**base_kwargs, "low_cpu_mem_usage": True},
        
        # 方案3: 使用 device_map="cpu"
        {**base_kwargs, "device_map": "cpu"},
        
        # 方案4: 添加 attn_implementation
        {**base_kwargs, "attn_implementation": "eager"},
        
        # 方案5: 完整参数
        {**base_kwargs, "device_map": "auto", "attn_implementation": "eager"},
    ]
    
    model = None
    for i, kwargs in enumerate(loading_strategies, 1):
        try:
            logger.info(f"尝试加载方案 {i}: {list(kwargs.keys())}")
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            logger.info(f"✅ 方案 {i} 加载成功")
            break
        except Exception as e:
            logger.warning(f"❌ 方案 {i} 失败: {e}")
            if i == len(loading_strategies):
                logger.error(f"所有加载方案都失败，最后错误: {e}")
                raise e
    
    # 如果使用QLoRA，准备模型进行k-bit训练
    if model_args.use_qlora:
        logger.info("准备模型进行k-bit训练...")
        model = prepare_model_for_kbit_training(model)
    
    # 如果使用LoRA或QLoRA，应用LoRA配置
    if model_args.use_lora or model_args.use_qlora:
        logger.info("应用LoRA配置...")
        lora_config = create_lora_config(model_args)
        
        # 如果没有指定target_modules，自动发现
        if lora_config.target_modules is None:
            logger.info("自动发现LoRA目标模块...")
            lora_config.target_modules = find_all_linear_names(model)
            logger.info(f"发现的目标模块: {lora_config.target_modules}")
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # 确保模型支持梯度检查点（仅在非LoRA模式下）
        model.gradient_checkpointing_enable()
    
    # 创建数据集
    logger.info("创建训练数据集...")
    train_dataset = SFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length
    )
    
    # 创建验证数据集
    if data_args.eval_data_path:
        logger.info(f"使用验证数据: {data_args.eval_data_path}")
        eval_dataset = SFTDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length
        )
    else:
        logger.info("未指定验证数据，使用训练数据作为验证集")
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
    if model_args.use_lora or model_args.use_qlora:
        # 保存LoRA适配器
        model.save_pretrained(training_args.output_dir)
        logger.info(f"LoRA适配器已保存到: {training_args.output_dir}")
        
        # 如果需要，也可以保存合并后的模型
        # merged_model = model.merge_and_unload()
        # merged_model.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))
    else:
        # 保存完整模型
        trainer.save_model()
    
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main() 