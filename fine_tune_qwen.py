#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 transformers 微调 Qwen2.5-0.5B 模型
用于小红书风格店铺文案生成任务
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

# 设置国内环境的镜像源和警告抑制
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', 'true')

# 禁用TensorFlow相关警告
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# DeepSpeed相关环境变量将在main函数中根据配置动态设置
os.environ.setdefault('DEEPSPEED_LOG_LEVEL', 'WARNING')

# 禁用Python警告
import warnings
warnings.filterwarnings('ignore')

# 导入必要的库
try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        set_seed,
        BitsAndBytesConfig,
        AutoConfig
    )
    from torch.utils.data import Dataset
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy
    
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    import bitsandbytes as bnb
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装依赖: pip install torch transformers peft bitsandbytes")
    raise

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 抑制DeepSpeed和其他第三方库的冗余日志
logging.getLogger('deepspeed').setLevel(logging.WARNING)
logging.getLogger('transformers.modeling_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.configuration_utils').setLevel(logging.WARNING)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.WARNING)
logging.getLogger('accelerate.utils.other').setLevel(logging.ERROR)  # 抑制内核版本警告

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
    use_deepspeed: bool = field(
        default=False,
        metadata={"help": "是否使用DeepSpeed进行训练"}
    )
    deepspeed_stage: int = field(
        default=2,
        metadata={"help": "DeepSpeed ZeRO阶段，支持1,2,3"}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "是否启用CPU offload优化器状态"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="data/zhc_store_recommend_doubao.jsonl",
        metadata={"help": "训练数据文件路径，支持单个文件或多个文件（逗号分隔）"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据文件路径，可选"}
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
    save_strategy: SaveStrategy = field(default=SaveStrategy.STEPS)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    fp16: bool = field(default=True)
    label_names: Optional[List[str]] = field(default_factory=lambda: ["labels"])
    deepspeed: Optional[str] = field(default=None)
    report_to: str = field(default="tensorboard")

class SFTDataset(Dataset):
    """SFT数据集类"""
    
    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        """加载JSONL格式的数据"""
        data = []
        
        # 支持逗号分隔的多个文件
        if ',' in data_path:
            file_paths = [path.strip() for path in data_path.split(',') if path.strip()]
        else:
            file_paths = [data_path.strip()]
        
        logger.info(f"准备加载 {len(file_paths)} 个数据文件")
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在，跳过: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if all(key in item for key in ['instruction', 'input', 'output']):
                                data.append(item)
                            else:
                                logger.warning(f"{file_path} 第{line_num}行缺少必要字段")
                        except json.JSONDecodeError as e:
                            logger.warning(f"{file_path} 第{line_num}行JSON解析失败: {e}")
        
        if not data:
            raise ValueError("没有加载到任何有效的训练数据")
            
        logger.info(f"成功加载 {len(data)} 条训练数据")
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
    """获取模型路径，优先使用本地模型"""
    logger.info(f"准备加载模型: {model_name}")
    
    # 检查本地模型
    local_paths = [
        f"{cache_dir}/{model_name}",
        f"{cache_dir}/{model_name.split('/')[-1]}",
        f"models/{model_name.split('/')[-1]}",
        model_name
    ]
    
    for local_path in local_paths:
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            logger.info(f"使用本地模型: {local_path}")
            return os.path.abspath(local_path)
    
    logger.info(f"从HuggingFace下载模型: {model_name}")
    return model_name

def create_quantization_config(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    """创建量化配置"""
    if not model_args.use_qlora:
        return None
    
    if model_args.quantization_bit == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.quantization_bit == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"不支持的量化位数: {model_args.quantization_bit}")

def create_lora_config(model_args: ModelArguments) -> Optional[LoraConfig]:
    """创建LoRA配置"""
    if not (model_args.use_lora or model_args.use_qlora):
        return None
    
    target_modules = model_args.lora_target_modules.split(',') if model_args.lora_target_modules else None
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

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

def create_deepspeed_config(model_args: ModelArguments, training_args: CustomTrainingArguments) -> Dict[str, Any]:
    """创建DeepSpeed配置"""
    if not model_args.use_deepspeed:
        return None
    
    # 基础配置
    config = {
        "bf16": {
            "enabled": not training_args.fp16
        },
        "fp16": {
            "enabled": training_args.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_args.weight_decay if hasattr(training_args, 'weight_decay') else 0.0
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_args.learning_rate,
                "warmup_num_steps": training_args.warmup_steps
            }
        },
        "zero_optimization": {
            "stage": model_args.deepspeed_stage,
            "offload_optimizer": {
                "device": "cpu" if (model_args.deepspeed_stage == 3 or model_args.cpu_offload) else "none"
            },
            "offload_param": {
                "device": "cpu" if model_args.deepspeed_stage == 3 else "none"
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": training_args.logging_steps,
        "train_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * torch.cuda.device_count(),
        "train_micro_batch_size_per_gpu": training_args.per_device_train_batch_size,
        "wall_clock_breakdown": False
    }
    
    # 如果使用LoRA，调整配置
    if model_args.use_lora or model_args.use_qlora:
        config["zero_optimization"]["stage"] = min(model_args.deepspeed_stage, 2)  # LoRA不支持stage 3
        
    logger.info(f"DeepSpeed配置: ZeRO Stage {config['zero_optimization']['stage']}")
    return config

def main():
    # 解析参数
    import sys
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    
    if len(sys.argv) >= 3 and sys.argv[1] == '--config_file':
        config_file = sys.argv[2]
        logger.info(f"使用配置文件: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 过滤掉注释字段
        filtered_config = {k: v for k, v in config.items() if not k.startswith('_')}
        model_args, data_args, training_args = parser.parse_dict(filtered_config)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 根据配置设置DeepSpeed相关环境变量
    if model_args.use_deepspeed:
        logger.info("启用DeepSpeed训练")
        os.environ.pop('ACCELERATE_USE_DEEPSPEED', None)  # 移除可能的禁用设置
        os.environ.pop('TRANSFORMERS_NO_DEEPSPEED', None)
    else:
        logger.info("使用标准训练（不使用DeepSpeed）")
        os.environ.setdefault('ACCELERATE_USE_DEEPSPEED', 'false')
        os.environ.setdefault('TRANSFORMERS_NO_DEEPSPEED', 'true')

    # 验证和修复训练参数
    if training_args.load_best_model_at_end:
        if data_args.eval_data_path is None:
            logger.warning("没有设置验证数据路径，但启用了load_best_model_at_end，将禁用此选项")
            training_args.load_best_model_at_end = False
        else:
            # 确保评估策略和保存策略匹配
            if training_args.evaluation_strategy == IntervalStrategy.NO:
                logger.info("检测到评估策略为NO，但需要进行评估，设置为STEPS")
                training_args.evaluation_strategy = IntervalStrategy.STEPS
            
            # 确保评估和保存策略匹配
            if training_args.evaluation_strategy != training_args.save_strategy:
                logger.info(f"调整评估策略以匹配保存策略: {training_args.save_strategy}")
                training_args.evaluation_strategy = training_args.save_strategy
    
    # 设置随机种子
    set_seed(training_args.seed if hasattr(training_args, 'seed') else 42)
    
    # 创建输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 设置TensorBoard日志目录
    if training_args.report_to == "tensorboard":
        tensorboard_log_dir = os.path.join(training_args.output_dir, "tensorboard_logs")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        os.environ['TENSORBOARD_LOG_DIR'] = tensorboard_log_dir
        logger.info(f"TensorBoard日志将保存到: {tensorboard_log_dir}")
        logger.info("启动TensorBoard查看训练进度: tensorboard --logdir=" + tensorboard_log_dir)
    
    # 创建DeepSpeed配置
    deepspeed_config = create_deepspeed_config(model_args, training_args)
    if deepspeed_config:
        # 保存DeepSpeed配置文件
        deepspeed_config_path = os.path.join(training_args.output_dir, "deepspeed_config.json")
        with open(deepspeed_config_path, 'w', encoding='utf-8') as f:
            json.dump(deepspeed_config, f, indent=2, ensure_ascii=False)
        training_args.deepspeed = deepspeed_config_path
        logger.info(f"DeepSpeed配置已保存到: {deepspeed_config_path}")
    
    # 获取模型路径
    model_path = get_model_path(model_args.model_name_or_path, model_args.cache_dir)
    
    # 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir
    )
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建量化配置
    quantization_config = create_quantization_config(model_args)
    
    # 加载模型
    logger.info("加载模型...")
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "torch_dtype": torch.float16,
    }
    
    # DeepSpeed会自动处理device_map，所以只在非DeepSpeed模式下设置
    if not model_args.use_deepspeed:
        model_kwargs["device_map"] = "auto"
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
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
            lora_config.target_modules = find_all_linear_names(model)
            logger.info(f"自动发现的目标模块: {lora_config.target_modules}")
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # 启用梯度检查点
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
        logger.info(f"创建验证数据集: {data_args.eval_data_path}")
        eval_dataset = SFTDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length
        )
    else:
        logger.info("使用训练数据作为验证集")
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
    if training_args.report_to == "tensorboard":
        tensorboard_log_dir = os.path.join(training_args.output_dir, "tensorboard_logs")
        logger.info("📊 TensorBoard已启用！")
        logger.info(f"📈 查看训练进度: tensorboard --logdir={tensorboard_log_dir}")
        logger.info("🌐 然后在浏览器打开: http://localhost:6006")
    
    trainer.train()
    
    # 保存模型
    logger.info("保存模型...")
    if model_args.use_lora or model_args.use_qlora:
        # 保存LoRA适配器
        model.save_pretrained(training_args.output_dir)
        logger.info(f"LoRA适配器已保存到: {training_args.output_dir}")
    else:
        # 保存完整模型
        trainer.save_model()
    
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("训练完成！")

if __name__ == "__main__":
    main() 