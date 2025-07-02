# LoRA/QLoRA 微调使用指南

本项目现已支持 LoRA (Low-Rank Adaptation) 和 QLoRA (Quantized LoRA) 微调方法，可以显著减少显存占用和训练参数量。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_stable.txt
```

确保安装了以下关键依赖：
- `peft>=0.6.2` - LoRA实现
- `bitsandbytes>=0.41.0` - 量化支持
- `transformers>=4.37.0` - 模型库

### 2. 选择微调方式

#### 方式一：全参数微调（原始方式）
- **特点**：微调所有模型参数
- **显存需求**：最高
- **训练效果**：通常最好
- **使用场景**：有充足GPU资源时

#### 方式二：LoRA 微调
- **特点**：只训练低秩适配器参数（~1-5%的原始参数）
- **显存需求**：中等
- **训练效果**：接近全参数微调
- **使用场景**：平衡性能和资源的首选

#### 方式三：QLoRA 微调
- **特点**：4/8位量化 + LoRA
- **显存需求**：最低（可减少50-75%显存）
- **训练效果**：略低于LoRA但仍然很好
- **使用场景**：显存有限的环境

## 📋 配置文件示例

### LoRA 微调配置
```bash
python fine_tune_qwen.py --config_file configs/train_config_lora.json
```

关键参数：
- `use_lora: true` - 启用LoRA
- `lora_r: 64` - LoRA rank，越大效果越好但参数越多
- `lora_alpha: 16` - LoRA scaling参数
- `lora_dropout: 0.1` - LoRA dropout率
- `learning_rate: 1e-4` - LoRA建议使用较高学习率

### QLoRA 微调配置

#### 4位量化 (推荐)
```bash
python fine_tune_qwen.py --config_file configs/train_config_qlora.json
```

#### 8位量化
```bash 
python fine_tune_qwen.py --config_file configs/train_config_qlora_8bit.json
```

关键参数：
- `use_qlora: true` - 启用QLoRA
- `quantization_bit: 4` - 量化位数 (4 或 8)
- `learning_rate: 2e-4` - QLoRA建议使用更高学习率

## 🎯 目标模块选择

LoRA可以选择性地应用到特定模块：

### 推荐配置
```json
"lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
```

### 轻量配置（显存更少）
```json
"lora_target_modules": "q_proj,k_proj,v_proj,o_proj"
```

### 自动发现（推荐）
```json
"lora_target_modules": null
```
- 系统会自动发现所有线性层

## 📊 显存对比

以 Qwen2.5-0.5B 为例：

| 方法 | 显存占用 | 可训练参数 | 相对性能 |
|------|----------|------------|----------|
| 全参数 | ~8GB | 100% | 100% |
| LoRA (r=64) | ~6GB | ~3% | 95-98% |
| QLoRA 4bit (r=64) | ~3GB | ~3% | 90-95% |
| QLoRA 8bit (r=64) | ~4GB | ~3% | 92-96% |

## 🔧 训练命令

### 命令行参数方式
```bash
python fine_tune_qwen.py \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 16 \
    --data_path data/store_xhs_sft_samples.jsonl \
    --output_dir ./output_lora \
    --learning_rate 1e-4
```

### 配置文件方式（推荐）
```bash
# 全参数微调
python fine_tune_qwen.py --config_file configs/train_config_full.json

# LoRA微调
python fine_tune_qwen.py --config_file configs/train_config_lora.json

# QLoRA微调  
python fine_tune_qwen.py --config_file configs/train_config_qlora.json
```

## 📤 模型保存与加载

### 保存
训练完成后，LoRA适配器会保存到输出目录：
```
output_qwen_lora/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files...
```

### 加载推理
```bash
# 基础推理
python inference_lora.py \
    --base_model ./models/Qwen2.5-0.5B-Instruct \
    --lora_model ./output_qwen_lora

# 量化推理
python inference_lora.py \
    --base_model ./models/Qwen2.5-0.5B-Instruct \
    --lora_model ./output_qwen_qlora \
    --load_in_4bit

# 交互式对话
python inference_lora.py \
    --base_model ./models/Qwen2.5-0.5B-Instruct \
    --lora_model ./output_qwen_lora \
    --chat
```

## 💡 调优建议

### LoRA 参数调优
- **rank (r)**：16-128，越大效果越好但参数越多
- **alpha**：通常设为 r/2 到 2*r
- **dropout**：0.05-0.1
- **学习率**：1e-4 到 5e-4，比全参数微调高

### QLoRA 特殊考虑
- 4位量化推荐用于显存极限情况
- 8位量化在精度和显存间平衡
- 学习率可以设置更高 (2e-4 到 1e-3)

### 目标模块选择
- **全覆盖**：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- **注意力层**：`q_proj,k_proj,v_proj,o_proj`
- **前馈层**：`gate_proj,up_proj,down_proj`

## ❓ 常见问题

### Q: LoRA和QLoRA选择哪个？
A: 
- 显存充足 → LoRA
- 显存紧张 → QLoRA
- 追求最佳效果 → LoRA
- 显存极限 → QLoRA 4bit

### Q: 如何确定最佳rank值？
A: 
- 小模型(1B以下)：16-64
- 中模型(1-7B)：64-128  
- 大模型(7B+)：128-256

### Q: 训练后如何合并模型？
A: 
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

### Q: QLoRA训练很慢怎么办？
A: 
- 使用更大的batch size
- 减少gradient accumulation steps
- 确保使用了合适的CUDA版本

## 📚 更多资源

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [QLoRA论文](https://arxiv.org/abs/2305.14314)  
- [PEFT库文档](https://huggingface.co/docs/peft)
- [bitsandbytes文档](https://github.com/TimDettmers/bitsandbytes)

---

**提示**：首次使用建议从LoRA开始，熟悉流程后再尝试QLoRA。 