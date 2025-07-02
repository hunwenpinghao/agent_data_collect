# 训练配置文件说明

本文件夹包含了不同微调方式的配置文件示例。

## 📁 配置文件列表

### 1. `train_config_full.json`
- **说明**：全参数微调配置
- **特点**：微调所有模型参数，效果最好但显存需求最高
- **适用场景**：有充足GPU资源，追求最佳效果
- **使用命令**：
```bash
python fine_tune_qwen.py --config_file configs/train_config_full.json
```

### 2. `train_config_lora.json`
- **说明**：LoRA微调配置
- **特点**：只训练约3%的参数，显存需求中等，效果接近全参数微调
- **适用场景**：平衡性能和资源的首选方案
- **使用命令**：
```bash
python fine_tune_qwen.py --config_file configs/train_config_lora.json
```

### 3. `train_config_qlora.json`
- **说明**：QLoRA 4位量化微调配置
- **特点**：4位量化+LoRA，显存需求最低，可减少60%显存占用
- **适用场景**：显存有限的环境，如消费级GPU
- **使用命令**：
```bash
python fine_tune_qwen.py --config_file configs/train_config_qlora.json
```

### 4. `train_config_qlora_8bit.json`
- **说明**：QLoRA 8位量化微调配置
- **特点**：8位量化+LoRA，在精度和显存间平衡
- **适用场景**：中等显存限制环境
- **使用命令**：
```bash
python fine_tune_qwen.py --config_file configs/train_config_qlora_8bit.json
```

## 🎯 选择建议

| 配置文件 | 显存需求 | 训练速度 | 效果保持 | 推荐场景 |
|----------|----------|----------|----------|----------|
| `train_config_full.json` | 高 (8GB+) | 中等 | 100% | 追求最佳效果 |
| `train_config_lora.json` | 中 (6GB+) | 快 | 95-98% | 平衡选择 |
| `train_config_qlora.json` | 低 (3GB+) | 慢 | 90-95% | 显存极限 |
| `train_config_qlora_8bit.json` | 中低 (4GB+) | 中等 | 92-96% | 轻量选择 |

## 🔧 自定义配置

你可以基于这些模板创建自己的配置文件：

1. **复制模板**：选择最接近你需求的配置文件
2. **修改参数**：根据你的数据和硬件调整参数
3. **保存配置**：以 `.json` 扩展名保存到 `configs/` 文件夹
4. **运行训练**：使用 `--config_file` 参数指定你的配置

## ⚙️ 关键参数说明

### 模型参数
- `model_name_or_path`: 基础模型路径
- `use_lora`: 是否启用LoRA
- `use_qlora`: 是否启用QLoRA
- `lora_r`: LoRA rank，控制适配器大小
- `lora_alpha`: LoRA缩放参数
- `quantization_bit`: 量化位数 (4或8)

### 训练参数  
- `learning_rate`: 学习率 (LoRA/QLoRA通常需要更高的学习率)
- `per_device_train_batch_size`: 每设备批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `num_train_epochs`: 训练轮数

### 数据参数
- `data_path`: 训练数据路径
- `max_seq_length`: 最大序列长度
- `output_dir`: 输出目录

---

**提示**：首次使用建议从 `train_config_lora.json` 开始！ 

```bash
# 默认LoRA微调
./run_train.sh

# 选择不同微调类型
./run_train.sh -t qlora      # QLoRA 4位量化
./run_train.sh -t full       # 全参数微调
./run_train.sh -t qlora_8bit # QLoRA 8位量化

# 使用自定义配置
./run_train.sh -c configs/my_custom_config.json

# 查看帮助
./run_train.sh -h 
```
