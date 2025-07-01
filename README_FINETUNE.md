# Qwen3-0.5B 微调指南

本项目用于微调 Qwen3-0.5B 模型，生成小红书风格的店铺推荐文案。

## 项目结构

```
├── fine_tune_qwen.py      # 主要的微调脚本
├── inference.py           # 推理测试脚本
├── train_config.json      # 训练配置文件
├── requirements.txt       # 依赖包列表
├── run_train.sh          # 训练启动脚本
├── store_xhs_sft_samples.jsonl  # 训练数据
└── README_FINETUNE.md    # 本文档
```

## 环境要求

- Python 3.8+
- CUDA 11.8+ (如果使用GPU)
- 至少16GB RAM (建议32GB+)
- 至少10GB显存 (建议16GB+)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置训练参数

编辑 `train_config.json` 文件，根据你的硬件配置调整参数：

```json
{
    "model_name_or_path": "qwen/Qwen3-0.5B-Instruct",
    "data_path": "store_xhs_sft_samples.jsonl",
    "output_dir": "./output_qwen",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "num_train_epochs": 3
}
```

**参数说明：**
- `per_device_train_batch_size`: 每个GPU的批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率
- `num_train_epochs`: 训练轮数

### 3. 开始训练

#### 方法1：使用启动脚本（推荐）

```bash
./run_train.sh
```

#### 方法2：手动运行

```bash
python3 fine_tune_qwen.py --config_file train_config.json
```

### 4. 监控训练进度

训练过程中可以使用 TensorBoard 监控：

```bash
tensorboard --logdir ./output_qwen/runs
```

## 推理测试

训练完成后，可以使用以下方式测试模型：

### 1. 默认测试

```bash
python3 inference.py --model_path ./output_qwen
```

### 2. 交互式测试

```bash
python3 inference.py --model_path ./output_qwen --interactive
```

### 3. 使用测试文件

```bash
python3 inference.py --model_path ./output_qwen --test_file store_xhs_sft_samples.jsonl
```

## 数据格式

训练数据应为JSONL格式，每行包含以下字段：

```json
{
    "instruction": "根据以下店铺信息，生成一段小红书风格的文案，风格为「甜美」：",
    "input": "店铺名称：星巴克\n品类：咖啡\n地址：郑州正弘城L8\n营业时间：10:00-22:00",
    "output": "✨郑州正弘城的星巴克真的太棒了！..."
}
```

## 配置说明

### 硬件配置建议

| 配置等级 | GPU | 显存 | 批次大小 | 梯度累积 |
|---------|-----|------|----------|----------|
| 低配置  | GTX 3060 | 12GB | 1 | 16 |
| 中等配置 | RTX 3080 | 16GB | 2 | 8 |
| 高配置  | RTX 4090 | 24GB | 4 | 4 |

### 训练参数调优

1. **学习率**: 建议范围 1e-5 到 5e-5
2. **训练轮数**: 通常 2-5 轮即可
3. **序列长度**: 根据数据特点调整，建议 1024-2048
4. **批次大小**: 根据显存大小调整

## 常见问题

### Q: 显存不足怎么办？
A: 
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 `fp16=true` 开启半精度训练
- 考虑使用 DeepSpeed 进行显存优化

### Q: 训练速度慢怎么办？
A:
- 增加 `dataloader_num_workers`
- 使用更快的存储设备
- 考虑使用多GPU训练

### Q: 如何评估模型效果？
A:
- 观察训练损失曲线
- 使用验证集评估
- 人工评估生成文案质量
- 使用BLEU、ROUGE等指标

## 模型部署

训练完成后，模型保存在 `./output_qwen` 目录下，可以：

1. **本地部署**: 使用 `inference.py` 脚本
2. **服务化部署**: 集成到FastAPI或Flask应用中
3. **量化部署**: 使用ONNX或TensorRT进行推理加速

## 版本历史

- v1.0: 初始版本，支持基础微调功能
- 后续版本将支持更多高级功能

## 许可证

本项目遵循原模型的许可证要求。

## 支持

如有问题，请查看：
1. 训练日志文件（在 `logs/` 目录下）
2. TensorBoard 监控面板
3. 相关文档和示例

---

**注意**: 首次运行会自动下载模型文件，可能需要较长时间，请确保网络连接稳定。 