# Qwen模型微调工具包

一个功能完整、易于使用的Qwen模型微调工具包，支持多种微调方式和内存优化策略。

## 📚 目录

- [主要特性](#-主要特性)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
- [配置类型对比](#-配置类型对比)
- [详细使用方法](#-详细使用方法)
- [内存优化指南](#-内存优化指南)
- [项目结构](#-项目结构)
- [使用示例](#-使用示例)
- [文档索引](#-文档索引)
- [故障排除](#-故障排除)
- [性能调优建议](#-性能调优建议)
- [常见问题快速解答](#-常见问题快速解答)

## 🌟 主要特性

- **多种微调方式**：支持全参数微调、LoRA、QLoRA、DeepSpeed分布式训练
- **内存优化**：提供4种不同程度的内存优化配置，适配不同GPU
- **自动配置**：智能参数验证和自动DeepSpeed配置生成
- **简化使用**：一键式训练脚本，无需复杂参数设置
- **完整文档**：详细的使用指南和故障排除文档

## 📋 环境要求

### 基础要求
- Python 3.8+
- CUDA 11.0+
- 4GB+ GPU内存（最低配置）

### 推荐配置
- Python 3.9+
- CUDA 11.8+
- 8GB+ GPU内存
- 16GB+ 系统内存

### 重要版本说明
- **transformers==4.51.3** (推荐，经过测试)
- **torch>=1.13.0** (支持CUDA 11.8+)
- **deepspeed>=0.9.0** (支持ZeRO优化)

### ⚠️ 配置参数兼容性说明
在transformers 4.51.3中，为了避免`--load_best_model_at_end`参数冲突，需要**同时设置**：
- `evaluation_strategy`: 标准参数名
- `eval_strategy`: 兼容性参数名

两个参数的值必须一致，都设置为`"steps"`或`"no"`。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements_stable.txt

# 或者使用最新版本
pip install -r requirements.txt
```

### 2. 准备数据

将训练数据保存为JSONL格式，每行包含以下字段：
```json
{
  "instruction": "你是一个美食推荐官。",
  "input": "用户的输入内容",
  "output": "期望的输出内容"
}
```

### 3. 选择配置并开始训练

```bash
# 使用默认LoRA配置
./run_train.sh

# 使用内存优化配置（推荐）
./run_train.sh -t stage2_offload

# 使用其他配置
./run_train.sh -t [配置类型]
```

### 4. 完整示例（从安装到训练）

```bash
# 1. 克隆或下载项目
git clone <repository-url>
cd agent_data_collect

# 2. 安装依赖
pip install -r requirements_stable.txt

# 3. 检查GPU状态
nvidia-smi

# 4. 开始训练（推荐配置）
./run_train.sh -t stage2_offload

# 5. 训练完成后，查看结果
ls -la output_qwen_deepspeed_stage2_offload/
```

## 📊 配置类型对比

| 配置类型 | GPU内存需求 | 训练速度 | Batch Size | 序列长度 | 适用场景 |
|----------|-------------|----------|------------|----------|----------|
| `full` | >16GB | 最快 | 2 | 2048 | 全参数微调，效果最好 |
| `lora` | 6-12GB | 快 | 2 | 2048 | 平衡选择，推荐日常使用 |
| `qlora` | 4-8GB | 中等 | 2 | 2048 | 量化微调，节省内存 |
| `deepspeed` | 8-16GB | 快 | 4 | 2048 | 多GPU分布式训练 |
| `stage2_offload` | 6-12GB | 中快 | 3 | 1792 | **推荐**，平衡内存和性能 |
| `stage3` | 4-6GB | 中慢 | 2 | 1536 | 最大内存优化 |
| `minimal` | <4GB | 慢 | 1 | 1024 | 紧急情况，最小内存需求 |

## 🔧 详细使用方法

### 训练脚本选项

```bash
./run_train.sh [选项]

选项:
  -t, --type TYPE        选择微调类型
  -c, --config FILE      使用自定义配置文件
  -h, --help            显示帮助信息
```

### 配置文件结构

每个配置文件包含以下主要参数：

```json
{
  "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
  "data_path": "data/your_data.jsonl",
  "eval_data_path": "data/your_eval_data.jsonl",
  "output_dir": "./output_qwen",
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 1e-4,
  "num_train_epochs": 3,
  "evaluation_strategy": "steps",
  "eval_strategy": "steps",
  "save_strategy": "steps",
  "eval_steps": 500,
  "save_steps": 500,
  "use_deepspeed": true,
  "deepspeed_stage": 2,
  "use_lora": true,
  "lora_r": 64
}
```

### 自定义配置

1. 复制现有配置文件：
```bash
cp configs/train_config_lora.json configs/my_config.json
```

2. 修改配置参数

3. 使用自定义配置：
```bash
./run_train.sh -c configs/my_config.json
```

## 💾 内存优化指南

### 遇到CUDA OOM错误？

按以下顺序尝试：

1. **首选方案**：使用平衡配置
```bash
./run_train.sh -t stage2_offload
```

2. **内存紧张**：使用最大优化
```bash
./run_train.sh -t stage3
```

3. **极端情况**：使用最小配置
```bash
./run_train.sh -t minimal
```

### 内存优化技巧

```bash
# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"

# 监控GPU使用
watch -n 1 nvidia-smi

# 检查进程占用
nvidia-smi pmon
```

## 📁 项目结构

```
agent_data_collect/
├── configs/                    # 配置文件目录
│   ├── train_config_lora.json           # LoRA配置
│   ├── train_config_deepspeed.json      # DeepSpeed配置
│   ├── train_config_deepspeed_stage2_offload.json  # 推荐配置
│   ├── train_config_deepspeed_stage3.json          # 最大内存优化
│   └── train_config_deepspeed_minimal.json         # 最小内存配置
├── data/                       # 数据文件目录
├── fine_tune_qwen.py          # 主训练脚本
├── run_train.sh               # 训练启动脚本
├── gradio_inference.py        # 模型推理界面
├── requirements_stable.txt    # 稳定版依赖
├── requirements.txt           # 最新版依赖
├── MEMORY_OPTIMIZATION_GUIDE.md  # 内存优化详细指南
└── README_*.md                # 各种专项文档
```

## 🎯 使用示例

### 基础训练

```bash
# 使用默认配置
./run_train.sh

# 使用QLoRA节省内存
./run_train.sh -t qlora

# 使用推荐的内存优化配置
./run_train.sh -t stage2_offload
```

### 高级训练

```bash
# 多GPU训练
./run_train.sh -t deepspeed

# 极限内存优化
./run_train.sh -t minimal

# 自定义配置
./run_train.sh -c configs/my_custom_config.json
```

### 模型推理

```bash
# 启动Gradio界面
python gradio_inference.py

# 或使用脚本
./run_gradio.sh
```

### 训练输出结构

训练完成后，输出目录包含以下文件：

```
output_qwen_*/
├── adapter_config.json        # LoRA配置文件
├── adapter_model.bin          # LoRA适配器权重
├── config.json               # 模型配置
├── tokenizer_config.json     # 分词器配置
├── tokenizer.json            # 分词器文件
├── special_tokens_map.json   # 特殊token映射
├── runs/                     # TensorBoard日志
├── checkpoint-*/             # 训练检查点
└── trainer_state.json        # 训练状态
```

## 📖 文档索引

| 文档 | 描述 |
|------|------|
| [README_FINETUNE.md](README_FINETUNE.md) | 微调详细指南 |
| [README_INFERENCE.md](README_INFERENCE.md) | 推理使用说明 |
| [README_GRADIO.md](README_GRADIO.md) | Gradio界面使用 |
| [README_LORA.md](README_LORA.md) | LoRA微调专项 |
| [README_ISSUE.md](README_ISSUE.md) | 常见问题解决 |
| [MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md) | 内存优化完整指南 |
| [ACCELERATE_VS_DEEPSPEED.md](ACCELERATE_VS_DEEPSPEED.md) | Accelerate vs DeepSpeed详细对比 |
| [TRAINING_COMPARISON_SUMMARY.md](TRAINING_COMPARISON_SUMMARY.md) | 训练方式快速对比 |
| [EVAL_README.md](EVAL_README.md) | 模型评估指南 |

## 🔨 故障排除

### 常见问题

1. **CUDA OOM错误**
   - 参考 [内存优化指南](MEMORY_OPTIMIZATION_GUIDE.md)
   - 尝试更小的batch_size或更高的DeepSpeed stage

2. **transformers版本问题**
   - 使用固定版本：`pip install transformers==4.51.3`

3. **DeepSpeed安装问题**
   - 重新安装：`pip install deepspeed --force-reinstall`

4. **配置文件错误**
   - 检查JSON格式是否正确
   - 确保所有必需字段都存在
   - ⚠️ **重要**: 需要同时设置`evaluation_strategy`和`eval_strategy`
   - 确保评估策略和保存策略匹配

### 获取帮助

```bash
# 查看脚本帮助
./run_train.sh -h

# 检查GPU状态
nvidia-smi

# 验证配置文件
python3 validate_configs.py

# 或者单独验证某个配置文件
python3 -c "import json; print(json.load(open('configs/train_config_lora.json')))"

# 查看错误日志
tail -f output_qwen/logs/train.log
```

## 📈 性能调优建议

### 训练效率优化

1. **使用合适的batch size**
   - 单GPU：2-8
   - 多GPU：根据GPU数量调整

2. **梯度累积**
   - 小batch size时增加gradient_accumulation_steps
   - 保持有效batch size = batch_size × accumulation_steps

3. **学习率调整**
   - LoRA：1e-4 ~ 5e-4
   - 全参数：1e-5 ~ 5e-5

### 内存使用优化

1. **序列长度**
   - 根据数据分布调整max_seq_length
   - 过长的序列会大幅增加内存使用

2. **模型选择**
   - 0.5B模型：适合快速实验
   - 1.8B模型：更好效果但需要更多内存

## ❓ 常见问题快速解答

### Q: 遇到CUDA OOM错误怎么办？
**A:** 按顺序尝试：`stage2_offload` → `stage3` → `minimal`

### Q: 训练速度太慢怎么办？
**A:** 检查GPU利用率，增加batch_size或减少序列长度

### Q: 配置文件报错怎么办？
**A:** 确保同时设置`evaluation_strategy`和`eval_strategy`，两者值要一致

### Q: 如何选择合适的配置？
**A:** 根据GPU内存：>12GB用`deepspeed`，6-12GB用`stage2_offload`，<6GB用`stage3`

### Q: Accelerate和DeepSpeed有什么区别？
**A:** Accelerate简单稳定适合新手，DeepSpeed内存优化强大适合生产环境

### Q: 训练结果在哪里？
**A:** 在`output_qwen_*/`目录下，LoRA适配器是`adapter_model.bin`

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Qwen团队](https://github.com/QwenLM/Qwen) 提供优秀的基础模型
- [Hugging Face](https://huggingface.co/) 提供transformers库
- [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed) 提供分布式训练支持
- [PEFT](https://github.com/huggingface/peft) 提供参数高效微调方法

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 邮件联系
- 或在项目中留言

## 🚀 快速参考

### 常用命令

```bash
# 检查环境
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# 训练命令
./run_train.sh -t stage2_offload    # 推荐配置
./run_train.sh -t stage3            # 最大内存优化
./run_train.sh -t minimal           # 极限内存配置

# 验证配置
python3 -c "import json; json.load(open('configs/train_config_lora.json'))"

# 监控训练
watch -n 1 nvidia-smi
tail -f output_qwen_*/logs/train.log
```

### 配置文件快速选择

```bash
# 根据GPU内存选择
>12GB  → deepspeed
6-12GB → stage2_offload  (推荐)
4-6GB  → stage3
<4GB   → minimal
```

## 📝 更新记录

- **最新版本**: 添加多种内存优化配置
- **主要改进**: 支持DeepSpeed ZeRO优化
- **配置修复**: 修复evaluation_strategy参数冲突
- **文档完善**: 提供完整的使用指南和故障排除

---

⭐ 如果这个项目对你有帮助，请给个Star！ 