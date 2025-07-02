# Qwen 模型评估脚本使用指南

## 概述

`run_eval.sh` 是一个全功能的 Qwen 模型评估脚本，支持生成任务的自动评估，包含多种评估指标和灵活的配置选项。

## 功能特点

- 🚀 **一键评估**: 简单命令即可完成模型评估
- 📊 **多种指标**: 支持 BLEU、ROUGE、精确匹配等评估指标
- 🔧 **自动修复**: 内置 transformers 兼容性修复
- 💾 **结果保存**: 自动保存评估结果和预测详情
- 🎯 **灵活配置**: 支持多种参数自定义
- 📱 **友好界面**: 彩色输出和详细进度显示

## 快速开始

### 1. 基本用法

```bash
# 使用默认设置评估模型
./run_eval.sh -m ./output/checkpoint-best -d ./data/test.jsonl

# 查看帮助信息
./run_eval.sh --help
```

### 2. 完整示例

```bash
# 完整配置的评估
./run_eval.sh \
  --model-path ./output/checkpoint-1000 \
  --data-path ./data/test_sample.jsonl \
  --output-dir ./eval_results_v1 \
  --batch-size 4 \
  --max-tokens 512 \
  --temperature 0.7 \
  --metrics bleu,rouge,exact_match \
  --save-predictions \
  --verbose
```

### 3. 使用测试数据

我们提供了一个示例测试文件：

```bash
# 使用内置测试数据
./run_eval.sh -m ./output/checkpoint-best -d ./data/test_sample.jsonl
```

## 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | `-m` | `./output/checkpoint-best` | 微调后的模型路径 |
| `--data-path` | `-d` | `./data/test.jsonl` | 评估数据文件路径 |
| `--output-dir` | `-o` | `./eval_results` | 评估结果输出目录 |
| `--batch-size` | `-b` | `4` | 批处理大小 |
| `--max-tokens` | `-t` | `512` | 最大生成token数 |
| `--temperature` | - | `0.7` | 生成温度（0-1） |
| `--top-p` | - | `0.9` | Top-p采样值 |
| `--device` | - | `auto` | 计算设备：cpu/cuda/auto |
| `--metrics` | - | `bleu,rouge` | 评估指标列表 |
| `--save-predictions` | - | `false` | 是否保存预测结果 |
| `--verbose` | - | `false` | 是否显示详细输出 |

## 数据格式

评估数据文件应为 JSONL 格式（每行一个JSON对象），支持以下字段：

### 格式1：指令-输入-输出
```json
{
  "instruction": "请解释什么是人工智能",
  "input": "",
  "output": "人工智能(AI)是指由机器展现出的智能..."
}
```

### 格式2：问题-答案
```json
{
  "question": "Python中如何定义一个函数？",
  "answer": "在Python中，使用def关键字来定义函数..."
}
```

### 格式3：提示-回复
```json
{
  "prompt": "请写一首关于春天的短诗",
  "response": "春风轻拂柳丝长..."
}
```

## 评估指标

### 支持的指标

- **BLEU**: 基于n-gram的文本相似度指标
- **ROUGE-1/2/L**: 基于重叠统计的文本摘要评估指标
- **精确匹配**: 预测与参考答案完全匹配的比例

### 指标配置示例

```bash
# 只使用BLEU
./run_eval.sh -m model_path -d data_path --metrics bleu

# 使用所有指标
./run_eval.sh -m model_path -d data_path --metrics bleu,rouge,exact_match
```

## 输出结果

### 控制台输出示例

```
============================== Qwen 模型评估脚本 ==============================

📁 模型路径: ./output/checkpoint-best
📄 数据路径: ./data/test_sample.jsonl
📂 输出目录: ./eval_results
🔢 批处理大小: 4
🎯 最大token数: 512
🌡️  生成温度: 0.7
📊 评估指标: bleu,rouge,exact_match
💻 计算设备: auto

============================================================
模型评估结果
============================================================
模型路径: ./output/checkpoint-best
数据路径: ./data/test_sample.jsonl
总样本数: 10
有效样本数: 10
时间戳: 2024-01-15 14:30:25

评估指标:
  BLEU: 0.3456
  ROUGE-1: 0.4532
  ROUGE-L: 0.4012
  EXACT_MATCH: 0.2000

统计信息:
  平均预测长度: 25.3 tokens
  平均参考长度: 28.7 tokens

结果文件: ./eval_results/evaluation_results.json
预测文件: ./eval_results/predictions.jsonl
============================================================
```

### 输出文件

1. **evaluation_results.json**: 完整的评估结果，包含指标和配置信息
2. **predictions.jsonl**: 详细的预测结果（如果启用 `--save-predictions`）
3. **eval_model.py**: 自动生成的Python评估脚本

## 高级用法

### 1. 对比不同模型

```bash
# 评估基础模型
./run_eval.sh -m Qwen/Qwen2.5-7B-Instruct -d ./data/test.jsonl -o ./eval_base

# 评估微调模型
./run_eval.sh -m ./output/checkpoint-best -d ./data/test.jsonl -o ./eval_tuned

# 比较结果
python -c "
import json
with open('./eval_base/evaluation_results.json') as f: base = json.load(f)
with open('./eval_tuned/evaluation_results.json') as f: tuned = json.load(f)
print('基础模型 BLEU:', base['metrics']['bleu'])
print('微调模型 BLEU:', tuned['metrics']['bleu'])
print('改进幅度:', tuned['metrics']['bleu'] - base['metrics']['bleu'])
"
```

### 2. 批量评估

```bash
# 评估多个检查点
for checkpoint in ./output/checkpoint-*; do
  if [ -d "$checkpoint" ]; then
    echo "评估: $checkpoint"
    ./run_eval.sh -m "$checkpoint" -d ./data/test.jsonl -o "./eval_results/$(basename $checkpoint)"
  fi
done
```

### 3. 不同生成参数测试

```bash
# 测试不同温度
for temp in 0.1 0.5 0.7 0.9; do
  ./run_eval.sh -m ./output/checkpoint-best -d ./data/test.jsonl \
    --temperature $temp -o "./eval_temp_$temp"
done
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的内存/显存
   - 查看是否需要下载模型文件

2. **依赖包缺失**
   ```bash
   pip install torch transformers datasets pandas tqdm
   pip install rouge nltk  # 可选：用于更准确的指标计算
   ```

3. **CUDA相关错误**
   ```bash
   # 强制使用CPU
   ./run_eval.sh -m model_path -d data_path --device cpu
   ```

4. **内存不足**
   ```bash
   # 减小批处理大小
   ./run_eval.sh -m model_path -d data_path --batch-size 1
   ```

### 调试模式

```bash
# 启用详细输出
./run_eval.sh -m model_path -d data_path --verbose

# 保存预测结果用于分析
./run_eval.sh -m model_path -d data_path --save-predictions
```

## 注意事项

1. **数据格式**: 确保评估数据为有效的JSONL格式
2. **模型兼容性**: 脚本内置了Qwen模型的兼容性修复
3. **计算资源**: 根据模型大小和数据量调整批处理大小
4. **评估公平性**: 使用相同的数据和参数对比不同模型

## 示例结果分析

评估完成后，可以使用以下脚本分析结果：

```python
import json
import pandas as pd

# 读取评估结果
with open('./eval_results/evaluation_results.json', 'r') as f:
    results = json.load(f)

print("评估摘要:")
print(f"模型: {results['model_path']}")
print(f"样本数: {results['num_samples']}")
print("\n指标详情:")
for metric, score in results['metrics'].items():
    if not metric.startswith('num_') and not metric.startswith('avg_'):
        print(f"  {metric}: {score:.4f}")

# 分析预测结果（如果保存了）
try:
    predictions = []
    with open('./eval_results/predictions.jsonl', 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    df = pd.DataFrame(predictions)
    print(f"\n预测统计:")
    print(f"平均预测长度: {df['prediction'].str.len().mean():.1f} 字符")
    print(f"平均参考长度: {df['reference'].str.len().mean():.1f} 字符")
    
except FileNotFoundError:
    print("未找到详细预测文件，请使用 --save-predictions 选项")
```

---

## 联系支持

如果遇到问题或需要帮助，请：

1. 检查错误日志
2. 查看生成的Python脚本 `eval_model.py`
3. 使用 `--verbose` 选项获取详细信息
4. 确保所有依赖都已正确安装

祝你评估顺利！🎉 