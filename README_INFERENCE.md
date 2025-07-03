# LoRA 模型推理使用指南

本文档介绍如何使用 `run_inference_lora.sh` 脚本进行 LoRA 模型推理。

## 快速开始

### 前提条件

1. 已完成 LoRA 模型训练
2. 安装了必要的 Python 依赖：
   ```bash
   pip install -r requirements_stable.txt
   ```

### 基本用法

```bash
# 最简单的用法：交互式对话模式（自动检测最新的训练模型）
./run_inference_lora.sh

# 查看帮助信息
./run_inference_lora.sh -h
```

## 详细用法

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | `-m` | LoRA模型目录 | 自动检测最新的output_qwen*目录 |
| `--base` | `-b` | 基础模型路径 | Qwen/Qwen2.5-0.5B-Instruct |
| `--quantization` | `-q` | 量化模式 (4bit/8bit/none) | none |
| `--mode` | `-t` | 推理模式 (chat/single/test) | chat |
| `--prompt` | `-p` | 单次推理的提示文本 | - |
| `--help` | `-h` | 显示帮助信息 | - |

### 推理模式

#### 1. 交互式对话模式 (chat)

最常用的模式，可以与模型进行多轮对话：

```bash
# 使用默认设置
./run_inference_lora.sh

# 指定特定的LoRA模型
./run_inference_lora.sh -m ./output_qwen_lora

# 使用4位量化节省显存
./run_inference_lora.sh -q 4bit -t chat
```

在交互模式下：
- 输入问题或指令
- 模型会生成回答
- 输入 `quit`、`exit` 或 `退出` 来结束对话

#### 2. 单次推理模式 (single)

适合脚本调用或批处理：

```bash
# 单次推理
./run_inference_lora.sh -t single -p "请为一家咖啡店写一段小红书风格的文案"

# 使用8位量化
./run_inference_lora.sh -q 8bit -t single -p "推荐几家上海的网红咖啡店"
```

#### 3. 批量测试模式 (test)

使用内置的测试样例来评估模型效果：

```bash
# 运行内置测试
./run_inference_lora.sh -t test

# 使用量化运行测试
./run_inference_lora.sh -q 4bit -t test
```

### 量化选项

| 量化模式 | 显存占用 | 推理速度 | 适用场景 |
|----------|----------|----------|----------|
| `none` | 高 | 最快 | 显存充足时使用 |
| `8bit` | 中等 | 中等 | 平衡选择 |
| `4bit` | 最低 | 较慢 | 显存不足时使用 |

## 使用示例

### 示例1：快速体验

```bash
# 最简单的使用方式
./run_inference_lora.sh
```

输出示例：
```
🔍 自动检测LoRA模型...
✅ 找到LoRA模型: ./output_qwen_lora
🔍 检查GPU信息...
Tesla V100, 32510, 31039
🔍 检查Python依赖...
✅ 依赖检查通过

==================== 推理配置 ====================
基础模型: Qwen/Qwen2.5-0.5B-Instruct
LoRA模型: ./output_qwen_lora
量化模式: none
推理模式: chat
==================================================

LoRA模型推理系统已启动！输入'quit'退出。
--------------------------------------------------
用户: 
```

### 示例2：单次推理用于自动化

```bash
# 生成文案
./run_inference_lora.sh -t single -p "为一家新开的奶茶店写一段吸引人的宣传文案"
```

### 示例3：使用特定模型和量化

```bash
# 使用特定的LoRA模型，开启4位量化
./run_inference_lora.sh -m ./output_qwen_qlora -q 4bit -t chat
```

### 示例4：测试模型效果

```bash
# 运行内置测试用例
./run_inference_lora.sh -t test
```

## 故障排除

### 常见问题

#### 1. 找不到LoRA模型

**错误信息**：`错误: 未找到LoRA模型目录 (output_qwen*)`

**解决方案**：
- 确保已完成LoRA训练
- 或使用 `-m` 参数指定模型路径

#### 2. 显存不足

**错误信息**：`CUDA out of memory`

**解决方案**：
- 使用量化选项：`-q 4bit` 或 `-q 8bit`
- 或在CPU上运行（会比较慢）

#### 3. 基础模型加载失败

**解决方案**：
- 检查网络连接（如果使用在线模型）
- 运行 `./download_model.sh` 预先下载模型
- 使用本地模型路径：`-b /path/to/local/model`

### 性能优化建议

1. **显存优化**：
   - 优先尝试4位量化：`-q 4bit`
   - 如果精度要求高，使用8位量化：`-q 8bit`

2. **速度优化**：
   - 确保使用GPU而非CPU
   - 预先下载模型到本地避免网络延迟

3. **模型选择**：
   - 使用最新训练的模型获得最佳效果
   - 可以通过 `-m` 参数指定特定的checkpoint

## 进阶用法

### 与其他脚本集成

```bash
# 在脚本中调用
RESPONSE=$(./run_inference_lora.sh -t single -p "你的问题" | tail -n 1)
echo "模型回答: $RESPONSE"
```

### 批量处理文件

```bash
# 读取问题文件并逐一推理
while IFS= read -r line; do
    echo "问题: $line"
    ./run_inference_lora.sh -t single -p "$line"
    echo "---"
done < questions.txt
```

## 相关文件

- `inference_lora.py` - 底层推理脚本
- `run_train.sh` - 训练脚本
- `configs/` - 训练配置文件
- `README_LORA.md` - LoRA训练指南 