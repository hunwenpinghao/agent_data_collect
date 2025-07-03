# 🤖 微调模型Gradio Web界面

一个基于Gradio的Web界面，用于加载和测试微调后的模型，支持LoRA、完整微调(Full Fine-tuning)和QLoRA模型。

## ✨ 特性

- 🎯 **多模型支持**：LoRA、完整微调、QLoRA
- 🔧 **量化支持**：4bit、8bit量化以节省显存
- 🖥️ **友好界面**：直观的Web界面，易于使用
- ⚡ **实时对话**：支持多轮对话和参数调整
- 📊 **模型信息**：显示详细的模型统计信息
- 🛡️ **兼容性修复**：内置transformers库兼容性补丁

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或安装特定包
pip install gradio>=4.0.0 bitsandbytes>=0.41.0
```

### 2. 启动Web界面

```bash
# 方式一：使用启动脚本（推荐）
chmod +x run_gradio.sh
./run_gradio.sh

# 方式二：直接运行
python3 gradio_inference.py
```

### 3. 访问界面

在浏览器中打开：`http://localhost:7860`

## 📋 使用说明

### 模型配置

1. **选择模型类型**：
   - `lora`：LoRA微调模型
   - `full_ft`：完整微调模型
   - `qlora`：QLoRA量化微调模型

2. **配置路径**：
   - **基础模型路径**：预训练模型的路径或HuggingFace模型名称
   - **LoRA适配器路径**：LoRA权重的保存路径（仅LoRA/QLoRA需要）

3. **量化选项**：
   - `none`：不使用量化
   - `4bit`：4位量化（节省最多显存）
   - `8bit`：8位量化（平衡性能和显存）

### 对话生成

1. **发送消息**：在输入框中输入问题
2. **调整参数**：
   - **最大长度**：生成文本的最大token数
   - **温度**：控制生成随机性（0.1-2.0）
   - **Top-p**：核采样参数（0.1-1.0）
   - **Top-k**：限制候选token数量
   - **重复惩罚**：避免重复生成

## 🔧 配置示例

### LoRA模型配置

```
模型类型: lora
基础模型路径: Qwen/Qwen2.5-0.5B-Instruct
LoRA适配器路径: ./output_qwen
量化类型: none
```

### 完整微调模型配置

```
模型类型: full_ft
基础模型路径: ./output_qwen_full
量化类型: 4bit
```

### QLoRA模型配置

```
模型类型: qlora
基础模型路径: Qwen/Qwen2.5-0.5B-Instruct
LoRA适配器路径: ./output_qwen_qlora
量化类型: 4bit
```

## 🎯 常见模型路径

### HuggingFace模型

```
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
```

### 本地模型

```
./models/Qwen2.5-0.5B-Instruct
./output_qwen
./checkpoints/checkpoint-1000
```

## 🛠️ 环境变量

```bash
# 自定义启动参数
export HOST=0.0.0.0          # 监听地址
export PORT=7860             # 端口号
export SHARE=false           # 是否公开分享
export BACKGROUND=false      # 是否后台运行

# 使用HuggingFace镜像
export USE_HF_MIRROR=true

# 启动
./run_gradio.sh
```

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查模型路径是否正确
   ls -la ./output_qwen
   
   # 检查网络连接
   ping hf-mirror.com
   ```

2. **内存不足**
   ```bash
   # 使用量化减少显存使用
   # 在界面中选择 4bit 或 8bit 量化
   
   # 或清理GPU缓存
   python3 -c "import torch; torch.cuda.empty_cache()"
   ```

3. **依赖包问题**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt --force-reinstall
   
   # 或使用稳定版本
   pip install -r requirements_stable.txt
   ```

### 日志检查

```bash
# 查看实时日志
tail -f logs/gradio_*.log

# 查看错误日志
grep -i error logs/gradio_*.log
```

## 📊 性能优化

### 显存优化

1. **使用量化**：选择4bit或8bit量化
2. **调整批大小**：减少同时处理的token数量
3. **清理缓存**：定期清理GPU缓存

### 速度优化

1. **GPU加速**：确保使用GPU而不是CPU
2. **模型本地化**：下载模型到本地避免网络延迟
3. **参数调整**：适当调整最大生成长度

## 🔄 API模式

Gradio界面也支持API调用：

```python
import requests

# 发送POST请求
response = requests.post(
    "http://localhost:7860/api/predict",
    json={
        "data": ["你的问题", 512, 0.7, 0.9, 50, 1.1]
    }
)

print(response.json())
```

## 🆕 版本更新

### v1.0.0
- 初始版本
- 支持LoRA、完整微调、QLoRA
- 基础Web界面

### v1.1.0
- 添加量化支持
- 兼容性补丁
- 优化界面布局

### v1.2.0
- 添加示例输入
- 改进错误处理
- 性能优化

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目基于MIT许可证开源。

## 📞 技术支持

如果遇到问题，请：

1. 检查日志文件
2. 查看故障排除部分
3. 提交Issue并附上详细信息

---

**享受使用微调模型进行对话的乐趣！** 🎉 