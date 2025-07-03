# 常见问题与解决方案

本文档记录了在使用本项目时可能遇到的常见问题及其解决方案。

## 1. Transformers 库兼容性问题

### 问题描述
运行训练脚本时出现以下错误：

```
File "/usr/local/lib/python3.11/site-packages/transformers/modeling_utils.py", line 1969, in post_init
    if v not in ALL_PARALLEL_STYLES:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: argument of type 'NoneType' is not iterable
```

### 错误原因
这是由于 transformers 库版本兼容性问题导致的。某些版本的 transformers 在处理模型配置时，对 `None` 值的处理存在问题。

### 解决方案
安装兼容的 transformers 版本：

```bash
pip install transformers==4.51.3
```

### 验证方法
安装完成后，可以通过以下命令验证版本：

```bash
python -c "import transformers; print(transformers.__version__)"
```

应该输出：`4.51.3`

### 相关信息
- 影响的文件：`fine_tune_qwen.py`
- 推荐的 transformers 版本：`4.51.3`
- Python 版本：`3.11+`

---

## 2. 其他常见问题

### 模型下载问题
如果遇到模型下载失败，可以：
1. 使用国内镜像源：`export HF_ENDPOINT=https://hf-mirror.com`
2. 手动下载模型到 `models/` 目录

### 内存不足问题
如果训练时内存不足，可以：
1. 减少 `per_device_train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 使用 QLoRA 量化训练

### GPU 相关问题
如果没有 GPU 或 GPU 内存不足：
1. 使用 CPU 训练（较慢）
2. 使用量化训练减少内存使用
3. 减少模型参数或序列长度

---

## 更新日志

- **2024-12-19**: 添加 transformers==4.51.3 兼容性问题解决方案 