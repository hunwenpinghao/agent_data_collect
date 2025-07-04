# LoRA 模型上传指南

## 🎯 问题解决

您的 LoRA 模型目录包含了很多训练临时文件，这些文件不应该上传到魔搭社区。我们为您准备了自动筛选工具！

## 📊 文件分析结果

### ✅ 需要上传 (149.4 MB)
```
adapter_model.safetensors    134.3 MB  - LoRA 适配器权重 ⭐
adapter_config.json            899 B   - LoRA 配置文件 ⭐
tokenizer.json                10.9 MB  - 分词器
tokenizer_config.json          7.2 KB  - 分词器配置
vocab.json                     2.6 MB  - 词汇表
merges.txt                     1.6 MB  - BPE 合并规则
special_tokens_map.json         613 B  - 特殊token映射
added_tokens.json               605 B  - 添加的token
configuration.json               41 B  - 模型配置
.gitattributes                  2.0 KB - Git属性（可选）
README.md                    自动生成  - 模型说明文档
```

### ❌ 不需要上传 (268.8 MB)
```
optimizer.pt                 268.8 MB  - 优化器状态 ❌
scaler.pt                       988 B  - 梯度缩放器 ❌
scheduler.pt                    1.0 KB - 调度器状态 ❌
rng_state.pth                  13.9 KB - 随机数状态 ❌
training_args.bin               5.3 KB - 训练参数 ❌
trainer_state.json              2.3 KB - 训练器状态 ❌
.git/                              目录 - Git版本控制 ❌
```

**节省空间：268.8 MB → 只上传 149.4 MB** 🎉

## 🚀 使用步骤

### 步骤1：进入工具目录
```bash
cd upload_to_modelscope
```

### 步骤2：自动筛选文件
```bash
python3 prepare_lora_upload.py
```

**自定义参数：**
```bash
python3 prepare_lora_upload.py \
  --source_dir ../output_qwen/zhc_xhs_qwen2.5_0.5b_instruct \
  --output_dir ../upload_ready/my_lora_model \
  --model_name my_custom_model_name
```

### 步骤3：验证筛选结果
```bash
python3 validate_model.py --model_dir ../upload_ready/zhc_xhs_qwen2.5_0.5b_instruct
```

### 步骤4：上传到魔搭社区
```bash
python3 upload_to_modelscope.py \
  --model_dir ../upload_ready/zhc_xhs_qwen2.5_0.5b_instruct \
  --model_id hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct \
  --token YOUR_TOKEN
```

## 🔧 自动化功能

### 智能文件筛选
- ✅ 自动复制 LoRA 必需文件
- ❌ 自动跳过训练临时文件
- 📝 自动生成专业的 README.md

### 生成的 README 包含
- 模型描述和用法示例
- LoRA 参数信息 (r=64, alpha=16)
- ModelScope 和 transformers 使用方法
- 完整的代码示例

## 📋 验证结果

运行筛选脚本后，您会看到：

```
🔧 LoRA模型上传准备工具
==============================
📁 源目录: ../output_qwen/zhc_xhs_qwen2.5_0.5b_instruct
📁 输出目录: ../upload_ready/zhc_xhs_qwen2.5_0.5b_instruct
============================================================
✅ adapter_model.safetensors        134.3 MB - 已复制
✅ adapter_config.json                 899 B - 已复制
...
⏭️  optimizer.pt                     268.8 MB - 已跳过（训练文件）
...
============================================================
📊 总结:
   已复制文件: 11
   跳过文件: 8
   总大小: 149.4 MB
```

## 💡 优势

1. **空间节省**：从 418MB 减少到 149MB，节省 64% 空间
2. **专业README**：自动生成包含使用示例的文档
3. **标准格式**：符合 Hugging Face/ModelScope 标准
4. **安全筛选**：确保不上传敏感的训练文件

## 🔍 使用 LoRA 模型

上传后，用户可以这样使用您的 LoRA 模型：

```python
from modelscope import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-0.5B-Instruct")

# 加载您的 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct")

# 生成文本
prompt = "小红书种草文案："
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 🆘 常见问题

**Q: 为什么不上传 optimizer.pt？**
A: 这是训练时的优化器状态，只用于恢复训练，推理时不需要。

**Q: LoRA 模型需要基础模型吗？**
A: 是的，LoRA 是适配器，需要与基础模型 Qwen2.5-0.5B-Instruct 配合使用。

**Q: 可以修改模型名称吗？**
A: 可以，使用 `--model_name` 参数自定义名称。

---

**下一步：运行 `python3 prepare_lora_upload.py` 开始准备您的 LoRA 模型！** 🚀 