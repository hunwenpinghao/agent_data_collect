# 魔搭社区模型上传指南

这个指南将帮助您将微调好的模型上传到魔搭社区（ModelScope）。

## 🚀 快速开始

### 步骤1：安装依赖

```bash
# 运行依赖安装脚本
./setup_upload_dependencies.sh

# 或者手动安装
pip install modelscope GitPython requests tqdm
```

### 步骤2：获取访问令牌

1. 访问 [魔搭社区个人中心](https://www.modelscope.cn/my/myaccesstoken)
2. 登录您的账号
3. 绑定阿里云账号（如果还没有绑定）
4. 生成或复制您的访问令牌

### 步骤3：上传模型

```bash
# 使用默认设置上传（推荐）
python3 upload_to_modelscope.py --token YOUR_TOKEN

# 或者设置环境变量
export MODELSCOPE_TOKEN=YOUR_TOKEN
python3 upload_to_modelscope.py
```

## 🛠️ 高级配置

### 完整命令参数

```bash
python3 upload_to_modelscope.py \
  --model_dir ../deploy_ark/output_qwen/Qwen2.5-0.5B-Instruct \
  --model_id hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct \
  --token YOUR_TOKEN \
  --method api \
  --description "基于小红书数据微调的Qwen2.5-0.5B模型" \
  --create_readme
```

### 参数说明

- `--model_dir`: 模型文件目录路径
- `--model_id`: 模型在魔搭社区的完整ID（用户名/模型名）
- `--token`: 魔搭社区访问令牌
- `--method`: 上传方法，可选 `api` 或 `git`
- `--description`: 模型描述
- `--create_readme`: 自动创建 README.md 文件

## 📋 上传前检查

脚本会自动检查以下内容：

1. **必需文件**：
   - `config.json` - 模型配置
   - `model.safetensors` - 模型权重

2. **可选文件**：
   - `tokenizer.json` - 分词器配置
   - `tokenizer_config.json` - 分词器配置
   - `generation_config.json` - 生成配置
   - `README.md` - 模型说明

## 📝 自动生成模型卡片

使用 `--create_readme` 参数会自动生成包含以下内容的 README.md：

- 模型描述
- 使用方法示例
- 训练详情
- 许可证信息
- 引用格式

## 🔧 两种上传方法

### 方法1：API 上传（推荐）

```bash
python3 upload_to_modelscope.py --method api --token YOUR_TOKEN
```

**优点**：
- 快速简单
- 自动处理文件上传
- 内置错误处理

### 方法2：Git 上传

```bash
python3 upload_to_modelscope.py --method git --token YOUR_TOKEN
```

**优点**：
- 更好的版本控制
- 可以查看上传历史
- 适合大文件上传

## 🐛 常见问题

### 1. 认证失败

```
ERROR - 登录 ModelScope 失败
```

**解决方案**：
- 检查访问令牌是否正确
- 确认魔搭社区账号已绑定阿里云账号
- 重新生成访问令牌

### 2. 模型仓库不存在

```
ERROR - 模型仓库不存在
```

**解决方案**：
- 先在魔搭社区创建模型仓库
- 确认模型ID格式正确（用户名/模型名）

### 3. 文件过大

```
ERROR - 文件上传失败
```

**解决方案**：
- 使用 Git 上传方法：`--method git`
- 检查网络连接
- 分批上传大文件

### 4. 权限问题

```
ERROR - 没有写入权限
```

**解决方案**：
- 确认您是模型仓库的所有者
- 检查访问令牌的权限范围

## 📊 上传进度监控

脚本会显示详细的上传进度：

```
2025-01-07 10:30:00 - INFO - 开始上传模型: hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct
2025-01-07 10:30:01 - INFO - 模型目录: deploy_ark/output_qwen/Qwen2.5-0.5B-Instruct
2025-01-07 10:30:02 - INFO - 上传方法: api
2025-01-07 10:30:03 - INFO - 登录 ModelScope 成功
2025-01-07 10:30:04 - INFO - 开始上传模型到: hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct
2025-01-07 10:35:00 - INFO - 模型上传成功
==================================================
🎉 模型上传成功！
模型地址: https://www.modelscope.cn/models/hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct
==================================================
```

## 🔍 验证上传结果

上传完成后，您可以：

1. 访问模型页面：`https://www.modelscope.cn/models/hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct`
2. 测试模型推理
3. 查看模型文件
4. 分享给其他用户

## 🤝 获取帮助

如果遇到问题，可以：

1. 查看上传日志：`upload_modelscope.log`
2. 运行帮助命令：`python3 upload_to_modelscope.py --help`
3. 访问 [魔搭社区文档](https://www.modelscope.cn/docs)
4. 在魔搭社区论坛寻求帮助

## 📚 相关链接

- [魔搭社区](https://www.modelscope.cn/)
- [获取访问令牌](https://www.modelscope.cn/my/myaccesstoken)
- [模型上传文档](https://www.modelscope.cn/docs/model-upload)
- [您的模型页面](https://www.modelscope.cn/models/hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct)

---

📄 **注意**：确保您的模型符合魔搭社区的使用条款和开源许可证要求。 