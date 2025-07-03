# 火山引擎AICC机密计算部署工具

基于火山引擎Jeddak AICC机密计算平台的安全模型部署解决方案，支持Qwen系列模型的加密部署和高并发推理服务。

## 🔒 核心特性

- **🛡️ 模型加密**: 使用Jeddak Secure Model SDK进行模型加密
- **☁️ 云端部署**: 部署到火山引擎AICC机密计算平台
- **⚡ 高并发**: 支持100+并发请求，QPS达50-100
- **🔐 安全推理**: 端到端加密推理，保护模型和数据安全
- **📊 监控告警**: 完整的性能监控和告警机制
- **🚀 一键部署**: 完整的四步部署流程自动化
- **✨ 自动配置**: 从CSV文件自动加载凭证，零手动输入
- **📦 智能安装**: 自动下载安装Jeddak SDK，无需手动操作

## 📁 文件结构

```
deploy_ark/
├── jeddak_model_encryptor.py  # Jeddak模型加密工具
├── ark_deploy.py              # AICC部署管理工具
├── ark_api_server.py          # 高性能API服务器
├── ark_client.py              # 并发客户端SDK
├── test_concurrent.py         # 并发测试工具
├── one_key_deploy.sh          # 一键部署脚本（含凭证自动加载）
├── start_ark.sh               # 快速启动脚本
├── aicc_config.json           # AICC配置文件
├── user.csv                   # 用户凭证文件
├── requirements.txt           # Python依赖
├── Dockerfile                 # 容器构建文件
├── docker-compose.yml         # 容器编排配置
├── AUTO_INSTALL_SDK_SUMMARY.md # SDK自动安装功能说明
└── README.md                  # 说明文档
```

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 设置火山引擎凭证
export VOLCANO_AK="your_access_key"
export VOLCANO_SK="your_secret_key"
export VOLCANO_APP_ID="your_app_id"
export VOLCANO_BUCKET_NAME="your_bucket_name"

# 可选：设置模型路径
export BASE_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
export LORA_MODEL_PATH="./output_qwen"
```

### 2. 安装依赖

```bash
cd deploy_ark
./start_ark.sh install
```

### 3. Jeddak SDK 安装 🚀

#### 🆕 自动安装 (推荐)

**新功能**: SDK 现在支持自动下载安装！无需手动下载。

```bash
# 使用一键部署，SDK会自动安装
./one_key_deploy.sh

# 或者在Python代码中自动安装
python3 -c "from jeddak_model_encryptor import auto_install_jeddak_sdk; auto_install_jeddak_sdk()"
```

#### 📋 手动安装 (备选)

如果自动安装失败，可以手动安装：

```bash
# 直接下载最新版本
curl -L -O https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl

# 安装SDK
pip install bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl

# 验证安装
python3 -c "from bytedance.jeddak_secure_model.model_encryption import JeddakModelEncrypter; print('✅ SDK 安装成功')"
```

#### 🔧 传统方式

```bash
# 获取SDK安装指导
./start_ark.sh sdk

# 或直接查看指导
python ark_deploy.py install
```

### 4. 部署到火山引擎AICC

```bash
# 完整AICC部署流程
./start_ark.sh deploy ./output_qwen qwen-finetune 高级版 your-bucket-name

# 或使用python脚本
python ark_deploy.py deploy \
    --model_path ./output_qwen \
    --model_name qwen-finetune \
    --aicc_spec 高级版 \
    --bucket_name your-bucket-name \
    --output deployment_result.json
```

## 🔧 AICC部署流程

### 四步部署流程

1. **🔐 模型加密和上传**
   - 使用Jeddak SDK加密模型文件
   - 上传加密模型到TOS对象存储
   - 生成密钥环和加密密钥

2. **📤 发布模型**
   - 将加密模型发布到AICC模型广场
   - 配置模型元数据和描述
   - 关联加密密钥信息

3. **🚀 部署模型服务**
   - 选择AICC规格（基础版/高级版/旗舰版）
   - 配置实例数量和自动扩缩容
   - 等待部署完成

4. **🧪 测试验证**
   - 自动测试模型可用性
   - 获取推理端点信息
   - 验证加密推理服务

### AICC规格说明

| 规格 | 适用模型 | 说明 |
|------|---------|------|
| 基础版 | 小尺寸模型 | 支持1.5B等小模型 |
| 高级版 | 中尺寸模型 | 支持32B等中型模型 |
| 旗舰版 | 大尺寸模型 | 支持DeepSeek R1-671B等大模型 |

## 🔧 命令行工具

### 部署管理

```bash
# 完整部署
python ark_deploy.py deploy \
    --model_path ./output_qwen \
    --model_name qwen-finetune \
    --aicc_spec 高级版

# 仅加密上传
python ark_deploy.py encrypt \
    --model_path ./output_qwen \
    --model_name qwen-finetune

# 发布已加密的模型
python ark_deploy.py publish \
    --encrypt_result qwen-finetune_encrypt_result.json \
    --model_name qwen-finetune

# 测试部署的模型
python ark_deploy.py test \
    --deployment_id your_deployment_id

# 查询端点信息
python ark_deploy.py info \
    --deployment_id your_deployment_id
```

### 启动脚本

```bash
# 本地服务
./start_ark.sh local

# Docker服务
./start_ark.sh docker

# AICC部署
./start_ark.sh deploy [模型路径] [模型名] [AICC规格] [存储桶]

# 测试服务
./start_ark.sh test [端点URL]

# 安装依赖
./start_ark.sh install

# SDK指导
./start_ark.sh sdk
```

## 🚢 本地服务

### 启动API服务

```bash
# 本地启动
./start_ark.sh local

# Docker启动
./start_ark.sh docker
```

### API接口

#### 生成文本
```http
POST /generate
Content-Type: application/json

{
  "prompt": "你好，请介绍一下自己",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

#### 批量生成
```http
POST /batch
Content-Type: application/json

{
  "requests": [
    {"prompt": "提示词1", "max_length": 512},
    {"prompt": "提示词2", "max_length": 512}
  ]
}
```

#### 健康检查
```http
GET /health
```

#### 模型信息
```http
GET /model/info
```

## 📊 性能测试

### 并发测试

```bash
# 基础测试
python test_concurrent.py \
  --endpoint http://localhost:8000 \
  --concurrent 5 \
  --total 20

# 压力测试
python test_concurrent.py \
  --endpoint http://localhost:8000 \
  --concurrent 10 \
  --total 100 \
  --output stress_test_results.json
```

### 客户端SDK使用

```python
from ark_client import ArkSyncClient, ArkClientConfig

config = ArkClientConfig(
    api_key="your_api_key",
    endpoint_url="http://localhost:8000"
)

with ArkSyncClient(config) as client:
    # 单个请求
    result = client.generate_single("你好")
    
    # 并发请求
    prompts = ["问题1", "问题2", "问题3"]
    results = client.concurrent_generate(prompts, max_workers=5)
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 必填 |
|--------|------|------|
| `VOLCANO_AK` | 火山引擎Access Key | ✅ |
| `VOLCANO_SK` | 火山引擎Secret Key | ✅ |
| `VOLCANO_APP_ID` | 火山账号ID | ✅ |
| `VOLCANO_BUCKET_NAME` | TOS存储桶名称 | ✅ |
| `BASE_MODEL_PATH` | 基础模型路径 | ❌ |
| `LORA_MODEL_PATH` | LoRA适配器路径 | ❌ |
| `AICC_SPEC` | AICC规格 | ❌ |
| `PORT` | 服务端口 | ❌ |
| `HOST` | 服务地址 | ❌ |
| `WORKERS` | 工作进程数 | ❌ |

### 配置文件

编辑 `aicc_config.json` 自定义配置：

```json
{
  "volc_ak": "${VOLCANO_AK}",
  "volc_sk": "${VOLCANO_SK}",
  "app_id": "${VOLCANO_APP_ID}",
  "bucket_name": "${VOLCANO_BUCKET_NAME}",
  "region": "cn-beijing",
  "aicc_api_endpoint": "https://aicc.volcengineapi.com"
}
```

## 🔍 故障排除

### 常见问题

1. **SDK未安装**
   ```bash
   # 解决方案
   ./start_ark.sh sdk
   # 按照指导下载并安装SDK
   ```

2. **环境变量未设置**
   ```bash
   # 解决方案
   export VOLCANO_AK="your_access_key"
   export VOLCANO_SK="your_secret_key"
   export VOLCANO_APP_ID="your_app_id"
   export VOLCANO_BUCKET_NAME="your_bucket_name"
   ```

3. **模型加密失败**
   - 检查模型路径是否正确
   - 确认TOS存储桶权限
   - 验证火山引擎凭证

4. **部署超时**
   - 大模型部署需要更长时间
   - 检查网络连接
   - 确认AICC规格选择

### 性能优化

1. **选择合适的AICC规格**
   - 根据模型大小选择规格
   - 考虑并发需求

2. **优化批处理**
   - 调整批次大小
   - 平衡延迟和吞吐量

3. **监控资源使用**
   - 观察GPU利用率
   - 跟踪内存消耗

## 📞 支持和文档

- **官方文档**: https://www.volcengine.com/docs/85010/1546894
- **Jeddak AICC**: 火山引擎机密计算平台
- **支持规格**: 基础版、高级版、旗舰版

## 📄 许可证

本项目基于 MIT 许可证开源。 