# 火山引擎AICC一键部署指南

## 🚀 快速开始

```bash
cd deploy_ark
./one_key_deploy.sh
```

## 📋 准备清单

在运行一键部署脚本之前，您需要准备以下信息：

### 1. 🔐 火山引擎账号信息

**✨ 新功能：自动加载凭证**

脚本现在可以自动从 `user.csv` 文件读取火山引擎凭证，无需手动输入！

#### 方式一：创建 user.csv 文件（推荐）

在 `deploy_ark` 目录或上级目录创建 `user.csv` 文件：

```csv
用户名,登录密码,登录地址,所属主账号ID,Access Key ID,Secret Access Key
wphu,Admin12345@,https://console.volcengine.com/auth/login/user/2108490487,2108490487,AKLTNDE1YjQwYjI5MmRlNDU0ZGJjOTMzMDI0MDI1ZWQ3MTQ,TWpJNVpEWTVOV0l5WVRJMk5EUm1ZVGs1TkdRM056UTRPRFV4WlRka1lUWQ==
```

**CSV文件字段说明**：
- 第4列：账号ID (APP_ID)
- 第5列：Access Key ID  
- 第6列：Secret Access Key

#### 方式二：手动获取凭证信息

如果没有 CSV 文件，脚本将引导您手动输入：

##### 获取 Access Key 和 Secret Key
1. 登录火山引擎控制台：https://console.volcengine.com/
2. 点击右上角头像 → **API访问密钥**
3. 点击 **新建访问密钥**
4. 记录下 **Access Key** 和 **Secret Key**

##### 获取账号ID
1. 在火山引擎控制台右上角点击头像
2. 在下拉菜单中可以看到您的 **账号ID**

### 2. 🪣 TOS对象存储桶

#### 创建TOS存储桶
1. 在火山引擎控制台进入 **对象存储TOS** 服务
2. 点击 **创建存储桶**
3. 配置信息：
   - **存储桶名称**: 全局唯一的名称（如：`my-aicc-models-2024`）
   - **地域**: 必须选择 **华北2（北京）**
   - **访问权限**: 私有读写
4. 点击创建

### 3. 🛠️ Jeddak SDK

#### SDK下载地址
- 官方文档：https://www.volcengine.com/docs/85010/1546894
- 当前推荐版本：`0.1.7.36`

#### 下载步骤
1. 访问上述文档链接
2. 下载对应版本的 SDK 文件：`bytedance.jeddak_secure_channel-0.1.7.36-py3-none-any.whl`
3. 脚本运行时会询问SDK文件路径

### 4. 📂 模型文件

确保您的模型目录包含以下文件：
- `config.json` - 模型配置文件
- `pytorch_model.bin` 或 `model.safetensors` - 模型权重文件
- `tokenizer.json` 或 `tokenizer_config.json` - 分词器配置
- `adapter_config.json` (可选) - LoRA适配器配置

## 🎯 使用流程

### 步骤1: 运行脚本
```bash
./one_key_deploy.sh
```

### 步骤2: 按引导配置

脚本会逐步引导您完成以下配置：

1. **检查火山引擎凭证**
   - 输入 Access Key
   - 输入 Secret Key  
   - 输入账号ID

2. **配置TOS存储桶**
   - 输入存储桶名称

3. **检查Jeddak SDK**
   - 检测是否已安装
   - 如未安装，引导您下载和安装

4. **选择模型文件**
   - 自动检测常见路径
   - 或手动指定模型路径

5. **选择AICC规格**
   - 基础版：适用于1.5B等小模型
   - 高级版：适用于32B等中型模型
   - 旗舰版：适用于DeepSeek R1-671B等大模型

6. **配置模型名称**
   - 设置部署后的模型名称

### 步骤3: 确认并部署

脚本会显示配置摘要，确认后开始部署。

## 📊 部署过程

部署包含四个阶段：

1. **🔐 模型加密和上传**
   - 使用Jeddak SDK加密模型
   - 上传到TOS对象存储
   - 生成密钥环和加密密钥

2. **📤 发布模型**
   - 发布到AICC模型广场
   - 关联加密信息

3. **🚀 部署模型服务**
   - 选择AICC规格部署
   - 配置自动扩缩容

4. **🧪 测试验证**
   - 自动测试模型可用性
   - 获取推理端点信息

## 📄 部署结果

部署成功后会生成结果文件：`{模型名称}_deployment_{时间戳}.json`

### 结果文件内容
```json
{
  "deployment_status": "success",
  "model_info": {
    "model_name": "qwen-finetune",
    "model_id": "model_xxx",
    "deployment_id": "deploy_xxx",
    "aicc_spec": "高级版"
  },
  "encryption_info": {
    "ring_id": "ring_xxx",
    "key_id": "key_xxx",
    "baseline": "baseline_xxx"
  },
  "endpoint_info": {
    "endpoint_url": "https://aicc.volcengineapi.com/inference/xxx",
    "service_name": "service-xxx",
    "access_token": "your_access_token"
  },
  "test_result": {
    "status": "success",
    "test_passed": true
  }
}
```

## 🔧 后续操作

### 测试推理服务
```bash
# 使用内置测试工具
python3 test_concurrent.py --endpoint <推理端点URL>

# 或使用curl测试
curl -X POST <推理端点URL>/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{"prompt": "你好", "max_length": 512}'
```

### 监控服务状态
```bash
python3 ark_deploy.py info --deployment_id <部署ID>
```

### 查询端点信息
```bash
python3 ark_deploy.py info --deployment_id <部署ID>
```

## 🔍 故障排除

### 常见问题

#### 1. 凭证配置错误
- **现象**: 提示AK/SK无效
- **解决**: 重新检查火山引擎控制台的凭证信息

#### 2. TOS存储桶权限问题
- **现象**: 上传失败
- **解决**: 确认存储桶在北京地区，且账号有权限

#### 3. Jeddak SDK安装失败
- **现象**: SDK导入失败
- **解决**: 检查SDK文件路径，或使用模拟模式

#### 4. 模型文件不完整
- **现象**: 模型加载失败
- **解决**: 检查模型目录是否包含必要文件

#### 5. 部署超时
- **现象**: 等待部署完成超时
- **解决**: 大模型需要更长时间，耐心等待或检查网络

### 日志查看
```bash
# 查看详细日志
tail -f ~/.volcano/logs/deploy.log

# 检查系统状态
python3 ark_deploy.py status
```

## 💡 高级选项

### 环境变量预设
您可以预先设置环境变量来跳过某些配置步骤：

```bash
# 设置火山引擎凭证
export VOLCANO_AK="your_access_key"
export VOLCANO_SK="your_secret_key"
export VOLCANO_APP_ID="your_app_id"

# 设置存储桶
export VOLCANO_BUCKET_NAME="your_bucket_name"

# 然后运行脚本
./one_key_deploy.sh
```

### 非交互模式
如果您想要完全自动化部署（适用于CI/CD），可以设置所有必要的环境变量。

### 配置文件模式
编辑 `aicc_config.json` 文件，预先配置所有参数：

```json
{
  "volc_ak": "your_access_key",
  "volc_sk": "your_secret_key", 
  "app_id": "your_app_id",
  "bucket_name": "your_bucket_name"
}
```

## 📞 获取帮助

- **官方文档**: https://www.volcengine.com/docs/85010/1546894
- **SDK下载**: 从官方文档获取最新版本
- **技术支持**: 火山引擎技术支持团队

## 🎉 恭喜

如果您成功完成部署，现在您的模型已经安全地部署在火山引擎AICC机密计算平台上，享受端到端的加密推理服务！ 