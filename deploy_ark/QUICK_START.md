# 🚀 快速开始 - 一键部署

## 💡 3分钟完成部署

```bash
cd deploy_ark
./one_key_deploy.sh
```

## 📋 需要准备的信息

### 🔐 火山引擎凭证（自动加载）
脚本会自动从 `user.csv` 文件读取凭证，无需手动输入！

**user.csv 格式**：
```csv
用户名,登录密码,登录地址,所属主账号ID,Access Key ID,Secret Access Key
wphu,Admin12345@,https://console.volcengine.com/auth/login/user/2108490487,2108490487,AKLTNDE1YjQwYjI5MmRlNDU0ZGJjOTMzMDI0MDI1ZWQ3MTQ,TWpJNVpEWTVOV0l5WVRJMk5EUm1ZVGs1TkdRM056UTRPRFV4WlRka1lUWQ==
```

**手动配置（如果没有CSV文件）**：
- **Access Key** - 在火山引擎控制台 → 头像 → API访问密钥 获取
- **Secret Key** - 同上
- **账号ID** - 在火山引擎控制台头像下拉菜单查看

### 🪣 TOS存储桶（自动推荐）
脚本会自动推荐存储桶名称：`aicc-models-用户名-年月`

**手动创建存储桶**：
- **存储桶名称** - 在火山引擎控制台 → 对象存储TOS → 创建存储桶
- **重要**: 必须选择华北2（北京）地区

### 🛠️ Jeddak SDK（可选）
- 下载地址：https://www.volcengine.com/docs/85010/1546894
- 文件名：`bytedance.jeddak_secure_channel-0.1.7.36-py3-none-any.whl`
- 如果没有SDK，会自动使用模拟模式

### 📂 模型文件
- 确保模型目录包含：`config.json`、`pytorch_model.bin`、`tokenizer.json`

## 🎯 一键命令

```bash
# 如果已有环境变量
export VOLCANO_AK="your_access_key"
export VOLCANO_SK="your_secret_key" 
export VOLCANO_APP_ID="your_app_id"
export VOLCANO_BUCKET_NAME="your_bucket_name"

# 运行一键部署
./one_key_deploy.sh
```

## 📱 获取帮助

详细指南请查看：[DEPLOY_GUIDE.md](./DEPLOY_GUIDE.md)

---
**⏰ 预计时间**: 首次部署10-30分钟（取决于模型大小）  
**🎯 结果**: 获得安全的加密推理端点 