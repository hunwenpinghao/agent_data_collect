# Jeddak SDK 自动下载安装功能

## 概述

为了简化用户体验，我们在部署脚本中添加了 Jeddak Secure Model SDK 的自动下载安装功能。用户无需手动下载 SDK 文件，脚本会自动处理下载和安装过程。

## 更新内容

### 1. 一键部署脚本 (`one_key_deploy.sh`)

#### 新增功能
- **`auto_install_jeddak_sdk()` 函数**: 自动下载和安装 SDK
- **智能下载工具检测**: 优先使用 `curl`，备选 `wget`
- **文件完整性验证**: 检查下载文件大小和完整性
- **自动清理**: 安装完成后自动删除下载文件

#### 改进的用户体验
- **全自动化**: 用户无需手动干预，一键完成 SDK 安装
- **友好提示**: 提供详细的下载和安装状态信息
- **向下兼容**: 自动安装失败时，提供手动安装指导

### 2. 模型加密工具 (`jeddak_model_encryptor.py`)

#### 新增功能
- **`auto_install_jeddak_sdk()` 函数**: Python 版本的自动安装功能
- **增强的 SDK 检查**: 自动尝试安装缺失的 SDK
- **改进的错误处理**: 更友好的错误提示和解决方案

#### 智能安装流程
```python
def _check_sdk(self):
    try:
        # 检查 SDK 是否已安装
        from bytedance.jeddak_secure_model.model_encryption import ...
        self.sdk_available = True
    except ImportError:
        # 自动尝试安装
        if auto_install_jeddak_sdk():
            # 验证安装成功
            self.sdk_available = True
        else:
            # 使用模拟模式
            self.sdk_available = False
```

## SDK 下载信息

### 下载地址
```
https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl
```

### 文件信息
- **版本**: 0.1.7.36
- **格式**: Python Wheel (.whl)
- **最小文件大小**: 10KB（用于验证下载完整性）

## 自动安装流程

### 1. 环境检查
- 检查系统是否有 `curl` 或 `wget`
- 检查 Python 和 pip 环境

### 2. 下载阶段
```bash
curl -L -o bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl \
     https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl
```

### 3. 验证阶段
- 检查文件是否存在
- 验证文件大小（至少 10KB）
- 确保下载完整

### 4. 安装阶段
```bash
pip install bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl
```

### 5. 验证阶段
```python
from bytedance.jeddak_secure_model.model_encryption import JeddakModelEncrypter
```

### 6. 清理阶段
- 删除下载的临时文件
- 释放存储空间

## 错误处理

### 常见错误和解决方案

#### 1. 网络连接问题
- **错误**: 下载超时或连接失败
- **解决**: 自动回退到手动安装模式
- **提示**: 提供手动下载命令

#### 2. 权限问题
- **错误**: pip 安装权限不足
- **解决**: 提示用户使用 `sudo` 或虚拟环境
- **建议**: 使用 `--user` 参数进行用户级安装

#### 3. 工具缺失
- **错误**: 系统没有 curl 或 wget
- **解决**: 显示工具安装指导
- **替代**: 提供手动下载链接

## 使用方法

### 自动安装 (推荐)
运行部署脚本，系统会自动处理 SDK 安装：
```bash
./one_key_deploy.sh
```

### 手动安装 (备选)
如果自动安装失败，可以手动安装：
```bash
# 下载 SDK
curl -L -O https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl

# 安装 SDK
pip install bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl
```

### Python 代码中使用
```python
from jeddak_model_encryptor import install_jeddak_sdk

# 自动安装 SDK
if install_jeddak_sdk():
    print("SDK 安装成功")
else:
    print("SDK 安装失败，请手动安装")
```

## 兼容性

### 支持的系统
- ✅ macOS (Darwin)
- ✅ Linux (Ubuntu, CentOS, 等)
- ✅ Windows (WSL)

### 支持的 Python 版本
- ✅ Python 3.7+
- ✅ Python 3.8+
- ✅ Python 3.9+
- ✅ Python 3.10+
- ✅ Python 3.11+

### 依赖工具
- `curl` (推荐) 或 `wget`
- `pip` (Python 包管理器)

## 安全考虑

### 下载安全
- 使用官方 CDN 地址
- HTTPS 加密传输
- 文件完整性验证

### 安装安全
- 使用官方 pip 安装方式
- 自动清理临时文件
- 不存储敏感信息

## 故障排除

### 常见问题

#### Q: 自动安装失败怎么办？
A: 脚本会自动回退到模拟模式，或者可以手动下载安装 SDK。

#### Q: 如何验证 SDK 是否正确安装？
A: 运行以下 Python 代码验证：
```python
try:
    from bytedance.jeddak_secure_model.model_encryption import JeddakModelEncrypter
    print("✅ SDK 安装成功")
except ImportError:
    print("❌ SDK 未安装或安装失败")
```

#### Q: 可以指定其他版本的 SDK 吗？
A: 目前脚本固定使用 0.1.7.36 版本，如需其他版本，请手动下载安装。

#### Q: 在虚拟环境中如何使用？
A: 激活虚拟环境后运行脚本，SDK 会安装到虚拟环境中。

## 更新日志

### v1.0 (2024-01-XX)
- ✅ 添加自动下载功能
- ✅ 添加文件完整性验证
- ✅ 添加智能错误处理
- ✅ 添加用户友好提示
- ✅ 支持多种下载工具

### 未来计划
- [ ] 支持多版本 SDK 选择
- [ ] 添加 SDK 更新检查
- [ ] 优化下载速度和稳定性
- [ ] 添加离线安装包支持 