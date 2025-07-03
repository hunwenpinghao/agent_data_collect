# 🚀 Qwen模型火山引擎部署指南

本文档提供了在火山引擎上部署微调后的Qwen2.5-0.5B-Instruct模型的完整解决方案，支持高并发生产环境。

## 📋 目录

- [概述](#概述)
- [部署方案](#部署方案)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [监控和运维](#监控和运维)
- [测试指南](#测试指南)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 🌟 概述

该部署方案提供了两种主要的部署方式：

1. **火山引擎机器学习平台 veMLP** - 完全控制的Kubernetes部署
2. **火山方舟** - 托管式API服务

### 支持功能
- ✅ 高并发推理（支持数千QPS）
- ✅ 自动扩缩容
- ✅ 负载均衡
- ✅ 健康检查和监控
- ✅ LoRA和完整微调模型
- ✅ GPU优化和量化支持

## 🛠️ 部署方案

### 方案对比

| 特性 | veMLP (Kubernetes) | 火山方舟 |
|------|-------------------|----------|
| 控制度 | 完全控制 | 托管服务 |
| 部署复杂度 | 中等 | 简单 |
| 自定义能力 | 高 | 中等 |
| 运维成本 | 中等 | 低 |
| 适用场景 | 定制化需求高 | 快速上线 |

## 📁 文件结构

```
deploy/
├── Dockerfile.production      # 生产环境Docker镜像
├── api_server.py             # 高性能API服务器
├── k8s-deployment.yaml       # Kubernetes部署配置
├── deploy_to_volcengine.sh   # 火山引擎部署脚本
├── convert_for_volcano.py    # 模型格式转换
├── deploy_to_ark.py          # 火山方舟部署
├── test_api.py              # API测试脚本
├── client_example.py         # 客户端使用示例
├── nginx.conf               # Nginx配置
├── supervisord.conf         # Supervisor配置
├── monitoring/              # 监控配置
│   ├── prometheus-config.yaml
│   └── grafana-dashboard.json
└── scaling/                 # 扩缩容配置
    └── vpa-config.yaml
```

## 🚀 快速开始

### 1. 环境准备

确保您已经：
- [x] 完成模型微调训练
- [x] 安装Docker和kubectl
- [x] 配置火山引擎访问权限
- [x] 准备GPU资源

### 2. 模型转换（LoRA模型）

如果您使用的是LoRA模型，需要先转换为完整模型：

```bash
cd deploy
python convert_for_volcano.py \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --lora_path "../output_qwen_lora" \
    --output_path "./merged_model"
```

### 3. 选择部署方式

#### 方式一：veMLP部署（推荐）

```bash
# 设置环境变量
export REGISTRY="your-volcengine-registry"
export IMAGE_NAME="qwen-model-api" 
export VERSION="v1.0.0"
export NAMESPACE="ai-models"

# 执行部署
./deploy_to_volcengine.sh
```

#### 方式二：火山方舟部署

```bash
# 设置API密钥
export VOLCANO_API_KEY="your-api-key"
export VOLCANO_API_SECRET="your-api-secret"

# 部署到方舟
python deploy_to_ark.py \
    --model_path "./merged_model" \
    --model_name "qwen2.5-finetuned" \
    --endpoint_name "qwen-production"
```

### 4. 测试部署

```bash
# 基础测试
python test_api.py --url "http://your-endpoint" --test_type health

# 完整测试
python test_api.py --url "http://your-endpoint" --test_type all
```

## 🔧 详细配置

### API服务器配置

主要环境变量：

```bash
# 模型配置
BASE_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL_PATH="/app/models/your-lora-model"
QUANTIZATION="none"  # none, 4bit, 8bit

# 服务配置
HOST="0.0.0.0"
PORT="8000"
WORKERS="4"

# Redis配置
REDIS_URL="redis://redis-service:6379"
```

### 资源配置示例

```yaml
# 基础配置（小规模）
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "4"
    nvidia.com/gpu: 1
```

## 📊 监控和运维

### 监控指标

- `api_requests_total` - 总请求数
- `api_request_duration_seconds` - 请求响应时间
- `api_active_connections` - 活跃连接数
- `model_inference_duration_seconds` - 模型推理时间

### 部署监控

```bash
# 部署Prometheus监控
kubectl apply -f monitoring/prometheus-config.yaml

# 访问监控面板
kubectl port-forward service/prometheus-service 9090:9090
```

## 🧪 测试指南

### 性能测试

```bash
# 健康检查
python test_api.py --url "http://localhost:8000" --test_type health

# 压力测试
python test_api.py --url "http://localhost:8000" --test_type stress --duration 300 --concurrent 20
```

### 客户端使用

```python
from deploy.client_example import QwenAPIClient

client = QwenAPIClient("http://your-endpoint")
result = client.generate("请为一家咖啡店写一段文案")
print(result['response'])
```

## ❗ 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径和GPU资源
2. **推理超时**: 调整超时配置或启用量化
3. **内存不足**: 增加内存限制或使用量化
4. **扩缩容不生效**: 检查HPA状态和指标服务器

### 调试命令

```bash
# 检查Pod状态
kubectl get pods -n ai-models

# 查看日志
kubectl logs -f deployment/qwen-model-api -n ai-models

# 进入Pod调试
kubectl exec -it <pod-name> -- /bin/bash
```

## 💡 最佳实践

### 性能优化
1. 使用4bit或8bit量化减少内存占用
2. 启用批处理提高吞吐量
3. 使用Redis缓存常见请求
4. 部署后进行模型预热

### 安全配置
1. 使用强API密钥并定期轮换
2. 配置Kubernetes网络策略
3. 设置合适的资源限制
4. 使用可信镜像仓库

### 运维建议
1. 配置关键指标告警
2. 使用日志聚合工具
3. 定期备份模型和配置
4. 使用蓝绿部署降低风险

## 📞 支持

如果遇到问题，请：
1. 检查故障排除部分
2. 查看项目日志文件
3. 使用测试脚本验证环境

---

**版本**: v1.0.0  
**最后更新**: 2025年1月