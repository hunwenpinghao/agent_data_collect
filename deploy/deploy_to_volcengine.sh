#!/bin/bash
# 火山引擎部署脚本

set -e

echo "🚀 开始部署到火山引擎"

# 配置变量
REGISTRY=${REGISTRY:-"your-volcengine-registry"}
IMAGE_NAME=${IMAGE_NAME:-"qwen-model-api"} 
VERSION=${VERSION:-"v1.0.0"}
NAMESPACE=${NAMESPACE:-"ai-models"}

# 1. 构建Docker镜像
echo "📦 构建Docker镜像..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} -f deploy/Dockerfile.production .

# 2. 推送到火山引擎镜像仓库
echo "📤 推送镜像到火山引擎..."
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}

# 3. 上传模型文件到火山引擎存储
echo "📂 上传模型文件..."
# 假设您已经配置了火山引擎CLI
# volcengine tos cp -r ./output_qwen/ tos://your-bucket/models/qwen/

# 4. 创建命名空间
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# 5. 创建PVC
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: ssd
EOF

# 6. 部署Redis
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF

# 7. 部署应用
sed "s|your-registry|${REGISTRY}|g; s|latest|${VERSION}|g" deploy/k8s-deployment.yaml | \
kubectl apply -f - -n ${NAMESPACE}

# 8. 等待部署完成
echo "⏳ 等待部署完成..."
kubectl rollout status deployment/qwen-model-api -n ${NAMESPACE}

# 9. 获取访问地址
echo "🎉 部署完成！"
echo "访问地址："
kubectl get service qwen-api-service -n ${NAMESPACE}

echo "📊 监控指标："
echo "kubectl port-forward service/qwen-api-service 8080:80 -n ${NAMESPACE}"
echo "访问 http://localhost:8080/metrics 查看监控指标"

echo "🔍 检查状态："
echo "kubectl get pods -n ${NAMESPACE}"
echo "kubectl logs -f deployment/qwen-model-api -n ${NAMESPACE}" 