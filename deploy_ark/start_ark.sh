#!/bin/bash
# 火山方舟快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必要的环境变量
check_env() {
    print_info "检查环境变量..."
    
    if [ -z "$BASE_MODEL_PATH" ]; then
        export BASE_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
        print_warn "BASE_MODEL_PATH 未设置，使用默认值: $BASE_MODEL_PATH"
    fi
    
    if [ -z "$VOLCANO_API_KEY" ]; then
        print_warn "VOLCANO_API_KEY 未设置，部分功能可能无法使用"
    fi
    
    print_info "环境变量检查完成"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查Docker（可选）
    if command -v docker &> /dev/null; then
        print_info "Docker 已安装"
        DOCKER_AVAILABLE=true
    else
        print_warn "Docker 未安装，将使用本地模式"
        DOCKER_AVAILABLE=false
    fi
    
    print_info "依赖检查完成"
}

# 安装Python依赖
install_python_deps() {
    print_info "安装Python依赖..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt 文件未找到"
        exit 1
    fi
    
    python3 -m pip install -r requirements.txt
    print_info "Python依赖安装完成"
}

# 启动本地服务
start_local() {
    print_info "启动本地API服务..."
    
    # 设置环境变量
    export PORT=${PORT:-8000}
    export HOST=${HOST:-0.0.0.0}
    export WORKERS=${WORKERS:-1}
    export MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-8}
    
    print_info "服务配置:"
    print_info "  地址: $HOST:$PORT"
    print_info "  工作进程: $WORKERS"
    print_info "  批次大小: $MAX_BATCH_SIZE"
    print_info "  模型路径: $BASE_MODEL_PATH"
    
    # 启动服务
    python3 ark_api_server.py
}

# 使用Docker启动
start_docker() {
    print_info "使用Docker启动服务..."
    
    # 检查docker-compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "docker-compose 或 docker compose 未找到"
        exit 1
    fi
    
    # 创建必要的目录
    mkdir -p models logs cache
    
    # 启动服务
    $COMPOSE_CMD up -d
    
    print_info "Docker服务启动完成"
    print_info "API服务地址: http://localhost:8000"
    print_info "Redis服务地址: localhost:6379"
    
    # 等待服务启动
    print_info "等待服务启动..."
    sleep 10
    
    # 健康检查
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_info "✅ 服务健康检查通过"
    else
        print_warn "⚠️  服务健康检查失败，请检查日志"
    fi
}

# 部署到火山引擎AICC
deploy_to_aicc() {
    print_info "部署到火山引擎AICC机密计算平台..."
    
    # 检查必要的环境变量
    if [ -z "$VOLCANO_AK" ] && [ -z "$VOLCANO_API_KEY" ]; then
        print_error "VOLCANO_AK 或 VOLCANO_API_KEY 环境变量未设置"
        print_info "设置方法:"
        print_info "  export VOLCANO_AK='your_access_key'"
        print_info "  export VOLCANO_SK='your_secret_key'"
        print_info "  export VOLCANO_APP_ID='your_app_id'"
        print_info "  export VOLCANO_BUCKET_NAME='your_bucket_name'"
        exit 1
    fi
    
    MODEL_PATH=${1:-"./output_qwen"}
    MODEL_NAME=${2:-"qwen-finetune"}
    AICC_SPEC=${3:-"高级版"}
    BUCKET_NAME=${4:-"$VOLCANO_BUCKET_NAME"}
    
    print_info "AICC部署参数:"
    print_info "  模型路径: $MODEL_PATH"
    print_info "  模型名称: $MODEL_NAME"
    print_info "  AICC规格: $AICC_SPEC"
    print_info "  存储桶: $BUCKET_NAME"
    
    # 检查Jeddak SDK
    python3 -c "from jeddak_model_encryptor import JeddakModelEncryptor" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warn "Jeddak SDK 未安装，将使用模拟模式"
        print_info "安装Jeddak SDK:"
        print_info "  python3 ark_deploy.py install"
    fi
    
    # 执行AICC部署
    python3 ark_deploy.py deploy \
        --config aicc_config.json \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --aicc_spec "$AICC_SPEC" \
        --bucket_name "$BUCKET_NAME" \
        --output "${MODEL_NAME}_deployment_result.json"
}

# 测试API服务
test_api() {
    ENDPOINT_URL=${1:-"http://localhost:8000"}
    
    print_info "测试API服务: $ENDPOINT_URL"
    
    python3 test_concurrent.py \
        --endpoint "$ENDPOINT_URL" \
        --concurrent 5 \
        --total 20 \
        --output test_results.json
}

# 显示帮助信息
show_help() {
    echo "火山引擎AICC机密计算部署工具"
    echo ""
    echo "用法:"
    echo "  $0 local              # 启动本地服务"
    echo "  $0 docker             # 使用Docker启动服务"
    echo "  $0 deploy [参数]      # 部署到火山引擎AICC"
    echo "  $0 test [端点URL]     # 测试API服务"
    echo "  $0 install            # 安装依赖"
    echo "  $0 sdk                # 安装Jeddak SDK指导"
    echo ""
    echo "部署示例:"
    echo "  $0 deploy ./output_qwen qwen-model 高级版 my-bucket"
    echo ""
    echo "环境变量:"
    echo "  BASE_MODEL_PATH       # 基础模型路径"
    echo "  LORA_MODEL_PATH       # LoRA模型路径"
    echo "  VOLCANO_AK            # 火山引擎Access Key"
    echo "  VOLCANO_SK            # 火山引擎Secret Key"
    echo "  VOLCANO_APP_ID        # 火山账号ID"
    echo "  VOLCANO_BUCKET_NAME   # TOS存储桶名称"
    echo "  PORT                  # 服务端口 (默认: 8000)"
    echo "  HOST                  # 服务地址 (默认: 0.0.0.0)"
    echo "  WORKERS               # 工作进程数 (默认: 1)"
    echo "  MAX_BATCH_SIZE        # 最大批次大小 (默认: 8)"
    echo ""
    echo "AICC规格:"
    echo "  基础版                # 支持小尺寸模型，如1.5B"
    echo "  高级版                # 支持中尺寸模型，如32B"
    echo "  旗舰版                # 支持大尺寸模型，如DeepSeek R1-671B"
    echo ""
    echo "更多信息:"
    echo "  文档：https://www.volcengine.com/docs/85010/1546894"
}

# 主函数
main() {
    case "${1:-help}" in
        "local")
            check_env
            check_dependencies
            start_local
            ;;
        "docker")
            check_env
            check_dependencies
            start_docker
            ;;
        "deploy")
            check_env
            check_dependencies
            deploy_to_aicc "$2" "$3" "$4" "$5"
            ;;
        "test")
            test_api "$2"
            ;;
        "install")
            check_dependencies
            install_python_deps
            ;;
        "sdk")
            print_info "Jeddak SDK 安装指导"
            python3 ark_deploy.py install
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"
