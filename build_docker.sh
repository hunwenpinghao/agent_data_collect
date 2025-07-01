#!/bin/bash

# Qwen3-0.5B 微调项目 Docker 构建和运行脚本

set -e

# 配置参数
IMAGE_NAME="qwen3-finetune"
TAG="latest"
CONTAINER_NAME="qwen3-finetune-container"

# 颜色输出函数
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# 检查Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装"
        exit 1
    fi
    print_info "Docker 检查通过"
}

# 构建镜像
build_image() {
    print_info "构建Docker镜像..."
    if docker build -t ${IMAGE_NAME}:${TAG} .; then
        print_success "镜像构建完成: ${IMAGE_NAME}:${TAG}"
    else
        print_error "镜像构建失败"
        exit 1
    fi
}

# 清理旧容器
cleanup_container() {
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "清理旧容器: ${CONTAINER_NAME}"
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    fi
}

# 运行容器（交互模式）
run_interactive() {
    cleanup_container
    print_info "启动交互式容器..."
    
    # 检查GPU支持
    GPU_ARGS=""
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        GPU_ARGS="--gpus all"
        print_info "启用GPU支持"
    fi
    
    # 创建目录
    mkdir -p ./data ./output_qwen ./logs ./.cache
    
    docker run -it --rm \
        ${GPU_ARGS} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/output_qwen:/app/output_qwen \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/.cache:/app/.cache \
        -p 6006:6006 \
        -p 8000:8000 \
        --shm-size=8g \
        ${IMAGE_NAME}:${TAG}
}

# 后台运行
run_background() {
    cleanup_container
    print_info "启动后台容器..."
    
    GPU_ARGS=""
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        GPU_ARGS="--gpus all"
    fi
    
    mkdir -p ./data ./output_qwen ./logs ./.cache
    
    docker run -d \
        ${GPU_ARGS} \
        --name ${CONTAINER_NAME} \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/output_qwen:/app/output_qwen \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/.cache:/app/.cache \
        -p 6006:6006 \
        -p 8000:8000 \
        --shm-size=8g \
        ${IMAGE_NAME}:${TAG} \
        tail -f /dev/null
    
    print_success "容器已启动: ${CONTAINER_NAME}"
    print_info "进入容器: docker exec -it ${CONTAINER_NAME} /bin/bash"
}

# 显示帮助
show_help() {
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build         构建Docker镜像"
    echo "  run           运行容器（交互模式）"
    echo "  run-bg        运行容器（后台模式）"
    echo "  shell         进入容器"
    echo "  train         开始训练"
    echo "  tensorboard   启动TensorBoard"
    echo "  compose       使用docker-compose启动"
    echo "  compose-cpu   使用docker-compose启动（CPU模式）"
    echo "  stop          停止容器"
    echo "  clean         清理容器和镜像"
    echo "  logs          查看日志"
    echo "  help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 build && $0 run     # 构建并运行"
    echo "  $0 train               # 开始训练"
    echo "  $0 compose             # 使用docker-compose"
    echo "  $0 compose-cpu         # CPU模式（无GPU要求）"
}

# 主函数
case "${1:-help}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        build_image
        run_interactive
        ;;
    "run-bg")
        check_docker
        build_image
        run_background
        ;;
    "shell")
        if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            docker exec -it ${CONTAINER_NAME} /bin/bash
        else
            print_error "容器未运行，请先执行: $0 run-bg"
        fi
        ;;
    "train")
        if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            docker exec -it ${CONTAINER_NAME} /bin/bash -c "./run_train.sh"
        else
            print_error "容器未运行，请先执行: $0 run-bg"
        fi
        ;;
    "tensorboard")
        if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            print_info "启动TensorBoard: http://localhost:6006"
            docker exec -d ${CONTAINER_NAME} tensorboard --logdir ./output_qwen/runs --host 0.0.0.0
        else
            print_error "容器未运行，请先执行: $0 run-bg"
        fi
        ;;
    "compose")
        print_info "使用docker-compose启动（GPU模式）..."
        docker-compose up -d --build
        print_success "服务已启动"
        print_info "进入容器: docker-compose exec qwen3-finetune /bin/bash"
        print_info "开始训练: docker-compose exec qwen3-finetune ./run_train.sh"
        print_info "TensorBoard: http://localhost:6006 (主服务) http://localhost:6007 (独立服务)"
        ;;
    "compose-cpu")
        print_info "使用docker-compose启动（CPU模式）..."
        docker-compose -f docker-compose.cpu.yml up -d --build
        print_success "服务已启动（CPU模式）"
        print_info "进入容器: docker-compose -f docker-compose.cpu.yml exec qwen3-finetune /bin/bash"
        print_info "开始训练: docker-compose -f docker-compose.cpu.yml exec qwen3-finetune ./run_train.sh"
        print_info "TensorBoard: http://localhost:6006 (主服务) http://localhost:6007 (独立服务)"
        ;;
    "stop")
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
        print_success "容器已停止"
        ;;
    "clean")
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
        docker rmi ${IMAGE_NAME}:${TAG} 2>/dev/null || true
        print_success "清理完成"
        ;;
    "logs")
        docker logs -f ${CONTAINER_NAME}
        ;;
    "help"|*)
        show_help
        ;;
esac 