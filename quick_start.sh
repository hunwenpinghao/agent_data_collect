#!/bin/bash

# Qwen3-0.5B 微调项目快速开始脚本
# 演示完整的 Docker 使用流程

set -e

echo "=================================="
echo "🚀 Qwen3-0.5B 微调项目快速开始"
echo "=================================="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker 检查通过"

# 显示选项
echo ""
echo "请选择运行方式:"
echo "1) 使用 build_docker.sh 脚本 (推荐)"
echo "2) 使用 docker-compose (适合生产环境)"
echo "3) 查看使用说明"
echo "4) 退出"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🔧 使用 build_docker.sh 脚本..."
        echo ""
        echo "步骤1: 构建并运行容器 (后台模式)"
        ./build_docker.sh run-bg
        
        echo ""
        echo "步骤2: 等待容器启动..."
        sleep 3
        
        echo ""
        echo "步骤3: 显示可用命令"
        echo "训练模型: ./build_docker.sh train"
        echo "进入容器: ./build_docker.sh shell"
        echo "启动TensorBoard: ./build_docker.sh tensorboard"
        echo "查看日志: ./build_docker.sh logs"
        echo "停止容器: ./build_docker.sh stop"
        
        echo ""
        read -p "是否立即开始训练? (y/n): " start_train
        if [[ $start_train == "y" || $start_train == "Y" ]]; then
            echo "🚀 开始训练..."
            ./build_docker.sh train
        else
            echo "💡 提示: 随时可以运行 './build_docker.sh train' 开始训练"
        fi
        ;;
        
    2)
        echo ""
        echo "🐳 使用 docker-compose..."
        echo ""
        echo "步骤1: 构建并启动服务"
        docker-compose up -d --build
        
        echo ""
        echo "步骤2: 等待服务启动..."
        sleep 5
        
        echo ""
        echo "步骤3: 显示可用命令"
        echo "进入主容器: docker-compose exec qwen3-finetune /bin/bash"
        echo "开始训练: docker-compose exec qwen3-finetune ./run_train.sh"
        echo "查看日志: docker-compose logs -f qwen3-finetune"
        echo "停止服务: docker-compose down"
        
        echo ""
        echo "🌐 TensorBoard 已启动:"
        echo "- 主服务: http://localhost:6006"
        echo "- 独立服务: http://localhost:6007"
        
        echo ""
        read -p "是否立即开始训练? (y/n): " start_train
        if [[ $start_train == "y" || $start_train == "Y" ]]; then
            echo "🚀 开始训练..."
            docker-compose exec qwen3-finetune ./run_train.sh
        else
            echo "💡 提示: 运行 'docker-compose exec qwen3-finetune ./run_train.sh' 开始训练"
        fi
        ;;
        
    3)
        echo ""
        echo "📖 使用说明"
        echo "============"
        echo ""
        echo "🎯 项目目标:"
        echo "   微调 Qwen3-0.5B 模型，生成小红书风格的店铺推荐文案"
        echo ""
        echo "📁 重要文件:"
        echo "   - store_xhs_sft_samples.jsonl: 训练数据"
        echo "   - train_config.json: 训练配置"
        echo "   - output_qwen/: 模型输出目录"
        echo "   - logs/: 训练日志"
        echo ""
        echo "🔧 主要命令:"
        echo "   ./build_docker.sh run      # 交互式运行"
        echo "   ./build_docker.sh run-bg   # 后台运行"
        echo "   ./build_docker.sh train    # 开始训练"
        echo "   ./build_docker.sh shell    # 进入容器"
        echo ""
        echo "📊 监控训练:"
        echo "   ./build_docker.sh tensorboard  # 启动 TensorBoard"
        echo "   访问: http://localhost:6006"
        echo ""
        echo "🧪 测试模型:"
        echo "   python3 inference.py --model_path ./output_qwen --interactive"
        echo ""
        echo "详细文档请查看: README_FINETUNE.md"
        ;;
        
    4)
        echo "👋 再见!"
        exit 0
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "✨ 设置完成! 祝训练顺利!"
echo "==================================" 