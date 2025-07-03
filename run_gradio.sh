#!/bin/bash

# 微调模型Gradio Web界面启动脚本
# 支持LoRA、完整微调和QLoRA模型

set -e

echo "🚀 启动微调模型Gradio Web界面"
echo "=================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

# 检查必要的Python包
echo "📦 检查Python依赖..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch未安装，请运行: pip install -r requirements.txt"
    exit 1
}

python3 -c "import transformers; print(f'Transformers版本: {transformers.__version__}')" 2>/dev/null || {
    echo "❌ Transformers未安装，请运行: pip install -r requirements.txt"
    exit 1
}

python3 -c "import gradio; print(f'Gradio版本: {gradio.__version__}')" 2>/dev/null || {
    echo "❌ Gradio未安装，请运行: pip install -r requirements.txt"
    exit 1
}

python3 -c "import peft; print(f'PEFT版本: {peft.__version__}')" 2>/dev/null || {
    echo "❌ PEFT未安装，请运行: pip install -r requirements.txt"
    exit 1
}

echo "✅ 所有依赖检查完成"

# 设置环境变量
echo "🔧 设置环境变量..."
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 设置HuggingFace镜像（如果需要）
if [ "${USE_HF_MIRROR:-false}" = "true" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "🌐 使用HuggingFace镜像: $HF_ENDPOINT"
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 检查GPU状态..."
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
        echo "GPU: $line"
    done
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU运行"
fi

# 创建日志目录
mkdir -p logs

# 启动参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-7860}
SHARE=${SHARE:-false}

echo ""
echo "🌟 启动参数:"
echo "   主机: $HOST"
echo "   端口: $PORT"
echo "   共享: $SHARE"
echo ""

# 显示使用说明
echo "📋 使用说明:"
echo "   1. 在浏览器中打开: http://localhost:$PORT"
echo "   2. 选择模型类型: LoRA/完整微调/QLoRA"
echo "   3. 配置模型路径和参数"
echo "   4. 点击'加载模型'按钮"
echo "   5. 开始对话测试"
echo ""

# 检查gradio_inference.py是否存在
if [ ! -f "gradio_inference.py" ]; then
    echo "❌ gradio_inference.py 文件不存在"
    echo "请确保在正确的目录中运行此脚本"
    exit 1
fi

# 启动Gradio应用
echo "🚀 启动Gradio Web界面..."
echo "按 Ctrl+C 停止服务"
echo "=================================="

# 使用nohup在后台运行（可选）
if [ "${BACKGROUND:-false}" = "true" ]; then
    echo "🔄 在后台启动服务..."
    nohup python3 gradio_inference.py > logs/gradio_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "📝 日志文件: logs/gradio_$(date +%Y%m%d_%H%M%S).log"
    echo "✅ 服务已在后台启动"
    echo "💡 查看日志: tail -f logs/gradio_*.log"
else
    # 前台运行
    python3 gradio_inference.py
fi 