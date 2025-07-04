#!/bin/bash

# TensorBoard启动脚本
# 用于查看模型训练进度

echo "🚀 TensorBoard 启动脚本"
echo "========================"

# 默认日志目录
DEFAULT_LOG_DIR="./output_qwen/tensorboard_logs"

# 获取日志目录参数
LOG_DIR=${1:-$DEFAULT_LOG_DIR}

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 日志目录不存在: $LOG_DIR"
    echo ""
    echo "📋 可用的日志目录:"
    find . -name "tensorboard_logs" -type d 2>/dev/null | head -10
    echo ""
    echo "💡 使用方法: ./start_tensorboard.sh [日志目录路径]"
    echo "   示例: ./start_tensorboard.sh ./output_qwen_simple/tensorboard_logs"
    exit 1
fi

echo "📊 启动TensorBoard..."
echo "📁 日志目录: $LOG_DIR"
echo "🌐 浏览器访问: http://localhost:6006"
echo ""
echo "💡 提示:"
echo "   - 按 Ctrl+C 停止TensorBoard"
echo "   - 如果端口被占用，会自动选择其他端口"
echo ""

# 启动TensorBoard
tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port=6006 