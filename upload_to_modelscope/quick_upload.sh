#!/bin/bash

# 魔搭社区模型快速上传脚本

set -e

echo "🚀 魔搭社区模型上传工具"
echo "================================"

# 检查是否已安装依赖
echo "📋 检查依赖..."
if ! python3 -c "import modelscope" 2>/dev/null; then
    echo "⚠️  ModelScope 未安装，正在自动安装..."
    ./setup_upload_dependencies.sh
fi

# 检查访问令牌
if [ -z "$MODELSCOPE_TOKEN" ]; then
    echo ""
    echo "🔑 请输入您的 ModelScope 访问令牌:"
    echo "   获取地址: https://www.modelscope.cn/my/myaccesstoken"
    read -p "Token: " TOKEN
    
    if [ -z "$TOKEN" ]; then
        echo "❌ 访问令牌不能为空！"
        exit 1
    fi
else
    TOKEN="$MODELSCOPE_TOKEN"
    echo "✅ 使用环境变量中的访问令牌"
fi

# 设置默认参数
MODEL_DIR="../deploy_ark/output_qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID="hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct"

echo ""
echo "📁 模型目录: $MODEL_DIR"
echo "🏷️  模型ID: $MODEL_ID"

# 询问是否需要修改参数
read -p "是否使用默认设置？(y/n) [y]: " USE_DEFAULT
USE_DEFAULT=${USE_DEFAULT:-y}

if [ "$USE_DEFAULT" != "y" ] && [ "$USE_DEFAULT" != "Y" ]; then
    read -p "模型目录路径 [$MODEL_DIR]: " CUSTOM_DIR
    MODEL_DIR=${CUSTOM_DIR:-$MODEL_DIR}
    
    read -p "模型ID [$MODEL_ID]: " CUSTOM_ID
    MODEL_ID=${CUSTOM_ID:-$MODEL_ID}
fi

# 询问是否创建 README
read -p "是否自动创建模型说明文档 README.md？(y/n) [y]: " CREATE_README
CREATE_README=${CREATE_README:-y}

# 构建命令
CMD="python3 upload_to_modelscope.py --model_dir \"$MODEL_DIR\" --model_id \"$MODEL_ID\" --token \"$TOKEN\""

if [ "$CREATE_README" = "y" ] || [ "$CREATE_README" = "Y" ]; then
    CMD="$CMD --create_readme"
fi

echo ""
echo "🚀 开始上传..."
echo "执行命令: $CMD"
echo ""

# 执行上传
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 上传完成！"
    echo "🌐 模型地址: https://www.modelscope.cn/models/$MODEL_ID"
    echo ""
    echo "接下来您可以："
    echo "1. 访问模型页面验证上传结果"
    echo "2. 测试模型推理功能"
    echo "3. 分享给其他用户使用"
else
    echo ""
    echo "❌ 上传失败！"
    echo "📋 请查看日志: upload_modelscope.log"
    echo "💡 常见解决方案:"
    echo "   1. 检查访问令牌是否正确"
    echo "   2. 确认网络连接正常"
    echo "   3. 验证模型文件完整性"
    exit 1
fi 