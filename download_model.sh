#!/bin/bash
# 手动下载Qwen2.5-0.5B-Instruct模型

set -e

MODEL_DIR="models/Qwen2.5-0.5B-Instruct"
MIRROR_BASE="https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"

echo "🚀 开始下载Qwen2.5-0.5B-Instruct模型..."

# 创建模型目录
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "📁 创建目录: $MODEL_DIR"

# 定义需要下载的文件列表
files=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "model.safetensors"
    "README.md"
)

# 下载文件
for file in "${files[@]}"; do
    echo "⬇️  下载: $file"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "$MIRROR_BASE/$file" -O "$file"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "$MIRROR_BASE/$file" -o "$file"
    else
        echo "❌ 错误: 需要安装 wget 或 curl"
        exit 1
    fi
    
    # 检查文件是否下载成功
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo "✅ $file 下载成功"
    else
        echo "❌ $file 下载失败"
        exit 1
    fi
done

echo ""
echo "🎉 模型下载完成！"
echo "📂 模型位置: $(pwd)"
echo ""
echo "📋 下载的文件："
ls -lh

echo ""
echo "🔧 接下来请修改配置文件中的 model_name_or_path:"
echo "   \"model_name_or_path\": \"$(pwd)\""
echo ""
echo "或者运行以下命令自动修改配置："
echo "   python update_config_for_local.py" 