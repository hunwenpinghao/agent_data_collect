#!/bin/bash

# 魔搭社区模型上传依赖安装脚本

echo "🚀 开始安装魔搭社区模型上传依赖..."

# 更新 pip
echo "📦 更新 pip..."
pip install --upgrade pip

# 安装 modelscope
echo "📦 安装 modelscope..."
pip install modelscope

# 安装 git python
echo "📦 安装 GitPython..."
pip install GitPython

# 安装其他可能需要的依赖
echo "📦 安装其他依赖..."
pip install requests tqdm

echo "✅ 依赖安装完成！"
echo ""
echo "🔧 使用方法:"
echo "1. 获取 ModelScope 访问令牌: https://www.modelscope.cn/my/myaccesstoken"
echo "2. 运行上传脚本:"
echo "   python upload_to_modelscope.py --token YOUR_TOKEN"
echo ""
echo "或者设置环境变量:"
echo "   export MODELSCOPE_TOKEN=YOUR_TOKEN"
echo "   python upload_to_modelscope.py"
echo ""
echo "🎯 更多选项:"
echo "   python upload_to_modelscope.py --help" 