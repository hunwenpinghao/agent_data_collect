#!/bin/bash

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 安装依赖
echo "安装依赖包..."
echo "检查是否存在版本兼容性问题..."

# 首先尝试稳定版本
if [ -f "requirements_stable.txt" ]; then
    echo "使用稳定版本依赖..."
    pip install -r requirements_stable.txt
else
    echo "使用默认版本依赖..."
    pip install -r requirements.txt
fi

# 检查安装是否成功
echo "验证安装..."
python3 -c "import torch; import transformers; import modelscope; print('✅ 依赖安装成功')" || {
    echo "❌ 依赖安装失败，尝试重新安装兼容版本..."
    pip install torch==2.1.0 transformers==4.36.2 modelscope==1.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 创建输出目录
mkdir -p output_qwen
mkdir -p logs

# 运行训练
echo "开始训练..."
python3 fine_tune_qwen.py \
    --config_file train_config_multi_files.json \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "训练完成！"
echo "模型保存在: ./output_qwen"
echo "日志保存在: ./logs"

# 运行测试
echo "运行推理测试..."
python3 inference.py --model_path ./output_qwen 