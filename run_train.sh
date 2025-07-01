#!/bin/bash

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

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