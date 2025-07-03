#!/bin/bash
# 训练脚本 - 支持多种微调方式

set -e

# 默认配置
CONFIG_TYPE="lora"
CUSTOM_CONFIG=""

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -t, --type TYPE        选择微调类型: full, lora, qlora, qlora_8bit, deepspeed (默认: lora)"
    echo "  -c, --config FILE      使用自定义配置文件"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "微调类型说明:"
    echo "  full         全参数微调 (显存需求高，效果最好)"
    echo "  lora         LoRA微调 (平衡选择，推荐)"
    echo "  qlora        QLoRA 4位量化微调 (显存需求最低)"
    echo "  qlora_8bit   QLoRA 8位量化微调 (中等显存需求)"
    echo "  deepspeed    DeepSpeed分布式训练 (多GPU高效训练)"
    echo ""
    echo "示例:"
    echo "  $0                              # 使用默认LoRA配置"
    echo "  $0 -t full                      # 使用全参数微调"
    echo "  $0 -t qlora                     # 使用QLoRA 4位量化"
    echo "  $0 -t deepspeed                 # 使用DeepSpeed分布式训练"
    echo "  $0 -c configs/my_config.json    # 使用自定义配置"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        -c|--config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查必要文件
if [[ ! -f "fine_tune_qwen.py" ]]; then
    echo "错误: 未找到 fine_tune_qwen.py 文件"
    exit 1
fi

# 确定配置文件
if [[ -n "$CUSTOM_CONFIG" ]]; then
    CONFIG_FILE="$CUSTOM_CONFIG"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "错误: 自定义配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    echo "使用自定义配置文件: $CONFIG_FILE"
else
    case $CONFIG_TYPE in
        full)
            CONFIG_FILE="configs/train_config_full.json"
            echo "使用全参数微调配置"
            ;;
        lora)
            CONFIG_FILE="configs/train_config_lora.json"
            echo "使用LoRA微调配置"
            ;;
        qlora)
            CONFIG_FILE="configs/train_config_qlora.json"
            echo "使用QLoRA 4位量化微调配置"
            ;;
        qlora_8bit)
            CONFIG_FILE="configs/train_config_qlora_8bit.json"
            echo "使用QLoRA 8位量化微调配置"
            ;;
        deepspeed)
            CONFIG_FILE="configs/train_config_deepspeed.json"
            echo "使用DeepSpeed分布式训练配置"
            ;;
        *)
            echo "错误: 不支持的微调类型: $CONFIG_TYPE"
            echo "支持的类型: full, lora, qlora, qlora_8bit, deepspeed"
            exit 1
            ;;
    esac
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "错误: 配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "警告: 未检测到NVIDIA GPU"
fi

# 检查依赖
echo "检查Python依赖..."
python -c "import torch, transformers, peft, bitsandbytes; print('✓ 依赖检查通过')" || {
    echo "错误: 缺少必要的Python依赖，请运行:"
    echo "pip install -r requirements_stable.txt"
    exit 1
}

# 显示配置信息
echo ""
echo "==================== 训练配置 ===================="
echo "配置文件: $CONFIG_FILE"
echo "微调类型: $CONFIG_TYPE"
python -c "
import json
with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
    config = json.load(f)
print(f\"模型: {config.get('model_name_or_path', 'N/A')}\")
print(f\"数据: {config.get('data_path', 'N/A')}\")
print(f\"输出目录: {config.get('output_dir', 'N/A')}\")
if config.get('use_deepspeed'):
    print(f\"DeepSpeed: 启用 (ZeRO Stage {config.get('deepspeed_stage', 'N/A')})\")
if config.get('use_lora'):
    print(f\"LoRA rank: {config.get('lora_r', 'N/A')}\")
elif config.get('use_qlora'):
    print(f\"QLoRA量化位数: {config.get('quantization_bit', 'N/A')}\")
    print(f\"LoRA rank: {config.get('lora_r', 'N/A')}\")
print(f\"学习率: {config.get('learning_rate', 'N/A')}\")
print(f\"训练轮数: {config.get('num_train_epochs', 'N/A')}\")
"
echo "=================================================="
echo ""

# 开始训练
echo "开始训练..."
echo "命令: python fine_tune_qwen.py --config_file $CONFIG_FILE"
echo ""

python fine_tune_qwen.py --config_file "$CONFIG_FILE"

echo ""
echo "训练完成！" 