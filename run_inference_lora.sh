#!/bin/bash
# LoRA模型推理脚本 - 支持多种推理模式

set -e

# 默认配置
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL=""
QUANTIZATION=""
MODE="chat"
PROMPT=""
CUSTOM_BASE=""

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model DIR        LoRA模型目录 (默认: 自动检测最新的输出目录)"
    echo "  -b, --base MODEL       基础模型路径 (默认: Qwen/Qwen2.5-0.5B-Instruct)"
    echo "  -q, --quantization Q   量化模式: 4bit, 8bit, none (默认: none)"
    echo "  -t, --mode MODE        推理模式: chat, single, test (默认: chat)"
    echo "  -p, --prompt TEXT      单次推理的提示文本 (当mode=single时使用)"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "推理模式说明:"
    echo "  chat         交互式对话模式"
    echo "  single       单次推理模式 (需要配合 -p 参数)"
    echo "  test         批量测试模式 (使用内置测试样例)"
    echo ""
    echo "量化选项:"
    echo "  4bit         使用4位量化加载基础模型 (节省显存)"
    echo "  8bit         使用8位量化加载基础模型 (中等显存节省)"
    echo "  none         不使用量化 (默认，需要更多显存)"
    echo ""
    echo "示例:"
    echo "  $0                                    # 交互式对话，自动检测模型"
    echo "  $0 -m ./output_qwen_lora              # 使用指定LoRA模型"
    echo "  $0 -q 4bit -t chat                    # 4位量化 + 交互式对话"
    echo "  $0 -t single -p \"写一段咖啡店文案\"    # 单次推理"
    echo "  $0 -t test                            # 批量测试"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            LORA_MODEL="$2"
            shift 2
            ;;
        -b|--base)
            CUSTOM_BASE="$2"
            shift 2
            ;;
        -q|--quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        -t|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
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
if [[ ! -f "inference_lora.py" ]]; then
    echo "错误: 未找到 inference_lora.py 文件"
    exit 1
fi

# 自动检测LoRA模型路径
if [[ -z "$LORA_MODEL" ]]; then
    echo "🔍 自动检测LoRA模型..."
    
    # 查找所有output目录
    LORA_DIRS=($(find . -maxdepth 1 -type d -name "output_qwen*" | sort -V))
    
    if [[ ${#LORA_DIRS[@]} -eq 0 ]]; then
        echo "错误: 未找到LoRA模型目录 (output_qwen*)"
        echo "请先运行训练或使用 -m 参数指定模型路径"
        exit 1
    fi
    
    # 使用最新的目录
    LORA_MODEL="${LORA_DIRS[-1]}"
    echo "✅ 找到LoRA模型: $LORA_MODEL"
fi

# 验证LoRA模型目录
if [[ ! -d "$LORA_MODEL" ]]; then
    echo "错误: LoRA模型目录不存在: $LORA_MODEL"
    exit 1
fi

# 检查是否包含必要的LoRA文件
if [[ ! -f "$LORA_MODEL/adapter_config.json" ]]; then
    echo "错误: $LORA_MODEL 不是有效的LoRA模型目录"
    echo "缺少文件: adapter_config.json"
    exit 1
fi

# 设置基础模型路径
if [[ -n "$CUSTOM_BASE" ]]; then
    BASE_MODEL="$CUSTOM_BASE"
fi

# 检查基础模型是否存在（如果是本地路径）
if [[ -d "$BASE_MODEL" && ! -f "$BASE_MODEL/config.json" ]]; then
    echo "错误: 基础模型目录无效: $BASE_MODEL"
    echo "缺少文件: config.json"
    exit 1
fi

# 验证量化参数
case $QUANTIZATION in
    "4bit"|"8bit"|""|"none")
        # 有效的量化选项
        ;;
    *)
        echo "错误: 无效的量化选项: $QUANTIZATION"
        echo "支持的选项: 4bit, 8bit, none"
        exit 1
        ;;
esac

# 验证模式参数
case $MODE in
    "chat"|"single"|"test")
        # 有效的模式
        ;;
    *)
        echo "错误: 无效的推理模式: $MODE"
        echo "支持的模式: chat, single, test"
        exit 1
        ;;
esac

# 单次推理模式需要提示文本
if [[ "$MODE" == "single" && -z "$PROMPT" ]]; then
    echo "错误: single模式需要提供提示文本 (-p 参数)"
    exit 1
fi

# 检查GPU
echo "🔍 检查GPU信息..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU推理"
fi

# 检查依赖
echo "🔍 检查Python依赖..."
python -c "import torch, transformers, peft; print('✅ 依赖检查通过')" || {
    echo "错误: 缺少必要的Python依赖，请运行:"
    echo "pip install -r requirements_stable.txt"
    exit 1
}

# 显示配置信息
echo ""
echo "==================== 推理配置 ===================="
echo "基础模型: $BASE_MODEL"
echo "LoRA模型: $LORA_MODEL"
echo "量化模式: ${QUANTIZATION:-none}"
echo "推理模式: $MODE"
if [[ "$MODE" == "single" ]]; then
    echo "提示文本: $PROMPT"
fi
echo "=================================================="
echo ""

# 构建推理命令
INFERENCE_CMD="python inference_lora.py --base_model \"$BASE_MODEL\" --lora_model \"$LORA_MODEL\""

# 添加量化参数
case $QUANTIZATION in
    "4bit")
        INFERENCE_CMD="$INFERENCE_CMD --load_in_4bit"
        ;;
    "8bit")
        INFERENCE_CMD="$INFERENCE_CMD --load_in_8bit"
        ;;
esac

# 添加模式参数
case $MODE in
    "chat")
        INFERENCE_CMD="$INFERENCE_CMD --chat"
        ;;
    "single")
        INFERENCE_CMD="$INFERENCE_CMD --prompt \"$PROMPT\""
        ;;
    "test")
        # test模式不需要额外参数，会运行内置测试
        ;;
esac

# 显示即将执行的命令
echo "🚀 执行推理命令:"
echo "$INFERENCE_CMD"
echo ""

# 执行推理
eval "$INFERENCE_CMD"

echo ""
echo "推理完成！" 