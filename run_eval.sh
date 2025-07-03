#!/bin/bash

# =============================================================================
# Qwen 模型评估脚本包装器
# 用法: ./run_eval.sh [选项]
# 实际评估由 eval_model.py 执行
# =============================================================================

set -e  # 遇到错误时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Qwen 模型评估脚本包装器

用法: $0 [选项]

选项:
    -m, --model-path PATH       微调后的模型路径 (必需)
    -d, --data-path PATH        评估数据路径 (必需)
    -o, --output-dir PATH       输出目录 (默认: ./eval_results)
    -b, --batch-size NUM        批处理大小 (默认: 4)
    --max-tokens NUM            最大生成token数 (默认: 512)
    --temperature FLOAT         生成温度 (默认: 0.7)
    --top-p FLOAT              Top-p采样 (默认: 0.9)
    --device DEVICE            设备: cpu/cuda/auto (默认: auto)
    --torch-dtype TYPE         数据类型: float16/float32/bfloat16 (默认: float16)
    --metrics METRICS          评估指标: bleu,rouge,exact_match (默认: bleu,rouge,exact_match)
    --save-predictions         保存预测结果
    --verbose                  详细输出
    -h, --help                 显示此帮助信息

示例:
    # 基本评估
    $0 -m ./output/checkpoint-1000 -d ./data/test.jsonl
    
    # 完整评估
    $0 -m ./output/checkpoint-best -d ./data/test.jsonl \\
       --metrics bleu,rouge,exact_match \\
       --save-predictions --verbose
    
    # 使用大批处理提高速度
    $0 -m ./output/checkpoint-best -d ./data/test.jsonl -b 8
    
    # 使用CPU评估
    $0 -m ./output/checkpoint-best -d ./data/test.jsonl --device cpu

注意: 
    - 此脚本是 eval_model.py 的包装器
    - 确保安装了必要的依赖: torch, transformers, tqdm
    - 可选依赖用于更准确的指标: nltk, rouge

EOF
}

# 检查eval_model.py是否存在
check_eval_script() {
    if [[ ! -f "eval_model.py" ]]; then
        log_error "评估脚本 eval_model.py 不存在！"
        log_error "请确保 eval_model.py 在当前目录中"
        exit 1
    fi
}

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    # 检查基础依赖
    python -c "
import sys
required_packages = ['torch', 'transformers']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print('❌ 缺少必需依赖: ' + ', '.join(missing_packages))
    print('请运行: pip install torch transformers tqdm')
    sys.exit(1)
" 2>/dev/null

    if [[ $? -ne 0 ]]; then
        log_error "缺少必需的Python依赖"
        log_error "请运行: pip install torch transformers tqdm"
        exit 1
    fi
}

# 设置环境变量
setup_environment() {
    # 设置缓存目录
    export HF_HOME="./cache"
    export TRANSFORMERS_CACHE="./cache"
    
    # 创建缓存目录
    mkdir -p ./cache
}

# 创建测试数据（如果不存在）
create_test_data() {
    local data_path="$1"
    local data_dir=$(dirname "$data_path")
    
    # 如果数据文件不存在且是默认路径，创建示例数据
    if [[ ! -f "$data_path" && "$data_path" == "./data/test.jsonl" ]]; then
        log_warning "测试数据文件不存在: $data_path"
        log_info "创建示例测试数据..."
        
        mkdir -p "$data_dir"
        cat > "$data_path" << 'EOF'
{"instruction": "请解释什么是人工智能", "input": "", "output": "人工智能(AI)是指由机器展现出的智能，它能够模拟人类的思维过程和认知能力。"}
{"instruction": "翻译下面的英文句子", "input": "The weather is nice today.", "output": "今天天气很好。"}
{"instruction": "计算以下数学表达式", "input": "5 + 3 * 2", "output": "5 + 3 * 2 = 5 + 6 = 11"}
{"question": "Python中如何定义一个函数？", "answer": "在Python中，使用def关键字来定义函数，语法是：def function_name(parameters):"}
{"prompt": "请写一首关于春天的短诗", "response": "春风轻拂柳丝长，\n花开满园香气扬。\n燕子归来筑新巢，\n万物复苏展新装。"}
EOF
        
        log_success "创建了示例测试数据: $data_path"
    fi
}

# 主函数
main() {
    echo ""
    log_info "==================== Qwen 模型评估脚本 ===================="
    echo ""
    
    # 检查基础环境
    check_eval_script
    check_python
    setup_environment
    
    # 解析参数并构建Python命令
    python_cmd="python eval_model.py"
    
    # 处理参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--model-path)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --model-path \"$2\""
                shift 2
                ;;
            -d|--data-path)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                DATA_PATH="$2"
                python_cmd+=" --data-path \"$2\""
                shift 2
                ;;
            -o|--output-dir)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --output-dir \"$2\""
                shift 2
                ;;
            -b|--batch-size)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --batch-size $2"
                shift 2
                ;;
            --max-tokens)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --max-tokens $2"
                shift 2
                ;;
            --temperature)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --temperature $2"
                shift 2
                ;;
            --top-p)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --top-p $2"
                shift 2
                ;;
            --device)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --device $2"
                shift 2
                ;;
            --torch-dtype)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --torch-dtype $2"
                shift 2
                ;;
            --metrics)
                if [[ -z "$2" ]]; then
                    log_error "选项 $1 需要参数"
                    exit 1
                fi
                python_cmd+=" --metrics \"$2\""
                shift 2
                ;;
            --save-predictions)
                python_cmd+=" --save-predictions"
                shift
                ;;
            --verbose)
                python_cmd+=" --verbose"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [[ ! "$python_cmd" =~ "--model-path" ]]; then
        log_error "缺少必需参数: --model-path"
        echo ""
        show_help
        exit 1
    fi
    
    if [[ ! "$python_cmd" =~ "--data-path" ]]; then
        log_warning "未指定数据路径，使用默认路径: ./data/test.jsonl"
        DATA_PATH="./data/test.jsonl"
        python_cmd+=" --data-path \"$DATA_PATH\""
    fi
    
    # 创建测试数据（如果需要）
    if [[ -n "$DATA_PATH" ]]; then
        create_test_data "$DATA_PATH"
    fi
    
    # 显示即将执行的命令
    log_info "执行评估命令:"
    echo "  $python_cmd"
    echo ""
    
    # 执行评估
    log_info "🚀 开始模型评估..."
    eval $python_cmd
    
    if [[ $? -eq 0 ]]; then
        echo ""
        log_success "🎉 评估完成！"
    else
        echo ""
        log_error "❌ 评估失败"
        exit 1
    fi
}

# 错误处理
trap 'log_error "脚本执行失败，行号: $LINENO"' ERR

# 检查是否直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 