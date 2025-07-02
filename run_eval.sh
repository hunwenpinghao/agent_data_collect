#!/bin/bash

# =============================================================================
# Qwen 模型评估脚本
# 用法: ./run_eval.sh [选项]
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
Qwen 模型评估脚本

用法: $0 [选项]

选项:
    -m, --model-path PATH       微调后的模型路径 (默认: ./output/checkpoint-best)
    -d, --data-path PATH        评估数据路径 (默认: ./data/test.jsonl)
    -o, --output-dir PATH       输出目录 (默认: ./eval_results)
    -b, --batch-size NUM        批处理大小 (默认: 4)
    -t, --max-tokens NUM        最大生成token数 (默认: 512)
    --base-model PATH           基础模型路径 (用于对比)
    --temperature FLOAT         生成温度 (默认: 0.7)
    --top-p FLOAT              Top-p采样 (默认: 0.9)
    --device DEVICE            设备: cpu|cuda|auto (默认: auto)
    --metrics METRICS          评估指标: bleu,rouge,exact_match (默认: bleu,rouge)
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
    
    # 对比评估
    $0 -m ./output/checkpoint-best --base-model Qwen/Qwen2.5-7B-Instruct \\
       -d ./data/test.jsonl

EOF
}

# 默认配置
MODEL_PATH="./output/checkpoint-best"
DATA_PATH="./data/test.jsonl"
OUTPUT_DIR="./eval_results"
BATCH_SIZE=4
MAX_TOKENS=512
BASE_MODEL=""
TEMPERATURE=0.7
TOP_P=0.9
DEVICE="auto"
METRICS="bleu,rouge"
SAVE_PREDICTIONS=false
VERBOSE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --save-predictions)
            SAVE_PREDICTIONS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证参数
validate_args() {
    log_info "验证参数..."
    
    # 检查模型路径
    if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
        log_error "模型路径不存在: $MODEL_PATH"
        exit 1
    fi
    
    # 检查数据路径
    if [[ ! -f "$DATA_PATH" ]]; then
        log_error "数据文件不存在: $DATA_PATH"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    log_success "参数验证通过"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    
    # 检查必要的Python包
    python -c "
import sys
required_packages = ['torch', 'transformers', 'datasets', 'numpy', 'pandas']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'缺少Python包: {', '.join(missing_packages)}')
    sys.exit(1)
" 2>/dev/null
    
    if [[ $? -ne 0 ]]; then
        log_warning "部分依赖包可能缺失，但继续执行..."
    fi
    
    log_success "依赖检查完成"
}

# 设置环境变量
setup_environment() {
    log_info "设置环境..."
    
    # 设置CUDA可见性
    if [[ "$DEVICE" == "auto" ]]; then
        export CUDA_VISIBLE_DEVICES="0"
    elif [[ "$DEVICE" == "cpu" ]]; then
        export CUDA_VISIBLE_DEVICES=""
    fi
    
    # 设置缓存目录
    export HF_HOME="./cache"
    export TRANSFORMERS_CACHE="./cache"
    
    # 设置日志级别
    if [[ "$VERBOSE" == "true" ]]; then
        export TRANSFORMERS_VERBOSITY="info"
    else
        export TRANSFORMERS_VERBOSITY="warning"
    fi
}

# 创建评估脚本
create_eval_script() {
    log_info "创建评估脚本..."
    
    cat > "${OUTPUT_DIR}/eval_model.py" << 'EOF'
#!/usr/bin/env python3
"""
模型评估脚本
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """加载模型和tokenizer"""
    logger.info(f"加载模型: {model_path}")
    
    try:
        # 应用之前的修复
        sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
        from fine_tune_qwen import apply_transformers_patch
        apply_transformers_patch()
        logger.info("已应用transformers修复补丁")
    except Exception as e:
        logger.warning(f"无法应用transformers修复补丁: {e}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to("cpu")
        
        logger.info("模型加载完成")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def load_eval_data(data_path: str) -> List[Dict]:
    """加载评估数据"""
    logger.info(f"加载评估数据: {data_path}")
    
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    logger.info(f"成功加载 {len(data)} 条评估数据")
    return data

def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """生成回复"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        
        if model.device.type != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 解码输出，去掉输入部分
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        logger.warning(f"生成失败: {e}")
        return ""

def evaluate_generation(
    model, 
    tokenizer, 
    eval_data: List[Dict],
    batch_size: int = 4,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    save_predictions: bool = False,
    output_dir: str = "./eval_results"
) -> Dict[str, Any]:
    """评估生成任务"""
    logger.info("开始生成评估...")
    
    predictions = []
    references = []
    inputs_list = []
    
    for item in tqdm(eval_data, desc="生成评估"):
        # 构建提示
        if "instruction" in item and "input" in item:
            if item["input"].strip():
                prompt = f"指令: {item['instruction']}\n输入: {item['input']}\n回答: "
            else:
                prompt = f"指令: {item['instruction']}\n回答: "
        elif "question" in item:
            prompt = f"问题: {item['question']}\n回答: "
        elif "prompt" in item:
            prompt = item["prompt"]
        else:
            prompt = str(item.get("input", ""))
        
        inputs_list.append(prompt)
        
        # 生成回复
        prediction = generate_response(
            model, tokenizer, prompt, 
            max_tokens, temperature, top_p
        )
        predictions.append(prediction)
        
        # 获取参考答案
        reference = item.get("output", item.get("answer", item.get("response", "")))
        references.append(str(reference))
    
    # 保存预测结果
    if save_predictions:
        results_file = os.path.join(output_dir, "predictions.jsonl")
        with open(results_file, 'w', encoding='utf-8') as f:
            for i, (inp, pred, ref) in enumerate(zip(inputs_list, predictions, references)):
                result = {
                    "id": i,
                    "input": inp,
                    "prediction": pred,
                    "reference": ref,
                    "original_data": eval_data[i]
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"预测结果保存到: {results_file}")
    
    return {
        "predictions": predictions,
        "references": references,
        "inputs": inputs_list,
        "num_samples": len(predictions)
    }

def calculate_metrics(predictions: List[str], references: List[str], metrics: List[str]) -> Dict[str, float]:
    """计算评估指标"""
    logger.info("计算评估指标...")
    
    results = {}
    
    # 过滤有效的预测和参考
    valid_pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) 
                   if p.strip() and r.strip()]
    
    if not valid_pairs:
        logger.warning("没有有效的预测-参考对")
        return {"error": "no_valid_pairs"}
    
    valid_predictions, valid_references = zip(*valid_pairs)
    
    # BLEU分数
    if "bleu" in metrics:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True)
            
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for pred, ref in zip(valid_predictions, valid_references):
                pred_tokens = pred.lower().split()
                ref_tokens = [ref.lower().split()]
                
                if len(pred_tokens) > 0 and len(ref_tokens[0]) > 0:
                    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(score)
            
            results["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            
        except Exception as e:
            logger.warning(f"BLEU计算失败: {e}")
            results["bleu"] = 0.0
    
    # ROUGE分数
    if "rouge" in metrics:
        try:
            from rouge import Rouge
            rouge = Rouge()
            
            scores = rouge.get_scores(list(valid_predictions), list(valid_references), avg=True)
            results["rouge-1"] = scores["rouge-1"]["f"]
            results["rouge-2"] = scores["rouge-2"]["f"]
            results["rouge-l"] = scores["rouge-l"]["f"]
            
        except Exception as e:
            logger.warning(f"ROUGE计算失败: {e}")
            try:
                # 简单的ROUGE-L近似
                def lcs_length(s1, s2):
                    m, n = len(s1), len(s2)
                    dp = [[0] * (n + 1) for _ in range(m + 1)]
                    for i in range(1, m + 1):
                        for j in range(1, n + 1):
                            if s1[i-1] == s2[j-1]:
                                dp[i][j] = dp[i-1][j-1] + 1
                            else:
                                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    return dp[m][n]
                
                rouge_l_scores = []
                for pred, ref in zip(valid_predictions, valid_references):
                    pred_tokens = pred.lower().split()
                    ref_tokens = ref.lower().split()
                    lcs_len = lcs_length(pred_tokens, ref_tokens)
                    if len(ref_tokens) > 0:
                        rouge_l = lcs_len / len(ref_tokens)
                        rouge_l_scores.append(rouge_l)
                
                results["rouge-l"] = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
                results["rouge-1"] = 0.0
                results["rouge-2"] = 0.0
                
            except Exception as e2:
                logger.warning(f"简单ROUGE计算也失败: {e2}")
                results["rouge-1"] = 0.0
                results["rouge-2"] = 0.0
                results["rouge-l"] = 0.0
    
    # 精确匹配
    if "exact_match" in metrics:
        exact_matches = sum(1 for p, r in zip(valid_predictions, valid_references) 
                           if p.lower().strip() == r.lower().strip())
        results["exact_match"] = exact_matches / len(valid_predictions) if valid_predictions else 0.0
    
    # 统计信息
    results["num_valid_pairs"] = len(valid_pairs)
    results["avg_prediction_length"] = sum(len(p.split()) for p in valid_predictions) / len(valid_predictions)
    results["avg_reference_length"] = sum(len(r.split()) for r in valid_references) / len(valid_references)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--data-path", required=True, help="评估数据路径")
    parser.add_argument("--output-dir", default="./eval_results", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=4, help="批处理大小")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p采样")
    parser.add_argument("--device", default="auto", help="设备")
    parser.add_argument("--metrics", default="bleu,rouge,exact_match", help="评估指标")
    parser.add_argument("--save-predictions", action="store_true", help="保存预测结果")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 加载模型和数据
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
        eval_data = load_eval_data(args.data_path)
        
        # 评估
        eval_results = evaluate_generation(
            model, tokenizer, eval_data,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            save_predictions=args.save_predictions,
            output_dir=args.output_dir
        )
        
        # 计算指标
        metrics_list = [m.strip() for m in args.metrics.split(',')]
        metrics = calculate_metrics(
            eval_results["predictions"],
            eval_results["references"],
            metrics_list
        )
        
        # 保存结果
        results = {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "num_samples": eval_results["num_samples"],
            "metrics": metrics,
            "config": {
                "batch_size": args.batch_size,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "device": args.device,
                "metrics": args.metrics
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印结果
        print("\n" + "="*60)
        print("模型评估结果")
        print("="*60)
        print(f"模型路径: {args.model_path}")
        print(f"数据路径: {args.data_path}")
        print(f"总样本数: {eval_results['num_samples']}")
        if "num_valid_pairs" in metrics:
            print(f"有效样本数: {metrics['num_valid_pairs']}")
        print(f"时间戳: {results['timestamp']}")
        print("\n评估指标:")
        for metric, score in metrics.items():
            if not metric.startswith("num_") and not metric.startswith("avg_"):
                print(f"  {metric.upper()}: {score:.4f}")
        
        print(f"\n统计信息:")
        if "avg_prediction_length" in metrics:
            print(f"  平均预测长度: {metrics['avg_prediction_length']:.1f} tokens")
        if "avg_reference_length" in metrics:
            print(f"  平均参考长度: {metrics['avg_reference_length']:.1f} tokens")
        
        print(f"\n结果文件: {results_file}")
        if args.save_predictions:
            print(f"预测文件: {os.path.join(args.output_dir, 'predictions.jsonl')}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"评估过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
EOF
    
    log_success "评估脚本创建完成"
}

# 运行评估
run_evaluation() {
    log_info "开始评估..."
    
    # 构建Python命令
    python_cmd="python ${OUTPUT_DIR}/eval_model.py"
    python_cmd+=" --model-path \"$MODEL_PATH\""
    python_cmd+=" --data-path \"$DATA_PATH\""
    python_cmd+=" --output-dir \"$OUTPUT_DIR\""
    python_cmd+=" --batch-size $BATCH_SIZE"
    python_cmd+=" --max-tokens $MAX_TOKENS"
    python_cmd+=" --temperature $TEMPERATURE"
    python_cmd+=" --top-p $TOP_P"
    python_cmd+=" --device $DEVICE"
    python_cmd+=" --metrics \"$METRICS\""
    
    if [[ "$SAVE_PREDICTIONS" == "true" ]]; then
        python_cmd+=" --save-predictions"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "执行命令: $python_cmd"
    fi
    
    # 执行评估
    eval $python_cmd
    
    if [[ $? -eq 0 ]]; then
        log_success "评估完成！"
        log_info "结果保存在: $OUTPUT_DIR"
        
        # 显示简要结果
        if [[ -f "$OUTPUT_DIR/evaluation_results.json" ]]; then
            echo ""
            echo "=== 快速查看结果 ==="
            python -c "
import json
try:
    with open('$OUTPUT_DIR/evaluation_results.json', 'r') as f:
        results = json.load(f)
    print(f\"✅ 评估成功完成\")
    print(f\"📊 样本数: {results['num_samples']}\")
    print(f\"🎯 主要指标:\")
    for metric, score in results['metrics'].items():
        if metric in ['bleu', 'rouge-1', 'rouge-l', 'exact_match']:
            print(f\"   {metric.upper()}: {score:.4f}\")
    print(f\"📁 详细结果: $OUTPUT_DIR/evaluation_results.json\")
except Exception as e:
    print(f\"❌ 无法读取结果文件: {e}\")
"
        fi
    else
        log_error "评估失败"
        exit 1
    fi
}

# 主函数
main() {
    echo ""
    log_info "==================== Qwen 模型评估脚本 ===================="
    echo ""
    
    # 显示配置
    log_info "评估配置:"
    echo "  📁 模型路径: $MODEL_PATH"
    echo "  📄 数据路径: $DATA_PATH"
    echo "  📂 输出目录: $OUTPUT_DIR"
    echo "  🔢 批处理大小: $BATCH_SIZE"
    echo "  🎯 最大token数: $MAX_TOKENS"
    echo "  🌡️  生成温度: $TEMPERATURE"
    echo "  📊 评估指标: $METRICS"
    echo "  💻 计算设备: $DEVICE"
    if [[ "$SAVE_PREDICTIONS" == "true" ]]; then
        echo "  💾 保存预测: 是"
    fi
    echo ""
    
    # 执行步骤
    validate_args
    check_dependencies
    setup_environment
    create_eval_script
    run_evaluation
    
    echo ""
    log_success "🎉 评估流程全部完成！"
    echo ""
}

# 错误处理
trap 'log_error "脚本执行失败，行号: $LINENO"' ERR

# 检查是否直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 执行主函数
    main "$@"
fi 