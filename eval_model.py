#!/usr/bin/env python3
"""
Qwen 模型评估脚本
支持多种评估指标和灵活的配置选项
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

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

def apply_transformers_patch():
    """应用transformers库的兼容性修复"""
    try:
        # 尝试导入现有的修复函数
        from fine_tune_qwen import apply_transformers_patch as patch_func
        patch_func()
        logger.info("✅ 已应用transformers修复补丁")
        return True
    except ImportError:
        logger.warning("⚠️ 无法导入fine_tune_qwen模块，尝试内置修复...")
        try:
            import transformers.modeling_utils as modeling_utils
            
            # 保存原始方法
            original_post_init = modeling_utils.PreTrainedModel.post_init
            
            def patched_post_init(self):
                """修复后的post_init方法"""
                try:
                    # 检查并修复可能的None值
                    if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp is None:
                        self.config.pretraining_tp = 1
                    
                    # 修复张量并行样式错误
                    tensor_parallel_attrs = ['tensor_parallel_style', 'parallel_style']
                    supported_styles = ['tp', 'dp', 'pp', 'cp']
                    
                    for attr in tensor_parallel_attrs:
                        if hasattr(self.config, attr):
                            current_value = getattr(self.config, attr)
                            if current_value is not None and current_value not in supported_styles:
                                setattr(self.config, attr, 'tp')
                    
                    # 其他常见修复
                    fixes = {
                        'attention_dropout': 0.0,
                        'rope_scaling': None,
                        'attn_implementation': 'eager',
                        'use_sliding_window': False,
                        'sliding_window': 4096
                    }
                    
                    for key, default_val in fixes.items():
                        if hasattr(self.config, key) and getattr(self.config, key) is None:
                            setattr(self.config, key, default_val)
                    
                    # 调用原始方法
                    return original_post_init(self)
                    
                except Exception as e:
                    logger.warning(f"post_init修复过程中出现错误: {e}")
                    pass
            
            # 应用patch
            modeling_utils.PreTrainedModel.post_init = patched_post_init
            logger.info("✅ 已应用内置transformers修复补丁")
            return True
            
        except Exception as e:
            logger.warning(f"❌ 无法应用transformers补丁: {e}")
            return False
    except Exception as e:
        logger.warning(f"❌ transformers补丁应用失败: {e}")
        return False

def load_model_and_tokenizer(model_path: str, device: str = "auto", torch_dtype: str = "float16"):
    """加载模型和tokenizer"""
    logger.info(f"🔄 加载模型: {model_path}")
    
    # 应用修复补丁
    apply_transformers_patch()
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 设置数据类型
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_obj,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to("cpu")
        
        logger.info(f"✅ 模型加载完成 (设备: {model.device.type})")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise

def load_eval_data(data_path: str) -> List[Dict]:
    """加载评估数据"""
    logger.info(f"📄 加载评估数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"评估数据文件不存在: {data_path}")
    
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    logger.info(f"✅ 成功加载 {len(data)} 条评估数据")
    return data

def build_prompt(item: Dict) -> str:
    """根据数据格式构建提示"""
    # 格式1: instruction + input + output
    if "instruction" in item:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        if input_text.strip():
            return f"指令: {instruction}\n输入: {input_text}\n回答: "
        else:
            return f"指令: {instruction}\n回答: "
    
    # 格式2: question + answer
    elif "question" in item:
        return f"问题: {item['question']}\n回答: "
    
    # 格式3: prompt + response
    elif "prompt" in item:
        return item["prompt"]
    
    # 格式4: 通用输入
    else:
        return str(item.get("input", item.get("text", "")))

def extract_reference(item: Dict) -> str:
    """提取参考答案"""
    # 优先级: output > answer > response > target
    for key in ["output", "answer", "response", "target"]:
        if key in item and item[key]:
            return str(item[key])
    
    logger.warning(f"⚠️ 无法找到参考答案: {item}")
    return ""

def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> str:
    """生成模型回复"""
    try:
        # 编码输入
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        )
        
        # 移动到设备
        if model.device.type != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3
        )
        
        # 生成回复
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
        logger.warning(f"⚠️ 生成失败: {e}")
        return ""

def generate_batch_responses(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> List[str]:
    """批量生成模型回复"""
    try:
        # 批量编码输入
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        )
        
        # 移动到设备
        if model.device.type != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3
        )
        
        # 批量生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 解码输出，去掉输入部分
        responses = []
        input_lengths = inputs['input_ids'].shape[1]
        
        for i, output in enumerate(outputs):
            # 对于批量生成，每个样本的输入长度可能不同
            # 使用attention_mask找到实际的输入长度
            actual_input_length = inputs['attention_mask'][i].sum().item()
            generated_tokens = output[actual_input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
        
    except Exception as e:
        logger.warning(f"⚠️ 批量生成失败: {e}")
        # 回退到单个生成
        return [generate_response(model, tokenizer, prompt, max_tokens, temperature, top_p) 
                for prompt in prompts]

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
    """执行生成评估"""
    logger.info(f"🚀 开始生成评估 (batch_size={batch_size})...")
    
    predictions = []
    references = []
    inputs_list = []
    
    # 准备所有数据
    all_prompts = []
    all_references = []
    
    for item in eval_data:
        prompt = build_prompt(item)
        reference = extract_reference(item)
        all_prompts.append(prompt)
        all_references.append(reference)
    
    # 批量处理
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(all_prompts), batch_size), 
                  desc="批量生成", unit="batch", total=total_batches):
        batch_prompts = all_prompts[i:i + batch_size]
        batch_references = all_references[i:i + batch_size]
        
        # 批量生成
        if len(batch_prompts) == 1:
            # 单个样本使用原来的方法
            batch_predictions = [generate_response(
                model, tokenizer, batch_prompts[0], 
                max_tokens, temperature, top_p
            )]
        else:
            # 多个样本使用批量生成
            batch_predictions = generate_batch_responses(
                model, tokenizer, batch_prompts,
                max_tokens, temperature, top_p
            )
        
        # 收集结果
        predictions.extend(batch_predictions)
        references.extend(batch_references)
        inputs_list.extend(batch_prompts)
    
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
        logger.info(f"💾 预测结果保存到: {results_file}")
    
    return {
        "predictions": predictions,
        "references": references,
        "inputs": inputs_list,
        "num_samples": len(predictions)
    }

def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """计算BLEU分数"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            
            if len(pred_tokens) > 0 and len(ref_tokens[0]) > 0:
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        
    except Exception as e:
        logger.warning(f"BLEU计算失败: {e}")
        return 0.0

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算ROUGE分数"""
    try:
        from rouge import Rouge
        rouge = Rouge()
        
        scores = rouge.get_scores(predictions, references, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
        
    except Exception as e:
        logger.warning(f"ROUGE计算失败: {e}")
        # 简单的ROUGE-L近似
        try:
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
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens = ref.lower().split()
                if len(ref_tokens) > 0:
                    lcs_len = lcs_length(pred_tokens, ref_tokens)
                    rouge_l = lcs_len / len(ref_tokens)
                    rouge_l_scores.append(rouge_l)
            
            return {
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
            }
            
        except Exception as e2:
            logger.warning(f"简单ROUGE计算也失败: {e2}")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

def calculate_metrics(predictions: List[str], references: List[str], metrics: List[str]) -> Dict[str, float]:
    """计算评估指标"""
    logger.info("📊 计算评估指标...")
    
    # 过滤有效的预测和参考
    valid_pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) 
                   if p.strip() and r.strip()]
    
    if not valid_pairs:
        logger.warning("⚠️ 没有有效的预测-参考对")
        return {"error": "no_valid_pairs"}
    
    valid_predictions, valid_references = zip(*valid_pairs)
    results = {}
    
    # BLEU分数
    if "bleu" in metrics:
        results["bleu"] = calculate_bleu_score(valid_predictions, valid_references)
    
    # ROUGE分数
    if "rouge" in metrics:
        rouge_scores = calculate_rouge_scores(valid_predictions, valid_references)
        results.update(rouge_scores)
    
    # 精确匹配
    if "exact_match" in metrics:
        exact_matches = sum(1 for p, r in zip(valid_predictions, valid_references) 
                           if p.lower().strip() == r.lower().strip())
        results["exact_match"] = exact_matches / len(valid_predictions)
    
    # 统计信息
    results["num_valid_pairs"] = len(valid_pairs)
    results["avg_prediction_length"] = sum(len(p.split()) for p in valid_predictions) / len(valid_predictions)
    results["avg_reference_length"] = sum(len(r.split()) for r in valid_references) / len(valid_references)
    
    return results

def print_results(results: Dict[str, Any], output_dir: str):
    """打印评估结果"""
    print("\n" + "="*60)
    print("🎯 模型评估结果")
    print("="*60)
    print(f"📁 模型路径: {results['model_path']}")
    print(f"📄 数据路径: {results['data_path']}")
    print(f"📊 总样本数: {results['num_samples']}")
    
    metrics = results['metrics']
    if "num_valid_pairs" in metrics:
        print(f"✅ 有效样本数: {metrics['num_valid_pairs']}")
    print(f"⏰ 时间戳: {results['timestamp']}")
    
    print(f"\n📈 评估指标:")
    main_metrics = ["bleu", "rouge-1", "rouge-2", "rouge-l", "exact_match"]
    for metric in main_metrics:
        if metric in metrics:
            print(f"  {metric.upper()}: {metrics[metric]:.4f}")
    
    print(f"\n📋 统计信息:")
    if "avg_prediction_length" in metrics:
        print(f"  平均预测长度: {metrics['avg_prediction_length']:.1f} tokens")
    if "avg_reference_length" in metrics:
        print(f"  平均参考长度: {metrics['avg_reference_length']:.1f} tokens")
    
    print(f"\n💾 结果文件: {os.path.join(output_dir, 'evaluation_results.json')}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Qwen 模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_model.py -m ./output/checkpoint-best -d ./data/test.jsonl
  python eval_model.py -m model_path -d data_path --metrics bleu,rouge,exact_match --save-predictions
        """
    )
    
    # 必需参数
    parser.add_argument("-m", "--model-path", required=True,
                       help="微调后的模型路径")
    parser.add_argument("-d", "--data-path", required=True,
                       help="评估数据文件路径 (JSONL格式)")
    
    # 可选参数
    parser.add_argument("-o", "--output-dir", default="./eval_results",
                       help="评估结果输出目录 (默认: ./eval_results)")
    parser.add_argument("-b", "--batch-size", type=int, default=4,
                       help="批处理大小 (默认: 4)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="最大生成token数 (默认: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度 (默认: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p采样 (默认: 0.9)")
    parser.add_argument("--device", default="auto",
                       help="计算设备: cpu/cuda/auto (默认: auto)")
    parser.add_argument("--torch-dtype", default="float16",
                       choices=["float16", "float32", "bfloat16"],
                       help="模型数据类型 (默认: float16)")
    parser.add_argument("--metrics", default="bleu,rouge,exact_match",
                       help="评估指标 (默认: bleu,rouge,exact_match)")
    parser.add_argument("--save-predictions", action="store_true",
                       help="保存详细的预测结果")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 加载模型和数据
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, args.device, args.torch_dtype
        )
        eval_data = load_eval_data(args.data_path)
        
        # 执行评估
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
        
        # 组织结果
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
                "torch_dtype": args.torch_dtype,
                "metrics": args.metrics
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印结果
        print_results(results, args.output_dir)
        
        if args.save_predictions:
            predictions_file = os.path.join(args.output_dir, "predictions.jsonl")
            print(f"📝 预测详情: {predictions_file}")
        
        logger.info("🎉 评估完成！")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断评估")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 评估过程中出现错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 