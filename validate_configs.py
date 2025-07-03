#!/usr/bin/env python3
"""
配置文件验证脚本
用于检查所有配置文件是否包含必要的参数，特别是eval_strategy参数
"""

import os
import json
import glob
from typing import Dict, List, Tuple

def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """验证单个配置文件"""
    errors = []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON解析错误: {e}"]
    except Exception as e:
        return False, [f"文件读取错误: {e}"]
    
    # 检查必需的参数
    required_params = [
        'model_name_or_path',
        'data_path',
        'output_dir',
        'per_device_train_batch_size',
        'gradient_accumulation_steps',
        'learning_rate',
        'num_train_epochs'
    ]
    
    for param in required_params:
        if param not in config:
            errors.append(f"缺少必需参数: {param}")
    
    # 检查评估策略参数
    if 'evaluation_strategy' in config:
        eval_strategy = config.get('evaluation_strategy')
        eval_strategy_alt = config.get('eval_strategy')
        
        # 检查是否有eval_strategy参数
        if eval_strategy_alt is None:
            errors.append("缺少兼容性参数: eval_strategy")
        elif eval_strategy != eval_strategy_alt:
            errors.append(f"evaluation_strategy ({eval_strategy}) 与 eval_strategy ({eval_strategy_alt}) 不匹配")
        
        # 检查load_best_model_at_end逻辑
        if config.get('load_best_model_at_end', False):
            if eval_strategy == 'no':
                errors.append("load_best_model_at_end=true 但 evaluation_strategy=no")
            if 'eval_data_path' not in config:
                errors.append("load_best_model_at_end=true 但缺少 eval_data_path")
            
            save_strategy = config.get('save_strategy', 'steps')
            if eval_strategy != save_strategy:
                errors.append(f"evaluation_strategy ({eval_strategy}) 与 save_strategy ({save_strategy}) 不匹配")
    
    # 检查DeepSpeed相关参数
    if config.get('use_deepspeed', False):
        if 'deepspeed_stage' not in config:
            errors.append("使用DeepSpeed但缺少 deepspeed_stage 参数")
    
    return len(errors) == 0, errors

def main():
    print("🔍 配置文件验证工具")
    print("=" * 50)
    
    # 找到所有配置文件
    config_files = glob.glob("configs/train_config_*.json")
    
    if not config_files:
        print("❌ 未找到配置文件")
        return
    
    total_files = len(config_files)
    valid_files = 0
    
    for config_file in sorted(config_files):
        print(f"\n📁 检查: {config_file}")
        
        is_valid, errors = validate_config_file(config_file)
        
        if is_valid:
            print("✅ 配置有效")
            valid_files += 1
        else:
            print("❌ 配置有误:")
            for error in errors:
                print(f"   - {error}")
    
    print("\n" + "=" * 50)
    print(f"📊 验证结果: {valid_files}/{total_files} 个配置文件有效")
    
    if valid_files == total_files:
        print("🎉 所有配置文件都通过验证！")
    else:
        print("⚠️  部分配置文件需要修复")
        
    # 显示关键参数统计
    print("\n📈 参数使用统计:")
    eval_strategies = {}
    deepspeed_usage = {'enabled': 0, 'disabled': 0}
    lora_usage = {'lora': 0, 'qlora': 0, 'full': 0}
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 统计评估策略
            strategy = config.get('evaluation_strategy', 'unknown')
            eval_strategies[strategy] = eval_strategies.get(strategy, 0) + 1
            
            # 统计DeepSpeed使用
            if config.get('use_deepspeed', False):
                deepspeed_usage['enabled'] += 1
            else:
                deepspeed_usage['disabled'] += 1
            
            # 统计微调方式
            if config.get('use_qlora', False):
                lora_usage['qlora'] += 1
            elif config.get('use_lora', False):
                lora_usage['lora'] += 1
            else:
                lora_usage['full'] += 1
                
        except Exception:
            continue
    
    print(f"   评估策略: {eval_strategies}")
    print(f"   DeepSpeed: {deepspeed_usage}")
    print(f"   微调方式: {lora_usage}")

if __name__ == "__main__":
    main() 