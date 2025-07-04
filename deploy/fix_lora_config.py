#!/usr/bin/env python3
"""
简单修复LoRA配置文件兼容性问题
"""

import os
import json
import argparse
import shutil

def fix_lora_config(lora_path):
    """修复LoRA配置文件中的兼容性问题"""
    
    config_path = os.path.join(lora_path, "adapter_config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 创建备份
    backup_path = config_path + ".backup"
    shutil.copy2(config_path, backup_path)
    print(f"✅ 已创建备份: {backup_path}")
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 移除不兼容的配置项
    problematic_keys = [
        'corda_config', 'eva_config', 'layer_replication',
        'layers_pattern', 'layers_to_transform', 'megatron_config',
        'megatron_core', 'trainable_token_indices', 'use_dora', 'use_rslora',
        'exclude_modules'
    ]
    
    removed_keys = []
    for key in problematic_keys:
        if key in config:
            del config[key]
            removed_keys.append(key)
    
    if removed_keys:
        print(f"🔧 移除的不兼容配置项: {', '.join(removed_keys)}")
        
        # 保存修复后的配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已修复配置文件: {config_path}")
        return True
    else:
        print("✅ 配置文件已经兼容，无需修改")
        return True

def main():
    parser = argparse.ArgumentParser(description="修复LoRA配置文件兼容性问题")
    parser.add_argument("--lora_path", type=str, 
                       default="output_deepspeed/zhc_xhs_qwen2.5_0.5b_lora",
                       help="LoRA模型路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.lora_path):
        print(f"❌ LoRA模型路径不存在: {args.lora_path}")
        return 1
    
    success = fix_lora_config(args.lora_path)
    
    if success:
        print("\n🎉 修复完成！现在可以运行转换脚本了")
        return 0
    else:
        print("\n❌ 修复失败")
        return 1

if __name__ == "__main__":
    exit(main()) 