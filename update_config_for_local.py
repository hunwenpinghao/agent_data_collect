#!/usr/bin/env python3
"""
自动更新配置文件以使用本地下载的模型
"""

import json
import os
from pathlib import Path

def update_config_files():
    """更新所有配置文件以使用本地模型路径"""
    
    # 检查本地模型是否存在
    local_model_path = "models/Qwen2.5-0.5B-Instruct"
    abs_model_path = os.path.abspath(local_model_path)
    
    if not os.path.exists(local_model_path):
        print(f"❌ 本地模型路径不存在: {local_model_path}")
        print("请先运行: ./download_model.sh")
        return False
    
    # 检查必要的模型文件
    required_files = [
        "config.json",
        "tokenizer.json", 
        "model.safetensors"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(local_model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要的模型文件: {missing_files}")
        print("请重新下载完整的模型文件")
        return False
    
    print(f"✅ 发现本地模型: {abs_model_path}")
    
    # 更新配置文件
    config_files = [
        "configs/train_config_full.json",
        "configs/train_config_lora.json", 
        "configs/train_config_qlora.json",
        "configs/train_config_qlora_8bit.json"
    ]
    
    updated_count = 0
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"⚠️  配置文件不存在: {config_file}")
            continue
            
        try:
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 备份原配置
            backup_file = config_file + ".backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            # 更新模型路径
            old_path = config.get("model_name_or_path", "")
            config["model_name_or_path"] = abs_model_path
            
            # 写回配置文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            print(f"✅ 已更新 {config_file}")
            print(f"   原路径: {old_path}")
            print(f"   新路径: {abs_model_path}")
            print(f"   备份至: {backup_file}")
            
            updated_count += 1
            
        except Exception as e:
            print(f"❌ 更新 {config_file} 失败: {e}")
    
    if updated_count > 0:
        print(f"\n🎉 成功更新了 {updated_count} 个配置文件")
        print("\n现在可以使用本地模型进行训练:")
        print("  ./run_train.sh lora")
        print("  ./run_train.sh full") 
        print("  ./run_train.sh qlora")
        return True
    else:
        print("\n❌ 没有成功更新任何配置文件")
        return False

if __name__ == "__main__":
    update_config_files() 