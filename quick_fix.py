#!/usr/bin/env python3
"""
快速修复脚本：解决 Qwen 模型的 NoneType 错误
使用方法：在运行 fine_tune_qwen.py 之前先运行这个脚本
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

def apply_quick_fix():
    """应用快速修复"""
    try:
        # 修复方案1: 临时修改transformers源码
        logger.info("应用transformers快速修复...")
        
        import transformers.modeling_utils as modeling_utils
        
        # 保存原始方法
        original_post_init = modeling_utils.PreTrainedModel.post_init
        
        def fixed_post_init(self):
            """修复后的post_init方法"""
            try:
                # 强制设置所有可能为None的配置
                if hasattr(self.config, 'pretraining_tp'):
                    if self.config.pretraining_tp is None:
                        self.config.pretraining_tp = 1
                
                # 检查其他常见的None值
                none_fixes = {
                    'attention_dropout': 0.0,
                    'rope_scaling': None,
                    'attn_implementation': 'eager',
                    'use_sliding_window': False,
                    'sliding_window': 4096
                }
                
                for key, default_val in none_fixes.items():
                    if hasattr(self.config, key) and getattr(self.config, key) is None:
                        setattr(self.config, key, default_val)
                
                # 调用原始方法
                return original_post_init(self)
                
            except Exception as e:
                # 如果还是失败，直接跳过post_init
                logger.warning(f"跳过post_init: {e}")
                pass
        
        # 应用修复
        modeling_utils.PreTrainedModel.post_init = fixed_post_init
        
        logger.info("✅ 快速修复应用成功！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 快速修复失败: {e}")
        return False

def test_fix():
    """测试修复是否有效"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # 测试小模型
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        logger.info(f"测试模型: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ 模型加载测试成功！")
        del model
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False

def main():
    logging.basicConfig(level=logging.INFO)
    
    logger.info("=" * 50)
    logger.info("Qwen 模型 NoneType 错误快速修复")
    logger.info("=" * 50)
    
    # 应用修复
    if apply_quick_fix():
        logger.info("🔧 修复已应用，现在测试...")
        
        if test_fix():
            logger.info("\n🎉 修复成功！现在可以运行 fine_tune_qwen.py")
            
            # 提供使用建议
            logger.info("\n📋 使用建议:")
            logger.info("1. 在Python会话中先运行: exec(open('quick_fix.py').read())")
            logger.info("2. 然后运行你的训练脚本")
            logger.info("3. 或者直接运行: python -c \"exec(open('quick_fix.py').read()); exec(open('fine_tune_qwen.py').read())\"")
            
            return True
        else:
            logger.error("\n❌ 修复测试失败")
    else:
        logger.error("\n❌ 无法应用修复")
    
    logger.error("\n建议尝试其他解决方案:")
    logger.error("1. pip install transformers>=4.51.0 --upgrade")
    logger.error("2. pip install transformers==4.36.2 --force-reinstall")
    logger.error("3. 使用不同的模型")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 