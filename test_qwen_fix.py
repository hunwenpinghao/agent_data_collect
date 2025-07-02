#!/usr/bin/env python3
"""
快速测试脚本：验证Qwen模型加载修复
"""

import sys
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """测试模型加载修复"""
    logger.info("开始测试Qwen模型加载修复...")
    
    # 导入修复的函数
    from fine_tune_qwen import load_model_with_patch, apply_transformers_patch
    
    try:
        # 首先测试补丁应用
        logger.info("测试transformers补丁应用...")
        patch_success = apply_transformers_patch()
        if patch_success:
            logger.info("✅ transformers补丁应用成功")
        else:
            logger.warning("⚠️  transformers补丁应用失败，但会继续测试")
        
        # 测试模型加载
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"  # 使用小模型以节省时间
        logger.info(f"测试模型加载: {model_path}")
        
        # 使用CPU和最小设置以避免内存问题
        model = load_model_with_patch(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ 模型加载成功!")
        logger.info(f"   模型类型: {type(model)}")
        logger.info(f"   配置类型: {type(model.config)}")
        
        # 测试tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("✅ Tokenizer加载成功!")
        
        # 简单测试
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        logger.info(f"✅ 文本编码测试通过: {test_text}")
        
        # 清理
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("\n🎉 所有测试通过！Qwen模型加载修复有效。")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False

def main():
    logger.info("Qwen模型加载修复测试")
    logger.info("="*40)
    
    if test_model_loading():
        logger.info("\n✅ 修复验证成功！可以运行fine_tune_qwen.py了。")
        return True
    else:
        logger.error("\n❌ 修复验证失败。请检查依赖和网络连接。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 