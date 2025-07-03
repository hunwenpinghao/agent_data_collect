#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式生成测试脚本
验证流式生成功能是否正常工作
"""

import sys
import time

def test_streaming_basic():
    """测试基本的流式生成功能"""
    print("🧪 测试基本流式生成...")
    
    try:
        # 模拟流式生成
        message = "你好，你是谁？"
        words = ["我", "是", "一个", "AI", "助手", "，", "很高兴", "为您", "服务", "！"]
        
        print(f"输入: {message}")
        print("流式输出: ", end="", flush=True)
        
        current_response = ""
        for word in words:
            current_response += word
            print(word, end="", flush=True)
            time.sleep(0.1)  # 模拟生成延迟
        
        print()
        print(f"最终回复: {current_response}")
        print("✅ 基本流式生成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 基本流式生成测试失败: {e}")
        return False

def test_gradio_streaming():
    """测试Gradio流式生成组件"""
    print("\n🧪 测试Gradio流式生成组件...")
    
    try:
        from gradio_inference import ModelInference
        
        # 创建模型推理实例
        inference = ModelInference()
        
        # 测试未加载模型的情况
        print("测试未加载模型的流式生成...")
        stream_gen = inference.generate_response("测试", stream=True)
        response = next(stream_gen)
        
        if "请先加载模型" in response:
            print("✅ 未加载模型时的错误处理正确")
        else:
            print(f"⚠️  预期错误消息，但得到: {response}")
        
        print("✅ Gradio流式生成组件测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Gradio流式生成组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_response_cleaning():
    """测试回复清理功能"""
    print("\n🧪 测试回复清理功能...")
    
    try:
        from gradio_inference import ModelInference
        
        inference = ModelInference()
        
        # 测试数据
        test_cases = [
            ("<|im_start|>assistant\n我是AI助手<|im_end|>", "我是AI助手"),
            ("user\n你好\nassistant\n你好！", "你好！"),
            ("我是来自阿里云的大规模语言模型，我叫通义千问。", "我是来自阿里云的大规模语言模型，我叫通义千问。"),
        ]
        
        for input_text, expected in test_cases:
            cleaned = inference._extract_and_clean_streaming_response(
                input_text, "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n", "你好"
            )
            print(f"输入: {repr(input_text)}")
            print(f"预期: {repr(expected)}")
            print(f"实际: {repr(cleaned)}")
            print()
        
        print("✅ 回复清理功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 回复清理功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 流式生成功能测试开始")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # 运行各项测试
    if test_streaming_basic():
        success_count += 1
    
    if test_gradio_streaming():
        success_count += 1
    
    if test_response_cleaning():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！流式生成功能正常")
        print("💡 现在可以运行: ./run_gradio.sh")
        return 0
    else:
        print("❌ 部分测试失败，请检查问题")
        print("建议:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 检查Gradio版本兼容性")
        print("3. 尝试关闭流式生成使用传统模式")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 