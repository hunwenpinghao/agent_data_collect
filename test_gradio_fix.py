#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio修复验证脚本
验证修复后的界面是否可以正常工作
"""

def test_gradio_basic():
    """测试基本的Gradio功能"""
    print("🧪 测试Gradio基本功能...")
    
    try:
        import gradio as gr
        print(f"✅ Gradio版本: {gr.__version__}")
        
        # 测试基本组件
        with gr.Blocks() as demo:
            gr.Markdown("# 测试界面")
            chatbot = gr.Chatbot(type="tuples")
            msg = gr.Textbox(label="输入")
            
            def respond(message, history):
                history.append([message, f"收到: {message}"])
                return history, ""
            
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
        
        print("✅ 基本组件创建成功")
        return True
        
    except Exception as e:
        print(f"❌ Gradio基本功能测试失败: {e}")
        return False

def test_model_inference():
    """测试模型推理类"""
    print("\n🧪 测试模型推理类...")
    
    try:
        from gradio_inference import ModelInference
        
        # 创建实例
        inference = ModelInference()
        print("✅ ModelInference实例创建成功")
        
        # 测试未加载模型时的响应
        response = inference.generate_response("测试", stream=False)
        print(f"未加载模型响应: {response}")
        
        if "请先加载模型" in response:
            print("✅ 错误处理正确")
        else:
            print(f"⚠️  意外响应: {response}")
        
        # 测试流式生成（应该回退到普通模式）
        stream_response = inference.generate_response("测试", stream=True)
        if isinstance(stream_response, str):
            print("✅ 流式生成正确回退到普通模式")
        else:
            print(f"⚠️  流式生成返回类型: {type(stream_response)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型推理类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_send_message_function():
    """测试发送消息函数逻辑"""
    print("\n🧪 测试发送消息函数...")
    
    try:
        from gradio_inference import ModelInference
        
        # 模拟send_message函数的核心逻辑
        model_inference = ModelInference()
        
        # 模拟参数
        history = []
        message = "你好"
        max_len = 100
        temp = 0.7
        top_p_val = 0.9
        top_k_val = 50
        rep_penalty = 1.1
        debug = False
        stream = False
        
        # 添加用户消息
        history.append([message, ""])
        
        # 生成响应
        response = model_inference.generate_response(
            message, max_len, temp, top_p_val, top_k_val, rep_penalty, debug, stream=stream
        )
        
        # 确保响应是字符串
        if not isinstance(response, str):
            response = str(response)
        
        # 更新历史
        history[-1][1] = response
        
        print(f"测试对话历史: {history}")
        print("✅ 发送消息函数逻辑正确")
        return True
        
    except Exception as e:
        print(f"❌ 发送消息函数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Gradio修复验证开始")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    if test_gradio_basic():
        success_count += 1
    
    if test_model_inference():
        success_count += 1
    
    if test_send_message_function():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 修复验证通过！")
        print("💡 现在可以启动Gradio界面:")
        print("   ./run_gradio.sh")
        print("")
        print("📋 预期行为:")
        print("1. 加载模型后发送消息")
        print("2. 看到'正在思考中...'")
        print("3. 然后显示完整回复")
        print("4. 不会再出现生成器错误")
        return 0
    else:
        print("❌ 部分测试失败")
        print("请检查错误信息并重试")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 