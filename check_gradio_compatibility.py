#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio兼容性检查脚本
检查Gradio版本和功能兼容性
"""

import sys
import subprocess
import pkg_resources
from packaging import version

def check_gradio_version():
    """检查Gradio版本"""
    try:
        import gradio as gr
        current_version = gr.__version__
        
        print(f"🔍 当前Gradio版本: {current_version}")
        
        # 检查版本兼容性
        min_version = "3.0.0"
        recommended_version = "4.0.0"
        
        if version.parse(current_version) < version.parse(min_version):
            print(f"❌ Gradio版本过低，最低要求: {min_version}")
            print(f"💡 请升级: pip install gradio>={recommended_version}")
            return False
        elif version.parse(current_version) < version.parse(recommended_version):
            print(f"⚠️  建议升级到: {recommended_version}")
            print(f"💡 执行: pip install gradio>={recommended_version}")
        else:
            print(f"✅ Gradio版本兼容")
            
        return True
        
    except ImportError:
        print("❌ Gradio未安装")
        print("💡 请安装: pip install gradio>=4.0.0")
        return False
    except Exception as e:
        print(f"❌ 检查Gradio版本时出错: {e}")
        return False

def check_gradio_features():
    """检查Gradio功能"""
    try:
        import gradio as gr
        
        # 检查关键功能
        features = {
            'Blocks': hasattr(gr, 'Blocks'),
            'Chatbot': hasattr(gr, 'Chatbot'),
            'Textbox': hasattr(gr, 'Textbox'),
            'Button': hasattr(gr, 'Button'),
            'Slider': hasattr(gr, 'Slider'),
            'Radio': hasattr(gr, 'Radio'),
            'Examples': hasattr(gr, 'Examples'),
        }
        
        print("\n🔧 功能检查:")
        all_ok = True
        for feature, available in features.items():
            status = "✅" if available else "❌"
            print(f"   {status} {feature}")
            if not available:
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"❌ 检查Gradio功能时出错: {e}")
        return False

def check_launch_params():
    """检查launch参数兼容性"""
    try:
        import gradio as gr
        import inspect
        
        # 获取launch方法的签名
        if hasattr(gr.Blocks, 'launch'):
            sig = inspect.signature(gr.Blocks.launch)
            params = list(sig.parameters.keys())
            
            print("\n🚀 Launch参数检查:")
            
            # 检查关键参数
            supported_params = {
                'server_name': 'server_name' in params,
                'server_port': 'server_port' in params,
                'share': 'share' in params,
                'show_error': 'show_error' in params,
                'show_tips': 'show_tips' in params,
                'enable_queue': 'enable_queue' in params,
            }
            
            for param, supported in supported_params.items():
                status = "✅" if supported else "❌"
                print(f"   {status} {param}")
            
            return supported_params
        else:
            print("❌ 找不到launch方法")
            return {}
            
    except Exception as e:
        print(f"❌ 检查launch参数时出错: {e}")
        return {}

def get_compatible_launch_params():
    """获取兼容的launch参数"""
    supported = check_launch_params()
    
    # 基础参数
    params = {
        'server_name': "0.0.0.0",
        'server_port': 7860,
        'share': False,
    }
    
    # 可选参数
    optional_params = {
        'show_error': True,
        'enable_queue': True,
        'show_tips': True,
    }
    
    # 只添加支持的参数
    for param, value in optional_params.items():
        if supported.get(param, False):
            params[param] = value
    
    return params

def main():
    """主函数"""
    print("🔍 Gradio兼容性检查")
    print("=" * 40)
    
    # 检查版本
    version_ok = check_gradio_version()
    
    # 检查功能
    features_ok = check_gradio_features()
    
    # 检查launch参数
    launch_params = get_compatible_launch_params()
    
    print("\n🎯 推荐的launch参数:")
    for param, value in launch_params.items():
        print(f"   {param}: {value}")
    
    print("\n📝 生成兼容代码:")
    print("demo.launch(")
    for param, value in launch_params.items():
        if isinstance(value, str):
            print(f'    {param}="{value}",')
        else:
            print(f'    {param}={value},')
    print(")")
    
    # 总结
    print("\n" + "=" * 40)
    if version_ok and features_ok:
        print("✅ 兼容性检查通过")
        return 0
    else:
        print("❌ 发现兼容性问题")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 