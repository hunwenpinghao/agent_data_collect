#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 transformers 库中 ALL_PARALLEL_STYLES 的 None 值问题
这个脚本可以独立运行来修复库文件，或作为模块导入使用
"""

import sys
import os
import importlib


def fix_transformers_parallel_styles():
    """修复 transformers 库中的 ALL_PARALLEL_STYLES None 值问题"""
    try:
        # 导入 transformers 库
        transformers = importlib.import_module('transformers')
        
        # 获取 modeling_utils 模块
        if hasattr(transformers, 'modeling_utils'):
            modeling_utils = transformers.modeling_utils
        else:
            modeling_utils = importlib.import_module('transformers.modeling_utils')
        
        # 检查 ALL_PARALLEL_STYLES 是否存在且不为 None
        current_value = getattr(modeling_utils, 'ALL_PARALLEL_STYLES', None)
        
        if current_value is None:
            # 设置默认的并行样式列表
            default_parallel_styles = ["tp", "dp", "pp", "cp"]
            setattr(modeling_utils, 'ALL_PARALLEL_STYLES', default_parallel_styles)
            print(f"✅ 成功修复 ALL_PARALLEL_STYLES: {default_parallel_styles}")
            return True
        else:
            print(f"ℹ️  ALL_PARALLEL_STYLES 已存在: {current_value}")
            return True
            
    except ImportError as e:
        print(f"❌ 无法导入 transformers 库: {e}")
        return False
    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        return False


def check_transformers_version():
    """检查 transformers 版本"""
    try:
        import transformers
        version = transformers.__version__
        print(f"🔍 Transformers 版本: {version}")
        
        # 检查版本是否过低
        from packaging import version as pkg_version
        if pkg_version.parse(version) < pkg_version.parse("4.40.0"):
            print("⚠️  建议升级到 transformers >= 4.40.0 以避免此问题")
            print("   升级命令: pip install transformers>=4.40.0")
        
        return version
    except ImportError:
        print("❌ transformers 库未安装")
        return None
    except Exception as e:
        print(f"❌ 无法检查版本: {e}")
        return None


def main():
    """主函数"""
    print("🔧 开始修复 transformers 库的 ALL_PARALLEL_STYLES 问题...")
    
    # 检查版本
    version = check_transformers_version()
    if version is None:
        print("❌ 无法继续，请先安装 transformers 库")
        sys.exit(1)
    
    # 应用修复
    success = fix_transformers_parallel_styles()
    
    if success:
        print("✅ 修复完成！现在可以正常加载模型了")
        print("💡 建议：考虑升级到更新版本的 transformers 库以彻底解决此问题")
    else:
        print("❌ 修复失败，请尝试升级 transformers 库")
        print("   升级命令: pip install transformers>=4.40.0")
        sys.exit(1)


if __name__ == "__main__":
    main()