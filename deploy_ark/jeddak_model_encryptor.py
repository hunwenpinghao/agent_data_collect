#!/usr/bin/env python3
"""
Jeddak模型加密工具
基于火山引擎Jeddak Secure Model SDK实现模型加密和上传
"""

import os
import json
import logging
import subprocess
import sys
from typing import Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def auto_install_jeddak_sdk() -> bool:
    """
    自动下载并安装Jeddak SDK
    
    Returns:
        bool: 安装成功返回True，失败返回False
    """
    sdk_url = "https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl"
    sdk_file = "bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl"
    
    logger.info("🚀 开始自动下载Jeddak SDK...")
    logger.info(f"下载地址: {sdk_url}")
    
    try:
        # 使用curl下载
        logger.info("正在下载SDK文件...")
        result = subprocess.run(['curl', '-L', '-o', sdk_file, sdk_url], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"下载失败: {result.stderr}")
            return False
            
        # 验证文件
        if not os.path.exists(sdk_file) or os.path.getsize(sdk_file) < 10000:
            logger.error("下载的文件异常")
            if os.path.exists(sdk_file):
                os.remove(sdk_file)
            return False
        
        logger.info(f"✅ SDK下载成功，文件大小: {os.path.getsize(sdk_file)} 字节")
        
        # 安装SDK
        logger.info("📦 开始安装SDK...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', sdk_file], 
                              capture_output=True, text=True)
        
        # 清理下载文件
        os.remove(sdk_file)
        
        if result.returncode != 0:
            logger.error(f"安装失败: {result.stderr}")
            return False
        
        logger.info("✅ SDK安装成功")
        
        # 验证安装
        try:
            from bytedance.jeddak_secure_model.model_encryption import JeddakModelEncrypter
            logger.info("✅ SDK验证成功")
            return True
        except ImportError:
            logger.error("❌ SDK验证失败")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ 下载超时")
        return False
    except FileNotFoundError:
        logger.error("❌ 系统中没有找到curl命令")
        return False
    except Exception as e:
        logger.error(f"❌ 安装过程出错: {e}")
        return False

@dataclass
class JeddakConfig:
    """Jeddak配置"""
    volc_ak: str  # 火山引擎AK
    volc_sk: str  # 火山引擎SK
    app_id: str   # 火山账号ID
    region: str = "cn-beijing"
    endpoint: str = "tos-cn-beijing.volces.com"
    service: str = "pcc"

class JeddakModelEncryptor:
    """Jeddak模型加密器"""
    
    def __init__(self, config: JeddakConfig):
        self.config = config
        self._check_sdk()
    
    def _check_sdk(self):
        """检查Jeddak SDK是否安装"""
        try:
            from bytedance.jeddak_secure_model.model_encryption import EncryptionConfig, JeddakModelEncrypter
            self.sdk_available = True
            logger.info("✅ Jeddak Secure Model SDK 已安装")
        except ImportError:
            logger.warning("⚠️  Jeddak Secure Model SDK 未安装")
            
            # 尝试自动安装
            logger.info("🚀 尝试自动下载并安装SDK...")
            if auto_install_jeddak_sdk():
                try:
                    from bytedance.jeddak_secure_model.model_encryption import EncryptionConfig, JeddakModelEncrypter
                    self.sdk_available = True
                    logger.info("🎉 SDK自动安装并验证成功！")
                    return
                except ImportError:
                    logger.error("SDK安装后验证失败")
            
            # 自动安装失败，使用模拟模式
            self.sdk_available = False
            logger.warning("⚠️  将使用模拟模式（不进行真实加密）")
            logger.info("💡 如需真实加密，请手动安装SDK：")
            logger.info("   curl -L -O https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl")
            logger.info("   pip install bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl")
    
    def encrypt_and_upload_model(self, 
                                model_path: str,
                                bucket_name: str,
                                ring_name: str,
                                key_name: str,
                                ring_desc: str = "",
                                key_desc: str = "",
                                ring_id: str = "") -> Dict:
        """
        加密并上传模型到TOS
        
        Args:
            model_path: 模型文件夹路径
            bucket_name: TOS存储桶名称
            ring_name: 密钥环名称
            key_name: 密钥名称
            ring_desc: 密钥环描述
            key_desc: 密钥描述
            ring_id: 已有密钥环ID（可选）
            
        Returns:
            Dict: 包含ring_id, key_id, baseline, model_name等信息
        """
        
        if self.sdk_available:
            return self._real_encrypt_and_upload(
                model_path, bucket_name, ring_name, key_name, 
                ring_desc, key_desc, ring_id
            )
        else:
            return self._simulate_encrypt_and_upload(
                model_path, bucket_name, ring_name, key_name
            )
    
    def _real_encrypt_and_upload(self, model_path: str, bucket_name: str, 
                                ring_name: str, key_name: str,
                                ring_desc: str, key_desc: str, ring_id: str) -> Dict:
        """真实的加密和上传流程"""
        try:
            from bytedance.jeddak_secure_model.model_encryption import EncryptionConfig, JeddakModelEncrypter
            
            logger.info(f"开始加密模型: {model_path}")
            logger.info(f"目标存储桶: {bucket_name}")
            logger.info(f"密钥环: {ring_name}")
            logger.info(f"密钥名称: {key_name}")
            
            # 创建加密配置
            config = EncryptionConfig("", "")
            encrypter = JeddakModelEncrypter(config)
            
            # 执行加密和上传
            result = encrypter.encrypt_model_and_upload(
                model_path=model_path,
                bucket_name=bucket_name,
                volc_ak=self.config.volc_ak,
                volc_sk=self.config.volc_sk,
                region=self.config.region,
                endpoint=self.config.endpoint,
                ring_id=ring_id,
                ring_name=ring_name,
                ring_desc=ring_desc,
                key_name=key_name,
                key_desc=key_desc,
                app_id=self.config.app_id,
                service=self.config.service
            )
            
            logger.info("✅ 模型加密和上传成功")
            logger.info(f"密钥环ID: {result.get('ring_id')}")
            logger.info(f"密钥ID: {result.get('key_id')}")
            logger.info(f"基线值: {result.get('baseline')}")
            logger.info(f"模型名称: {result.get('model_name')}")
            
            return result
            
        except Exception as e:
            logger.error(f"模型加密和上传失败: {e}")
            raise
    
    def _simulate_encrypt_and_upload(self, model_path: str, bucket_name: str,
                                   ring_name: str, key_name: str) -> Dict:
        """模拟加密和上传流程"""
        import time
        import uuid
        
        logger.info("🔄 模拟模型加密和上传流程...")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"存储桶: {bucket_name}")
        logger.info(f"密钥环: {ring_name}")
        logger.info(f"密钥名称: {key_name}")
        
        # 模拟处理时间
        time.sleep(2)
        
        model_name = os.path.basename(model_path)
        
        # 生成模拟结果
        result = {
            "ring_id": f"ring_{ring_name}_{uuid.uuid4().hex[:8]}",
            "key_id": f"key_{key_name}_{uuid.uuid4().hex[:8]}",
            "baseline": f"baseline_{uuid.uuid4().hex[:16]}",
            "model_name": model_name,
            "model_path": f"https://{bucket_name}.{self.config.endpoint}/{model_name}/"
        }
        
        logger.info("✅ 模拟加密和上传完成")
        logger.info(f"密钥环ID: {result['ring_id']}")
        logger.info(f"密钥ID: {result['key_id']}")
        logger.info(f"基线值: {result['baseline']}")
        logger.info(f"模型名称: {result['model_name']}")
        logger.info(f"模型路径: {result['model_path']}")
        
        return result

def install_jeddak_sdk():
    """安装Jeddak SDK的辅助函数"""
    logger.info("📦 Jeddak Secure Model SDK 安装指南:")
    logger.info("")
    logger.info("方法1 - 自动安装:")
    if auto_install_jeddak_sdk():
        logger.info("✅ SDK自动安装成功！")
        return True
    else:
        logger.warning("❌ 自动安装失败，请使用手动方法")
    
    logger.info("")
    logger.info("方法2 - 手动安装:")
    logger.info("curl -L -O https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl")
    logger.info("pip install bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl")
    logger.info("")
    logger.info("方法3 - 从官方文档:")
    logger.info("1. 访问火山引擎文档: https://www.volcengine.com/docs/85010/1546894")
    logger.info("2. 下载对应版本的SDK文件")
    logger.info("3. pip install <下载的文件>")
    
    return False

if __name__ == "__main__":
    # 示例用法
    config = JeddakConfig(
        volc_ak=os.getenv("VOLCANO_AK", "your_ak"),
        volc_sk=os.getenv("VOLCANO_SK", "your_sk"), 
        app_id=os.getenv("VOLCANO_APP_ID", "your_app_id")
    )
    
    encryptor = JeddakModelEncryptor(config)
    
    # 示例加密和上传
    try:
        result = encryptor.encrypt_and_upload_model(
            model_path="./output_qwen",
            bucket_name="your-bucket-name",
            ring_name="Qwen-Ring",
            key_name="QwenKey",
            ring_desc="Ring for Qwen model",
            key_desc="Encryption key for Qwen model"
        )
        
        print("加密结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"加密失败: {e}")
        install_jeddak_sdk() 