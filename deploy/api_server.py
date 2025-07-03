#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产级API服务器 for 火山引擎部署
支持多并发推理和负载均衡
"""

import os
import json
import time
import torch
import asyncio
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus 指标
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration in seconds')
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Number of active connections')
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time')

class ModelManager:
    """模型管理器 - 支持模型热加载和缓存"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.lora_path = None
        self.load_time = None
        
    async def load_model(self, base_model_path: str, lora_path: Optional[str] = None, quantization: str = "none"):
        """异步加载模型"""
        try:
            logger.info(f"Loading model: {base_model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 设置模型加载参数
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # 量化配置
            if quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
            
            # 加载LoRA适配器
            if lora_path and os.path.exists(lora_path):
                logger.info(f"Loading LoRA adapter: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            
            self.model_path = base_model_path
            self.lora_path = lora_path
            self.load_time = time.time()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    async def generate_response(self, prompt: str, max_length: int = 512, 
                               temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成响应"""
        if self.model is None or self.tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # 构建对话格式
            conversation = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 编码输入
            inputs = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取assistant的回复
            assistant_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            response = generated_text[assistant_start:].strip()
            
            # 记录推理时间
            inference_time = time.time() - start_time
            MODEL_INFERENCE_TIME.observe(inference_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# 全局模型管理器
model_manager = ModelManager()

# Redis连接池
redis_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global redis_pool
    
    # 启动时初始化
    logger.info("Starting up application...")
    
    # 初始化Redis连接
    try:
        redis_pool = redis.ConnectionPool.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            max_connections=20
        )
        logger.info("Redis connection pool initialized")
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}")
    
    # 加载模型
    try:
        base_model = os.getenv("BASE_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
        lora_path = os.getenv("LORA_MODEL_PATH")
        quantization = os.getenv("QUANTIZATION", "none")
        
        await model_manager.load_model(base_model, lora_path, quantization)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down application...")
    if redis_pool:
        await redis_pool.disconnect()

# FastAPI应用
app = FastAPI(
    title="Qwen Model API Server",
    description="高性能Qwen模型推理API服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="输入提示")
    max_length: int = Field(512, ge=1, le=2048, description="最大生成长度")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="温度参数")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="top_p采样参数")
    stream: bool = Field(False, description="是否流式输出")

class GenerateResponse(BaseModel):
    response: str
    model_info: Dict[str, Any]
    inference_time: float
    timestamp: float

# API路由
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "timestamp": time.time()
    }

@app.get("/model/info")
async def get_model_info():
    """获取模型信息"""
    if model_manager.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "base_model_path": model_manager.model_path,
        "lora_path": model_manager.lora_path,
        "load_time": model_manager.load_time,
        "device": str(next(model_manager.model.parameters()).device),
        "model_type": type(model_manager.model).__name__
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """生成文本"""
    start_time = time.time()
    
    try:
        # 增加活跃连接数
        ACTIVE_CONNECTIONS.inc()
        
        # 生成响应
        response = await model_manager.generate_response(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        inference_time = time.time() - start_time
        
        # 记录指标
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="success").inc()
        REQUEST_DURATION.observe(inference_time)
        
        return GenerateResponse(
            response=response,
            model_info={
                "model_path": model_manager.model_path,
                "lora_path": model_manager.lora_path
            },
            inference_time=inference_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="error").inc()
        raise e
    finally:
        ACTIVE_CONNECTIONS.dec()

@app.get("/metrics")
async def get_metrics():
    """Prometheus指标"""
    return generate_latest()

@app.post("/model/reload")
async def reload_model(base_model_path: str, lora_path: Optional[str] = None):
    """重新加载模型"""
    try:
        await model_manager.load_model(base_model_path, lora_path)
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    # 运行配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    if workers > 1:
        # 多进程模式
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
    else:
        # 单进程模式
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        ) 