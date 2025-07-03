#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火山方舟专用API服务器
支持高并发推理和自动扩缩容
"""

import os
import json
import time
import torch
import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
try:
    import redis.asyncio as redis
except ImportError:
    redis = None
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus 指标
REQUEST_COUNT = Counter('ark_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ark_api_request_duration_seconds', 'Request duration in seconds')
ACTIVE_CONNECTIONS = Gauge('ark_api_active_connections', 'Number of active connections')
MODEL_INFERENCE_TIME = Histogram('ark_model_inference_duration_seconds', 'Model inference time')
QUEUE_SIZE = Gauge('ark_request_queue_size', 'Current request queue size')

class RequestBatch:
    """请求批处理类"""
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.requests = []
        self.futures = []
        self.lock = threading.Lock()
        
    def add_request(self, request_data: Dict, future):
        """添加请求到批次"""
        with self.lock:
            self.requests.append(request_data)
            self.futures.append(future)
            
            if len(self.requests) >= self.max_batch_size:
                return True
        return False
    
    def get_batch(self):
        """获取当前批次"""
        with self.lock:
            if not self.requests:
                return [], []
            
            requests = self.requests.copy()
            futures = self.futures.copy()
            self.requests.clear()
            self.futures.clear()
            return requests, futures

class ModelManager:
    """增强的模型管理器 - 支持批处理推理"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.lora_path = None
        self.load_time = None
        self.device = None
        self.batch_processor = None
        self.request_queue = queue.Queue()
        self.batch_size = int(os.getenv("MAX_BATCH_SIZE", "8"))
        self.max_wait_time = float(os.getenv("MAX_WAIT_TIME", "0.1"))
        
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
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except ImportError:
                    logger.warning("BitsAndBytesConfig not available, skipping quantization")
            
            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
            
            # 加载LoRA适配器
            if lora_path and os.path.exists(lora_path) and PeftModel:
                logger.info(f"Loading LoRA adapter: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            
            self.device = next(self.model.parameters()).device
            self.model_path = base_model_path
            self.lora_path = lora_path
            self.load_time = time.time()
            
            # 启动批处理器
            self.start_batch_processor()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def start_batch_processor(self):
        """启动批处理器"""
        self.batch_processor = RequestBatch(self.batch_size, self.max_wait_time)
        
        def process_batches():
            while True:
                try:
                    requests, futures = self.batch_processor.get_batch()
                    if requests:
                        self._process_batch(requests, futures)
                    time.sleep(0.01)  # 小延迟避免CPU占用过高
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        thread = threading.Thread(target=process_batches, daemon=True)
        thread.start()
    
    def _process_batch(self, requests: List[Dict], futures: List):
        """处理批次请求"""
        try:
            start_time = time.time()
            
            # 准备批次输入
            prompts = []
            for req in requests:
                conversation = f"<|im_start|>user\n{req['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(conversation)
            
            # 批量编码
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=requests[0].get('max_length', 512),
                    temperature=requests[0].get('temperature', 0.7),
                    top_p=requests[0].get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 批量解码
            responses = []
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                assistant_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                response = generated_text[assistant_start:].strip()
                responses.append(response)
            
            # 返回结果
            inference_time = time.time() - start_time
            MODEL_INFERENCE_TIME.observe(inference_time)
            
            for i, future in enumerate(futures):
                if i < len(responses):
                    future.set_result({
                        "response": responses[i],
                        "inference_time": inference_time,
                        "batch_size": len(requests)
                    })
                else:
                    future.set_exception(Exception("Batch processing failed"))
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for future in futures:
                future.set_exception(e)
    
    async def generate_response(self, prompt: str, max_length: int = 512, 
                               temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """生成响应 - 支持批处理"""
        if self.model is None or self.tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # 创建Future对象
        future = asyncio.get_event_loop().create_future()
        
        # 添加到批处理队列
        request_data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # 如果批次已满，立即处理
        if self.batch_processor.add_request(request_data, future):
            # 触发批处理
            pass
        
        # 等待结果
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
    
    async def generate_stream(self, prompt: str, max_length: int = 512, 
                             temperature: float = 0.7, top_p: float = 0.9) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        if self.model is None or self.tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            conversation = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(conversation, return_tensors="pt").to(self.device)
            
            # 流式生成
            with torch.no_grad():
                for _ in range(max_length):
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=1,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # 获取新生成的token
                    new_token = outputs[0][-1:]
                    token_text = self.tokenizer.decode(new_token, skip_special_tokens=True)
                    
                    if token_text == self.tokenizer.eos_token:
                        break
                    
                    yield token_text
                    
                    # 更新输入序列
                    inputs.input_ids = outputs
                    
                    # 添加小延迟
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Stream generation failed: {str(e)}")

# 全局模型管理器
model_manager = ModelManager()

# Redis连接池
redis_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global redis_pool
    
    # 启动时初始化
    logger.info("Starting up Ark API server...")
    
    # 初始化Redis连接
    if redis:
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
    logger.info("Shutting down Ark API server...")
    if redis_pool:
        await redis_pool.disconnect()

# FastAPI应用
app = FastAPI(
    title="火山方舟 Qwen Model API",
    description="专为火山方舟优化的高性能Qwen模型推理API服务",
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
    batch_size: int
    timestamp: float

class BatchRequest(BaseModel):
    requests: List[GenerateRequest] = Field(..., description="批量请求")

class BatchResponse(BaseModel):
    responses: List[GenerateResponse]
    total_time: float
    batch_size: int

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model_manager.model is not None,
        "version": "1.0.0"
    }

@app.get("/model/info")
async def get_model_info():
    """获取模型信息"""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": model_manager.model_path,
        "lora_path": model_manager.lora_path,
        "load_time": model_manager.load_time,
        "device": str(model_manager.device),
        "batch_size": model_manager.batch_size,
        "max_wait_time": model_manager.max_wait_time
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """生成文本"""
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        if request.stream:
            # 流式响应
            async def generate():
                async for token in model_manager.generate_stream(
                    request.prompt, 
                    request.max_length, 
                    request.temperature, 
                    request.top_p
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # 普通响应
            result = await model_manager.generate_response(
                request.prompt,
                request.max_length,
                request.temperature,
                request.top_p
            )
            
            response = GenerateResponse(
                response=result["response"],
                model_info={
                    "model_path": model_manager.model_path,
                    "device": str(model_manager.device)
                },
                inference_time=result["inference_time"],
                batch_size=result.get("batch_size", 1),
                timestamp=time.time()
            )
            
            REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="success").inc()
            REQUEST_DURATION.observe(time.time() - start_time)
            
            return response
            
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="error").inc()
        logger.error(f"Generate request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_CONNECTIONS.dec()

@app.post("/batch", response_model=BatchResponse)
async def batch_generate(request: BatchRequest):
    """批量生成"""
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        # 并发处理多个请求
        tasks = []
        for req in request.requests:
            task = model_manager.generate_response(
                req.prompt,
                req.max_length,
                req.temperature,
                req.top_p
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 构建响应
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {result}")
                continue
            
            responses.append(GenerateResponse(
                response=result["response"],
                model_info={
                    "model_path": model_manager.model_path,
                    "device": str(model_manager.device)
                },
                inference_time=result["inference_time"],
                batch_size=result.get("batch_size", 1),
                timestamp=time.time()
            ))
        
        batch_response = BatchResponse(
            responses=responses,
            total_time=time.time() - start_time,
            batch_size=len(request.requests)
        )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/batch", status="success").inc()
        REQUEST_DURATION.observe(time.time() - start_time)
        
        return batch_response
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/batch", status="error").inc()
        logger.error(f"Batch request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_CONNECTIONS.dec()

@app.get("/metrics")
async def get_metrics():
    """获取Prometheus指标"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/model/reload")
async def reload_model(base_model_path: str, lora_path: Optional[str] = None):
    """重新加载模型"""
    try:
        await model_manager.load_model(base_model_path, lora_path)
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", 1))
    
    uvicorn.run(
        "ark_api_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    ) 