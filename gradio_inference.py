#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Web界面用于微调模型推理
支持LoRA、完整微调(Full FT)和QLoRA模型
"""

import os
import json
import torch
import gradio as gr
import logging
import threading
import time
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import PeftModel, PeftConfig
import gc

# 设置兼容性环境变量
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', 'true')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_transformers_patch():
    """
    应用monkey patch来修复transformers库的NoneType错误
    """
    try:
        import transformers.modeling_utils as modeling_utils
        
        # 保存原始的post_init方法
        original_post_init = modeling_utils.PreTrainedModel.post_init
        
        def patched_post_init(self):
            """修复后的post_init方法"""
            try:
                # 检查并修复可能的None值
                if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp is None:
                    self.config.pretraining_tp = 1
                
                # 修复张量并行样式错误
                tensor_parallel_attrs = ['tensor_parallel_style', 'parallel_style']
                supported_styles = ['tp', 'dp', 'pp', 'cp']
                
                for attr in tensor_parallel_attrs:
                    if hasattr(self.config, attr):
                        current_value = getattr(self.config, attr)
                        if current_value is not None and current_value not in supported_styles:
                            setattr(self.config, attr, 'tp')
                
                # 检查其他可能的None值和无效值
                config_fixes = {
                    'attn_implementation': 'eager',
                    'rope_scaling': None,
                    'use_sliding_window': False,
                    'sliding_window': 4096,
                    'max_window_layers': 28,
                    'attention_dropout': 0.0,
                    'tensor_parallel_style': 'tp',
                    'parallel_style': 'tp', 
                    'tensor_parallel': False,
                    'sequence_parallel': False,
                }
                
                for key, default_value in config_fixes.items():
                    if hasattr(self.config, key) and getattr(self.config, key) is None:
                        setattr(self.config, key, default_value)
                
                return original_post_init(self)
                
            except Exception as e:
                logger.warning(f"post_init修复过程中出现错误: {e}")
                pass
        
        # 应用patch
        modeling_utils.PreTrainedModel.post_init = patched_post_init
        logger.info("✅ 已应用transformers post_init修复补丁")
        return True
        
    except Exception as e:
        logger.warning(f"❌ 无法应用transformers补丁: {e}")
        return False

def load_model_with_patch(model_path: str, **kwargs):
    """
    加载模型并修复可能的配置问题
    """
    try:
        # 应用transformers补丁
        apply_transformers_patch()
        
        logger.info(f"正在加载配置: {model_path}")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # 基础修复
        config_fixes = {
            'attn_implementation': 'eager',
            'pretraining_tp': 1,
            'torch_dtype': torch.float16,
            'use_cache': True,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.0,
            'rope_scaling': None,
            'tie_word_embeddings': False,
            '_name_or_path': model_path,
            'tensor_parallel_style': 'tp',
            'parallel_style': 'tp',
            'tensor_parallel': False,
            'sequence_parallel': False,
        }
        
        # 应用修复
        for key, default_value in config_fixes.items():
            if not hasattr(config, key) or getattr(config, key) is None:
                setattr(config, key, default_value)
        
        # 使用修复后的配置
        kwargs['config'] = config
        kwargs.setdefault('trust_remote_code', True)
        kwargs.setdefault('torch_dtype', torch.float16)
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        logger.info("✅ 模型加载成功！")
        
        return model
        
    except Exception as e:
        logger.error(f"使用补丁方法加载模型失败: {e}")
        raise e

class ModelInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.loaded_model_path = None
        self.loaded_lora_path = None
        
    def load_model(self, model_path: str, model_type: str, lora_path: Optional[str] = None, 
                   quantization: str = "none") -> Tuple[str, str]:
        """
        加载模型
        
        Args:
            model_path: 基础模型路径
            model_type: 模型类型 (lora, full_ft, qlora)
            lora_path: LoRA适配器路径（仅当model_type为lora或qlora时需要）
            quantization: 量化类型 (none, 4bit, 8bit)
            
        Returns:
            加载状态信息
        """
        try:
            # 清理之前的模型
            if self.model is not None:
                del self.model
                del self.tokenizer
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 设置量化配置
            quantization_config = None
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # 设置模型加载参数
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            # 加载基础模型
            logger.info(f"加载基础模型: {model_path}")
            if model_type == "full_ft":
                # 完整微调模型直接加载
                self.model = load_model_with_patch(model_path, **model_kwargs)
            else:
                # LoRA或QLoRA需要先加载基础模型
                self.model = load_model_with_patch(model_path, **model_kwargs)
                
                # 加载LoRA适配器
                if lora_path and os.path.exists(lora_path):
                    logger.info(f"加载LoRA适配器: {lora_path}")
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                else:
                    return "❌ 错误", f"LoRA路径不存在或未提供: {lora_path}"
            
            self.model_type = model_type
            self.loaded_model_path = model_path
            self.loaded_lora_path = lora_path
            
            # 模型信息
            device_info = f"设备: {next(self.model.parameters()).device}"
            model_info = f"模型类型: {model_type}, 量化: {quantization}"
            
            success_msg = f"✅ 模型加载成功！\n{model_info}\n{device_info}"
            logger.info(success_msg)
            
            return "✅ 成功", success_msg
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            logger.error(error_msg)
            return "❌ 失败", error_msg
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7,
                         top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.1, debug: bool = False, stream: bool = False):
        """
        生成回复，支持流式或一次性生成
        
        Args:
            stream: 是否使用流式生成
            
        Returns:
            如果stream=True，返回生成器；否则返回字符串
        """
        if self.model is None or self.tokenizer is None:
            if stream:
                yield "❌ 请先加载模型！"
                return
            else:
                return "❌ 请先加载模型！"
        
        try:
            # 构建对话格式
            conversation = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 编码输入
            inputs = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            
            # 生成参数
            generate_kwargs = {
                "max_length": min(max_length, 2048),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "no_repeat_ngram_size": 3,
            }
            
            if stream:
                # 流式生成 - 暂时禁用，避免兼容性问题
                logger.warning("流式生成暂时禁用，使用普通模式")
                # return self._generate_stream(inputs, generate_kwargs, conversation, prompt, input_length, debug)
            
            # 使用普通模式生成
                # 一次性生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **generate_kwargs
                    )
                
                # 解码输出
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 调试输出
            if debug:
                print(f"🔍 原始生成文本: {repr(generated_text)}")
                print(f"🔍 输入提示: {repr(conversation)}")
            
            # 提取assistant的回复 - 改进的提取逻辑
            response = generated_text
            
            # 方法1：基于对话格式提取
            if "<|im_start|>assistant\n" in generated_text:
                parts = generated_text.split("<|im_start|>assistant\n", 1)
                if len(parts) > 1:
                    response = parts[1]
                    # 移除结束标记
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0]
            
            # 方法2：如果没有找到标准格式，尝试移除输入部分
            elif conversation in generated_text:
                response = generated_text.replace(conversation, "").strip()
            
            # 方法3：如果包含用户输入，移除用户输入部分
            elif prompt in response:
                # 找到用户输入后的内容
                prompt_index = response.find(prompt)
                if prompt_index != -1:
                    # 从用户输入结束后开始提取
                    after_prompt = response[prompt_index + len(prompt):].strip()
                    if after_prompt:
                        response = after_prompt
            
            # 清理回复文本
            response = self._clean_response(response, conversation, prompt)
            
            # 调试输出
            if debug:
                print(f"🔍 清理后回复: {repr(response)}")
            
            # 如果回复为空或太短，返回友好消息
            if not response or len(response.strip()) < 2:
                return "抱歉，我无法生成有效的回复。请尝试调整参数或重新输入。"
            
            return response
                
        except Exception as e:
            error_msg = f"❌ 生成失败: {str(e)}"
            logger.error(error_msg)
            if stream:
                yield error_msg
            else:
                return error_msg
    
    def _generate_stream(self, inputs, generate_kwargs, conversation, prompt, input_length, debug):
        """
        流式生成回复，使用简单的逐步生成实现
        """
        try:
            # 计算最大新token数
            max_new_tokens = min(generate_kwargs.get("max_length", 512) - input_length, 1024)
            
            current_ids = inputs.input_ids.clone()
            current_attention = inputs.attention_mask.clone()
            current_response = ""
            
            # 逐步生成
            with torch.no_grad():
                for step in range(max_new_tokens):
                    # 生成下一个token
                    outputs = self.model.generate(
                        current_ids,
                        attention_mask=current_attention,
                        max_new_tokens=1,
                        temperature=generate_kwargs["temperature"],
                        top_p=generate_kwargs["top_p"],
                        top_k=generate_kwargs["top_k"],
                        repetition_penalty=generate_kwargs["repetition_penalty"],
                        do_sample=generate_kwargs["do_sample"],
                        pad_token_id=generate_kwargs["pad_token_id"],
                        eos_token_id=generate_kwargs["eos_token_id"],
                    )
                    
                    # 检查是否生成了结束token
                    new_token_id = outputs[0, -1].item()
                    if new_token_id == self.tokenizer.eos_token_id:
                        break
                    
                    # 更新序列
                    current_ids = outputs
                    current_attention = torch.ones_like(current_ids)
                    
                    # 解码到目前为止的生成内容
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 提取回复部分
                    new_response = self._extract_and_clean_streaming_response(
                        generated_text, conversation, prompt
                    )
                    
                    # 只有当回复有实际更新时才yield
                    if new_response != current_response and len(new_response.strip()) > 0:
                        current_response = new_response
                        
                        # 调试输出
                        if debug:
                            new_token = self.tokenizer.decode([new_token_id], skip_special_tokens=True)
                            logger.info(f"🔍 新token: {repr(new_token)} -> 当前回复: {repr(current_response[:50])}...")
                        
                        yield current_response
                        
                        # 小延迟让界面更新
                        time.sleep(0.05)
            
            # 最终清理
            if current_response:
                final_response = self._clean_response(current_response, conversation, prompt)
                if final_response != current_response and len(final_response.strip()) > 0:
                    yield final_response
            
            # 如果没有生成任何内容
            if not current_response.strip():
                yield "抱歉，模型没有生成有效内容，请重试。"
            
        except Exception as e:
            error_msg = f"❌ 流式生成失败: {str(e)}"
            logger.error(error_msg)
            yield error_msg
    
    def _extract_and_clean_streaming_response(self, buffer, conversation, prompt):
        """
        从流式生成的缓冲区中提取并清理回复
        """
        # 移除可能的对话格式标记
        response = buffer
        
        # 移除常见的标记
        markers_to_remove = [
            "<|im_start|>", "<|im_end|>", 
            "user\n", "assistant\n", "system\n",
            "User:", "Assistant:", "System:"
        ]
        
        for marker in markers_to_remove:
            response = response.replace(marker, "")
        
        # 移除开头的冒号和空格
        response = response.lstrip(': \n\t')
        
        # 如果包含用户输入，尝试移除
        if prompt in response:
            parts = response.split(prompt, 1)
            if len(parts) > 1:
                response = parts[1].strip()
        
        # 移除多余的换行
        response = response.strip()
        
        return response
    
    def _extract_response_part(self, generated_text, conversation, prompt):
        """
        从生成的文本中提取回复部分
        """
        response = generated_text
        
        # 方法1：基于对话格式提取
        if "<|im_start|>assistant\n" in generated_text:
            parts = generated_text.split("<|im_start|>assistant\n", 1)
            if len(parts) > 1:
                response = parts[1]
                # 移除结束标记
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
        
        # 方法2：如果没有找到标准格式，尝试移除输入部分
        elif conversation in generated_text:
            response = generated_text.replace(conversation, "").strip()
        
        # 方法3：如果包含用户输入，移除用户输入部分
        elif prompt in response:
            prompt_index = response.find(prompt)
            if prompt_index != -1:
                after_prompt = response[prompt_index + len(prompt):].strip()
                if after_prompt:
                    response = after_prompt
        
        # 基本清理
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|im_end|>", "")
        response = response.replace("user\n", "")
        response = response.replace("assistant\n", "")
        response = response.strip()
        
        return response
    
    def _clean_response(self, response: str, conversation: str, prompt: str) -> str:
        """
        清理回复文本，移除不必要的格式标记和重复内容
        """
        # 移除对话格式标记
        response = response.replace("<|im_start|>", "")
        response = response.replace("<|im_end|>", "")
        response = response.replace("user\n", "")
        response = response.replace("assistant\n", "")
        
        # 移除可能的系统角色标记
        response = response.replace("system\n", "")
        response = response.replace("System:", "")
        response = response.replace("User:", "")
        response = response.replace("Assistant:", "")
        
        # 移除重复的输入内容
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # 移除可能的对话开始符号
        if response.startswith("user\n") or response.startswith("assistant\n"):
            lines = response.split('\n', 1)
            if len(lines) > 1:
                response = lines[1]
        
        # 移除多余的换行和空格
        response = response.strip()
        
        # 移除开头的冒号和空格
        response = response.lstrip(': \n\t')
        
        # 如果回复太长，可能包含了多轮对话，只取第一部分
        if len(response) > 1000:
            # 查找可能的对话分割点
            split_markers = [
                "<|im_start|>user",
                "<|im_start|>assistant", 
                "user\n",
                "assistant\n",
                "\n\nuser:",
                "\n\nassistant:"
            ]
            
            for marker in split_markers:
                if marker in response:
                    response = response.split(marker)[0].strip()
                    break
        
        return response
    
    def get_model_info(self) -> str:
        """获取当前模型信息"""
        if self.model is None:
            return "❌ 未加载模型"
        
        try:
            # 基本信息
            info = []
            info.append(f"📊 模型信息")
            info.append(f"─────────────")
            info.append(f"类型: {self.model_type}")
            info.append(f"基础模型: {self.loaded_model_path}")
            
            if self.loaded_lora_path:
                info.append(f"LoRA路径: {self.loaded_lora_path}")
            
            # 设备信息
            device = next(self.model.parameters()).device
            info.append(f"设备: {device}")
            
            # 参数统计
            if hasattr(self.model, 'print_trainable_parameters'):
                # PEFT模型
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                info.append(f"可训练参数: {trainable_params:,}")
                info.append(f"总参数: {total_params:,}")
                info.append(f"训练参数比例: {100 * trainable_params / total_params:.2f}%")
            else:
                # 普通模型
                total_params = sum(p.numel() for p in self.model.parameters())
                info.append(f"总参数: {total_params:,}")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"❌ 获取模型信息失败: {str(e)}"

# 全局模型实例
model_inference = ModelInference()

def create_gradio_interface():
    """创建Gradio界面"""
    
    # 自定义CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .model-info {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🤖 微调模型推理系统") as demo:
        gr.Markdown("# 🤖 微调模型推理系统")
        gr.Markdown("支持LoRA、完整微调(Full FT)和QLoRA模型的加载与推理")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📂 模型配置")
                
                # 模型类型选择
                model_type = gr.Radio(
                    choices=["lora", "full_ft", "qlora"],
                    value="lora",
                    label="模型类型",
                    info="选择微调模型类型"
                )
                
                # 基础模型路径
                base_model_path = gr.Textbox(
                    value="Qwen/Qwen2.5-0.5B-Instruct",
                    label="基础模型路径",
                    info="HuggingFace模型名称或本地路径"
                )
                
                # LoRA适配器路径
                lora_path = gr.Textbox(
                    value="./output_qwen",
                    label="LoRA适配器路径",
                    info="LoRA/QLoRA适配器的保存路径",
                    visible=True
                )
                
                # 量化选项
                quantization = gr.Radio(
                    choices=["none", "4bit", "8bit"],
                    value="none",
                    label="量化类型",
                    info="选择量化方式以节省显存"
                )
                
                # 加载按钮
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                
                # 加载状态
                load_status = gr.Textbox(
                    label="加载状态",
                    interactive=False,
                    lines=3
                )
                
                # 模型信息
                model_info_display = gr.Textbox(
                    label="模型信息",
                    interactive=False,
                    lines=8,
                    elem_classes=["model-info"]
                )
                
                # 获取模型信息按钮
                info_btn = gr.Button("📊 获取模型信息")
                
            with gr.Column(scale=2):
                gr.Markdown("## 💬 对话生成")
                
                # 聊天历史
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=400,
                    avatar_images=["👤", "🤖"],
                    type="tuples"
                )
                
                # 用户输入
                user_input = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的问题...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 发送", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空对话")
                
                gr.Markdown("## ⚙️ 生成参数")
                
                with gr.Row():
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=2048,
                        value=512,
                        step=50,
                        label="最大长度"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="温度"
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p"
                    )
                    
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k"
                    )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.1,
                    label="重复惩罚"
                )
                
                # 调试选项
                with gr.Row():
                    debug_mode = gr.Checkbox(
                        label="🔍 调试模式",
                        value=False,
                        info="显示详细的生成过程信息"
                    )
                    
                    stream_mode = gr.Checkbox(
                        label="⚡ 流式生成",
                        value=False,  # 默认关闭，避免兼容性问题
                        info="暂时禁用，使用普通生成模式",
                        interactive=False  # 暂时禁用交互
                    )
        
        # 控制LoRA路径显示
        def update_lora_visibility(model_type_value):
            return gr.update(visible=model_type_value in ["lora", "qlora"])
        
        model_type.change(
            fn=update_lora_visibility,
            inputs=[model_type],
            outputs=[lora_path]
        )
        
        # 加载模型
        def load_model_wrapper(model_type_val, base_model_val, lora_path_val, quantization_val):
            if model_type_val in ["lora", "qlora"] and not lora_path_val:
                return "❌ 错误", "请提供LoRA适配器路径"
            
            status, message = model_inference.load_model(
                base_model_val, 
                model_type_val, 
                lora_path_val if model_type_val in ["lora", "qlora"] else None,
                quantization_val
            )
            return message
        
        load_btn.click(
            fn=load_model_wrapper,
            inputs=[model_type, base_model_path, lora_path, quantization],
            outputs=[load_status]
        )
        
        # 获取模型信息
        info_btn.click(
            fn=lambda: model_inference.get_model_info(),
            outputs=[model_info_display]
        )
        
        # 发送消息
        def send_message(history, message, max_len, temp, top_p_val, top_k_val, rep_penalty, debug, stream):
            if not message.strip():
                return history, ""
            
            # 添加用户消息到历史记录
            history.append([message, ""])
            
            try:
                if stream:
                    # 流式生成
                    try:
                        response_generator = model_inference.generate_response(
                            message, max_len, temp, top_p_val, top_k_val, rep_penalty, debug, stream=True
                        )
                        
                        # 检查是否真的是生成器
                        if hasattr(response_generator, '__iter__') and hasattr(response_generator, '__next__'):
                            # 先返回空的回复，然后逐步更新
                            yield history, ""
                            
                            for partial_response in response_generator:
                                # 确保partial_response是字符串
                                if not isinstance(partial_response, str):
                                    partial_response = str(partial_response)
                                
                                # 在调试模式下，添加额外信息
                                display_response = partial_response
                                if debug and not partial_response.startswith("❌"):
                                    debug_info = f"\n\n[调试信息] 模型类型: {model_inference.model_type or '未加载'}"
                                    if model_inference.loaded_model_path:
                                        debug_info += f"\n[调试信息] 基础模型: {model_inference.loaded_model_path}"
                                    if model_inference.loaded_lora_path:
                                        debug_info += f"\n[调试信息] LoRA路径: {model_inference.loaded_lora_path}"
                                    display_response = partial_response + debug_info
                                
                                # 更新对话历史中的最后一条消息
                                history[-1][1] = display_response
                                yield history, ""
                        else:
                            # 如果不是生成器，当作普通响应处理
                            response = str(response_generator)
                            history[-1][1] = response
                            yield history, ""
                            
                    except Exception as stream_error:
                        logger.error(f"流式生成错误: {stream_error}")
                        # 回退到非流式模式
                        history[-1][1] = "⚠️ 流式生成失败，使用普通模式..."
                        yield history, ""
                        
                        response = model_inference.generate_response(
                            message, max_len, temp, top_p_val, top_k_val, rep_penalty, debug, stream=False
                        )
                        
                        if not isinstance(response, str):
                            response = str(response)
                        
                        history[-1][1] = response
                        yield history, ""
                        
                else:
                    # 一次性生成 - 先显示正在生成的状态
                    history[-1][1] = "🤔 正在思考中..."
                    yield history, ""
                    
                    response = model_inference.generate_response(
                        message, max_len, temp, top_p_val, top_k_val, rep_penalty, debug, stream=False
                    )
                    
                    # 确保响应是字符串
                    if not isinstance(response, str):
                        response = str(response)
                    
                    # 在调试模式下，添加额外信息
                    if debug and not response.startswith("❌"):
                        response += f"\n\n[调试信息] 模型类型: {model_inference.model_type or '未加载'}"
                        if model_inference.loaded_model_path:
                            response += f"\n[调试信息] 基础模型: {model_inference.loaded_model_path}"
                        if model_inference.loaded_lora_path:
                            response += f"\n[调试信息] LoRA路径: {model_inference.loaded_lora_path}"
                    
                    # 更新对话历史
                    history[-1][1] = response
                    yield history, ""
                    
            except Exception as e:
                error_msg = f"❌ 生成失败: {str(e)}"
                logger.error(f"发送消息错误: {e}")
                history[-1][1] = error_msg
                yield history, ""
        
        send_btn.click(
            fn=send_message,
            inputs=[chatbot, user_input, max_length, temperature, top_p, top_k, repetition_penalty, debug_mode, stream_mode],
            outputs=[chatbot, user_input]
        )
        
        # 回车发送
        user_input.submit(
            fn=send_message,
            inputs=[chatbot, user_input, max_length, temperature, top_p, top_k, repetition_penalty, debug_mode, stream_mode],
            outputs=[chatbot, user_input]
        )
        
        # 清空对话
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        # 示例输入
        gr.Markdown("## 📝 示例输入")
        example_inputs = [
            "请为一家咖啡店写一段小红书风格的文案",
            "如何制作一杯完美的拿铁咖啡？",
            "推荐几家上海的网红咖啡店",
            "写一个关于春天的短诗",
            "解释一下人工智能的基本概念"
        ]
        
        examples = gr.Examples(
            examples=example_inputs,
            inputs=user_input,
            label="点击示例快速输入"
        )
        
        # 页面底部信息
        gr.Markdown("""
        ---
        ### 📖 使用说明
        1. **选择模型类型**：LoRA、完整微调(Full FT)或QLoRA
        2. **配置模型路径**：设置基础模型和LoRA适配器路径
        3. **选择量化方式**：可选择4bit或8bit量化以节省显存
        4. **加载模型**：点击"加载模型"按钮
        5. **开始对话**：在输入框中输入问题并发送
        6. **调整参数**：根据需要调整生成参数
        
        ### 💡 提示
        - 首次加载模型可能需要较长时间下载
        - 量化可以显著减少显存使用，但可能略微影响质量
        - 温度越高生成越随机，越低越确定
        - Top-p和Top-k控制生成的多样性
        
        ### ⚡ 流式生成（暂时禁用）
        - **当前状态**：流式生成功能暂时禁用，使用普通生成模式
        - **生成过程**：会显示"正在思考中..."然后显示完整回复
        - **稳定性**：普通模式更稳定，避免兼容性问题
        - **后续优化**：将在后续版本中重新启用流式生成
        
        ### 🔧 问题排查
        - **如果回复包含多余内容**：开启调试模式查看详细信息
        - **如果回复不是期望的风格**：检查是否加载了正确的微调模型
        - **如果模型回复"我是通义千问"**：说明加载的是基础模型，请检查LoRA路径
        - **回复格式异常**：尝试调整温度和重复惩罚参数
        - **流式生成卡住**：关闭流式生成或重新加载模型
        """)
    
    return demo

def main():
    """主函数"""
    # 创建界面
    demo = create_gradio_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 