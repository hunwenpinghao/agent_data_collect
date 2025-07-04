#!/usr/bin/env python3
"""
魔搭社区模型上传脚本
用于将微调好的模型上传到魔搭社区 ModelScope
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import time
import logging
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upload_modelscope.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查必要的依赖"""
    try:
        import modelscope
        logger.info(f"ModelScope 版本: {modelscope.__version__}")
    except ImportError:
        logger.error("ModelScope 未安装，请先安装: pip install modelscope")
        sys.exit(1)

    try:
        import git
        logger.info("git-python 可用")
    except ImportError:
        logger.error("git-python 未安装，请先安装: pip install GitPython")
        sys.exit(1)

def validate_model_directory(model_dir: str) -> bool:
    """验证模型目录是否包含必要的文件"""
    required_files = ['config.json', 'model.safetensors']
    model_path = Path(model_dir)
    
    if not model_path.exists():
        logger.error(f"模型目录不存在: {model_dir}")
        return False
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"缺少文件: {missing_files}")
        logger.info("但仍将尝试上传...")
    
    return True

def create_model_card(model_dir: str, model_name: str, model_description: str = None):
    """创建或更新模型卡片 README.md"""
    readme_path = Path(model_dir) / "README.md"
    
    if readme_path.exists():
        logger.info("README.md 已存在，跳过创建")
        return
    
    model_card_content = f"""---
license: apache-2.0
language:
- zh
- en
tags:
- qwen
- instruct
- fine-tuned
pipeline_tag: text-generation
---

# {model_name}

## 模型描述

{model_description or f"这是基于 Qwen2.5-0.5B-Instruct 微调的模型"}

## 使用方法

### 通过 ModelScope 使用

```python
from modelscope import AutoTokenizer, AutoModelForCausalLM
from modelscope import GenerationConfig

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained('hunwenpinghao/{model_name}')
model = AutoModelForCausalLM.from_pretrained('hunwenpinghao/{model_name}')

# 生成文本
messages = [
    {{"role": "user", "content": "你好！"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### 通过 Pipeline 使用

```python
from modelscope import pipeline

pipe = pipeline('text-generation', model='hunwenpinghao/{model_name}')
result = pipe('你好！', max_length=200)
print(result)
```

## 训练详情

- **基础模型**: Qwen2.5-0.5B-Instruct
- **微调方法**: LoRA/QLoRA
- **训练数据**: 自定义数据集
- **训练时间**: {time.strftime('%Y-%m-%d')}

## 许可证

Apache-2.0

## 引用

如果您使用了这个模型，请引用：

```bibtex
@misc{{{model_name.replace('-', '_')},
  author = {{hunwenpinghao}},
  title = {{{model_name}}},
  year = {{2025}},
  publisher = {{ModelScope}},
  journal = {{ModelScope Repository}},
  howpublished = {{\\url{{https://modelscope.cn/models/hunwenpinghao/{model_name}}}}}
}}
```
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)
    
    logger.info(f"已创建模型卡片: {readme_path}")

def upload_model_git(model_dir: str, repo_url: str, token: str):
    """使用 Git 方式上传模型"""
    try:
        from git import Repo
        
        # 创建临时目录
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"克隆仓库到临时目录: {temp_dir}")
            
            # 克隆仓库
            repo_url_with_token = repo_url.replace('https://www.modelscope.cn', f'https://oauth2:{token}@www.modelscope.cn')
            repo = Repo.clone_from(repo_url_with_token, temp_dir)
            
            # 复制模型文件
            import shutil
            for item in Path(model_dir).iterdir():
                if item.is_file():
                    shutil.copy2(item, Path(temp_dir) / item.name)
                    logger.info(f"复制文件: {item.name}")
            
            # 添加文件到 Git
            repo.git.add(A=True)
            
            # 提交更改
            try:
                repo.index.commit(f"Upload model files - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("提交更改成功")
            except Exception as e:
                logger.warning(f"提交时出现警告: {e}")
                # 如果没有更改，继续推送
            
            # 推送到远程仓库
            origin = repo.remote(name='origin')
            origin.push()
            logger.info("推送到远程仓库成功")
            
    except Exception as e:
        logger.error(f"Git 上传失败: {e}")
        raise

def upload_model_api(model_dir: str, model_id: str, token: str):
    """使用 ModelScope API 上传模型"""
    try:
        from modelscope.hub.api import HubApi
        
        # 创建 API 实例
        api = HubApi()
        api.login(token)
        logger.info("登录 ModelScope 成功")
        
        # 上传模型
        logger.info(f"开始上传模型到: {model_id}")
        api.push_model(
            model_id=model_id,
            model_dir=model_dir,
            commit_message=f"Upload model - {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("模型上传成功")
        
    except Exception as e:
        logger.error(f"API 上传失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="上传模型到魔搭社区")
    parser.add_argument("--model_dir", type=str, 
                       default="../deploy_ark/output_qwen/Qwen2.5-0.5B-Instruct",
                       help="模型文件目录")
    parser.add_argument("--model_id", type=str,
                       default="hunwenpinghao/zhc_xhs_qwen2.5_0.5b_instruct",
                       help="模型ID (用户名/模型名)")
    parser.add_argument("--token", type=str,
                       help="ModelScope 访问令牌")
    parser.add_argument("--method", type=str, choices=['api', 'git'], 
                       default='api',
                       help="上传方法: api 或 git")
    parser.add_argument("--description", type=str,
                       help="模型描述")
    parser.add_argument("--create_readme", action='store_true',
                       help="创建 README.md 文件")
    
    args = parser.parse_args()
    
    # 从环境变量获取 token
    if not args.token:
        args.token = os.getenv('MODELSCOPE_TOKEN')
    
    if not args.token:
        logger.error("请提供 ModelScope 访问令牌")
        logger.error("方法1: --token YOUR_TOKEN")
        logger.error("方法2: export MODELSCOPE_TOKEN=YOUR_TOKEN")
        logger.error("获取令牌: https://www.modelscope.cn/my/myaccesstoken")
        sys.exit(1)
    
    # 检查依赖
    check_dependencies()
    
    # 验证模型目录
    if not validate_model_directory(args.model_dir):
        sys.exit(1)
    
    # 创建模型卡片
    if args.create_readme:
        create_model_card(args.model_dir, args.model_id.split('/')[-1], args.description)
    
    try:
        logger.info(f"开始上传模型: {args.model_id}")
        logger.info(f"模型目录: {args.model_dir}")
        logger.info(f"上传方法: {args.method}")
        
        if args.method == 'api':
            upload_model_api(args.model_dir, args.model_id, args.token)
        else:
            repo_url = f"https://www.modelscope.cn/models/{args.model_id}.git"
            upload_model_git(args.model_dir, repo_url, args.token)
        
        logger.info("=" * 50)
        logger.info("🎉 模型上传成功！")
        logger.info(f"模型地址: https://www.modelscope.cn/models/{args.model_id}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"上传失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 