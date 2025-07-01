# 使用官方 PyTorch CUDA 镜像作为基础镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/app/.cache/huggingface
ENV MODELSCOPE_CACHE=/app/.cache/modelscope

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --upgrade pip

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs output_qwen .cache/huggingface .cache/modelscope

# 设置权限
RUN chmod +x run_train.sh

# 暴露端口（用于TensorBoard和可能的推理服务）
EXPOSE 6006 8000

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "=== Qwen3-0.5B 微调容器已启动 ==="\n\
echo "可用命令:"\n\
echo "  训练模型: ./run_train.sh"\n\
echo "  交互式推理: python3 inference.py --model_path ./output_qwen --interactive"\n\
echo "  启动TensorBoard: tensorboard --logdir ./output_qwen/runs --host 0.0.0.0"\n\
echo "  查看帮助: python3 fine_tune_qwen.py --help"\n\
echo "==================================="\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 设置入口点
ENTRYPOINT ["/app/entrypoint.sh"]

# 默认命令
CMD ["/bin/bash"] 