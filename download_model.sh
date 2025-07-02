#!/bin/bash
# 多源下载Qwen2.5-0.5B-Instruct模型 (支持HuggingFace镜像和ModelScope)

set -e

# 配置参数
MODEL_NAME="Qwen2.5-0.5B-Instruct"
MODEL_DIR="models/$MODEL_NAME"
HF_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
MS_MODEL_PATH="qwen/Qwen2.5-0.5B-Instruct"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色消息
print_info() { echo -e "${BLUE}$1${NC}"; }
print_success() { echo -e "${GREEN}$1${NC}"; }
print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_error() { echo -e "${RED}$1${NC}"; }

# 显示标题
echo "🚀 Qwen2.5-0.5B-Instruct 模型多源下载工具"
echo "=================================================="

# 检查现有模型
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    print_warning "⚠️  发现已存在的模型: $MODEL_DIR"
    echo -n "是否覆盖现有模型? (y/N): "
    read -r overwrite
    if [[ ! "$overwrite" =~ ^[Yy]$ ]]; then
        print_info "取消下载，使用现有模型"
        exit 0
    fi
    print_info "🗑️  清理现有模型..."
    rm -rf "$MODEL_DIR"
fi

# 选择下载源
echo ""
print_info "请选择下载源:"
echo "1) 🌐 HuggingFace镜像 (hf-mirror.com) - 推荐国内用户"
echo "2) 🇨🇳 ModelScope (modelscope.cn) - 国内官方源"
echo "3) 🔄 自动选择 (先试HF镜像，失败则用ModelScope)"
echo ""
echo -n "请输入选择 (1-3, 默认3): "
read -r choice

# 设置默认选择
choice=${choice:-3}

# 下载函数定义
download_with_hf_mirror() {
    print_info "📥 使用HuggingFace镜像下载..."
    
    local mirror_base="https://hf-mirror.com/$HF_MODEL_PATH/resolve/main"
    local files=(
        "config.json"
        "generation_config.json"
        "tokenizer.json"
        "tokenizer_config.json"
        "model.safetensors"
        "README.md"
    )
    
    # 创建目录
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    
    # 下载文件
    for file in "${files[@]}"; do
        print_info "⬇️  下载: $file"
        
        if command -v wget >/dev/null 2>&1; then
            if ! wget -q --show-progress --timeout=30 "$mirror_base/$file" -O "$file"; then
                print_error "❌ wget下载失败: $file"
                return 1
            fi
        elif command -v curl >/dev/null 2>&1; then
            if ! curl -L --connect-timeout 30 --max-time 300 "$mirror_base/$file" -o "$file"; then
                print_error "❌ curl下载失败: $file"
                return 1
            fi
        else
            print_error "❌ 错误: 需要安装 wget 或 curl"
            return 1
        fi
        
        # 验证文件
        if [ -f "$file" ] && [ -s "$file" ]; then
            print_success "✅ $file 下载成功"
        else
            print_error "❌ $file 下载失败或文件为空"
            return 1
        fi
    done
    
    cd - > /dev/null
    return 0
}

download_with_modelscope() {
    print_info "📥 使用ModelScope下载..."
    
    # 方法1: 尝试使用Python + ModelScope库
    if command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then
        print_info "🐍 尝试使用Python + ModelScope库..."
        
        # 确定Python命令
        PYTHON_CMD="python3"
        if ! command -v python3 >/dev/null 2>&1; then
            PYTHON_CMD="python"
        fi
        
        # 创建临时下载脚本
        cat > /tmp/download_with_modelscope.py << EOF
import os
import sys

try:
    from modelscope import snapshot_download
    
    model_name = "$MS_MODEL_PATH"
    cache_dir = "models"
    
    print(f"下载模型: {model_name}")
    print(f"保存到: {cache_dir}")
    
    model_dir = snapshot_download(model_name, cache_dir=cache_dir)
    print(f"✅ 下载成功: {model_dir}")
    
except ImportError:
    print("❌ ModelScope库未安装")
    print("安装命令: pip install modelscope")
    sys.exit(1)
except Exception as e:
    print(f"❌ 下载失败: {e}")
    sys.exit(1)
EOF
        
        if $PYTHON_CMD /tmp/download_with_modelscope.py; then
            rm -f /tmp/download_with_modelscope.py
            return 0
        else
            rm -f /tmp/download_with_modelscope.py
            print_warning "Python方式失败，尝试git方式..."
        fi
    fi
    
    # 方法2: 使用git clone
    if command -v git >/dev/null 2>&1; then
        print_info "📦 使用git克隆ModelScope仓库..."
        
        # 确保git-lfs可用
        if ! command -v git-lfs >/dev/null 2>&1; then
            print_warning "⚠️  git-lfs未安装，大文件可能下载不完整"
            print_info "安装命令: brew install git-lfs  # macOS"
            print_info "安装命令: apt install git-lfs   # Ubuntu"
        else
            git lfs install > /dev/null 2>&1
        fi
        
        # 克隆仓库
        local repo_url="https://www.modelscope.cn/$MS_MODEL_PATH.git"
        if git clone "$repo_url" "$MODEL_DIR"; then
            # 清理git信息
            rm -rf "$MODEL_DIR/.git"
            print_success "✅ git克隆成功"
            return 0
        else
            print_error "❌ git克隆失败"
            return 1
        fi
    else
        print_error "❌ git命令不可用"
        return 1
    fi
}

# 验证下载结果
verify_model() {
    local model_path="$1"
    
    print_info "🔍 验证模型文件..."
    
    local required_files=(
        "config.json"
        "tokenizer.json"
        "model.safetensors"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$model_path/$file" ]; then
            print_error "❌ 缺少必要文件: $file"
            return 1
        fi
        
        if [ ! -s "$model_path/$file" ]; then
            print_error "❌ 文件为空: $file"
            return 1
        fi
    done
    
    print_success "✅ 模型文件验证通过"
    return 0
}

# 执行下载
download_success=false

case $choice in
    1)
        print_info "选择: HuggingFace镜像"
        if download_with_hf_mirror; then
            download_success=true
        fi
        ;;
    2)
        print_info "选择: ModelScope"
        if download_with_modelscope; then
            download_success=true
        fi
        ;;
    3)
        print_info "选择: 自动选择"
        print_info "🔄 首先尝试HuggingFace镜像..."
        if download_with_hf_mirror; then
            download_success=true
        else
            print_warning "HuggingFace镜像失败，尝试ModelScope..."
            if download_with_modelscope; then
                download_success=true
            fi
        fi
        ;;
    *)
        print_error "❌ 无效选择"
        exit 1
        ;;
esac

# 检查下载结果
if [ "$download_success" = true ]; then
    if verify_model "$MODEL_DIR"; then
        print_success "🎉 模型下载完成！"
        echo ""
        print_info "📂 模型位置: $(realpath "$MODEL_DIR")"
        echo ""
        print_info "📋 下载的文件："
        ls -lh "$MODEL_DIR"
        echo ""
        print_info "🔧 接下来的步骤："
        echo "1. 运行配置更新: python update_config_for_local.py"
        echo "2. 开始训练: ./run_train.sh lora"
    else
        print_error "❌ 模型验证失败，请重新下载"
        exit 1
    fi
else
    print_error "❌ 所有下载方式都失败了"
    echo ""
    print_info "💡 建议的解决方案："
    echo "1. 检查网络连接"
    echo "2. 安装ModelScope: pip install modelscope"
    echo "3. 安装git-lfs: brew install git-lfs"
    echo "4. 使用代理或VPN"
    exit 1
fi 