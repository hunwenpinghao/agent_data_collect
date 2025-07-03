#!/bin/bash
# 火山引擎AICC一键部署脚本
# 交互式引导用户完成配置和部署

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_guide() {
    echo -e "${CYAN}[GUIDE]${NC} $1"
}

# 打印分隔线
print_separator() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# 打印标题
print_title() {
    echo ""
    print_separator
    echo -e "${CYAN}🚀 火山引擎AICC机密计算平台一键部署工具${NC}"
    print_separator
    echo ""
}

# 等待用户确认
wait_for_confirm() {
    echo ""
    read -p "$(echo -e ${YELLOW}请按 Enter 键继续，或输入 'q' 退出...${NC})" input
    if [[ "$input" == "q" || "$input" == "Q" ]]; then
        print_info "用户取消部署"
        exit 0
    fi
}

# 读取用户输入
read_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    if [[ -n "$default" ]]; then
        read -p "$(echo -e ${CYAN}$prompt [默认: $default]: ${NC})" input
        if [[ -z "$input" ]]; then
            input="$default"
        fi
    else
        read -p "$(echo -e ${CYAN}$prompt: ${NC})" input
    fi
    
    # 使用eval将值赋给指定的变量（兼容旧版bash）
    eval "$var_name=\"\$input\""
}

# 从user.csv文件读取火山引擎凭证
load_credentials_from_csv() {
    local csv_file="user.csv"
    local parent_csv="../user.csv"
    
    # 检查当前目录或上级目录的user.csv文件
    if [[ -f "$csv_file" ]]; then
        csv_file="$csv_file"
    elif [[ -f "$parent_csv" ]]; then
        csv_file="$parent_csv"
    else
        return 1
    fi
    
    print_info "检测到凭证文件: $csv_file"
    
    # 读取CSV文件的第二行（跳过标题行）
    local csv_line=$(sed -n '2p' "$csv_file")
    
    if [[ -z "$csv_line" ]]; then
        print_warn "CSV文件为空或格式不正确"
        return 1
    fi
    
    # 使用逗号分割CSV行
    IFS=',' read -ra FIELDS <<< "$csv_line"
    
    # 检查字段数量
    if [[ ${#FIELDS[@]} -lt 6 ]]; then
        print_warn "CSV文件格式不正确，字段数量不足"
        return 1
    fi
    
    # 提取凭证信息 (数组索引从0开始)
    # 第4列(索引3): 所属主账号ID
    # 第5列(索引4): Access Key ID  
    # 第6列(索引5): Secret Access Key
    local app_id="${FIELDS[3]}"
    local access_key="${FIELDS[4]}"
    local secret_key="${FIELDS[5]}"
    
    # 验证提取的信息
    if [[ -z "$app_id" || -z "$access_key" || -z "$secret_key" ]]; then
        print_warn "CSV文件中的凭证信息不完整"
        return 1
    fi
    
    # 设置环境变量
    export VOLCANO_APP_ID="$app_id"
    export VOLCANO_AK="$access_key"
    export VOLCANO_SK="$secret_key"
    
    print_info "✅ 成功从CSV文件加载火山引擎凭证"
    print_info "  AK: ${VOLCANO_AK:0:8}****"
    print_info "  SK: ${VOLCANO_SK:0:8}****"
    print_info "  APP_ID: $VOLCANO_APP_ID"
    
    return 0
}

# 检查火山引擎凭证配置
check_volcano_credentials() {
    print_step "步骤1: 检查火山引擎凭证配置"
    
    # 首先尝试从CSV文件加载凭证
    if load_credentials_from_csv; then
        print_info "✅ 火山引擎凭证已从CSV文件自动加载"
        return 0
    fi
    
    # 检查现有环境变量
    if [[ -n "$VOLCANO_AK" && -n "$VOLCANO_SK" && -n "$VOLCANO_APP_ID" ]]; then
        print_info "检测到现有的火山引擎凭证配置"
        echo "  AK: ${VOLCANO_AK:0:8}****"
        echo "  SK: ${VOLCANO_SK:0:8}****" 
        echo "  APP_ID: $VOLCANO_APP_ID"
        
        read -p "$(echo -e ${YELLOW}是否使用现有配置? [Y/n]: ${NC})" use_existing
        if [[ "$use_existing" != "n" && "$use_existing" != "N" ]]; then
            return 0
        fi
    fi
    
    print_warn "需要配置火山引擎凭证信息"
    print_guide "请按照以下步骤获取火山引擎凭证："
    echo ""
    echo "1. 登录火山引擎控制台: https://console.volcengine.com/"
    echo "2. 点击右上角头像 -> API访问密钥"
    echo "3. 新建访问密钥，获取 Access Key 和 Secret Key"
    echo "4. 在控制台右上角点击头像，查看账号ID"
    echo ""
    echo "💡 提示: 您也可以创建 user.csv 文件来自动加载凭证信息"
    echo ""
    
    wait_for_confirm
    
    # 输入凭证信息
    print_info "请输入火山引擎凭证信息:"
    read_input "Access Key" "" "VOLCANO_AK"
    read_input "Secret Key" "" "VOLCANO_SK"
    read_input "账号ID (APP_ID)" "" "VOLCANO_APP_ID"
    
    # 验证输入
    if [[ -z "$VOLCANO_AK" || -z "$VOLCANO_SK" || -z "$VOLCANO_APP_ID" ]]; then
        print_error "凭证信息不能为空"
        exit 1
    fi
    
    # 导出环境变量
    export VOLCANO_AK="$VOLCANO_AK"
    export VOLCANO_SK="$VOLCANO_SK"
    export VOLCANO_APP_ID="$VOLCANO_APP_ID"
    
    print_info "✅ 火山引擎凭证配置完成"
}

# 检查TOS存储桶配置
check_tos_bucket() {
    print_step "步骤2: 检查TOS存储桶配置"
    
    if [[ -n "$VOLCANO_BUCKET_NAME" ]]; then
        print_info "检测到现有的TOS存储桶: $VOLCANO_BUCKET_NAME"
        read -p "$(echo -e ${YELLOW}是否使用现有存储桶? [Y/n]: ${NC})" use_existing
        if [[ "$use_existing" != "n" && "$use_existing" != "N" ]]; then
            return 0
        fi
    fi
    
    # 生成推荐的存储桶名称
    local username=$(whoami)
    local timestamp=$(date +"%Y%m")
    local recommended_bucket="aicc-models-${username}-${timestamp}"
    
    print_info "推荐使用存储桶名称: $recommended_bucket"
    read -p "$(echo -e ${YELLOW}是否使用推荐的存储桶名称? [Y/n]: ${NC})" use_recommended
    
    if [[ "$use_recommended" != "n" && "$use_recommended" != "N" ]]; then
        export VOLCANO_BUCKET_NAME="$recommended_bucket"
        print_info "✅ 使用推荐存储桶: $VOLCANO_BUCKET_NAME"
        print_warn "请确保在火山引擎控制台创建此存储桶（华北2-北京地区）"
        return 0
    fi
    
    print_warn "需要配置TOS对象存储桶"
    print_guide "请按照以下步骤创建TOS存储桶："
    echo ""
    echo "1. 登录火山引擎控制台: https://console.volcengine.com/"
    echo "2. 进入 '对象存储TOS' 服务"
    echo "3. 点击 '创建存储桶'"
    echo "4. 配置存储桶:"
    echo "   - 存储桶名称: 全局唯一的名称"
    echo "   - 地域: 选择 '华北2（北京）'"
    echo "   - 访问权限: 私有读写"
    echo "5. 点击创建"
    echo ""
    
    wait_for_confirm
    
    read_input "TOS存储桶名称" "$recommended_bucket" "VOLCANO_BUCKET_NAME"
    
    if [[ -z "$VOLCANO_BUCKET_NAME" ]]; then
        print_error "存储桶名称不能为空"
        exit 1
    fi
    
    export VOLCANO_BUCKET_NAME="$VOLCANO_BUCKET_NAME"
    print_info "✅ TOS存储桶配置完成"
}

# 自动下载和安装Jeddak SDK
auto_install_jeddak_sdk() {
    local sdk_url="https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl"
    local sdk_file="bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl"
    
    print_info "🚀 开始自动下载安装Jeddak SDK..."
    print_info "下载地址: $sdk_url"
    
    # 步骤1: 检查并安装依赖
    print_info "📋 检查依赖环境..."
    local dependencies=("cryptography>=38.0.0" "requests>=2.22" "typing_extensions>=4.12")
    
    for dep in "${dependencies[@]}"; do
        print_info "检查依赖: $dep"
        if ! pip install "$dep" >/dev/null 2>&1; then
            print_warn "依赖安装可能有问题: $dep"
        fi
    done
    
    # 步骤2: 下载SDK
    print_info "📦 正在下载SDK文件..."
    
    # 检查是否有下载工具
    if command -v curl >/dev/null 2>&1; then
        print_info "使用curl下载SDK..."
        if curl -L -o "$sdk_file" "$sdk_url"; then
            print_info "✅ SDK下载成功"
        else
            print_warn "❌ 使用curl下载失败"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        print_info "使用wget下载SDK..."
        if wget -O "$sdk_file" "$sdk_url"; then
            print_info "✅ SDK下载成功"
        else
            print_warn "❌ 使用wget下载失败"
            return 1
        fi
    else
        print_warn "❌ 系统中没有找到curl或wget，无法自动下载"
        print_warn "请安装curl: brew install curl (macOS) 或 apt-get install curl (Ubuntu)"
        return 1
    fi
    
    # 验证下载的文件
    if [[ ! -f "$sdk_file" ]]; then
        print_warn "❌ 下载的SDK文件不存在"
        return 1
    fi
    
    local file_size=$(wc -c < "$sdk_file" 2>/dev/null || echo "0")
    if [[ "$file_size" -lt 10000 ]]; then
        print_warn "❌ 下载的SDK文件大小异常（可能下载失败）"
        rm -f "$sdk_file"
        return 1
    fi
    
    print_info "SDK文件大小: $file_size 字节"
    
    # 步骤3: 安装SDK（带重试机制）
    print_info "🔧 开始安装SDK..."
    local install_success=false
    
    for attempt in 1 2; do
        if [[ $attempt -gt 1 ]]; then
            print_info "第${attempt}次安装尝试..."
            # 尝试更新pip
            pip install --upgrade pip >/dev/null 2>&1
        fi
        
        if pip install --force-reinstall "$sdk_file"; then
            install_success=true
            break
        else
            print_warn "第${attempt}次安装失败"
        fi
    done
    
    # 清理下载文件
    rm -f "$sdk_file"
    
    if [[ "$install_success" == "false" ]]; then
        print_warn "❌ SDK安装失败"
        print_info "💡 可能的解决方案:"
        print_info "1. 手动安装: pip install --upgrade pip"
        print_info "2. 检查网络连接"
        print_info "3. 使用虚拟环境"
        return 1
    fi
    
    print_info "✅ SDK安装成功"
    
    # 步骤4: 验证安装
    if python3 -c "from bytedance.jeddak_secure_model.model_encryption import JeddakModelEncrypter" 2>/dev/null; then
        print_info "✅ SDK安装验证成功"
        return 0
    else
        print_warn "❌ SDK安装验证失败"
        print_info "💡 建议重启终端后重试"
        return 1
    fi
}

# 检查Jeddak SDK
check_jeddak_sdk() {
    print_step "步骤3: 检查Jeddak SDK安装"
    
    # 检查SDK是否已安装
    if python3 -c "from bytedance.jeddak_secure_model.model_encryption import JeddakModelEncrypter" 2>/dev/null; then
        print_info "✅ Jeddak SDK 已安装"
        return 0
    fi
    
    print_warn "Jeddak SDK 未安装"
    print_info "🚀 尝试自动下载并安装Jeddak SDK..."
    
    # 尝试自动安装
    if auto_install_jeddak_sdk; then
        print_info "🎉 Jeddak SDK 自动安装成功！"
        return 0
    fi
    
    # 如果自动安装失败，提供手动安装选项
    print_warn "自动安装失败，提供手动安装选项："
    print_guide "手动安装方法："
    echo ""
    echo "1. 访问火山引擎文档: https://www.volcengine.com/docs/85010/1546894"
    echo "2. 下载 Jeddak Secure Model SDK"
    echo "3. 当前推荐版本: 0.1.7.36"
    echo ""
    echo "或者直接下载:"
    echo "  curl -L -O https://lf3-static.bytednsdoc.com/obj/eden-cn/jzeh7vhobenuhog/bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl"
    echo "  pip install bytedance_jeddak_secure_channel-0.1.7.36-py3-none-any.whl"
    echo ""
    
    read -p "$(echo -e ${YELLOW}是否继续使用模拟模式进行部署? [Y/n]: ${NC})" use_simulation
    
    if [[ "$use_simulation" == "n" || "$use_simulation" == "N" ]]; then
        print_error "用户选择退出部署"
        exit 1
    else
        print_warn "将使用模拟模式进行部署（仅用于测试）"
        print_info "💡 提示: 模拟模式不会进行真实的模型加密，仅用于测试部署流程"
    fi
}

# 检查模型文件
check_model_files() {
    print_step "步骤4: 检查模型文件"
    
    # 检查默认路径
    local default_paths=("./output_qwen" "./models" "./fine_tuned_model")
    local found_path=""
    
    for path in "${default_paths[@]}"; do
        if [[ -d "$path" ]]; then
            found_path="$path"
            break
        fi
    done
    
    if [[ -n "$found_path" ]]; then
        print_info "检测到模型目录: $found_path"
        read -p "$(echo -e ${YELLOW}是否使用此模型目录? [Y/n]: ${NC})" use_found
        if [[ "$use_found" != "n" && "$use_found" != "N" ]]; then
            MODEL_PATH="$found_path"
            export MODEL_PATH
            print_info "✅ 使用模型路径: $MODEL_PATH"
            return 0
        fi
    fi
    
    print_guide "请指定模型文件路径:"
    echo ""
    echo "模型目录应包含以下文件:"
    echo "  - config.json"
    echo "  - pytorch_model.bin 或 model.safetensors"
    echo "  - tokenizer.json 或 tokenizer_config.json"
    echo "  - (可选) adapter_config.json (LoRA模型)"
    echo ""
    
    read_input "模型文件路径" "./output_qwen" "MODEL_PATH"
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        print_error "模型目录不存在: $MODEL_PATH"
        exit 1
    fi
    
    export MODEL_PATH
    print_info "✅ 模型路径配置完成: $MODEL_PATH"
}

# 选择AICC规格
select_aicc_spec() {
    print_step "步骤5: 选择AICC规格"
    
    echo ""
    echo "请选择AICC规格:"
    echo "1. 基础版 - 适用于小尺寸模型（如1.5B）"
    echo "2. 高级版 - 适用于中尺寸模型（如32B）"
    echo "3. 旗舰版 - 适用于大尺寸模型（如DeepSeek R1-671B）"
    echo ""
    
    read -p "$(echo -e ${CYAN}请选择规格 [1-3, 默认: 2]: ${NC})" spec_choice
    
    case "$spec_choice" in
        "1")
            AICC_SPEC="基础版"
            ;;
        "3")
            AICC_SPEC="旗舰版"
            ;;
        *)
            AICC_SPEC="高级版"
            ;;
    esac
    
    export AICC_SPEC
    print_info "✅ 选择AICC规格: $AICC_SPEC"
}

# 配置模型名称
configure_model_name() {
    print_step "步骤6: 配置模型名称"
    
    local default_name=$(basename "$MODEL_PATH")
    if [[ "$default_name" == "." ]]; then
        default_name="qwen-finetune"
    fi
    
    read_input "模型名称" "$default_name" "MODEL_NAME"
    
    export MODEL_NAME
    print_info "✅ 模型名称配置完成: $MODEL_NAME"
}

# 显示配置摘要
show_config_summary() {
    print_step "配置摘要"
    
    echo ""
    echo "部署配置信息:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔐 火山引擎AK:     ${VOLCANO_AK:0:8}****"
    echo "🔐 火山引擎SK:     ${VOLCANO_SK:0:8}****"
    echo "🆔 账号ID:         $VOLCANO_APP_ID"
    echo "🪣 TOS存储桶:      $VOLCANO_BUCKET_NAME"
    echo "📂 模型路径:       $MODEL_PATH"
    echo "🏷️  模型名称:       $MODEL_NAME"
    echo "⚙️  AICC规格:       $AICC_SPEC"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    read -p "$(echo -e ${YELLOW}确认以上配置正确，开始部署? [Y/n]: ${NC})" confirm_deploy
    if [[ "$confirm_deploy" == "n" || "$confirm_deploy" == "N" ]]; then
        print_info "用户取消部署"
        exit 0
    fi
}

# 执行部署
execute_deployment() {
    print_step "开始AICC部署"
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_file="${MODEL_NAME}_deployment_${timestamp}.json"
    
    print_info "🚀 启动火山引擎AICC部署流程..."
    print_info "📊 部署结果将保存到: $output_file"
    
    echo ""
    
    # 执行部署命令
    if python3 ark_deploy.py deploy \
        --config aicc_config.json \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --aicc_spec "$AICC_SPEC" \
        --bucket_name "$VOLCANO_BUCKET_NAME" \
        --output "$output_file"; then
        
        print_separator
        print_info "🎉 部署成功完成！"
        print_separator
        
        if [[ -f "$output_file" ]]; then
            print_info "📋 部署结果文件: $output_file"
            
            # 提取关键信息
            if command -v jq >/dev/null 2>&1; then
                echo ""
                echo "🔗 推理端点信息:"
                jq -r '.endpoint_info.endpoint_url // "未知"' "$output_file" 2>/dev/null || echo "  无法解析端点信息"
                echo ""
                echo "🆔 部署ID:"
                jq -r '.model_info.deployment_id // "未知"' "$output_file" 2>/dev/null || echo "  无法解析部署ID"
            else
                echo ""
                echo "💡 提示: 安装 jq 工具可以更好地显示部署结果"
                echo "   安装命令: brew install jq (macOS) 或 apt install jq (Ubuntu)"
            fi
        fi
        
        echo ""
        echo "🎯 下一步操作:"
        echo "  1. 查看部署结果: cat $output_file"
        echo "  2. 测试推理服务: python3 test_concurrent.py --endpoint <推理端点>"
        echo "  3. 监控服务状态: python3 ark_deploy.py info --deployment_id <部署ID>"
        
    else
        print_error "❌ 部署失败"
        echo ""
        echo "🔍 故障排除建议:"
        echo "  1. 检查网络连接"
        echo "  2. 验证火山引擎凭证"
        echo "  3. 确认TOS存储桶权限"
        echo "  4. 查看详细日志"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    print_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip >/dev/null 2>&1 && ! python3 -m pip --version >/dev/null 2>&1; then
        print_error "pip 未安装"
        exit 1
    fi
    
    # 检查必要的Python包
    if ! python3 -c "import requests" 2>/dev/null; then
        print_info "安装Python依赖..."
        pip install -r requirements.txt
    fi
    
    print_info "✅ 依赖检查完成"
}

# 主函数
main() {
    print_title
    
    print_info "欢迎使用火山引擎AICC机密计算平台一键部署工具！"
    print_info "此工具将引导您完成模型的安全部署流程。"
    
    wait_for_confirm
    
    # 执行部署步骤
    check_dependencies
    check_volcano_credentials
    check_tos_bucket
    check_jeddak_sdk
    check_model_files
    select_aicc_spec
    configure_model_name
    show_config_summary
    execute_deployment
    
    echo ""
    print_separator
    print_info "🎊 感谢使用火山引擎AICC一键部署工具！"
    print_separator
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 