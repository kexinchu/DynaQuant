#!/bin/bash

# SGLang混合精度服务器启动脚本
# 真正集成到SGLang架构中

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "SGLang混合精度服务器启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH          模型路径 (必需)"
    echo "  -c, --config PATH         混合精度配置文件路径 (必需)"
    echo "  -d, --device DEVICE       设备 (默认: cuda)"
    echo "  -t, --dtype TYPE          数据类型 (默认: auto)"
    echo "  -p, --port PORT           端口 (默认: 8080)"
    echo "  -h, --host HOST           主机 (默认: 127.0.0.1)"
    echo "  --test                    运行测试生成"
    echo "  --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -m /path/to/model -c mixed_precision_config.yaml"
    echo "  $0 -m /path/to/model -c mixed_precision_config.yaml --test"
    echo "  $0 -m /path/to/model -c mixed_precision_config.yaml -d cpu"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未找到"
        exit 1
    fi
    
    # 检查必要的Python包
    python3 -c "import torch" 2>/dev/null || {
        print_error "PyTorch 未安装"
        exit 1
    }
    
    python3 -c "import transformers" 2>/dev/null || {
        print_error "Transformers 未安装"
        exit 1
    }
    
    python3 -c "import yaml" 2>/dev/null || {
        print_error "PyYAML 未安装"
        exit 1
    }
    
    print_success "依赖检查通过"
}

# 检查文件是否存在
check_files() {
    local model_path="$1"
    local config_path="$2"
    
    print_info "检查文件..."
    
    if [[ ! -d "$model_path" ]] && [[ ! -f "$model_path" ]]; then
        print_error "模型路径不存在: $model_path"
        exit 1
    fi
    
    if [[ ! -f "$config_path" ]]; then
        print_error "配置文件不存在: $config_path"
        exit 1
    fi
    
    # 检查配置文件格式
    if ! python3 -c "import yaml; yaml.safe_load(open('$config_path'))" 2>/dev/null; then
        print_error "配置文件格式错误: $config_path"
        exit 1
    fi
    
    print_success "文件检查通过"
}

# 检查CUDA可用性
check_cuda() {
    local device="$1"
    
    if [[ "$device" == "cuda" ]]; then
        print_info "检查CUDA可用性..."
        
        if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            print_warning "CUDA不可用，将使用CPU"
            device="cpu"
        else
            print_success "CUDA可用"
        fi
    fi
    
    echo "$device"
}

# 主函数
main() {
    local model_path=""
    local config_path=""
    local device="cuda"
    local dtype="auto"
    local port="8080"
    local host="127.0.0.1"
    local test_mode=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model)
                model_path="$2"
                shift 2
                ;;
            -c|--config)
                config_path="$2"
                shift 2
                ;;
            -d|--device)
                device="$2"
                shift 2
                ;;
            -t|--dtype)
                dtype="$2"
                shift 2
                ;;
            -p|--port)
                port="$2"
                shift 2
                ;;
            -h|--host)
                host="$2"
                shift 2
                ;;
            --test)
                test_mode=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [[ -z "$model_path" ]]; then
        print_error "模型路径是必需的"
        show_help
        exit 1
    fi
    
    if [[ -z "$config_path" ]]; then
        print_error "配置文件路径是必需的"
        show_help
        exit 1
    fi
    
    # 检查依赖
    check_dependencies
    
    # 检查文件
    check_files "$model_path" "$config_path"
    
    # 检查CUDA
    device=$(check_cuda "$device")
    
    # 显示配置信息
    print_info "启动配置:"
    echo "  模型路径: $model_path"
    echo "  配置文件: $config_path"
    echo "  设备: $device"
    echo "  数据类型: $dtype"
    echo "  主机: $host"
    echo "  端口: $port"
    echo "  测试模式: $test_mode"
    
    # 构建Python命令
    local python_cmd="python3 launch_sglang_mixed_precision.py"
    python_cmd="$python_cmd --model '$model_path'"
    python_cmd="$python_cmd --mixed-precision-config '$config_path'"
    python_cmd="$python_cmd --device '$device'"
    python_cmd="$python_cmd --dtype '$dtype'"
    python_cmd="$python_cmd --host '$host'"
    python_cmd="$python_cmd --port '$port'"
    
    if [[ "$test_mode" == true ]]; then
        python_cmd="$python_cmd --test"
    fi
    
    print_info "启动SGLang混合精度服务器..."
    print_info "命令: $python_cmd"
    
    # 执行Python脚本
    eval "$python_cmd"
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
