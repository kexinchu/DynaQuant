#!/bin/bash

# 混合精度Transformer模型部署脚本

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

# 检查Python版本
check_python_version() {
    print_info "检查Python版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python版本: $PYTHON_VERSION"
    else
        print_error "Python3未安装，请先安装Python3"
        exit 1
    fi
}

# 检查CUDA
check_cuda() {
    print_info "检查CUDA环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        print_success "CUDA驱动版本: $CUDA_VERSION"
        
        # 检查GPU
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        print_success "GPU数量: $GPU_COUNT"
    else
        print_warning "CUDA未检测到，将使用CPU模式"
    fi
}

# 安装依赖
install_dependencies() {
    print_info "安装Python依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "依赖安装完成"
    else
        print_error "requirements.txt文件不存在"
        exit 1
    fi
}

# 检查配置文件
check_config() {
    print_info "检查配置文件..."
    
    if [ ! -f "config/model_config.yaml" ]; then
        print_error "配置文件 config/model_config.yaml 不存在"
        print_info "请根据README.md中的说明创建配置文件"
        exit 1
    fi
    
    print_success "配置文件检查通过"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    mkdir -p logs
    mkdir -p models
    mkdir -p data
    
    print_success "目录创建完成"
}

# 设置环境变量
setup_environment() {
    print_info "设置环境变量..."
    
    # 设置CUDA相关环境变量
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    # 设置Python路径
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    
    # 设置日志级别
    export LOG_LEVEL=INFO
    
    print_success "环境变量设置完成"
}

# 启动服务器
start_server() {
    print_info "启动混合精度Transformer模型服务器..."
    
    # 获取命令行参数
    CONFIG_FILE=${1:-"config/model_config.yaml"}
    HOST=${2:-"127.0.0.1"}
    PORT=${3:-"8080"}
    WORKERS=${4:-"4"}
    
    print_info "配置参数:"
    print_info "  配置文件: $CONFIG_FILE"
    print_info "  服务器地址: $HOST:$PORT"
    print_info "  工作线程数: $WORKERS"
    
    # 启动服务器
    python3 main.py \
        --config "$CONFIG_FILE" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        2>&1 | tee logs/server.log
}

# 运行测试
run_tests() {
    print_info "运行测试..."
    
    if [ -f "examples/test_client.py" ]; then
        python3 examples/test_client.py
        print_success "测试完成"
    else
        print_warning "测试文件不存在，跳过测试"
    fi
}

# 显示帮助信息
show_help() {
    echo "混合精度Transformer模型部署脚本"
    echo ""
    echo "用法: $0 [命令] [参数]"
    echo ""
    echo "命令:"
    echo "  install    安装依赖和初始化环境"
    echo "  start      启动服务器"
    echo "  test       运行测试"
    echo "  help       显示帮助信息"
    echo ""
    echo "参数:"
    echo "  --config   配置文件路径 (默认: config/model_config.yaml)"
    echo "  --host     服务器主机地址 (默认: 127.0.0.1)"
    echo "  --port     服务器端口 (默认: 8080)"
    echo "  --workers  工作线程数 (默认: 4)"
    echo ""
    echo "示例:"
    echo "  $0 install"
    echo "  $0 start --config config/model_config.yaml --host 0.0.0.0 --port 8080"
    echo "  $0 test"
}

# 解析命令行参数
parse_args() {
    CONFIG_FILE="config/model_config.yaml"
    HOST="127.0.0.1"
    PORT="8080"
    WORKERS="4"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
}

# 主函数
main() {
    case "${1:-help}" in
        install)
            print_info "开始安装和初始化..."
            check_python_version
            check_cuda
            install_dependencies
            check_config
            create_directories
            setup_environment
            print_success "安装和初始化完成"
            ;;
        start)
            parse_args "$@"
            check_python_version
            check_cuda
            check_config
            setup_environment
            start_server "$CONFIG_FILE" "$HOST" "$PORT" "$WORKERS"
            ;;
        test)
            check_python_version
            setup_environment
            run_tests
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"
