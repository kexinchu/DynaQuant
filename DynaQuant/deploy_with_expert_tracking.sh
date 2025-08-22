#!/bin/bash

# 混合精度模型部署脚本（包含专家激活跟踪）
# 作者: AI Assistant
# 版本: 2.0

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8081"
DEFAULT_WORKERS="4"
DEFAULT_CONFIG="config/model_config.yaml"

# 函数定义
print_header() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  混合精度模型部署系统 (专家激活跟踪版)"
    echo "=========================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[步骤 $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

check_python_version() {
    print_step "1" "检查Python版本..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python版本: $PYTHON_VERSION"
        else
            print_error "需要Python 3.8或更高版本，当前版本: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "未找到Python3，请先安装Python 3.8+"
        exit 1
    fi
}

check_cuda() {
    print_step "2" "检查CUDA环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
        print_success "CUDA版本: $CUDA_VERSION"
        
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        print_success "GPU数量: $GPU_COUNT"
        
        # 显示GPU信息
        echo "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    else
        print_warning "未检测到NVIDIA GPU，将使用CPU模式"
    fi
}

install_dependencies() {
    print_step "3" "安装依赖包..."
    
    if [ -f "requirements.txt" ]; then
        print_success "找到requirements.txt，开始安装依赖..."
        pip3 install -r requirements.txt
        print_success "依赖安装完成"
    else
        print_error "未找到requirements.txt文件"
        exit 1
    fi
}

check_config() {
    print_step "4" "检查配置文件..."
    
    if [ -f "$DEFAULT_CONFIG" ]; then
        print_success "找到配置文件: $DEFAULT_CONFIG"
        
        # 检查配置文件格式
        if command -v python3 &> /dev/null; then
            python3 -c "import yaml; yaml.safe_load(open('$DEFAULT_CONFIG'))" 2>/dev/null
            if [ $? -eq 0 ]; then
                print_success "配置文件格式正确"
            else
                print_error "配置文件格式错误"
                exit 1
            fi
        fi
    else
        print_warning "未找到配置文件: $DEFAULT_CONFIG"
        print_warning "将使用默认配置"
    fi
}

create_directories() {
    print_step "5" "创建必要目录..."
    
    mkdir -p logs
    mkdir -p expert_stats_plots
    mkdir -p config
    
    print_success "目录创建完成"
}

setup_environment() {
    print_step "6" "设置环境变量..."
    
    # 设置CUDA相关环境变量
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    
    # 设置Python路径
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    # 设置日志级别
    export LOG_LEVEL=INFO
    
    print_success "环境变量设置完成"
}

start_server() {
    print_step "7" "启动混合精度模型服务器..."
    
    local host=${1:-$DEFAULT_HOST}
    local port=${2:-$DEFAULT_PORT}
    local workers=${3:-$DEFAULT_WORKERS}
    local config=${4:-$DEFAULT_CONFIG}
    
    print_success "启动参数:"
    echo "  主机: $host"
    echo "  端口: $port"
    echo "  工作进程: $workers"
    echo "  配置文件: $config"
    
    # 启动服务器
    nohup python3 main.py \
        --config "$config" \
        --host "$host" \
        --port "$port" \
        --workers "$workers" \
        > logs/server.log 2>&1 &
    
    SERVER_PID=$!
    echo $SERVER_PID > logs/server.pid
    
    print_success "服务器已启动，PID: $SERVER_PID"
    print_success "日志文件: logs/server.log"
    
    # 仅当健康检查通过后才退出，否则每10s重试
    print_step "等待服务器健康检查(/health)通过..."
    while true; do
        # 健康检查（加超时，避免 curl 挂死）
        if curl -fsS --connect-timeout 2 --max-time 50 "http://$host:$port/health" > /dev/null; then
            print_success "服务器启动成功，健康检查通过"
            break
        fi

        # 若进程已退出，立即报错返回
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            print_warning "服务器进程已退出（PID: $SERVER_PID）。请检查 logs/server.log"
            return 1
        fi

        # 未就绪则等待 10s 后重试
        sleep 60
    done
}

run_tests() {
    print_step "8" "运行功能测试..."
    
    # 等待服务器完全启动
    sleep 10
    
    # 运行基本功能测试
    if [ -f "examples/test_client.py" ]; then
        print_success "运行基本功能测试..."
        python3 examples/test_client.py 8081
    fi
    
    # 运行专家激活跟踪测试
    if [ -f "examples/test_expert_tracking.py" ]; then
        print_success "运行专家激活跟踪测试..."
        python3 examples/test_expert_tracking.py 8081
    fi
    
    print_success "测试完成"
}

generate_visualization() {
    print_step "9" "生成专家激活统计可视化..."
    
    if [ -f "examples/visualize_expert_stats.py" ]; then
        print_success "生成可视化图表..."
        python3 examples/visualize_expert_stats.py 8081
        
        if [ -d "expert_stats_plots" ]; then
            print_success "可视化图表已保存到 expert_stats_plots/ 目录"
            echo "生成的文件:"
            ls -la expert_stats_plots/
        fi
    else
        print_warning "未找到可视化脚本"
    fi
}

show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --host HOST      服务器主机地址 (默认: $DEFAULT_HOST)"
    echo "  -p, --port PORT      服务器端口 (默认: $DEFAULT_PORT)"
    echo "  -w, --workers NUM    工作进程数 (默认: $DEFAULT_WORKERS)"
    echo "  -c, --config FILE    配置文件路径 (默认: $DEFAULT_CONFIG)"
    echo "  --install-only       仅安装依赖，不启动服务器"
    echo "  --test-only          仅运行测试，不启动服务器"
    echo "  --visualize-only     仅生成可视化，不启动服务器"
    echo "  --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置启动"
    echo "  $0 -h 0.0.0.0 -p 9000 -w 8           # 自定义配置启动"
    echo "  $0 --install-only                     # 仅安装依赖"
    echo "  $0 --test-only                        # 仅运行测试"
}

parse_args() {
    HOST=$DEFAULT_HOST
    PORT=$DEFAULT_PORT
    WORKERS=$DEFAULT_WORKERS
    CONFIG=$DEFAULT_CONFIG
    INSTALL_ONLY=false
    TEST_ONLY=false
    VISUALIZE_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--host)
                HOST="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            -w|--workers)
                WORKERS="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG="$2"
                shift 2
                ;;
            --install-only)
                INSTALL_ONLY=true
                shift
                ;;
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --visualize-only)
                VISUALIZE_ONLY=true
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
}

main() {
    print_header
    
    # 解析命令行参数
    parse_args "$@"
    
    # 检查Python版本
    check_python_version
    
    # 检查CUDA环境
    check_cuda
    
    # 安装依赖
    # install_dependencies
    
    # 检查配置文件
    check_config
    
    # 创建目录
    create_directories
    
    # 设置环境
    setup_environment
    
    if [ "$INSTALL_ONLY" = true ]; then
        print_success "依赖安装完成，退出"
        exit 0
    fi
    
    if [ "$TEST_ONLY" = true ]; then
        run_tests
        exit 0
    fi
    
    if [ "$VISUALIZE_ONLY" = true ]; then
        generate_visualization
        exit 0
    fi
    
    # 启动服务器
    start_server "$HOST" "$PORT" "$WORKERS" "$CONFIG"
    
    # 运行测试
    run_tests
    
    # 生成可视化
    generate_visualization
    
    print_success "部署完成！"
    echo ""
    echo "服务器信息:"
    echo "  地址: http://$HOST:$PORT"
    echo "  健康检查: http://$HOST:$PORT/health"
    echo "  专家统计: http://$HOST:$PORT/expert_stats"
    echo ""
    echo "日志文件: logs/server.log"
    echo "可视化图表: expert_stats_plots/"
    echo ""
    echo "停止服务器: kill \$(cat logs/server.pid)"
}

# 运行主函数
main "$@"
