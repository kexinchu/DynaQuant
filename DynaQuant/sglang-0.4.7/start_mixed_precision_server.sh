#!/bin/bash

# SGLang混合精度服务器启动脚本

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

# 默认参数
MODEL_PATH="/dcar-vepfs-trans-models/Qwen3-235B-A22B"
MIXED_PRECISION_CONFIG="mixed_precision_config.yaml"
HOST="127.0.0.1"
PORT="8080"
TP_SIZE=1
DP_SIZE=4
MAX_RUNNING_REQUESTS=32
MAX_TOTAL_TOKENS=40960
DTYPE="bfloat16"
ENABLE_MIXED_PRECISION=false

# 显示帮助信息
show_help() {
    echo "SGLang混合精度服务器启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model-path PATH          模型路径 (默认: $MODEL_PATH)"
    echo "  --mixed-precision-config PATH  混合精度配置文件 (默认: $MIXED_PRECISION_CONFIG)"
    echo "  --enable-mixed-precision   启用混合精度加载"
    echo "  --host HOST                服务器主机地址 (默认: $HOST)"
    echo "  --port PORT                服务器端口 (默认: $PORT)"
    echo "  --tp-size SIZE             张量并行大小 (默认: $TP_SIZE)"
    echo "  --dp-size SIZE             数据并行大小 (默认: $DP_SIZE)"
    echo "  --max-running-requests N   最大运行请求数 (默认: $MAX_RUNNING_REQUESTS)"
    echo "  --max-total-tokens N       最大总token数 (默认: $MAX_TOTAL_TOKENS)"
    echo "  --dtype DTYPE              数据类型 (默认: $DTYPE)"
    echo "  --help                     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --enable-mixed-precision --host 0.0.0.0 --port 8080"
    echo "  $0 --model-path /path/to/model --mixed-precision-config config.yaml --enable-mixed-precision"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mixed-precision-config)
            MIXED_PRECISION_CONFIG="$2"
            shift 2
            ;;
        --enable-mixed-precision)
            ENABLE_MIXED_PRECISION=true
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --dp-size)
            DP_SIZE="$2"
            shift 2
            ;;
        --max-running-requests)
            MAX_RUNNING_REQUESTS="$2"
            shift 2
            ;;
        --max-total-tokens)
            MAX_TOTAL_TOKENS="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    print_error "模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 检查混合精度配置文件
if [ "$ENABLE_MIXED_PRECISION" = true ] && [ ! -f "$MIXED_PRECISION_CONFIG" ]; then
    print_warning "混合精度配置文件不存在: $MIXED_PRECISION_CONFIG"
    print_info "将使用标准模式启动"
    ENABLE_MIXED_PRECISION=false
fi

# 显示启动信息
print_info "=" * 60
print_info "SGLang混合精度服务器启动"
print_info "=" * 60
print_info "模型路径: $MODEL_PATH"
print_info "混合精度配置: $MIXED_PRECISION_CONFIG"
print_info "启用混合精度: $ENABLE_MIXED_PRECISION"
print_info "服务器地址: $HOST:$PORT"
print_info "张量并行大小: $TP_SIZE"
print_info "数据并行大小: $DP_SIZE"
print_info "最大运行请求数: $MAX_RUNNING_REQUESTS"
print_info "最大总token数: $MAX_TOTAL_TOKENS"
print_info "数据类型: $DTYPE"
print_info "=" * 60

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# 构建启动命令
LAUNCH_CMD="python3 launch_mixed_precision_server.py"

if [ "$ENABLE_MIXED_PRECISION" = true ]; then
    LAUNCH_CMD="$LAUNCH_CMD --enable-mixed-precision --mixed-precision-config $MIXED_PRECISION_CONFIG"
fi

LAUNCH_CMD="$LAUNCH_CMD --model-path $MODEL_PATH"
LAUNCH_CMD="$LAUNCH_CMD --tp-size $TP_SIZE"
LAUNCH_CMD="$LAUNCH_CMD --dp-size $DP_SIZE"
LAUNCH_CMD="$LAUNCH_CMD --max-running-requests $MAX_RUNNING_REQUESTS"
LAUNCH_CMD="$LAUNCH_CMD --host $HOST"
LAUNCH_CMD="$LAUNCH_CMD --port $PORT"
LAUNCH_CMD="$LAUNCH_CMD --max-total-tokens $MAX_TOTAL_TOKENS"
LAUNCH_CMD="$LAUNCH_CMD --dtype $DTYPE"
LAUNCH_CMD="$LAUNCH_CMD --trust-remote-code"
LAUNCH_CMD="$LAUNCH_CMD --attention-backend torch_native"
LAUNCH_CMD="$LAUNCH_CMD --sampling-backend pytorch"
LAUNCH_CMD="$LAUNCH_CMD --disable-cuda-graph"
LAUNCH_CMD="$LAUNCH_CMD --disable-cuda-graph-padding"
LAUNCH_CMD="$LAUNCH_CMD --kv-cache-dtype auto"
LAUNCH_CMD="$LAUNCH_CMD --allow-auto-truncate"
LAUNCH_CMD="$LAUNCH_CMD --chunked-prefill-size 16384"

# 启动服务器
print_info "启动服务器..."
print_info "命令: $LAUNCH_CMD"

eval $LAUNCH_CMD
