#!/bin/bash

# 兼容SGLang原生启动方式的混合精度启动脚本
# 支持TP/DP/EP等SGLang原生功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
MODEL_PATH=""
MIXED_PRECISION_CONFIG=""
TP_SIZE=4
DP_SIZE=2
EP_SIZE=1
HOST="127.0.0.1"
PORT=8080
DTYPE="bfloat16"
MAX_RUNNING_REQUESTS=32
MAX_TOTAL_TOKENS=40960
ENABLE_MIXED_PRECISION=false

# 显示帮助信息
show_help() {
    echo -e "${BLUE}兼容SGLang原生启动方式的混合精度启动脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH              模型路径 (必需)"
    echo "  -c, --config PATH             混合精度配置文件路径"
    echo "  -t, --tp-size SIZE            张量并行大小 (默认: 4)"
    echo "  -d, --dp-size SIZE            数据并行大小 (默认: 2)"
    echo "  -e, --ep-size SIZE            专家并行大小 (默认: 1)"
    echo "  --host HOST                   服务器主机 (默认: 127.0.0.1)"
    echo "  --port PORT                   服务器端口 (默认: 8080)"
    echo "  --dtype TYPE                  数据类型 (默认: bfloat16)"
    echo "  --max-running-requests NUM    最大运行请求数 (默认: 32)"
    echo "  --max-total-tokens NUM        最大总token数 (默认: 40960)"
    echo "  --enable-mixed-precision      启用混合精度加载"
    echo "  -h, --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 标准SGLang启动 (无混合精度)"
    echo "  $0 -m /path/to/model -t 4 -d 2"
    echo ""
    echo "  # 启用混合精度的SGLang启动"
    echo "  $0 -m /path/to/model -c mixed_precision_config.yaml --enable-mixed-precision -t 4 -d 2"
    echo ""
    echo "  # 兼容原生SGLang命令格式"
    echo "  $0 -m /path/to/model --enable-mixed-precision -c config.yaml \\"
    echo "     --tp-size 4 --dp-size 2 --ep-size 1 \\"
    echo "     --max-running-requests 32 --max-total-tokens 40960 \\"
    echo "     --dtype bfloat16 --trust-remote-code"
    echo ""
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}检查依赖...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到python3${NC}"
        exit 1
    fi
    
    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}警告: 未找到nvidia-smi，可能没有CUDA环境${NC}"
    else
        echo -e "${GREEN}CUDA环境检查通过${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    fi
    
    # 检查PyTorch
    if ! python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null; then
        echo -e "${RED}错误: 未找到PyTorch${NC}"
        exit 1
    fi
    
    # 检查CUDA可用性
    if python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')" 2>/dev/null | grep -q "True"; then
        echo -e "${GREEN}PyTorch CUDA支持检查通过${NC}"
    else
        echo -e "${YELLOW}警告: PyTorch CUDA支持不可用${NC}"
    fi
}

# 检查文件
check_files() {
    echo -e "${BLUE}检查文件...${NC}"
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
        exit 1
    fi
    
    if [[ "$ENABLE_MIXED_PRECISION" = true ]] && [[ ! -f "$MIXED_PRECISION_CONFIG" ]]; then
        echo -e "${RED}错误: 混合精度配置文件不存在: $MIXED_PRECISION_CONFIG${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}文件检查通过${NC}"
}

# 检查GPU数量
check_gpu_count() {
    echo -e "${BLUE}检查GPU数量...${NC}"
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo 0)
    local required_gpus=$((TP_SIZE * DP_SIZE))
    
    echo -e "可用GPU数量: $gpu_count"
    echo -e "需要GPU数量: $required_gpus (TP=$TP_SIZE × DP=$DP_SIZE)"
    
    if [[ $gpu_count -lt $required_gpus ]]; then
        echo -e "${RED}错误: GPU数量不足${NC}"
        echo -e "需要至少 $required_gpus 个GPU，但只有 $gpu_count 个可用"
        exit 1
    fi
    
    echo -e "${GREEN}GPU数量检查通过${NC}"
}

# 显示配置信息
show_config() {
    echo -e "${BLUE}配置信息:${NC}"
    echo -e "  模型路径: $MODEL_PATH"
    if [[ "$ENABLE_MIXED_PRECISION" = true ]]; then
        echo -e "  混合精度: 启用"
        echo -e "  配置文件: $MIXED_PRECISION_CONFIG"
    else
        echo -e "  混合精度: 禁用"
    fi
    echo -e "  张量并行: $TP_SIZE"
    echo -e "  数据并行: $DP_SIZE"
    echo -e "  专家并行: $EP_SIZE"
    echo -e "  服务器地址: $HOST:$PORT"
    echo -e "  数据类型: $DTYPE"
    echo -e "  最大运行请求数: $MAX_RUNNING_REQUESTS"
    echo -e "  最大总token数: $MAX_TOTAL_TOKENS"
    echo ""
}

# 启动服务器
start_server() {
    echo -e "${BLUE}启动兼容SGLang原生方式的混合精度服务器...${NC}"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
    
    # 构建启动命令
    LAUNCH_CMD="python3 launch_mixed_precision_sglang.py"
    
    # 添加混合精度参数
    if [[ "$ENABLE_MIXED_PRECISION" = true ]]; then
        LAUNCH_CMD="$LAUNCH_CMD --enable-mixed-precision --mixed-precision-config $MIXED_PRECISION_CONFIG"
    fi
    
    # 添加标准SGLang参数
    LAUNCH_CMD="$LAUNCH_CMD --model-path $MODEL_PATH"
    LAUNCH_CMD="$LAUNCH_CMD --tp-size $TP_SIZE"
    LAUNCH_CMD="$LAUNCH_CMD --dp-size $DP_SIZE"
    LAUNCH_CMD="$LAUNCH_CMD --ep-size $EP_SIZE"
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
    
    echo -e "${GREEN}启动命令:${NC}"
    echo "$LAUNCH_CMD"
    echo ""
    echo -e "${BLUE}服务器将在 http://$HOST:$PORT 启动${NC}"
    echo -e "${YELLOW}按 Ctrl+C 停止服务器${NC}"
    echo ""
    
    # 启动服务器
    eval $LAUNCH_CMD
}

# 主函数
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}兼容SGLang原生启动方式的混合精度服务器${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model)
                MODEL_PATH="$2"
                shift 2
                ;;
            -c|--config)
                MIXED_PRECISION_CONFIG="$2"
                shift 2
                ;;
            -t|--tp-size)
                TP_SIZE="$2"
                shift 2
                ;;
            -d|--dp-size)
                DP_SIZE="$2"
                shift 2
                ;;
            -e|--ep-size)
                EP_SIZE="$2"
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
            --dtype)
                DTYPE="$2"
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
            --enable-mixed-precision)
                ENABLE_MIXED_PRECISION=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}未知参数: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [[ -z "$MODEL_PATH" ]]; then
        echo -e "${RED}错误: 必须指定模型路径 (-m/--model)${NC}"
        show_help
        exit 1
    fi
    
    if [[ "$ENABLE_MIXED_PRECISION" = true ]] && [[ -z "$MIXED_PRECISION_CONFIG" ]]; then
        echo -e "${RED}错误: 启用混合精度时必须指定配置文件 (-c/--config)${NC}"
        show_help
        exit 1
    fi
    
    # 检查Python环境
    check_dependencies
    echo ""
    
    # 检查配置文件
    check_files
    echo ""
    
    # 检查GPU数量
    check_gpu_count
    echo ""
    
    # 显示配置信息
    show_config
    
    # 启动服务器
    start_server
}

# 捕获Ctrl+C信号
trap 'echo -e "\n${YELLOW}服务器已停止${NC}"; exit 0' INT

# 运行主函数
main "$@"
