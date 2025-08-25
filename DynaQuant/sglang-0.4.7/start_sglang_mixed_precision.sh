#!/bin/bash

# 混合精度TP/DP启动脚本
# 支持TP=4, DP=2的配置

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
WORLD_SIZE=8
DIST_INIT_ADDR="127.0.0.1:50000"
DTYPE="auto"
TEST_MODE=false

# 进程管理
PIDS=()

# 显示帮助信息
show_help() {
    echo -e "${BLUE}混合精度TP/DP启动脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH              模型路径 (必需)"
    echo "  -c, --config PATH             混合精度配置文件路径 (必需)"
    echo "  -t, --tp-size SIZE            张量并行大小 (默认: 4)"
    echo "  -d, --dp-size SIZE            数据并行大小 (默认: 2)"
    echo "  -a, --dist-addr ADDR:PORT     分布式初始化地址 (默认: 127.0.0.1:50000)"
    echo "  --dtype TYPE                  数据类型 (默认: auto)"
    echo "  --test                        运行测试模式"
    echo "  -h, --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -m /path/to/model -c mixed_precision_config.yaml"
    echo "  $0 -m /path/to/model -c mixed_precision_config.yaml -t 4 -d 2 --test"
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
    
    if [[ ! -f "$MIXED_PRECISION_CONFIG" ]]; then
        echo -e "${RED}错误: 混合精度配置文件不存在: $MIXED_PRECISION_CONFIG${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}文件检查通过${NC}"
}

# 检查GPU数量
check_gpu_count() {
    local required_gpus=$WORLD_SIZE
    local available_gpus=0
    
    if command -v nvidia-smi &> /dev/null; then
        available_gpus=$(nvidia-smi --list-gpus | wc -l)
    fi
    
    echo -e "${BLUE}GPU检查:${NC}"
    echo "  需要GPU数量: $required_gpus"
    echo "  可用GPU数量: $available_gpus"
    
    if [[ $available_gpus -lt $required_gpus ]]; then
        echo -e "${YELLOW}警告: 可用GPU数量($available_gpus)少于需要数量($required_gpus)${NC}"
        echo "  将使用单GPU模式运行多个进程"
    else
        echo -e "${GREEN}GPU数量检查通过${NC}"
    fi
}

# 启动单个进程
start_process() {
    local rank=$1
    local local_rank=$((rank % $(nvidia-smi --list-gpus | wc -l 2>/dev/null || echo 1)))
    
    echo -e "${BLUE}启动进程 rank=$rank (local_rank=$local_rank)...${NC}"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$local_rank
    
    # 启动进程
    python3 launch_mixed_precision_tp_dp.py \
        --model "$MODEL_PATH" \
        --mixed-precision-config "$MIXED_PRECISION_CONFIG" \
        --tp-size "$TP_SIZE" \
        --dp-size "$DP_SIZE" \
        --rank "$rank" \
        --world-size "$WORLD_SIZE" \
        --dist-init-addr "$DIST_INIT_ADDR" \
        --dtype "$DTYPE" \
        ${TEST_MODE:+--test} &
    
    local pid=$!
    PIDS+=($pid)
    
    echo -e "${GREEN}进程 rank=$rank 已启动 (PID: $pid)${NC}"
}

# 启动所有进程
start_all_processes() {
    echo -e "${BLUE}启动所有进程...${NC}"
    echo "配置: TP=$TP_SIZE, DP=$DP_SIZE, World Size=$WORLD_SIZE"
    echo "分布式地址: $DIST_INIT_ADDR"
    echo ""
    
    # 启动所有进程
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
        start_process $rank
        sleep 2  # 等待进程启动
    done
    
    echo ""
    echo -e "${GREEN}所有进程已启动${NC}"
    echo "进程PID: ${PIDS[@]}"
}

# 清理进程
cleanup() {
    echo -e "${YELLOW}清理进程...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止进程 PID: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # 等待进程结束
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    
    echo -e "${GREEN}所有进程已清理${NC}"
}

# 监控进程
monitor_processes() {
    echo -e "${BLUE}监控进程状态...${NC}"
    echo "按 Ctrl+C 停止所有进程"
    echo ""
    
    # 设置信号处理
    trap cleanup EXIT INT TERM
    
    # 监控循环
    while true; do
        local running=0
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running++))
            fi
        done
        
        if [[ $running -eq 0 ]]; then
            echo -e "${RED}所有进程已停止${NC}"
            break
        fi
        
        echo -e "${GREEN}运行中进程: $running/$WORLD_SIZE${NC}"
        sleep 10
    done
}

# 主函数
main() {
    echo -e "${BLUE}=== 混合精度TP/DP启动脚本 ===${NC}"
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
            -a|--dist-addr)
                DIST_INIT_ADDR="$2"
                shift 2
                ;;
            --dtype)
                DTYPE="$2"
                shift 2
                ;;
            --test)
                TEST_MODE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证必需参数
    if [[ -z "$MODEL_PATH" ]]; then
        echo -e "${RED}错误: 必须指定模型路径 (-m/--model)${NC}"
        show_help
        exit 1
    fi
    
    if [[ -z "$MIXED_PRECISION_CONFIG" ]]; then
        echo -e "${RED}错误: 必须指定混合精度配置文件 (-c/--config)${NC}"
        show_help
        exit 1
    fi
    
    # 计算world size
    WORLD_SIZE=$((TP_SIZE * DP_SIZE))
    
    echo -e "${BLUE}配置信息:${NC}"
    echo "  模型路径: $MODEL_PATH"
    echo "  混合精度配置: $MIXED_PRECISION_CONFIG"
    echo "  张量并行大小: $TP_SIZE"
    echo "  数据并行大小: $DP_SIZE"
    echo "  总进程数: $WORLD_SIZE"
    echo "  分布式地址: $DIST_INIT_ADDR"
    echo "  数据类型: $DTYPE"
    echo "  测试模式: $TEST_MODE"
    echo ""
    
    # 检查依赖和文件
    check_dependencies
    check_files
    check_gpu_count
    
    echo ""
    echo -e "${BLUE}开始启动混合精度TP/DP服务器...${NC}"
    echo ""
    
    # 启动所有进程
    start_all_processes
    
    # 监控进程
    monitor_processes
}

# 运行主函数
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
