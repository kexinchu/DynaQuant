#!/bin/bash

# 混合并行测试启动脚本
# 基于Qwen3-30B-A3B单expert模型的并行策略测试

set -e

# 配置
MODEL_PATH="/dev/shm/Qwen3-30B-A3B"  # 修改为你的模型路径
SGLANG_PATH="sglang-0.4.7"  # sglang路径

# 颜色输出
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

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch 未安装"
        exit 1
    fi
    
    if ! python3 -c "import requests" &> /dev/null; then
        print_error "requests 库未安装，正在安装..."
        pip3 install requests
    fi
    
    print_success "依赖检查完成"
}

# 检查模型路径
check_model_path() {
    print_info "检查模型路径: $MODEL_PATH"
    
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "模型路径不存在: $MODEL_PATH"
        print_info "请修改脚本中的 MODEL_PATH 变量为正确的模型路径"
        exit 1
    fi
    
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        print_error "模型配置文件不存在: $MODEL_PATH/config.json"
        exit 1
    fi
    
    print_success "模型路径检查完成"
}

# 检查GPU
check_gpu() {
    print_info "检查GPU..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi 未找到，可能没有GPU或驱动未安装"
        return
    fi
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_info "检测到 $GPU_COUNT 个GPU"
    
    if [ $GPU_COUNT -lt 8 ]; then
        print_warning "GPU数量少于8个，测试可能无法正常运行"
    fi
}

# 运行基础测试
run_basic_test() {
    print_info "运行基础模型测试..."
    
    if [ -f "OneExpertTest.py" ]; then
        python3 OneExpertTest.py
        print_success "基础测试完成"
    else
        print_warning "OneExpertTest.py 不存在，跳过基础测试"
    fi
}

# 运行Expert Parallel测试
run_ep_test() {
    print_info "运行 Expert Parallel 测试..."
    
    if [ -f "test_single_expert_ep.py" ]; then
        python3 test_single_expert_ep.py
        print_success "Expert Parallel 测试完成"
    else
        print_error "test_single_expert_ep.py 不存在"
    fi
}

# 运行Tensor Parallel测试
run_tp_test() {
    print_info "运行 Tensor Parallel 测试..."
    
    if [ -f "test_single_expert_tp.py" ]; then
        python3 test_single_expert_tp.py
        print_success "Tensor Parallel 测试完成"
    else
        print_error "test_single_expert_tp.py 不存在"
    fi
}

# 运行混合并行测试
run_hybrid_test() {
    print_info "运行混合并行测试..."
    
    if [ -f "test_hybrid_parallel.py" ]; then
        python3 test_hybrid_parallel.py --model-path "$MODEL_PATH" --config both
        print_success "混合并行测试完成"
    else
        print_error "test_hybrid_parallel.py 不存在"
    fi
}

# 运行性能基准测试
run_benchmark() {
    print_info "运行性能基准测试..."
    
    if [ -f "moe_experiment.py" ]; then
        print_info "运行 DP 基准测试..."
        python3 moe_experiment.py --mode dp --sequence-length 128 --qps 4 --duration 10
        
        print_info "运行 TP 基准测试..."
        python3 moe_experiment.py --mode tp --sequence-length 128 --qps 4 --duration 10
        
        print_success "性能基准测试完成"
    else
        print_warning "moe_experiment.py 不存在，跳过性能基准测试"
    fi
}

# 显示帮助信息
show_help() {
    echo "混合并行测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -c, --check         只检查环境和依赖"
    echo "  -b, --basic         运行基础模型测试"
    echo "  -e, --ep            运行 Expert Parallel 测试"
    echo "  -t, --tp            运行 Tensor Parallel 测试"
    echo "  -a, --all           运行所有测试"
    echo "  -m, --model PATH    指定模型路径"
    echo ""
    echo "示例:"
    echo "  $0 --all                    # 运行所有测试"
    echo "  $0 --ep                     # 只运行 Expert Parallel 测试"
    echo "  $0 --model /path/to/model   # 指定模型路径并运行所有测试"
}

# 主函数
main() {
    local run_all=false
    local run_check=false
    local run_basic=false
    local run_ep=false
    local run_tp=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                run_check=true
                shift
                ;;
            -b|--basic)
                run_basic=true
                shift
                ;;
            -e|--ep)
                run_ep=true
                shift
                ;;
            -t|--tp)
                run_tp=true
                shift
                ;;
            -a|--all)
                run_all=true
                shift
                ;;
            -m|--model)
                MODEL_PATH="$2"
                shift 2
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定任何选项，默认运行所有测试
    if [ "$run_check" = false ] && [ "$run_basic" = false ] && [ "$run_ep" = false ] && [ "$run_tp" = false ] && [ "$run_all" = false ]; then
        run_all=true
    fi
    
    print_info "开始混合并行测试"
    print_info "模型路径: $MODEL_PATH"
    
    # 检查环境和依赖
    check_dependencies
    check_model_path
    check_gpu
    
    if [ "$run_check" = true ]; then
        print_success "环境检查完成"
        exit 0
    fi
    
    # 运行测试
    if [ "$run_basic" = true ] || [ "$run_all" = true ]; then
        run_basic_test
    fi
    
    if [ "$run_ep" = true ] || [ "$run_all" = true ]; then
        run_ep_test
    fi
    
    if [ "$run_tp" = true ] || [ "$run_all" = true ]; then
        run_tp_test
    fi
    
    if [ "$run_all" = true ]; then
        run_hybrid_test
        run_benchmark
    fi
    
    print_success "所有测试完成！"
}

# 运行主函数
main "$@"
