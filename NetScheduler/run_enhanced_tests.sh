#!/bin/bash

# EP vs TP 增强测试快速启动脚本
# 自动化运行部署验证和性能测试流程

set -e  # 遇到错误时退出

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

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi 未找到，GPU信息获取可能失败"
    fi
    
    # 检查Python包
    python3 -c "import requests" 2>/dev/null || {
        print_warning "requests 包未安装，正在安装..."
        pip install requests
    }
    
    python3 -c "import matplotlib" 2>/dev/null || {
        print_warning "matplotlib 包未安装，图表生成功能将不可用"
        print_info "安装命令: pip install matplotlib numpy"
    }
    
    # 检查Flask（内部状态检查器需要）
    python3 -c "import flask" 2>/dev/null || {
        print_warning "Flask 包未安装，内部状态检查器将自动安装"
    }
    
    print_success "依赖检查完成"
}

# 设置内部状态检查器
setup_internal_state_checker() {
    print_info "设置内部状态检查器..."
    
    # 检查SGLang源码是否存在
    if [ ! -d "sglang-0.4.7" ]; then
        print_warning "SGLang源码目录不存在，内部状态检查器将使用回退方法"
        return
    fi
    
    # 检查内部状态检查器脚本
    if [ ! -f "sglang_internal_state_checker.py" ]; then
        print_error "内部状态检查器脚本不存在: sglang_internal_state_checker.py"
        exit 1
    fi
    
    print_info "内部状态检查器已准备就绪"
}

# 清理内部状态检查器
cleanup_internal_state_checker() {
    print_info "清理内部状态检查器..."
    
    # 停止可能的API服务器进程
    pkill -f "internal_state_api.py" 2>/dev/null || true
    
    # 清理可能的临时文件
    rm -f sglang_internal_state_checker_*.log 2>/dev/null || true
    
    print_success "内部状态检查器清理完成"
}

# 检查GPU状态
check_gpu_status() {
    print_info "检查GPU状态..."
    
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_info "检测到 $gpu_count 张GPU"
        
        if [ $gpu_count -lt 8 ]; then
            print_warning "GPU数量少于8张，可能影响测试效果"
        fi
        
        # 显示GPU使用情况
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader,nounits | while IFS=', ' read -r gpu_id mem_used mem_total util; do
            print_info "GPU $gpu_id: 内存使用 ${mem_used}MB/${mem_total}MB, 利用率 ${util}%"
        done
    else
        print_warning "无法获取GPU信息"
    fi
}

# 检查模型路径
check_model_path() {
    print_info "检查模型路径..."
    
    # 检查测试脚本中的模型路径
    if grep -q "/dev/shm/Qwen3-30B-A3B" test_single_expert_ep.py; then
        print_warning "检测到默认模型路径，请确认路径是否正确"
        print_info "请修改 test_single_expert_ep.py 和 test_single_expert_tp.py 中的模型路径"
    fi
}

# 运行EP测试
run_ep_test() {
    print_info "开始运行 Expert Parallel 测试..."
    
    if [ -f "ep_test_results.json" ]; then
        print_warning "发现已存在的EP测试结果，是否覆盖? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "跳过EP测试"
            return
        fi
    fi
    
    print_info "启动EP服务器并运行测试..."
    python3 test_single_expert_ep.py
    
    if [ -f "ep_test_results.json" ]; then
        print_success "EP测试完成，结果已保存到 ep_test_results.json"
    else
        print_error "EP测试失败，未生成结果文件"
        exit 1
    fi
}

# 运行TP测试
run_tp_test() {
    print_info "开始运行 Tensor Parallel 测试..."
    
    if [ -f "tp_test_results.json" ]; then
        print_warning "发现已存在的TP测试结果，是否覆盖? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "跳过TP测试"
            return
        fi
    fi
    
    print_info "启动TP服务器并运行测试..."
    python3 test_single_expert_tp.py
    
    if [ -f "tp_test_results.json" ]; then
        print_success "TP测试完成，结果已保存到 tp_test_results.json"
    else
        print_error "TP测试失败，未生成结果文件"
        exit 1
    fi
}

# 运行性能对比分析
run_comparison() {
    print_info "开始运行性能对比分析..."
    
    if [ ! -f "ep_test_results.json" ] || [ ! -f "tp_test_results.json" ]; then
        print_error "缺少测试结果文件，无法进行对比分析"
        exit 1
    fi
    
    python3 compare_ep_tp_performance.py
    
    if [ -f "ep_tp_comparison.json" ]; then
        print_success "对比分析完成，结果已保存到 ep_tp_comparison.json"
    fi
    
    if [ -f "performance_comparison.png" ]; then
        print_success "性能对比图表已生成: performance_comparison.png"
    fi
}

# 显示结果摘要
show_summary() {
    print_info "测试结果摘要:"
    echo "=================================="
    
    if [ -f "ep_test_results.json" ]; then
        print_info "EP测试结果: ep_test_results.json"
    fi
    
    if [ -f "tp_test_results.json" ]; then
        print_info "TP测试结果: tp_test_results.json"
    fi
    
    if [ -f "ep_tp_comparison.json" ]; then
        print_info "对比分析结果: ep_tp_comparison.json"
    fi
    
    if [ -f "performance_comparison.png" ]; then
        print_info "性能对比图表: performance_comparison.png"
    fi
    
    echo "=================================="
    print_success "所有测试完成！"
}

# 清理函数
cleanup() {
    print_info "清理临时文件..."
    
    # 清理可能的临时文件
    rm -f *.tmp
    rm -f *.log
    
    print_success "清理完成"
}

# 显示帮助信息
show_help() {
    echo "EP vs TP 增强测试快速启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --check    仅检查环境和依赖"
    echo "  -e, --ep       仅运行EP测试"
    echo "  -t, --tp       仅运行TP测试"
    echo "  -a, --all      运行完整测试流程 (默认)"
    echo "  -r, --compare  仅运行对比分析 (需要已有测试结果)"
    echo "  --clean        清理临时文件"
    echo ""
    echo "示例:"
    echo "  $0              # 运行完整测试流程"
    echo "  $0 -c           # 仅检查环境"
    echo "  $0 -e           # 仅运行EP测试"
    echo "  $0 -t           # 仅运行TP测试"
    echo "  $0 -r           # 仅运行对比分析"
}

# 主函数
main() {
    echo "=================================="
    echo "EP vs TP 增强测试快速启动脚本"
    echo "=================================="
    
    # 解析命令行参数
    case "${1:-all}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check)
            check_dependencies
            check_gpu_status
            check_model_path
            exit 0
            ;;
        -e|--ep)
            check_dependencies
            check_gpu_status
            check_model_path
            setup_internal_state_checker
            run_ep_test
            cleanup_internal_state_checker
            show_summary
            ;;
        -t|--tp)
            check_dependencies
            check_gpu_status
            check_model_path
            setup_internal_state_checker
            run_tp_test
            cleanup_internal_state_checker
            show_summary
            ;;
        -r|--compare)
            run_comparison
            show_summary
            ;;
        -a|--all)
            check_dependencies
            check_gpu_status
            check_model_path
            setup_internal_state_checker
            run_ep_test
            run_tp_test
            run_comparison
            cleanup_internal_state_checker
            show_summary
            ;;
        --clean)
            cleanup
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 设置信号处理
trap 'print_error "测试被中断"; cleanup; exit 1' INT TERM

# 运行主函数
main "$@"
