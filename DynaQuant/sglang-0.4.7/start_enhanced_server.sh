#!/bin/bash

# 增强的SGLang服务器启动脚本
# 集成混合精度权重加载和专家激活跟踪功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_CONFIG="mixed_precision_config.yaml"
DEFAULT_MODEL_PATH=""
DEFAULT_PORT=8080
DEFAULT_HOST="127.0.0.1"
DEFAULT_ENABLE_EXPERT_TRACKING=true

# 显示帮助信息
show_help() {
    echo -e "${BLUE}增强的SGLang服务器启动脚本${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config PATH        混合精度配置文件路径 (默认: $DEFAULT_CONFIG)"
    echo "  -m, --model PATH         模型路径 (必需)"
    echo "  -p, --port PORT          服务器端口 (默认: $DEFAULT_PORT)"
    echo "  -h, --host HOST          服务器主机地址 (默认: $DEFAULT_HOST)"
    echo "  --no-expert-tracking     禁用专家激活跟踪"
    echo "  --test                   运行功能测试"
    echo "  --help                   显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -m /path/to/model -c config.yaml"
    echo "  $0 --model /path/to/model --port 8081 --no-expert-tracking"
    echo "  $0 --test"
}

# 解析命令行参数
CONFIG_PATH=$DEFAULT_CONFIG
MODEL_PATH=""
PORT=$DEFAULT_PORT
HOST=$DEFAULT_HOST
ENABLE_EXPERT_TRACKING=$DEFAULT_ENABLE_EXPERT_TRACKING
RUN_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        --no-expert-tracking)
            ENABLE_EXPERT_TRACKING=false
            shift
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查Python环境
check_python_env() {
    echo -e "${BLUE}检查Python环境...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到python3${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Python3 版本: $(python3 --version)${NC}"
    
    # 检查必要的包
    required_packages=("torch" "transformers" "safetensors" "yaml")
    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            echo -e "${GREEN}✓ $package 已安装${NC}"
        else
            echo -e "${YELLOW}⚠ $package 未安装${NC}"
        fi
    done
}

# 运行功能测试
run_tests() {
    echo -e "${BLUE}运行增强功能测试...${NC}"
    
    if [ ! -f "test_enhanced_features.py" ]; then
        echo -e "${RED}错误: 测试文件 test_enhanced_features.py 不存在${NC}"
        exit 1
    fi
    
    python3 test_enhanced_features.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 所有测试通过${NC}"
    else
        echo -e "${RED}✗ 部分测试失败${NC}"
        exit 1
    fi
}

# 检查配置文件
check_config() {
    echo -e "${BLUE}检查配置文件...${NC}"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${RED}错误: 配置文件 $CONFIG_PATH 不存在${NC}"
        echo -e "${YELLOW}请创建配置文件或使用 --config 指定正确的路径${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 配置文件存在: $CONFIG_PATH${NC}"
    
    # 验证配置文件格式
    if python3 -c "import yaml; yaml.safe_load(open('$CONFIG_PATH'))" 2>/dev/null; then
        echo -e "${GREEN}✓ 配置文件格式正确${NC}"
    else
        echo -e "${RED}错误: 配置文件格式不正确${NC}"
        exit 1
    fi
}

# 检查模型路径
check_model() {
    echo -e "${BLUE}检查模型路径...${NC}"
    
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}错误: 未指定模型路径${NC}"
        echo -e "${YELLOW}请使用 -m 或 --model 参数指定模型路径${NC}"
        exit 1
    fi
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}错误: 模型路径 $MODEL_PATH 不存在${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 模型路径存在: $MODEL_PATH${NC}"
    
    # 检查模型文件
    model_files=("config.json" "tokenizer.json" "tokenizer_config.json")
    for file in "${model_files[@]}"; do
        if [ -f "$MODEL_PATH/$file" ]; then
            echo -e "${GREEN}✓ 模型文件存在: $file${NC}"
        else
            echo -e "${YELLOW}⚠ 模型文件不存在: $file${NC}"
        fi
    done
}

# 启动服务器
start_server() {
    echo -e "${BLUE}启动增强的SGLang服务器...${NC}"
    
    if [ ! -f "launch_enhanced_server.py" ]; then
        echo -e "${RED}错误: 服务器启动脚本 launch_enhanced_server.py 不存在${NC}"
        exit 1
    fi
    
    # 构建启动命令
    cmd="python3 launch_enhanced_server.py"
    cmd="$cmd --config $CONFIG_PATH"
    cmd="$cmd --model $MODEL_PATH"
    cmd="$cmd --port $PORT"
    cmd="$cmd --host $HOST"
    
    if [ "$ENABLE_EXPERT_TRACKING" = true ]; then
        cmd="$cmd --enable-expert-tracking"
    fi
    
    echo -e "${GREEN}启动命令: $cmd${NC}"
    echo -e "${BLUE}服务器将在 http://$HOST:$PORT 启动${NC}"
    echo -e "${YELLOW}按 Ctrl+C 停止服务器${NC}"
    echo ""
    
    # 启动服务器
    eval $cmd
}

# 主函数
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}增强的SGLang服务器启动脚本${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # 检查Python环境
    check_python_env
    echo ""
    
    # 如果只是运行测试
    if [ "$RUN_TEST" = true ]; then
        run_tests
        exit 0
    fi
    
    # 检查配置文件
    check_config
    echo ""
    
    # 检查模型路径
    check_model
    echo ""
    
    # 显示配置信息
    echo -e "${BLUE}配置信息:${NC}"
    echo -e "  配置文件: $CONFIG_PATH"
    echo -e "  模型路径: $MODEL_PATH"
    echo -e "  服务器地址: $HOST:$PORT"
    echo -e "  专家跟踪: $ENABLE_EXPERT_TRACKING"
    echo ""
    
    # 启动服务器
    start_server
}

# 捕获Ctrl+C信号
trap 'echo -e "\n${YELLOW}服务器已停止${NC}"; exit 0' INT

# 运行主函数
main
