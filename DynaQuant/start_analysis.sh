#!/bin/bash

# 完整的Qwen模型测试和Coze API分析流程启动脚本

echo "=========================================="
echo "Qwen模型测试 + Coze API分析 完整流程"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip3 install -r requirements_test.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖安装失败"
        exit 1
    fi
fi

# 检查配置文件
if [ ! -f "coze_config.json" ]; then
    echo "错误: 未找到 coze_config.json 配置文件"
    echo "请先配置你的Coze API信息"
    exit 1
fi

# 检查测试数据文件
if [ ! -f "test_data.txt" ] && [ ! -f "test_data.jsonl" ]; then
    echo "错误: 未找到测试数据文件 (test_data.txt 或 test_data.jsonl)"
    exit 1
fi

# 显示使用说明
echo ""
echo "完整流程说明:"
echo "1. 启动Qwen模型服务: bash Qwen3-235B-A22B.sh"
echo "2. 运行模型测试程序"
echo "3. 运行Coze API分析程序"
echo "4. 查看分析结果"
echo ""

# 检查服务是否运行
echo "检查模型服务状态..."
if curl -s http://127.0.0.1:8080/health &> /dev/null; then
    echo "✓ 模型服务正在运行"
    echo ""
    
    # 询问用户是否继续
    read -p "是否继续运行完整流程？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "用户取消操作"
        exit 0
    fi
    
    echo ""
    echo "开始运行完整流程..."
    echo ""
    
    # 步骤1: 运行模型测试
    echo "步骤1: 运行Qwen模型测试..."
    if [ -f "test_data.txt" ]; then
        echo "使用 test_data.txt 进行测试"
        python3 test_qwen_service.py -i test_data.txt -o test_results.jsonl
    elif [ -f "test_data.jsonl" ]; then
        echo "使用 test_data.jsonl 进行测试"
        python3 test_qwen_service.py -i test_data.jsonl -o test_results.jsonl
    fi
    
    # 检查测试结果
    if [ -f "test_results.jsonl" ]; then
        echo ""
        echo "✓ 模型测试完成，生成了 test_results.jsonl"
        echo ""
        
        # 步骤2: 运行Coze API分析
        echo "步骤2: 运行Coze API分析..."
        python3 run_coze_analysis.py test_results.jsonl
        
        # 检查分析结果
        if [ -f "coze_results.jsonl" ] && [ -f "summary_report.json" ]; then
            echo ""
            echo "✓ Coze API分析完成！"
            echo ""
            echo "生成的文件:"
            echo "  - test_results.jsonl: 模型测试结果"
            echo "  - coze_results.jsonl: Coze API分析结果"
            echo "  - summary_report.json: 分析摘要报告"
            echo ""
            echo "可以查看这些文件来了解详细结果"
        else
            echo ""
            echo "✗ Coze API分析失败，请检查配置和网络连接"
        fi
    else
        echo ""
        echo "✗ 模型测试失败，未生成结果文件"
    fi
    
else
    echo "✗ 模型服务未运行"
    echo ""
    echo "请先启动模型服务:"
    echo "bash Qwen3-235B-A22B.sh"
    echo ""
    echo "或者手动启动后再次运行此脚本"
fi

echo ""
echo "=========================================="
echo "流程结束"
echo "=========================================="
