# Qwen3-30B-A3B 单Expert模型并行策略测试

本项目基于sglang-0.4.7实现了对Qwen3-30B-A3B单expert模型的两种并行策略测试：

1. **Expert Parallel (EP) 方式**: experts层在8张GPU上都创建备份，使用随机routing
2. **Tensor Parallel (TP) 方式**: experts层在8张卡上进行TP=8切分，均匀部署

## 项目结构

```
NetScheduler/
├── OneExpertTest.py              # 基础模型测试
├── test_single_expert_ep.py      # Expert Parallel测试
├── test_single_expert_tp.py      # Tensor Parallel测试
├── test_hybrid_parallel.py       # 混合并行测试
├── moe_experiment.py             # 性能基准测试
├── run_tests.sh                  # 启动脚本
└── README.md                     # 说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- sglang-0.4.7
- 8张GPU (推荐)
- CUDA 11.8+

## 安装依赖

```bash
# 安装基础依赖
pip install torch transformers requests

# 安装sglang (如果还没有安装)
# 假设sglang-0.4.7在当前目录
cd sglang-0.4.7
pip install -e .
```

## 配置

### 1. 修改模型路径

在所有测试脚本中，将模型路径修改为你的实际路径：

```python
# 在脚本中修改这一行
model_path = "/path/to/your/Qwen3-30B-A3B"
```

### 2. 检查GPU配置

确保你的系统有足够的GPU资源：

```bash
nvidia-smi
```

## 使用方法

### 方法1: 使用启动脚本 (推荐)

```bash
# 给脚本执行权限
chmod +x run_tests.sh

# 运行所有测试
./run_tests.sh --all

# 只运行Expert Parallel测试
./run_tests.sh --ep

# 只运行Tensor Parallel测试
./run_tests.sh --tp

# 检查环境
./run_tests.sh --check

# 指定模型路径
./run_tests.sh --model /path/to/model --all
```

### 方法2: 单独运行测试脚本

```bash
# 基础测试
python3 OneExpertTest.py

# Expert Parallel测试
python3 test_single_expert_ep.py

# Tensor Parallel测试
python3 test_single_expert_tp.py

# 混合并行测试
python3 test_hybrid_parallel.py --config both

# 性能基准测试
python3 moe_experiment.py --mode dp --sequence-length 128 --qps 4 --duration 10
python3 moe_experiment.py --mode tp --sequence-length 128 --qps 4 --duration 10
```

## 测试配置说明

### 配置1: Expert Parallel (EP=8)

- **Expert层**: 使用EP=8，每个expert在8张GPU上都有备份
- **其他层**: TP=4, DP=2
- **路由策略**: 随机routing
- **优势**: 减少通信开销，提高吞吐量

启动命令：
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/Qwen3-30B-A3B \
  --tp-size 4 \
  --dp-size 2 \
  --enable-ep-moe \
  --ep-size 8 \
  --host 127.0.0.1 --port 8080
```

### 配置2: Tensor Parallel (TP=8)

- **Expert层**: 使用TP=8，expert在8张GPU上切分
- **其他层**: TP=8 (所有层统一使用TP=8)
- **切分策略**: 均匀部署在8张卡上
- **优势**: 减少内存占用，支持更大模型

启动命令：
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/Qwen3-30B-A3B \
  --tp-size 8 \
  --dp-size 1 \
  --host 127.0.0.1 --port 8081
```

## 性能指标

测试脚本会收集以下性能指标：

- **TTFT (Time To First Token)**: 首token生成时间
- **TPOT (Time Per Output Token)**: 每个输出token的平均时间
- **吞吐量**: 每秒处理的请求数
- **延迟**: 请求响应时间
- **内存使用**: GPU内存占用情况

## 注意事项

### 1. 内存管理

- 确保有足够的GPU内存
- 可以调整 `--max-total-tokens` 和 `--mem-fraction-static` 参数
- 对于长序列，考虑调整 `--chunked-prefill-size`

### 2. 网络配置

- 确保GPU间通信正常
- 如果遇到peer access错误，添加 `--enable-p2p-check` 参数

### 3. 模型兼容性

- 确保模型已经通过 `prune_qwen3_a22B-to_single_expert.py` 处理
- 检查模型配置文件中的expert数量

### 4. 环境变量

测试脚本会自动设置以下环境变量：

```bash
export SGLANG_DISABLE_MARLIN=1
export SGL_DISABLE_AWQ_MARLIN=1
export SGLANG_DISABLE_SGL_KERNEL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少batch size或序列长度
   --max-total-tokens 20480
   --chunked-prefill-size 8192
   ```

2. **GPU通信错误**
   ```bash
   # 添加P2P检查
   --enable-p2p-check
   ```

3. **模型加载失败**
   ```bash
   # 检查模型路径和权限
   ls -la /path/to/model
   ```

4. **端口被占用**
   ```bash
   # 修改端口号
   --port 8082
   ```

### 调试模式

启用详细日志：

```bash
export NCCL_DEBUG=INFO
export SGLANG_LOG_LEVEL=DEBUG
```

## 结果分析

测试完成后，你会得到：

1. **功能测试结果**: 验证模型推理的正确性
2. **性能基准测试**: 对比两种并行策略的性能
3. **资源使用情况**: GPU内存和计算利用率
4. **延迟统计**: 详细的延迟分布信息

## 扩展测试

### 自定义测试

你可以修改测试脚本中的参数：

```python
# 修改测试prompt
test_prompts = [
    "你的自定义prompt1",
    "你的自定义prompt2"
]

# 修改生成参数
'max_tokens': 200,
'temperature': 0.5,
```

### 批量测试

```bash
# 运行多次测试取平均值
for i in {1..5}; do
    echo "运行第 $i 次测试"
    python3 test_hybrid_parallel.py --config both
    sleep 10
done
```

## 贡献

欢迎提交Issue和Pull Request来改进测试脚本。

## 许可证

本项目遵循MIT许可证。
