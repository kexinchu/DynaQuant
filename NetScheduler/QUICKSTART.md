# 快速开始指南

本指南将帮助你快速设置和运行Qwen3-30B-A3B单expert模型的并行策略测试。

## 1. 环境准备

### 1.1 检查系统要求

- **操作系统**: Linux (推荐) 或 Windows
- **Python**: 3.8+
- **GPU**: 8张GPU (推荐)
- **内存**: 至少64GB系统内存
- **存储**: 至少100GB可用空间

### 1.2 安装依赖

```bash
# 安装Python依赖
pip install torch transformers requests

# 如果使用conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 1.3 准备模型

确保你已经有了经过`prune_qwen3_a22B-to_single_expert.py`处理的Qwen3-30B-A3B模型。

## 2. 快速测试

### 2.1 基础测试

首先运行基础测试，确保模型可以正常加载：

```bash
# Linux/Mac
python3 OneExpertTest.py

# Windows
python OneExpertTest.py
```

### 2.2 使用启动脚本

#### Linux/Mac:
```bash
# 给脚本执行权限
chmod +x run_tests.sh

# 运行所有测试
./run_tests.sh --all

# 只运行Expert Parallel测试
./run_tests.sh --ep

# 只运行Tensor Parallel测试
./run_tests.sh --tp
```

#### Windows:
```cmd
# 运行所有测试
run_tests.bat --all

# 只运行Expert Parallel测试
run_tests.bat --ep

# 只运行Tensor Parallel测试
run_tests.bat --tp
```

### 2.3 单独运行测试

```bash
# Expert Parallel测试
python3 test_single_expert_ep.py

# Tensor Parallel测试
python3 test_single_expert_tp.py

# 混合并行测试
python3 test_hybrid_parallel.py --config both
```

## 3. 配置调整

### 3.1 修改模型路径

在所有测试脚本中，找到并修改模型路径：

```python
# 在脚本中找到这一行并修改
model_path = "/path/to/your/Qwen3-30B-A3B"
```

### 3.2 调整GPU配置

如果你的GPU数量不是8张，需要调整配置：

```python
# 修改GPU设备列表
gpu_devices = "0,1,2,3"  # 如果只有4张GPU
```

### 3.3 内存优化

如果遇到内存不足，可以调整以下参数：

```python
# 减少最大token数
max_total_tokens = 20480

# 减少chunked prefill size
chunked_prefill_size = 8192
```

## 4. 常见问题解决

### 4.1 CUDA内存不足

```bash
# 减少batch size
--max-total-tokens 20480
--chunked-prefill-size 8192
```

### 4.2 GPU通信错误

```bash
# 添加P2P检查
--enable-p2p-check
```

### 4.3 模型加载失败

检查模型路径和权限：
```bash
ls -la /path/to/model
```

### 4.4 端口被占用

修改端口号：
```bash
--port 8082
```

## 5. 性能监控

### 5.1 GPU监控

```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 或者使用htop查看进程
htop
```

### 5.2 网络监控

```bash
# 监控网络流量
iftop

# 或者使用nethogs
nethogs
```

## 6. 结果分析

测试完成后，你会得到：

1. **功能测试结果**: 验证模型推理的正确性
2. **性能基准测试**: 对比两种并行策略的性能
3. **延迟统计**: 详细的延迟分布信息
4. **资源使用情况**: GPU内存和计算利用率

### 6.1 性能指标说明

- **TTFT (Time To First Token)**: 首token生成时间，越低越好
- **TPOT (Time Per Output Token)**: 每个输出token的平均时间，越低越好
- **吞吐量**: 每秒处理的请求数，越高越好
- **内存使用**: GPU内存占用情况

### 6.2 结果对比

比较两种配置的性能：

| 配置 | TTFT (ms) | TPOT (ms) | 吞吐量 (req/s) | 内存使用 |
|------|-----------|-----------|----------------|----------|
| Expert Parallel | - | - | - | - |
| Tensor Parallel | - | - | - | - |

## 7. 高级配置

### 7.1 自定义测试

修改测试prompts：

```python
test_prompts = [
    "你的自定义prompt1",
    "你的自定义prompt2"
]
```

### 7.2 批量测试

```bash
# 运行多次测试取平均值
for i in {1..5}; do
    echo "运行第 $i 次测试"
    python3 test_hybrid_parallel.py --config both
    sleep 10
done
```

### 7.3 配置文件

使用配置文件管理参数：

```bash
# 生成配置文件
python3 config.py

# 修改配置文件
vim test_config.json

# 使用配置文件运行测试
python3 test_hybrid_parallel.py --config-file test_config.json
```

## 8. 故障排除

### 8.1 调试模式

启用详细日志：

```bash
export NCCL_DEBUG=INFO
export SGLANG_LOG_LEVEL=DEBUG
```

### 8.2 检查日志

查看详细的错误信息：

```bash
# 查看Python错误
python3 -u test_single_expert_ep.py 2>&1 | tee ep_test.log

# 查看系统日志
dmesg | tail -20
```

### 8.3 环境检查

运行环境检查脚本：

```bash
# Linux/Mac
./run_tests.sh --check

# Windows
run_tests.bat --check
```

## 9. 下一步

完成基础测试后，你可以：

1. **性能调优**: 根据结果调整参数以获得最佳性能
2. **扩展测试**: 添加更多测试场景和负载
3. **生产部署**: 将测试结果应用到生产环境
4. **贡献代码**: 提交改进和bug修复

## 10. 获取帮助

如果遇到问题：

1. 查看README.md获取详细文档
2. 检查故障排除部分
3. 查看日志文件获取错误信息
4. 提交Issue到项目仓库

---

**注意**: 确保在运行测试前已经正确配置了所有环境变量和路径。如果遇到问题，请先运行环境检查脚本。
