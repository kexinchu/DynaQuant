# EP vs TP 部署验证和性能测试使用指南

本指南详细说明如何使用增强后的测试脚本来验证EP和TP部署配置，并进行全面的性能测试。

## 功能概述

增强后的测试脚本提供以下功能：

### 1. 部署验证功能
- **GPU内存使用分析**: 检查各GPU的内存使用情况，验证部署是否均匀
- **部署类型识别**: 自动识别是EP还是TP部署方式
- **模型加载验证**: 确认模型是否正确加载并响应请求

### 2. 性能测试功能
- **TTFT (Time To First Token)**: 首token生成时间
- **TPOT (Time Per Output Token)**: 每个输出token的平均时间
- **Overall Latency**: 总体延迟时间
- **多维度测试**: 不同query长度和QPS的测试

### 3. 结果分析功能
- **统计分析**: 平均值、中位数、标准差等统计指标
- **性能对比**: EP vs TP的性能差异分析
- **可视化图表**: 生成性能对比图表
- **总结报告**: 自动生成部署建议

## 文件结构

```
NetScheduler/
├── test_single_expert_ep.py          # EP部署测试脚本
├── test_single_expert_tp.py          # TP部署测试脚本
├── compare_ep_tp_performance.py      # 性能对比分析脚本
├── ep_test_results.json              # EP测试结果 (运行后生成)
├── tp_test_results.json              # TP测试结果 (运行后生成)
├── ep_tp_comparison.json             # 对比分析结果 (运行后生成)
├── performance_comparison.png        # 性能对比图表 (运行后生成)
└── USAGE_GUIDE.md                   # 本使用指南
```

## 使用步骤

### 步骤1: 环境准备

确保你的环境满足以下要求：

```bash
# 检查GPU状态
nvidia-smi

# 检查Python依赖
pip install requests matplotlib numpy

# 确保sglang已安装
cd sglang-0.4.7
pip install -e .
```

### 步骤2: 修改模型路径

在所有测试脚本中，将模型路径修改为你的实际路径：

```python
# 在 test_single_expert_ep.py 和 test_single_expert_tp.py 中找到并修改
'--model-path', '/path/to/your/Qwen3-30B-A3B'
```

### 步骤3: 运行EP部署测试

```bash
python3 test_single_expert_ep.py
```

**测试内容**:
- 启动EP服务器 (Expert Parallel)
- 验证部署配置 (GPU内存使用均匀性)
- 运行性能测试:
  - 测试组1: 不同query长度 (128, 256, 512, 1024, 2048, 4096) @ QPS=1
  - 测试组2: 不同QPS (1, 2, 4, 8, 16, 32, 64) @ query长度=256
- 生成结果文件: `ep_test_results.json`

**预期输出**:
```
=== 启动 Expert Parallel 服务器 ===
启动命令: python3 -m sglang.launch_server --model-path /path/to/model ...
等待服务器启动...

=== 验证 Expert Parallel 部署配置 ===
GPU数量: 8
平均GPU内存使用率: 45.23%
GPU内存使用率标准差: 2.15%
✅ EP部署验证通过: GPU内存使用相对均匀，符合expert备份分布特征
GPU 0: 内存使用 44.8%, 利用率 12.3%
GPU 1: 内存使用 45.1%, 利用率 11.9%
...

=== 开始性能测试 ===
--- 测试组1: 不同query长度 (QPS=1) ---
测试query长度: 128
  请求 1: TTFT=245.67ms, TPOT=12.34ms, Overall=1489.23ms
  请求 2: TTFT=238.91ms, TPOT=11.89ms, Overall=1423.45ms
...

=== 测试结果分析 ===
--- Query长度测试结果 ---
Query长度 128:
  TTFT: 平均=242.29ms, 中位数=245.67ms, 标准差=3.45ms
  TPOT: 平均=12.12ms, 中位数=12.34ms, 标准差=0.23ms
  Overall: 平均=1456.34ms, 中位数=1489.23ms, 标准差=32.89ms
...
```

### 步骤4: 运行TP部署测试

```bash
python3 test_single_expert_tp.py
```

**测试内容**:
- 启动TP服务器 (Tensor Parallel)
- 验证部署配置 (GPU内存使用均匀性)
- 运行相同的性能测试
- 生成结果文件: `tp_test_results.json`

**预期输出**:
```
=== 启动 Tensor Parallel 服务器 ===
启动命令: python3 -m sglang.launch_server --model-path /path/to/model ...
等待服务器启动...

=== 验证 Tensor Parallel 部署配置 ===
GPU数量: 8
平均GPU内存使用率: 42.18%
GPU内存使用率标准差: 3.67%
✅ TP部署验证通过: GPU内存使用相对均匀，符合expert切分分布特征
GPU 0: 内存使用 41.2%, 利用率 15.7%
GPU 1: 内存使用 42.8%, 利用率 14.9%
...
```

### 步骤5: 运行性能对比分析

```bash
python3 compare_ep_tp_performance.py
```

**分析内容**:
- 加载EP和TP测试结果
- 对比部署配置差异
- 分析性能差异
- 生成总结报告
- 创建可视化图表
- 保存对比结果: `ep_tp_comparison.json`

**预期输出**:
```
EP vs TP 性能对比分析
==================================================

=== 部署配置对比分析 ===
GPU内存使用对比:
  EP平均使用率: 45.23%
  TP平均使用率: 42.18%
  EP标准差: 2.15%
  TP标准差: 3.67%

部署类型对比:
  EP类型: expert_parallel
  TP类型: tensor_parallel
  EP分布: uniform
  TP分布: uniform

=== 性能测试结果对比分析 ===
--- Query长度性能对比 ---
Query长度 128:
  EP - TTFT: 242.29ms, TPOT: 12.12ms, Overall: 1456.34ms
  TP - TTFT: 198.45ms, TPOT: 10.89ms, Overall: 1234.56ms
  改进 - TTFT: -18.10%, TPOT: -10.15%, Overall: -15.25%

=== 总结报告 ===
1. 部署验证结果:
   - EP平均GPU内存使用率: 45.23%
   - TP平均GPU内存使用率: 42.18%
   - 内存使用差异: 3.05%

2. 性能测试总结:
   - Query长度测试平均改进: -12.45%
   - QPS测试平均改进: -8.67%

3. 推荐建议:
   - TP部署方式在性能上明显优于EP部署方式
   - 建议在生产环境中使用TP部署方式
```

## 部署验证说明

### EP部署验证特征

**Expert Parallel (EP) 部署应该显示以下特征**:

1. **GPU内存使用均匀**: 所有GPU的内存使用率应该相对均匀
   - 标准差 < 10% 认为是均匀分布
   - 每个GPU都有expert的完整副本

2. **部署类型识别**: 
   - `expert_distribution.type`: "expert_parallel"
   - `expert_distribution.distribution`: "uniform"

3. **配置信息**:
   - `parallel_config.mode`: "expert_parallel"
   - `parallel_config.ep_size`: 8
   - `parallel_config.tp_size`: 4
   - `parallel_config.dp_size`: 2

### TP部署验证特征

**Tensor Parallel (TP) 部署应该显示以下特征**:

1. **GPU内存使用均匀**: 所有GPU的内存使用率应该相对均匀
   - 标准差 < 15% 认为是均匀分布 (TP允许稍大的差异)
   - expert被均匀切分到各个GPU

2. **部署类型识别**:
   - `expert_distribution.type`: "tensor_parallel"
   - `expert_distribution.distribution`: "uniform"

3. **配置信息**:
   - `parallel_config.mode`: "tensor_parallel"
   - `parallel_config.tp_size`: 8
   - `parallel_config.dp_size`: 1

## 性能指标说明

### TTFT (Time To First Token)
- **定义**: 从发送请求到收到第一个输出token的时间
- **重要性**: 影响用户体验，越低越好
- **影响因素**: 模型加载、预处理、首次推理

### TPOT (Time Per Output Token)
- **定义**: 生成每个后续token的平均时间
- **重要性**: 影响生成速度，越低越好
- **影响因素**: 模型推理效率、GPU计算能力

### Overall Latency
- **定义**: 整个请求的响应时间
- **重要性**: 总体性能指标
- **计算公式**: TTFT + (TPOT × 输出token数量)

## 测试配置说明

### 测试组1: Query长度测试
- **目的**: 测试不同输入长度对性能的影响
- **配置**: QPS=1, 输出token=100
- **测试长度**: 128, 256, 512, 1024, 2048, 4096
- **每个长度测试次数**: 5次 (可配置)

### 测试组2: QPS测试
- **目的**: 测试不同并发量对性能的影响
- **配置**: query长度=256, 输出token=100
- **测试QPS**: 1, 2, 4, 8, 16, 32, 64
- **每个QPS测试次数**: 5次 (可配置)

## 结果文件说明

### ep_test_results.json / tp_test_results.json
```json
{
  "deployment_info": {
    "gpu_memory_usage": {"0": 45.2, "1": 44.8, ...},
    "gpu_utilization": {"0": 12.3, "1": 11.9, ...},
    "model_loaded": true,
    "expert_distribution": {
      "type": "expert_parallel",
      "distribution": "uniform",
      "memory_std": 2.15,
      "memory_mean": 45.23
    },
    "parallel_config": {
      "mode": "expert_parallel",
      "ep_size": 8,
      "tp_size": 4,
      "dp_size": 2
    }
  },
  "test_results": {
    "query_length_test": [...],
    "qps_test": [...]
  }
}
```

### ep_tp_comparison.json
```json
{
  "deployment_comparison": {
    "gpu_memory_usage": {
      "ep_mean": 45.23,
      "tp_mean": 42.18,
      "ep_std": 2.15,
      "tp_std": 3.67
    },
    "deployment_type": {
      "ep_type": "expert_parallel",
      "tp_type": "tensor_parallel"
    }
  },
  "performance_comparison": {
    "query_length_performance": {...},
    "qps_performance": {...}
  }
}
```

## 故障排除

### 常见问题

1. **GPU信息获取失败**
   ```
   获取GPU信息失败: [Errno 2] No such file or directory: 'nvidia-smi'
   ```
   **解决方案**: 确保nvidia-smi在PATH中，或修改脚本中的路径

2. **模型加载失败**
   ```
   ❌ 模型加载失败，请检查服务器状态
   ```
   **解决方案**: 
   - 检查模型路径是否正确
   - 确认服务器是否正常启动
   - 检查端口是否被占用

3. **内存不足**
   ```
   CUDA out of memory
   ```
   **解决方案**:
   - 减少 `--max-total-tokens` 参数
   - 减少 `--chunked-prefill-size` 参数
   - 减少测试的query长度

4. **matplotlib导入失败**
   ```
   警告: matplotlib未安装，跳过图表生成
   ```
   **解决方案**: `pip install matplotlib`

### 性能调优建议

1. **减少测试时间**: 修改 `num_requests_per_test` 参数
2. **调整QPS范围**: 根据实际需求修改 `qps_values`
3. **调整query长度**: 根据实际使用场景修改 `query_lengths`
4. **优化服务器参数**: 调整 `--max-running-requests` 等参数

## 高级用法

### 自定义测试配置

```python
# 在测试脚本中修改这些参数
query_lengths = [64, 128, 256, 512]  # 自定义query长度
qps_values = [1, 4, 8, 16]          # 自定义QPS
num_requests_per_test = 3           # 自定义测试次数
```

### 批量测试

```bash
# 创建批量测试脚本
#!/bin/bash
for i in {1..5}; do
    echo "运行第 $i 轮测试"
    python3 test_single_expert_ep.py
    python3 test_single_expert_tp.py
    python3 compare_ep_tp_performance.py
    sleep 60
done
```

### 自动化分析

```python
# 创建自定义分析脚本
import json
from compare_ep_tp_performance import PerformanceComparison

# 加载结果
comparison = PerformanceComparison(ep_results, tp_results)

# 自定义分析
deployment_diff = comparison.analyze_deployment_differences()
perf_diff = comparison.analyze_performance_differences()

# 生成自定义报告
print(f"TP相对于EP的性能改进: {perf_diff['overall_improvement']:.2f}%")
```

## 总结

通过使用这些增强的测试脚本，你可以：

1. **验证部署配置**: 确认EP和TP部署是否符合预期
2. **全面性能测试**: 测试不同场景下的性能表现
3. **详细结果分析**: 获得统计分析和可视化图表
4. **智能建议**: 根据测试结果获得部署建议

这些工具将帮助你更好地理解和优化单expert模型的并行部署策略。
