# EP vs TP 增强测试功能总结

## 概述

我已经为你的EP和TP测试场景创建了一套完整的增强测试系统，包含部署验证、性能测试、结果分析和可视化功能。

## 新增功能

### 1. 部署验证功能

#### EP部署验证
- **验证目标**: 确认唯一的expert部署8个备份，每个GPU部署一个
- **验证方法**: 
  - 检查GPU内存使用均匀性 (标准差 < 10%)
  - 识别部署类型为 "expert_parallel"
  - 验证配置信息 (EP=8, TP=4, DP=2)

#### TP部署验证
- **验证目标**: 确认唯一的expert切分成8块，每块一个GPU
- **验证方法**:
  - 检查GPU内存使用均匀性 (标准差 < 15%)
  - 识别部署类型为 "tensor_parallel"
  - 验证配置信息 (TP=8, DP=1)

### 2. 性能测试功能

#### 测试组1: 不同Query长度测试
- **QPS**: 1
- **Query长度**: 128, 256, 512, 1024, 2048, 4096
- **输出token**: 100
- **每个长度测试次数**: 5次

#### 测试组2: 不同QPS测试
- **Query长度**: 256
- **QPS**: 1, 2, 4, 8, 16, 32, 64
- **输出token**: 100
- **每个QPS测试次数**: 5次

#### 性能指标
- **TTFT (Time To First Token)**: 首token生成时间
- **TPOT (Time Per Output Token)**: 每个输出token的平均时间
- **Overall Latency**: 总体延迟时间

### 3. 结果分析功能

#### 统计分析
- 平均值、中位数、标准差
- 性能改进百分比计算
- 部署配置对比

#### 可视化图表
- Query长度 vs 性能指标对比
- QPS vs 性能指标对比
- 性能改进百分比图表

#### 智能建议
- 根据测试结果自动生成部署建议
- 性能优劣分析

## 文件结构

```
NetScheduler/
├── test_single_expert_ep.py          # EP部署测试脚本 (增强版)
├── test_single_expert_tp.py          # TP部署测试脚本 (增强版)
├── compare_ep_tp_performance.py      # 性能对比分析脚本 (新增)
├── run_enhanced_tests.sh             # Linux/Mac启动脚本 (新增)
├── run_enhanced_tests.bat            # Windows启动脚本 (新增)
├── USAGE_GUIDE.md                   # 详细使用指南 (新增)
├── ENHANCED_TESTING_SUMMARY.md      # 本总结文档 (新增)
├── ep_test_results.json              # EP测试结果 (运行后生成)
├── tp_test_results.json              # TP测试结果 (运行后生成)
├── ep_tp_comparison.json             # 对比分析结果 (运行后生成)
└── performance_comparison.png        # 性能对比图表 (运行后生成)
```

## 使用方法

### 快速开始

#### Linux/Mac:
```bash
# 给脚本执行权限
chmod +x run_enhanced_tests.sh

# 运行完整测试流程
./run_enhanced_tests.sh

# 仅检查环境
./run_enhanced_tests.sh -c

# 仅运行EP测试
./run_enhanced_tests.sh -e

# 仅运行TP测试
./run_enhanced_tests.sh -t

# 仅运行对比分析
./run_enhanced_tests.sh -r
```

#### Windows:
```cmd
# 运行完整测试流程
run_enhanced_tests.bat

# 仅检查环境
run_enhanced_tests.bat -c

# 仅运行EP测试
run_enhanced_tests.bat -e

# 仅运行TP测试
run_enhanced_tests.bat -t

# 仅运行对比分析
run_enhanced_tests.bat -r
```

### 手动运行

```bash
# 1. 运行EP测试
python test_single_expert_ep.py

# 2. 运行TP测试
python test_single_expert_tp.py

# 3. 运行对比分析
python compare_ep_tp_performance.py
```

## 验证部署是否符合预期

### EP场景验证
运行EP测试后，你应该看到类似输出：
```
=== 验证 Expert Parallel 部署配置 ===
GPU数量: 8
平均GPU内存使用率: 45.23%
GPU内存使用率标准差: 2.15%
✅ EP部署验证通过: GPU内存使用相对均匀，符合expert备份分布特征
```

**验证要点**:
- ✅ GPU内存使用率标准差 < 10%
- ✅ 部署类型识别为 "expert_parallel"
- ✅ 每个GPU都有相似的内存使用

### TP场景验证
运行TP测试后，你应该看到类似输出：
```
=== 验证 Tensor Parallel 部署配置 ===
GPU数量: 8
平均GPU内存使用率: 42.18%
GPU内存使用率标准差: 3.67%
✅ TP部署验证通过: GPU内存使用相对均匀，符合expert切分分布特征
```

**验证要点**:
- ✅ GPU内存使用率标准差 < 15%
- ✅ 部署类型识别为 "tensor_parallel"
- ✅ 每个GPU都有相似的内存使用

## 性能测试结果

### 测试组1: Query长度测试
```
--- Query长度测试结果 ---
Query长度 128:
  TTFT: 平均=242.29ms, 中位数=245.67ms, 标准差=3.45ms
  TPOT: 平均=12.12ms, 中位数=12.34ms, 标准差=0.23ms
  Overall: 平均=1456.34ms, 中位数=1489.23ms, 标准差=32.89ms
```

### 测试组2: QPS测试
```
--- QPS测试结果 ---
QPS 1:
  TTFT: 平均=245.67ms, 中位数=242.29ms, 标准差=5.23ms
  TPOT: 平均=12.34ms, 中位数=12.12ms, 标准差=0.45ms
  Overall: 平均=1489.23ms, 中位数=1456.34ms, 标准差=45.67ms
```

## 对比分析结果

运行对比分析后，你会看到：
```
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

## 结果文件说明

### ep_test_results.json / tp_test_results.json
包含完整的测试结果，包括：
- 部署验证信息
- GPU内存使用情况
- 性能测试数据
- 统计分析结果

### ep_tp_comparison.json
包含对比分析结果，包括：
- 部署配置对比
- 性能差异分析
- 改进百分比计算

### performance_comparison.png
包含4个子图的性能对比图表：
1. Query长度 vs TTFT
2. Query长度 vs TPOT
3. QPS vs Overall Latency
4. 性能改进百分比

## 自定义配置

### 修改测试参数
在测试脚本中修改以下参数：
```python
# 测试配置
query_lengths = [128, 256, 512, 1024, 2048, 4096]  # 自定义query长度
qps_values = [1, 2, 4, 8, 16, 32, 64]              # 自定义QPS
num_requests_per_test = 5                           # 自定义测试次数
```

### 修改模型路径
```python
# 在测试脚本中修改
'--model-path', '/path/to/your/Qwen3-30B-A3B'
```

## 故障排除

### 常见问题

1. **GPU信息获取失败**
   - 确保nvidia-smi在PATH中
   - 检查GPU驱动是否正确安装

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确认服务器是否正常启动
   - 检查端口是否被占用

3. **内存不足**
   - 减少 `--max-total-tokens` 参数
   - 减少 `--chunked-prefill-size` 参数
   - 减少测试的query长度

4. **matplotlib导入失败**
   - 安装matplotlib: `pip install matplotlib numpy`

## 总结

这套增强测试系统提供了：

1. **完整的部署验证**: 自动验证EP和TP部署是否符合预期
2. **全面的性能测试**: 多维度测试不同场景下的性能表现
3. **详细的结果分析**: 统计分析和可视化图表
4. **智能的建议系统**: 根据测试结果提供部署建议
5. **易用的自动化工具**: 一键运行完整测试流程

通过这些工具，你可以：
- 确认部署配置是否正确
- 全面了解两种部署方式的性能差异
- 获得数据支持的部署决策建议
- 持续监控和优化模型性能

建议按照以下顺序使用：
1. 先运行环境检查: `./run_enhanced_tests.sh -c`
2. 修改模型路径
3. 运行完整测试: `./run_enhanced_tests.sh`
4. 查看结果文件和图表
5. 根据建议选择最优部署方式
