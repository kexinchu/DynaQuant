# GPU使用率不均匀深度分析

## 问题现象

即使添加了`--ep-dispatch-algorithm dynamic`参数，GPU使用率仍然不均匀：
- 一个GPU使用率60%+
- 另一个GPU使用率只有16%
- 大部分时间device 0使用率更高
- 偶尔其他GPU使用率也会更高

## 根本原因分析

### 1. 配置冲突问题

**当前EP配置存在冲突**：
```bash
# 环境变量设置
SINGLE_EXPERT_MODE=dp  # 使用DP模式

# 启动参数
--tp-size 8           # 其他层使用TP=8
--dp-size 1           # 其他层使用DP=1
--enable-ep-moe       # 启用expert parallel
--ep-size 8           # expert parallel size = 8
--ep-dispatch-algorithm dynamic  # 动态分配
```

**问题分析**：
- `SINGLE_EXPERT_MODE=dp` 表示expert层使用DP模式
- 但其他层使用 `--tp-size 8`，这意味着非expert层（如attention层）使用TP=8
- 这种混合配置可能导致负载不均

### 2. 模型架构影响

**MoE模型的特点**：
- 只有部分层是MoE层（expert层）
- 其他层（attention、norm等）仍然使用传统的TP/DP
- 不同层的计算密度不同

**负载分布**：
- **Expert层**：使用EP，应该均匀分布
- **Attention层**：使用TP=8，可能存在通信开销不均
- **Norm层**：使用TP=8，计算量相对较小

### 3. 通信开销不均

**TP通信模式**：
- TP需要在GPU间进行all-reduce操作
- 不同GPU的通信延迟可能不同
- 网络拓扑可能导致某些GPU成为瓶颈

**EP通信模式**：
- EP主要进行expert选择和数据分发
- 通信开销相对较小，但仍有影响

## 详细诊断方法

### 1. 检查实际配置

```bash
# 检查环境变量
echo $SINGLE_EXPERT_MODE
echo $CUDA_VISIBLE_DEVICES

# 检查服务器配置
curl http://127.0.0.1:8080/health
```

### 2. 监控GPU使用率模式

```bash
# 使用监控脚本
python gpu_utilization_monitor.py --interval 3 --duration 60
```

### 3. 分析使用率分布

**正常情况下的使用率分布**：
- 所有GPU使用率应该在 ±10% 范围内
- 标准差应该小于10%

**当前问题表现**：
- 使用率差异超过40%
- 标准差大于15%
- 存在明显的负载倾斜

## 解决方案

### 方案1: 纯EP配置（推荐）

**修改配置**：
```bash
# 环境变量
export SINGLE_EXPERT_MODE=dp

# 启动命令
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-235B-A22B/ \
  --tp-size 1 \           # 其他层不使用TP
  --dp-size 8 \           # 其他层使用DP=8
  --enable-ep-moe \
  --ep-size 8 \
  --ep-dispatch-algorithm dynamic \
  --port 8080
```

**优势**：
- 减少TP通信开销
- 所有层都使用DP，负载更均匀
- 简化配置，减少冲突

### 方案2: 纯TP配置

**修改配置**：
```bash
# 环境变量
export SINGLE_EXPERT_MODE=tp

# 启动命令
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-235B-A22B/ \
  --tp-size 8 \           # 所有层都使用TP=8
  --dp-size 1 \           # 不使用DP
  --port 8081
```

**优势**：
- 内存使用更少
- 支持更大模型
- 配置简单

### 方案3: 混合配置优化

**修改配置**：
```bash
# 环境变量
export SINGLE_EXPERT_MODE=dp

# 启动命令
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-235B-A22B/ \
  --tp-size 4 \           # 其他层使用TP=4
  --dp-size 2 \           # 其他层使用DP=2
  --enable-ep-moe \
  --ep-size 8 \
  --ep-dispatch-algorithm dynamic \
  --port 8080
```

**优势**：
- 平衡TP和DP的优势
- 减少TP通信开销
- 保持一定的内存效率

## 验证方法

### 1. 使用率均匀性检查

```python
def check_uniformity(utilizations):
    """检查使用率均匀性"""
    mean_util = statistics.mean(utilizations)
    std_dev = statistics.stdev(utilizations)
    cv = (std_dev / mean_util) * 100  # 变异系数
    
    print(f"平均使用率: {mean_util:.1f}%")
    print(f"标准差: {std_dev:.1f}%")
    print(f"变异系数: {cv:.1f}%")
    
    # 判断标准
    if cv < 10:
        return "优秀"
    elif cv < 15:
        return "良好"
    elif cv < 25:
        return "一般"
    else:
        return "差"
```

### 2. 性能对比测试

**测试指标**：
- GPU使用率均匀性
- 整体吞吐量
- 平均延迟
- 内存使用情况

**测试方法**：
```bash
# 测试不同配置
python test_single_expert_ep.py    # 当前配置
python test_single_expert_tp.py    # TP配置
python test_hybrid_parallel.py     # 混合配置
```

## 预期效果

### 修复前
- GPU使用率差异：40%+
- 标准差：>15%
- 负载倾斜：明显

### 修复后
- GPU使用率差异：<10%
- 标准差：<10%
- 负载倾斜：轻微

## 监控建议

### 1. 持续监控
```bash
# 实时监控
python gpu_utilization_monitor.py --interval 5 --duration 300
```

### 2. 关键指标
- GPU使用率标准差
- 内存使用均匀性
- 温度分布
- 功耗分布

### 3. 告警阈值
- 使用率标准差 > 15%
- 单个GPU使用率 < 10%
- 温度差异 > 10°C

## 总结

GPU使用率不均匀的主要原因是：
1. **配置冲突**：混合使用TP和DP导致负载不均
2. **通信开销**：TP模式下的all-reduce操作不均匀
3. **模型架构**：不同层的计算密度不同

**推荐解决方案**：
1. 使用纯EP配置（`--tp-size 1 --dp-size 8`）
2. 或使用纯TP配置（`--tp-size 8 --dp-size 1`）
3. 避免复杂的混合配置

这样可以显著改善GPU使用率的均匀性，提高整体系统性能。

