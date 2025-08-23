# 修复说明

本文档说明了为解决Qwen3-30B-A3B单expert模型并行策略测试中的问题所做的修改。

## 问题1: Expert分配除零错误

### 问题描述
当只有一个expert时，sglang的MoE专家分配逻辑会出现除零错误：
```
ZeroDivisionError: integer division or modulo by zero
```

### 错误位置
- `sglang-0.4.7/python/sglang/srt/managers/expert_location.py`
- `_compute_gpu_id_of_physical_expert`函数
- `compute_logical_to_rank_dispatch_physical_map`函数
- `_init_common`函数

### 修复内容

#### 1. 修复`_compute_gpu_id_of_physical_expert`函数
```python
def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_physical_experts: int
) -> int:
    # 修复除零错误：当num_local_physical_experts为0时，直接返回physical_expert_id
    if num_local_physical_experts <= 0:
        return physical_expert_id % 8  # 假设有8个GPU，可以根据实际情况调整
    return physical_expert_id // num_local_physical_experts
```

#### 2. 修复`compute_logical_to_rank_dispatch_physical_map`函数
```python
def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    r = random.Random(seed)

    # 修复除零错误：当只有一个expert时，每个GPU分配一个expert
    if num_physical_experts <= num_gpus:
        num_local_physical_experts = 1
    else:
        num_local_physical_experts = num_physical_experts // num_gpus
    
    # ... 其余代码保持不变
```

#### 3. 修复`_init_common`函数
```python
@staticmethod
def _init_common(server_args: ServerArgs, model_config: ModelConfig):
    # ... 前面的代码保持不变
    
    # 修复除零错误：当expert数量小于ep_size时，每个GPU分配一个expert
    if num_physical_experts <= ep_size:
        num_local_physical_experts = 1
    else:
        num_local_physical_experts = num_physical_experts // ep_size
    
    # ... 其余代码保持不变
```

## 问题2: MoE不支持TP

### 问题描述
sglang的MoE实现默认不支持tensor parallelism，需要特殊配置。

### 解决方案
sglang已经内置了`SingleExpertMoE`类，支持两种模式：
- `dp`模式：每个GPU都有expert的完整副本
- `tp`模式：expert在多个GPU上切分

### 配置方法

#### 1. 环境变量配置
```bash
# Expert Parallel (DP模式)
export SINGLE_EXPERT_MODE=dp

# Tensor Parallel (TP模式)
export SINGLE_EXPERT_MODE=tp
```

#### 2. 测试脚本修改
修改了以下测试脚本，添加了正确的环境变量配置：

- `test_single_expert_ep.py`
- `test_single_expert_tp.py`
- `test_hybrid_parallel.py`

#### 3. 启动命令示例

**Expert Parallel配置**:
```bash
export SINGLE_EXPERT_MODE=dp
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-30B-A3B \
  --tp-size 4 \
  --dp-size 2 \
  --enable-ep-moe \
  --ep-size 8 \
  --host 127.0.0.1 --port 8080
```

**Tensor Parallel配置**:
```bash
export SINGLE_EXPERT_MODE=tp
python3 -m sglang.launch_server \
  --model-path /dev/shm/Qwen3-30B-A3B \
  --tp-size 8 \
  --dp-size 1 \
  --host 127.0.0.1 --port 8081
```

## 修改的文件列表

### sglang源码修改
1. `sglang-0.4.7/python/sglang/srt/managers/expert_location.py`
   - 修复了除零错误
   - 改进了expert分配逻辑

### 测试脚本修改
1. `test_single_expert_ep.py`
   - 添加了`SINGLE_EXPERT_MODE=dp`环境变量
   - 使用正确的EP配置

2. `test_single_expert_tp.py`
   - 添加了`SINGLE_EXPERT_MODE=tp`环境变量
   - 使用正确的TP配置

3. `test_hybrid_parallel.py`
   - 添加了环境变量配置
   - 改进了配置管理

## 测试验证

### 1. 基础测试
```bash
# 测试Expert Parallel
python3 test_single_expert_ep.py

# 测试Tensor Parallel
python3 test_single_expert_tp.py
```

### 2. 混合测试
```bash
# 运行两种配置的对比测试
python3 test_hybrid_parallel.py --config both
```

### 3. 使用启动脚本
```bash
# Linux/Mac
./run_tests.sh --ep
./run_tests.sh --tp
./run_tests.sh --all
```

## 注意事项

1. **模型路径**: 确保模型路径正确，并且模型已经通过`prune_qwen3_a22B-to_single_expert.py`处理
2. **GPU数量**: 确保有足够的GPU资源（推荐8张GPU）
3. **内存配置**: 根据实际情况调整`max_total_tokens`和`chunked_prefill_size`参数
4. **环境变量**: 确保正确设置了`SINGLE_EXPERT_MODE`环境变量

## 性能对比

修复后的配置应该能够正常运行，并提供以下性能对比：

| 配置 | Expert层策略 | 其他层策略 | 优势 |
|------|-------------|-----------|------|
| Expert Parallel | DP=8 (每GPU完整副本) | TP=4, DP=2 | 减少通信开销，提高吞吐量 |
| Tensor Parallel | TP=8 (GPU切分) | TP=8 | 减少内存占用，支持更大模型 |

## 故障排除

如果仍然遇到问题，请检查：

1. **日志信息**: 查看详细的错误日志
2. **环境变量**: 确认`SINGLE_EXPERT_MODE`设置正确
3. **模型配置**: 确认模型配置文件中的expert数量
4. **GPU资源**: 确认GPU数量和内存充足

## 后续优化

1. **动态配置**: 支持运行时动态切换并行策略
2. **性能调优**: 根据实际硬件优化参数配置
3. **监控指标**: 添加更详细的性能监控
4. **自动化测试**: 完善自动化测试流程
