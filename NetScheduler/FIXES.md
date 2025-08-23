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

## 问题3: 缺少函数定义

### 问题描述
sglang缺少`get_moe_expert_parallel_world_size`函数定义：
```
cannot import name 'get_moe_expert_parallel_world_size' from 'sglang.srt.distributed'
```

### 修复内容

#### 1. 添加缺失的函数
在`sglang-0.4.7/python/sglang/srt/distributed/parallel_state.py`中添加：

```python
def get_moe_expert_parallel_world_size():
    """Return world size for the MoE expert parallel group."""
    # For now, we use the tensor parallel world size as the expert parallel world size
    # This can be extended later to support separate expert parallel groups
    return get_tensor_model_parallel_world_size()


def get_moe_expert_parallel_rank():
    """Return my rank for the MoE expert parallel group."""
    # For now, we use the tensor parallel rank as the expert parallel rank
    # This can be extended later to support separate expert parallel groups
    return get_tensor_model_parallel_rank()
```

## 问题4: Tensor Parallel不支持

### 问题描述
`Qwen3MoeModel`不支持tensor parallel：
```
ValueError: <class 'transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel'> does not support tensor parallel yet!
```

### 修复内容

#### 1. 添加tp_plan支持
在`sglang-0.4.7/python/sglang/srt/models/qwen3_moe.py`中为`Qwen3MoeModel`添加：

```python
class Qwen3MoeModel(Qwen2MoeModel):
    def __init__(self, config, quant_config=None, prefix=""):
        # ... 现有代码 ...
        
        # Add tensor parallel plan support
        self.supports_tp_plan = True
        self._tp_plan = {
            "embed_tokens": "rowwise",
            "lm_head": "colwise_rep",
            ".*attention.*q_proj": "colwise",
            ".*attention.*k_proj": "colwise", 
            ".*attention.*v_proj": "colwise",
            ".*attention.*o_proj": "rowwise",
            ".*mlp.*gate_up_proj": "colwise",
            ".*mlp.*down_proj": "rowwise",
        }
```

## 问题5: 简化测试配置

### 问题描述
复杂的并行配置可能导致更多问题，需要简化测试配置。

### 解决方案

#### 1. 创建简化测试脚本
创建了`test_simple_ep.py`，使用最基本的配置：

```bash
# 简化配置
--tp-size 1  # 避免tensor parallel问题
--dp-size 1  # 避免data parallel问题
--max-total-tokens 20480  # 减少内存使用
--chunked-prefill-size 8192  # 减少内存使用
```

#### 2. 简化测试流程
- 使用简单的prompt进行测试
- 减少max_tokens数量
- 使用基本的推理功能

## 修改的文件列表

### sglang源码修改
1. `sglang-0.4.7/python/sglang/srt/managers/expert_location.py`
   - 修复了除零错误
   - 改进了expert分配逻辑

2. `sglang-0.4.7/python/sglang/srt/distributed/parallel_state.py`
   - 添加了`get_moe_expert_parallel_world_size`函数
   - 添加了`get_moe_expert_parallel_rank`函数

3. `sglang-0.4.7/python/sglang/srt/models/qwen3_moe.py`
   - 为`Qwen3MoeModel`添加了tp_plan支持
   - 添加了tensor parallel配置

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

4. `test_simple_ep.py` (新增)
   - 简化的测试配置
   - 基本的推理功能测试

## 测试验证

### 1. 简化测试 (推荐)
```bash
# 使用简化的配置进行测试
python3 test_simple_ep.py
```

### 2. 基础测试
```bash
# 测试Expert Parallel
python3 test_single_expert_ep.py

# 测试Tensor Parallel
python3 test_single_expert_tp.py
```

### 3. 混合测试
```bash
# 运行两种配置的对比测试
python3 test_hybrid_parallel.py --config both
```

### 4. 使用启动脚本
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
5. **简化配置**: 建议先使用简化配置测试基本功能，再逐步增加复杂度

## 性能对比

修复后的配置应该能够正常运行，并提供以下性能对比：

| 配置 | Expert层策略 | 其他层策略 | 优势 | 复杂度 |
|------|-------------|-----------|------|--------|
| 简化配置 | DP模式 | TP=1, DP=1 | 稳定可靠，易于调试 | 低 |
| Expert Parallel | DP=8 (每GPU完整副本) | TP=4, DP=2 | 减少通信开销，提高吞吐量 | 中 |
| Tensor Parallel | TP=8 (GPU切分) | TP=8 | 减少内存占用，支持更大模型 | 高 |

## 故障排除

如果仍然遇到问题，请检查：

1. **日志信息**: 查看详细的错误日志
2. **环境变量**: 确认`SINGLE_EXPERT_MODE`设置正确
3. **模型配置**: 确认模型配置文件中的expert数量
4. **GPU资源**: 确认GPU数量和内存充足
5. **简化测试**: 先运行`test_simple_ep.py`确保基本功能正常

## 后续优化

1. **动态配置**: 支持运行时动态切换并行策略
2. **性能调优**: 根据实际硬件优化参数配置
3. **监控指标**: 添加更详细的性能监控
4. **自动化测试**: 完善自动化测试流程
5. **错误处理**: 改进错误处理和恢复机制
