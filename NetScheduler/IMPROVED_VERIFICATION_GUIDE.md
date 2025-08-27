# 改进的EP/TP部署验证指南

## 问题分析

你提出的问题非常正确。当前的验证逻辑确实存在一些局限性：

### 1. 当前验证逻辑的问题

#### GPU内存使用均匀性验证的局限性
- **EP部署**: 理论上每个GPU都有完整的expert副本，内存使用应该更均匀
- **TP部署**: expert被切分到不同GPU，但其他层（如attention、embedding等）可能在不同GPU上分布不均匀

但实际上，这种判断可能不够准确，因为：
- 模型的其他组件（非expert层）的内存分布可能掩盖expert层的分布特征
- GPU内存使用还受到KV cache、中间激活值等因素影响
- 不同层的并行策略可能不同，导致内存使用模式复杂

### 2. 更准确的验证方法

我提供了三种改进的验证方法：

## 方法1: 增强的部署验证器

### 功能特点
- **多维度检查**: 不仅检查GPU内存使用，还检查配置、环境变量、并行状态
- **配置推断**: 从进程信息和环境变量推断实际配置
- **综合判断**: 基于多个指标进行综合判断，而不是单一指标

### 使用方法
```bash
# 验证EP部署
python enhanced_deployment_verifier.py --deployment-type ep --server-url http://127.0.0.1:8080

# 验证TP部署
python enhanced_deployment_verifier.py --deployment-type tp --server-url http://127.0.0.1:8081

# 输出结果到文件
python enhanced_deployment_verifier.py --deployment-type ep --output ep_verification.json
```

### 验证指标
1. **配置检查**: 验证启动参数是否正确
2. **环境变量检查**: 验证`SINGLE_EXPERT_MODE`等环境变量
3. **GPU内存使用检查**: 检查内存使用的均匀性
4. **Expert分布检查**: 推断expert的分布模式
5. **并行状态检查**: 验证并行配置

## 方法2: SGLang内部状态检查器（推荐）

### 功能特点
- **直接访问内部状态**: 通过修改sglang源码，直接获取内部并行状态
- **实时状态查询**: 通过API接口实时查询运行时的状态
- **最准确的验证**: 直接访问sglang的内部数据结构

### 使用方法

#### 步骤1: 设置内部状态检查功能
```bash
# 添加内部状态检查API到sglang
python sglang_internal_state_checker.py --action setup --sglang-path sglang-0.4.7
```

#### 步骤2: 验证部署
```bash
# 验证EP部署
python sglang_internal_state_checker.py --action verify --deployment-type ep

# 验证TP部署
python sglang_internal_state_checker.py --action verify --deployment-type tp
```

#### 步骤3: 清理（可选）
```bash
# 清理修改的文件
python sglang_internal_state_checker.py --action cleanup
```

### 验证指标
1. **内部并行状态**: 直接查询sglang的并行状态
2. **环境信息**: 获取运行时环境变量和配置
3. **并行组信息**: 查询所有并行组的详细信息
4. **实时状态**: 获取运行时的实时状态信息

## 方法3: 修改sglang源码添加验证功能

### 直接修改sglang源码的优势
1. **最准确的验证**: 直接访问内部数据结构
2. **实时监控**: 可以在运行时实时监控状态
3. **详细信息**: 获取最详细的并行和分布信息

### 修改的文件
1. **`parallel_state.py`**: 添加状态查询函数
2. **`qwen3_moe.py`**: 添加expert分布查询函数
3. **新增API模块**: 提供HTTP接口查询状态

### 添加的功能
```python
# 获取内部并行状态
def get_internal_parallel_state() -> Dict[str, Any]:
    """获取内部并行状态信息"""
    return {
        'tensor_parallel_world_size': get_tensor_model_parallel_world_size(),
        'tensor_parallel_rank': get_tensor_model_parallel_rank(),
        'moe_expert_parallel_world_size': get_moe_expert_parallel_world_size(),
        'moe_expert_parallel_rank': get_moe_expert_parallel_rank(),
        'is_initialized': _TP is not None,
        # ... 更多详细信息
    }

# 获取expert分布信息
def get_expert_distribution_info(self) -> Dict[str, Any]:
    """获取expert分布信息"""
    return {
        'layer_id': self.layer_id,
        'tp_size': self.tp_size,
        'num_experts': self.num_experts,
        'local_expert_ids': list(range(self.start_expert_id, self.end_expert_id + 1)),
        'distribution_type': 'tp' if self.tp_size > 1 else 'dp',
        # ... 更多详细信息
    }
```

## 验证标准

### EP部署验证标准
1. **并行配置**: `moe_expert_parallel_world_size > 1`
2. **环境配置**: `SINGLE_EXPERT_MODE = 'dp'`
3. **Expert分布**: `distribution_type = 'dp'`
4. **内存使用**: 相对均匀（CV < 15%）

### TP部署验证标准
1. **并行配置**: `tensor_parallel_world_size > 1`
2. **环境配置**: `SINGLE_EXPERT_MODE = 'tp'`
3. **Expert分布**: `distribution_type = 'tp'`
4. **内存使用**: 相对均匀（CV < 25%）

## 使用建议

### 1. 推荐使用顺序
1. **首先使用方法2**: SGLang内部状态检查器，这是最准确的方法
2. **备选使用方法1**: 增强的部署验证器，不需要修改源码
3. **最后使用方法3**: 直接修改源码，适合需要深度定制的情况

### 2. 验证流程
```bash
# 1. 设置内部状态检查功能
python sglang_internal_state_checker.py --action setup

# 2. 启动EP服务器
export SINGLE_EXPERT_MODE=dp
python -m sglang.launch_server --model-path /path/to/model --tp-size 4 --dp-size 2 --enable-ep-moe --ep-size 8

# 3. 验证EP部署
python sglang_internal_state_checker.py --action verify --deployment-type ep

# 4. 启动TP服务器
export SINGLE_EXPERT_MODE=tp
python -m sglang.launch_server --model-path /path/to/model --tp-size 8 --dp-size 1

# 5. 验证TP部署
python sglang_internal_state_checker.py --action verify --deployment-type tp
```

### 3. 结果解读
- **验证通过**: 所有检查项都通过，部署符合预期
- **验证失败**: 部分检查项失败，需要检查配置
- **警告信息**: 某些检查项有警告，但不影响整体验证

## 优势对比

| 方法 | 准确性 | 复杂度 | 是否需要修改源码 | 实时性 | 推荐度 |
|------|--------|--------|------------------|--------|--------|
| 原始方法 | 中等 | 低 | 否 | 否 | ⭐⭐ |
| 增强验证器 | 高 | 中 | 否 | 否 | ⭐⭐⭐ |
| 内部状态检查器 | 最高 | 中 | 是 | 是 | ⭐⭐⭐⭐⭐ |
| 直接修改源码 | 最高 | 高 | 是 | 是 | ⭐⭐⭐⭐ |

## 总结

通过使用这些改进的验证方法，你可以：

1. **更准确地验证部署**: 直接访问sglang的内部状态
2. **实时监控状态**: 在运行时实时查询状态信息
3. **多维度验证**: 从多个角度验证部署的正确性
4. **详细的结果分析**: 获得详细的验证结果和建议

这些方法解决了原始验证逻辑的局限性，提供了更准确、更全面的EP/TP部署验证功能。
