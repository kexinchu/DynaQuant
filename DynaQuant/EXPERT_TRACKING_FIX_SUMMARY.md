# Expert Tracking 修复总结

## 问题描述

在 Qwen3-235B-A22B（MoE）推理时，逐层逐 expert 激活统计不正确，表现为：
- 只导出了 layer_0，且只有 expert 1
- 数字明显不对
- 缺少正确的 hot_cold_score 计算

## 根本原因分析

### 1. Hook 位置错误
- 原 hook 在 `select_experts` 函数中，但可能被多次调用或某些路径下不被调用
- 缺少在 MoE 层中的直接统计

### 2. 层索引获取失败
- 代码中尝试从多个来源获取层索引，但都不可靠
- `recorder._current_layer_idx.value` 可能为 None
- 调用栈推断方法不够稳定

### 3. 分布式映射问题
- 在 EP/TP/DP 环境下，物理 expert ID 和逻辑 expert ID 的映射不正确
- 缺少正确的 expert ID 转换逻辑

### 4. 统计覆盖问题
- 多个请求可能覆盖同一个统计结果
- 缺少请求级别的隔离

## 修复方案

### 1. 修复 Hook 位置

#### 在 Qwen3MoeSparseMoeBlock 中添加直接统计
- 在 `op_select_experts` 方法中添加 `_record_expert_activations` 调用
- 在 `forward_normal` 方法中添加统计逻辑
- 确保在路由计算完成后、expert 执行前进行统计

#### 在 FusedMoE 中添加统计
- 在 `forward_cuda` 方法中添加 `_record_expert_activations_in_fused_moe` 调用
- 确保覆盖所有可能的执行路径

### 2. 修复分布式映射

#### 改进 Expert ID 映射逻辑
- 在 `_map_physical_to_logical_expert_id` 方法中添加分布式配置支持
- 正确处理 EP/TP/DP 环境下的 expert ID 转换
- 支持 expert location metadata 和分布式配置两种映射方式

#### 在 MoE 层中添加映射逻辑
- 在 `_map_to_global_expert_id` 方法中实现正确的分布式映射
- 考虑 EP 环境下的 expert 分布

### 3. 改进统计聚合

#### 完善 hot_cold_score 计算
- 在 `get_expert_stats_by_layer` 方法中正确计算 hot_cold_score
- 确保按层分组统计，每层内计算相对分数
- 支持多进程统计聚合

#### 增强统计验证
- 在 `calculate_hot_cold_scores` 方法中添加统计完整性验证
- 显示详细的统计摘要信息

## 修改的文件

### 1. `/sglang-0.4.7/python/sglang/srt/models/qwen3_moe.py`
- 在 `Qwen3MoeSparseMoeBlock` 类中添加 `_record_expert_activations` 方法
- 在 `op_select_experts` 和 `forward_normal` 方法中添加统计调用
- 添加 `_map_to_global_expert_id` 方法处理分布式映射

### 2. `/sglang-0.4.7/python/sglang/srt/layers/moe/topk.py`
- 简化 `select_experts` 函数中的 hook 调用逻辑
- 移除重复的统计代码，避免冲突

### 3. `/sglang-0.4.7/python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- 在 `UnquantizedFusedMoEMethod` 类中添加 `_record_expert_activations_in_fused_moe` 方法
- 在 `forward_cuda` 方法中添加统计调用

### 4. `/sglang-0.4.7/python/sglang/srt/managers/expert_distribution.py`
- 改进 `_map_physical_to_logical_expert_id` 方法，支持分布式配置
- 完善 `_ExpertDistributionRecorderNoop` 类的 expert tracking 支持

### 5. `/expert_tracking_launcher.py`
- 改进 `calculate_hot_cold_scores` 方法，添加统计验证和详细日志

## 测试计划

### 1. 快速验证测试
```bash
python quick_test_expert_tracking.py
```
- 发送单个请求
- 检查是否能获取到 expert 统计信息
- 验证统计数据的完整性

### 2. 综合测试
```bash
python test_expert_tracking_fix.py
```
- 运行多个不同类型的测试用例
- 分析 expert 统计变化
- 验证 hot_cold_score 计算正确性

### 3. 完整测试
```bash
python expert_tracking_launcher.py --workers 4
```
- 使用原有的完整测试流程
- 验证多线程环境下的统计正确性

## 预期效果

修复后应该能够看到：

1. **完整的层统计**：所有 MoE 层都有统计信息
2. **正确的 Expert 数量**：每层显示正确的 expert 数量
3. **准确的激活统计**：每个 expert 的激活次数和 token 数量正确
4. **正确的 hot_cold_score**：基于每层内最大激活次数计算的相对分数
5. **分布式支持**：在 EP/TP/DP 环境下正确映射 expert ID

## 输出格式

修复后的输出应该符合目标 Schema：
```json
{
  "layer_0": {
    "experts": {
      "0": {"activation_count": 128, "total_tokens": 576, "hot_cold_score": 0.67},
      "1": {"activation_count": 95, "total_tokens": 432, "hot_cold_score": 0.50},
      ...
    }
  },
  "layer_1": {
    "experts": {
      ...
    }
  }
}
```

## 注意事项

1. **最小侵入性**：所有修改都是最小侵入的，不影响原有功能
2. **错误处理**：所有统计代码都有完善的错误处理，不会影响正常推理
3. **性能影响**：统计操作的开销很小，对推理性能影响可忽略
4. **兼容性**：支持各种分布式配置和 MoE 实现方式

## 验证步骤

1. 启动 SGLang 服务
2. 运行快速验证测试
3. 检查日志输出，确认有 expert 统计信息
4. 运行完整测试，验证统计数据的正确性
5. 检查导出的 JSON 文件，确认格式和内容正确