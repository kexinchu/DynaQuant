# GPTQ量化支持

## 概述

本系统现已支持GPTQ-Int4量化模型格式，可以自动检测和反量化GPTQ权重，实现与FP格式的兼容。

## GPTQ格式说明

GPTQ量化模型使用特殊的权重格式，包含以下组件：

- `qweight`: 量化的权重（int32格式，packed int4）
- `qzeros`: 量化的零点
- `scales`: 缩放因子
- `g_idx`: 分组索引（可选）

## 支持的权重名称格式

系统支持以下GPTQ权重名称格式：

```
model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}.weight
```

例如：
- `model.layers.0.mlp.experts.0.down_proj.weight`
- `model.layers.0.mlp.experts.0.gate_proj.weight`
- `model.layers.0.mlp.experts.0.up_proj.weight`

## 自动组件映射

系统会自动将权重名称映射到对应的GPTQ组件：

```
model.layers.0.mlp.experts.0.down_proj.weight
↓
model.layers.0.mlp.experts.0.down_proj.qweight
model.layers.0.mlp.experts.0.down_proj.qzeros
model.layers.0.mlp.experts.0.down_proj.scales
model.layers.0.mlp.experts.0.down_proj.g_idx (可选)
```

## 配置示例

```yaml
model:
  mixed_precision:
    int4_path: "/path/to/gptq-model"
    weight_mapping:
      "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
      "model.layers.0.mlp.experts.0.gate_proj.weight": "int4"
      "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
```

## 反量化算法

系统使用以下算法进行GPTQ反量化：

1. **解包int32**: 将packed int32解包为int4
2. **反量化公式**: `weight = scale * (qweight - qzero)`
3. **维度处理**: 自动处理不同维度的张量

## 测试方法

运行GPTQ权重加载测试：

```bash
python examples/test_gptq_loading.py
```

测试内容包括：
- GPTQ格式检测
- 组件解析
- 反量化验证
- 权重加载测试

## 错误处理

系统提供详细的错误信息：

- 组件缺失警告
- 形状不匹配提示
- 反量化失败处理
- 加载统计信息

## 性能优化

- 优化的int32到int4解包算法
- 权重文件缓存机制
- 批量加载支持
- 内存高效处理

## 兼容性

- 支持多种GPTQ变体
- 向后兼容FP格式
- 自动格式检测
- 灵活的配置选项
