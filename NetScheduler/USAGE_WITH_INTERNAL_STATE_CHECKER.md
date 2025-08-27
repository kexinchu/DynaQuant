# 使用内部状态检查器的EP/TP测试指南

## 概述

本指南说明如何使用集成了SGLang内部状态检查器的测试代码，这是最准确的EP/TP部署验证方法。

## 文件结构

```
├── test_single_expert_ep.py          # EP测试文件（已集成内部状态检查器）
├── test_single_expert_tp.py          # TP测试文件（已集成内部状态检查器）
├── sglang_internal_state_checker.py  # 内部状态检查器
├── enhanced_deployment_verifier.py   # 增强的部署验证器（备选方案）
└── USAGE_WITH_INTERNAL_STATE_CHECKER.md  # 本指南
```

## 前置条件

### 1. 确保SGLang源码存在
```bash
# 检查sglang-0.4.7目录是否存在
ls -la sglang-0.4.7/
```

### 2. 检查依赖
```bash
# 检查Python内置模块（内部状态API使用Python内置的http.server）
python -c "import http.server, urllib.parse, threading; print('✅ 依赖检查通过')"
```

## 依赖要求

- Python 3.7+
- SGLang 0.4.7
- requests（用于HTTP请求）
- Python内置模块：http.server, urllib.parse, threading

## 使用方法

### 方法1: 直接运行测试文件（推荐）

#### 运行EP测试
```bash
# 设置环境变量
export SINGLE_EXPERT_MODE=dp

# 运行EP测试
python test_single_expert_ep.py
```

#### 运行TP测试
```bash
# 设置环境变量
export SINGLE_EXPERT_MODE=tp

# 运行TP测试
python test_single_expert_tp.py
```

### 方法2: 手动设置和验证

#### 步骤1: 设置内部状态检查功能
```bash
# 设置SGLang内部状态检查功能
python sglang_internal_state_checker.py --action setup --sglang-path sglang-0.4.7
```

#### 步骤2: 启动服务器
```bash
# EP服务器
export SINGLE_EXPERT_MODE=dp
python -m sglang.launch_server --model-path /path/to/model --tp-size 8 --dp-size 1 --enable-ep-moe --ep-size 8 --port 8080

# 或TP服务器
export SINGLE_EXPERT_MODE=tp
python -m sglang.launch_server --model-path /path/to/model --tp-size 8 --dp-size 1 --port 8081
```

#### 步骤3: 验证部署
```bash
# 验证EP部署
python sglang_internal_state_checker.py --action verify --deployment-type ep

# 验证TP部署
python sglang_internal_state_checker.py --action verify --deployment-type tp
```

## 验证流程说明

### 1. 自动设置阶段
测试文件会自动执行以下步骤：
- 检查`sglang_internal_state_checker.py`是否存在
- 运行设置命令，修改SGLang源码添加内部状态API
- 检查Python内置模块是否可用

### 2. 内部状态验证阶段
- 启动内部状态API服务器（端口8082）
- 查询SGLang的内部并行状态
- 查询环境变量和配置信息
- 执行多维度验证

### 3. 性能测试阶段
- 执行传统的性能测试
- 收集TTFT、TPOT、Overall Latency等指标
- 生成详细的测试报告

### 4. 清理阶段
- 停止内部状态API服务器
- 清理所有相关进程

## 验证指标

### EP部署验证标准
1. **并行配置**: `moe_expert_parallel_world_size > 1`
2. **环境配置**: `SINGLE_EXPERT_MODE = 'dp'`
3. **GPU内存使用**: 相对均匀（CV < 15%）

### TP部署验证标准
1. **并行配置**: `tensor_parallel_world_size > 1`
2. **环境配置**: `SINGLE_EXPERT_MODE = 'tp'`
3. **GPU内存使用**: 相对均匀（CV < 25%）

## 输出结果

### 1. 控制台输出
```
=== 设置SGLang内部状态检查功能 ===
✅ SGLang内部状态检查功能设置完成

=== 启动内部状态API服务器 (端口: 8082) ===
✅ 内部状态API服务器启动成功: http://127.0.0.1:8082

=== 使用内部状态验证 EP 部署 ===
并行状态信息:
  Tensor Parallel World Size: 8
  Data Parallel World Size: 1
  MoE Expert Parallel World Size: 8
  是否已初始化: True

环境信息:
  Single Expert Mode: dp
  CUDA Visible Devices: 0,1,2,3,4,5,6,7

✅ EP部署验证通过 (2/2 项检查通过)
```

### 2. 结果文件
- `ep_test_results.json`: EP测试结果
- `tp_test_results.json`: TP测试结果

结果文件包含：
- 部署验证信息（包括内部状态验证结果）
- 性能测试数据
- GPU使用情况
- 详细的统计信息

## 故障排除

### 1. 找不到sglang_internal_state_checker.py
```
❌ 找不到 sglang_internal_state_checker.py，请确保该文件存在
```
**解决方案**: 确保`sglang_internal_state_checker.py`文件在当前目录中。

### 2. 设置失败
```
❌ 设置失败: [错误信息]
```
**解决方案**: 
- 检查SGLang源码路径是否正确
- 确保有足够的权限修改SGLang源码
- 检查Python环境和依赖

### 3. 内部状态API服务器启动失败
```
❌ 内部状态API服务器启动失败: [错误信息]
```
**解决方案**:
- 检查端口8082是否被占用
- 确保Flask已正确安装
- 检查防火墙设置

### 4. 验证失败
```
❌ EP部署验证失败 (0/2 项检查通过)
```
**解决方案**:
- 检查服务器是否正确启动
- 验证环境变量设置
- 检查并行配置参数

## 高级用法

### 1. 自定义端口
```bash
# 使用自定义端口
python sglang_internal_state_checker.py --action verify --deployment-type ep --api-port 8083
```

### 2. 输出到文件
```bash
# 将验证结果保存到文件
python sglang_internal_state_checker.py --action verify --deployment-type ep --output ep_verification.json
```

### 3. 清理修改
```bash
# 清理对SGLang源码的修改
python sglang_internal_state_checker.py --action cleanup
```

## 优势对比

| 特性 | 传统验证方法 | 内部状态检查器 |
|------|-------------|---------------|
| 准确性 | 中等 | 最高 |
| 实时性 | 否 | 是 |
| 详细信息 | 有限 | 丰富 |
| 设置复杂度 | 低 | 中等 |
| 依赖修改源码 | 否 | 是 |

## 注意事项

1. **源码修改**: 此方法会修改SGLang源码，建议在测试环境中使用
2. **端口冲突**: 内部状态API使用端口8082，确保该端口可用
3. **权限要求**: 需要足够的权限来修改SGLang源码和安装依赖
4. **环境隔离**: 建议在虚拟环境中使用，避免影响其他项目

## 总结

使用内部状态检查器的测试代码提供了最准确的EP/TP部署验证方法。通过直接访问SGLang的内部状态，可以获得最可靠的验证结果。虽然设置过程稍复杂，但验证的准确性和详细程度远超传统方法。
