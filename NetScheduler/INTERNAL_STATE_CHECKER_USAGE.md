# 内部状态检查器使用指南

## 概述

本指南介绍如何使用新集成的内部状态检查器来验证EP和TP部署配置。内部状态检查器提供了最准确的部署验证方法，通过直接查询SGLang的内部并行状态和expert分布信息。

## 主要特性

### 1. 准确的部署验证
- **内部状态查询**: 直接访问SGLang的内部并行组信息
- **Expert分布检查**: 验证expert在GPU上的实际分布情况
- **环境变量验证**: 检查并行配置的环境变量设置

### 2. 自动回退机制
- 如果内部状态检查器不可用，自动回退到GPU内存使用分析
- 确保测试的可靠性

### 3. 详细的验证报告
- 并行组信息
- Expert分布状态
- 环境变量配置
- 验证结果和错误信息

## 使用方法

### 方法1: 使用自动化脚本 (推荐)

#### Linux/Mac
```bash
# 运行完整测试流程
./run_enhanced_tests.sh

# 仅运行EP测试
./run_enhanced_tests.sh -e

# 仅运行TP测试
./run_enhanced_tests.sh -t

# 仅检查环境
./run_enhanced_tests.sh -c
```

#### Windows
```cmd
# 运行完整测试流程
run_enhanced_tests.bat

# 仅运行EP测试
run_enhanced_tests.bat -e

# 仅运行TP测试
run_enhanced_tests.bat -t

# 仅检查环境
run_enhanced_tests.bat -c
```

### 方法2: 直接运行测试脚本

```bash
# EP测试
python test_single_expert_ep.py

# TP测试
python test_single_expert_tp.py

# 性能对比分析
python compare_ep_tp_performance.py
```

## 验证结果解读

### 成功验证示例
```
=== 内部状态验证结果 ===
✅ EP部署验证通过: 内部状态检查确认配置正确

并行组信息: {
  'tp_size': 8,
  'dp_size': 1,
  'ep_size': 8,
  'world_size': 8
}

环境变量信息: {
  'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
  'SINGLE_EXPERT_MODE': 'ep'
}
```

### 验证失败示例
```
=== 内部状态验证结果 ===
❌ EP部署验证失败: 内部状态检查发现配置问题
错误信息: Expert分布不符合EP模式，检测到TP切分特征
```

## 文件结构

```
├── test_single_expert_ep.py          # EP测试脚本 (已更新)
├── test_single_expert_tp.py          # TP测试脚本 (已更新)
├── sglang_internal_state_checker.py  # 内部状态检查器
├── compare_ep_tp_performance.py      # 性能对比分析 (已更新)
├── run_enhanced_tests.sh             # Linux/Mac自动化脚本 (已更新)
├── run_enhanced_tests.bat            # Windows自动化脚本 (已更新)
└── sglang-0.4.7/                     # SGLang源码目录
    ├── python/sglang/srt/distributed/parallel_state.py
    ├── python/sglang/srt/models/qwen3_moe.py
    └── python/sglang/srt/internal_state_api.py
```

## 输出文件

### 测试结果文件
- `ep_test_results.json`: EP测试结果，包含内部状态验证信息
- `tp_test_results.json`: TP测试结果，包含内部状态验证信息
- `ep_tp_comparison.json`: 性能对比分析结果
- `performance_comparison.png`: 性能对比图表

### 内部状态信息结构
```json
{
  "deployment_info": {
    "internal_state": {
      "parallel_state": {
        "tp_size": 8,
        "dp_size": 1,
        "ep_size": 8
      },
      "environment_info": {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "SINGLE_EXPERT_MODE": "ep"
      },
      "verification_result": {
        "is_valid": true,
        "expert_distribution": {
          "type": "expert_parallel",
          "distribution": "uniform"
        },
        "parallel_config": {
          "mode": "expert_parallel",
          "ep_size": 8
        }
      }
    }
  }
}
```

## 故障排除

### 1. 内部状态检查器不可用
**症状**: 显示"回退到基础验证方法"
**解决方案**: 
- 确保SGLang源码目录存在
- 检查`sglang_internal_state_checker.py`文件
- 确保有足够的权限修改SGLang源码

### 2. API服务器启动失败
**症状**: 无法连接到内部API服务器
**解决方案**:
- 检查端口8080/8081是否被占用
- 确保Flask已正确安装
- 检查防火墙设置

### 3. 验证结果不准确
**症状**: 验证结果与预期不符
**解决方案**:
- 检查SGLang版本兼容性
- 确认模型配置正确
- 查看详细的错误信息

## 性能影响

### 内部状态检查器开销
- **设置时间**: ~30秒 (修改源码、安装依赖、重新安装SGLang)
- **验证时间**: ~5秒 (查询内部状态)
- **内存开销**: 最小 (仅启动Flask API服务器)

### 回退方法开销
- **验证时间**: ~2秒 (GPU信息查询)
- **内存开销**: 无额外开销

## 最佳实践

### 1. 首次使用
```bash
# 1. 检查环境
./run_enhanced_tests.sh -c

# 2. 运行完整测试
./run_enhanced_tests.sh
```

### 2. 开发调试
```bash
# 仅运行单个测试进行调试
./run_enhanced_tests.sh -e  # 或 -t
```

### 3. 批量测试
```bash
# 运行完整流程并生成报告
./run_enhanced_tests.sh -a
```

## 技术细节

### 内部状态检查器工作原理
1. **源码修改**: 在SGLang源码中添加内部状态查询函数
2. **API服务器**: 启动Flask服务器提供HTTP接口
3. **状态查询**: 通过HTTP请求获取并行状态和expert分布信息
4. **结果验证**: 根据EP/TP特征验证配置正确性

### 回退机制
1. **GPU内存分析**: 分析各GPU内存使用模式
2. **均匀性检查**: 计算内存使用标准差
3. **模式识别**: 根据内存分布特征推断部署模式

## 更新日志

### v2.0 (当前版本)
- ✅ 集成内部状态检查器
- ✅ 自动回退机制
- ✅ 详细的验证报告
- ✅ 跨平台支持 (Linux/Mac/Windows)
- ✅ 自动化脚本更新

### v1.0 (之前版本)
- ✅ 基础GPU内存验证
- ✅ 性能测试功能
- ✅ 结果对比分析
