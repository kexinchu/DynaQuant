# DynaQuant - 基于SGLang的混合精度MoE模型部署系统

DynaQuant是一个基于SGLang 0.4.7的混合精度MoE（Mixture of Experts）模型部署系统，支持混合精度推理、选择性权重加载和专家激活跟踪。系统可以根据配置文件从不同精度的权重文件中选择性加载参数，支持FP16、FP8、Int4等不同精度的混合使用，并提供实时专家激活统计功能。

## 📋 项目概述

本项目基于原始SGLang进行了大量扩展和修改，主要专注于：
- **混合精度推理**: 支持不同精度权重的混合使用
- **专家激活跟踪**: 实时监控MoE模型中每个expert的激活情况
- **动态量化策略**: 基于expert激活热度进行动态量化
- **Hot-Cold分析**: 计算expert的激活热度分数
- **GPTQ量化支持**: 支持GPTQ-Int4量化模型格式

## 🔄 相对于原始SGLang的主要修改

### 1. 新增的核心模块

#### 混合精度加载器 (`sglang-0.4.7/python/sglang/srt/model_loader/`)

这些模块实现了核心的混合精度权重加载功能，允许从不同精度的权重文件中选择性加载参数：

- **`enhanced_mixed_precision_loader.py`** : 
  - **功能**: 增强的混合精度权重加载器，是系统的核心组件
  - **特性**: 
    - 集成GPTQ支持和专家激活跟踪功能
    - 支持FP16、FP8、Int4等多种精度格式的混合加载
    - 提供权重缓存机制，避免重复加载
    - 支持动态权重映射配置更新
    - 内置专家激活统计功能
  - **核心类**: `EnhancedMixedPrecisionWeightLoader`, `ExpertActivationTracker`
  - **使用场景**: 模型加载时根据配置文件选择不同精度的权重

- **`mixed_precision_quantizer.py`**:
  - **功能**: 混合精度量化器，实现基于expert激活热度的动态量化策略
  - **特性**:
    - 支持Hot-Cold专家分析，根据激活频率动态调整量化精度
    - 提供三种量化策略：Hot experts (8位)、Medium experts (6位)、Cold experts (4位)
    - 支持专家量化配置档案管理
    - 提供量化性能影响评估
  - **核心类**: `MixedPrecisionQuantizer`, `ExpertQuantizationManager`
  - **使用场景**: 运行时根据expert使用情况动态调整量化精度

- **`sglang_mixed_precision_loader.py`**:
  - **功能**: SGLang兼容的混合精度加载器
  - **特性**: 提供与SGLang原生加载器的兼容接口
  - **使用场景**: 作为SGLang和混合精度系统的桥梁

- **`mixed_precision_loader.py`**:
  - **功能**: 基础混合精度加载器
  - **特性**: 提供基础的混合精度加载功能
  - **使用场景**: 作为其他加载器的基础类

#### 专家跟踪系统 (`sglang-0.4.7/python/sglang/srt/models/`)

这些模块实现了专家激活跟踪和统计功能，用于分析MoE模型的行为：

- **`enhanced_expert_tracker.py`**:
  - **功能**: 增强版专家激活跟踪器，是专家跟踪系统的核心
  - **特性**:
    - 支持hot-cold分数计算，识别热点和冷点专家
    - 实时跟踪每个expert的激活次数和token处理量
    - 提供多线程安全的激活记录机制
    - 支持激活历史的衰减计算
    - 提供详细的统计报告导出功能
  - **核心类**: `EnhancedExpertTracker`, `ExpertActivationRecord`
  - **使用场景**: 在模型推理过程中实时收集专家激活数据

- **`moe_tracker.py`**:
  - **功能**: MoE模块跟踪器，用于包装MoE模块进行激活统计
  - **特性**:
    - 自动包装MoE层，拦截专家选择过程
    - 记录每个专家的激活情况和权重
    - 支持多种MoE实现（EP-MoE、FusedMoE等）
    - 提供透明的跟踪机制，不影响模型性能
  - **核心类**: `MoETracker`, `ExpertTrackingWrapper`
  - **使用场景**: 在模型初始化时自动包装MoE层

#### 模型实现修改

- **`qwen3_moe.py`** (931行):
  - **功能**: 修改了Qwen3MoE模型实现，集成了专家激活跟踪功能
  - **主要修改**:
    - 在`Qwen3MoeSparseMoeBlock`中集成了专家激活跟踪
    - 修改了前向传播过程，在专家选择后记录激活信息
    - 添加了专家分布记录功能
    - 支持与全局专家跟踪器的集成
  - **核心修改点**:
    - `forward`方法中添加了激活记录逻辑
    - 集成了`get_global_expert_tracker()`调用
    - 添加了专家选择的调试信息输出
  - **使用场景**: 作为Qwen3MoE模型的增强版本，支持专家跟踪

#### 管理层扩展 (`sglang-0.4.7/python/sglang/srt/managers/`)

这些模块扩展了SGLang的管理层功能，添加了专家管理和量化管理能力：

- **`expert_distribution.py`**:
  - **功能**: 专家分布管理器，负责记录和分析专家激活分布
  - **特性**:
    - 支持全局专家分布记录器
    - 提供专家激活统计API
    - 支持专家分布数据的导出和分析
    - 集成调试功能，便于问题排查
  - **核心类**: `ExpertDistributionRecorder`, `ExpertDistributionManager`
  - **使用场景**: 在模型推理过程中收集专家使用模式数据

- **`expert_location.py`**:
  - **功能**: 专家位置管理器，管理专家在分布式系统中的位置信息
  - **特性**:
    - 跟踪专家在不同GPU/节点上的分布
    - 支持专家位置优化策略
    - 提供专家迁移和负载均衡功能
  - **使用场景**: 在分布式MoE系统中管理专家位置

- **`expert_location_dispatch.py`**:
  - **功能**: 专家位置分发管理器，负责专家请求的路由和分发
  - **特性**:
    - 根据专家位置信息路由请求
    - 支持专家负载均衡
    - 提供专家可用性检查
  - **使用场景**: 在分布式环境中分发专家计算任务

- **`dynamic_quantization_manager.py`**:
  - **功能**: 动态量化管理器，根据运行时情况动态调整量化策略
  - **特性**:
    - 基于专家激活热度调整量化精度
    - 支持运行时量化配置更新
    - 提供量化性能监控
  - **使用场景**: 运行时根据模型使用情况优化量化策略

- **`non_expert_fp16_initializer.py`** (1237行):
  - **功能**: 非专家FP16初始化器，专门处理非专家权重的FP16初始化
  - **特性**:
    - 优化非专家权重的初始化过程
    - 支持混合精度初始化策略
    - 提供内存使用优化
  - **使用场景**: 在模型加载时优化非专家权重的初始化

#### 层实现修改 (`sglang-0.4.7/python/sglang/srt/layers/`)

这些模块修改了SGLang的层实现，添加了混合精度和专家跟踪支持：

- **`moe/enhanced_ep_moe.py`**:
  - **功能**: 增强的EP（Expert Parallel）MoE层实现
  - **特性**:
    - 支持专家并行计算
    - 集成专家激活跟踪功能
    - 优化专家通信和同步
    - 支持混合精度专家计算
  - **使用场景**: 在大规模分布式环境中实现专家并行

- **`mixed_precision_epmoe.py`**:
  - **功能**: 混合精度EP MoE层，支持不同精度的专家计算
  - **特性**:
    - 支持专家权重的混合精度加载
    - 提供精度转换和计算优化
    - 集成专家跟踪功能
  - **使用场景**: 在内存受限环境中使用混合精度专家计算

- **`mixed_precision_linear.py`**:
  - **功能**: 混合精度线性层，支持不同精度的权重和激活
  - **特性**:
    - 支持权重和激活的不同精度组合
    - 提供精度转换优化
    - 支持动态精度调整
  - **使用场景**: 在需要内存优化的场景中使用混合精度计算

### 2. 启动脚本和配置文件

#### 启动脚本

这些脚本提供了多种启动方式，支持不同的部署场景：

- **`launch_sglang_mixed_precision.py`** (81行):
  - **功能**: 兼容SGLang原生启动方式的混合精度启动脚本
  - **特性**:
    - 支持SGLang原生命令行参数
    - 集成混合精度配置选项
    - 提供环境变量设置和路径配置
    - 支持渐进式功能启用
  - **使用场景**: 需要与SGLang原生功能完全兼容的场景

- **`launch_mixed_precision_server.py`**:
  - **功能**: 混合精度服务器启动脚本
  - **特性**:
    - 专门针对混合精度功能优化
    - 支持混合精度配置加载
    - 提供专家跟踪功能集成
  - **使用场景**: 专门使用混合精度功能的场景

- **`launch_enhanced_server.py`**:
  - **功能**: 增强功能服务器启动脚本
  - **特性**:
    - 集成所有增强功能
    - 支持专家跟踪和混合精度
    - 提供完整的调试和监控功能
  - **使用场景**: 需要完整功能集的开发和测试场景

- **`start_sglang_mixed_precision.sh`** (300行):
  - **功能**: Bash启动脚本，支持TP/DP/EP等SGLang原生功能
  - **特性**:
    - 完整的Bash脚本，支持参数解析
    - 支持张量并行(TP)、数据并行(DP)、专家并行(EP)
    - 提供依赖检查、GPU数量验证
    - 支持颜色输出和详细的帮助信息
    - 集成环境变量设置和路径配置
  - **使用场景**: 生产环境部署和自动化脚本

#### 配置文件

- **`mixed_precision_config.yaml`** (19行):
  - **功能**: 混合精度配置文件，定义权重映射和专家跟踪设置
  - **配置内容**:
    - 不同精度模型的路径配置（FP16、FP8、Int4）
    - 权重映射规则，指定哪些层使用哪种精度
    - 专家跟踪配置（启用状态、历史长度、衰减因子）
    - 服务器配置（主机、端口、超时设置）
    - 输出配置（基础目录设置）
  - **使用场景**: 所有混合精度功能的配置中心

- **`README_MIXED_PRECISION_MOE.md`** (402行):
  - **功能**: 混合精度MoE系统的详细文档
  - **内容**:
    - 完整的系统架构说明
    - 详细的使用指南和API文档
    - 配置示例和最佳实践
    - 故障排除指南
    - 性能优化建议
  - **使用场景**: 开发者参考和用户指南

### 3. 测试和工具脚本

#### 测试脚本

这些脚本提供了全面的功能测试和验证：

- **`test_mixed_precision.py`**:
  - **功能**: 混合精度功能测试
  - **测试内容**:
    - 混合精度权重加载测试
    - 不同精度格式的转换测试
    - 权重映射配置验证
    - 内存使用优化验证
  - **使用场景**: 验证混合精度功能的正确性

- **`test_enhanced_features.py`**:
  - **功能**: 增强功能测试
  - **测试内容**:
    - 专家激活跟踪功能测试
    - 混合精度量化器测试
    - 专家分布管理器测试
    - 动态量化策略测试
  - **使用场景**: 验证所有增强功能的集成

- **`test_sglang_integration.py`**:
  - **功能**: SGLang集成测试
  - **测试内容**:
    - 与SGLang原生功能的兼容性测试
    - API接口兼容性测试
    - 启动脚本集成测试
    - 配置文件兼容性测试
  - **使用场景**: 确保与SGLang的完全兼容

- **`test_true_mixed_precision.py`**:
  - **功能**: 真实混合精度测试
  - **测试内容**:
    - 使用真实模型的混合精度推理测试
    - 性能基准测试
    - 内存使用对比测试
    - 精度损失评估测试
  - **使用场景**: 验证混合精度的实际效果

#### 工具脚本

- **`analyze_safetensors_index.py`**:
  - **功能**: Safetensors索引文件分析工具
  - **特性**:
    - 解析safetensors索引文件
    - 分析模型结构和权重分布
    - 生成混合精度配置建议
    - 支持不同模型格式的兼容性
  - **使用场景**: 自动分析模型结构并生成配置

- **`convert_weights_for_sglang.py`**:
  - **功能**: 权重转换工具
  - **特性**:
    - 支持多种权重格式转换
    - 优化权重存储格式
    - 支持不同精度之间的转换
    - 提供批量转换功能
  - **使用场景**: 将其他格式的模型转换为SGLang兼容格式

### 4. 根目录工具脚本

#### 专家跟踪工具

这些工具专门用于专家激活跟踪功能的启用、测试和管理：

- **`enable_expert_tracking.py`** (172行):
  - **功能**: 在SGLang启动时启用expert tracking功能
  - **特性**:
    - 初始化全局专家跟踪器
    - 验证跟踪器状态和配置
    - 提供调试和状态检查功能
    - 支持MoE模块跟踪设置
  - **核心功能**:
    - `enable_expert_tracking()`: 启用专家跟踪
    - `setup_moe_tracking()`: 设置MoE跟踪
    - `get_tracking_status()`: 获取跟踪状态
    - `export_current_stats()`: 导出统计信息
  - **使用场景**: 在模型启动前初始化专家跟踪功能

- **`enhanced_expert_tracker.py`** (242行):
  - **功能**: 增强版专家激活跟踪器（独立版本）
  - **特性**:
    - 支持hot-cold分数计算和实时跟踪
    - 提供高效的激活统计数据结构
    - 支持多线程安全的激活记录
    - 提供详细的统计报告导出
  - **核心类**:
    - `EnhancedExpertTracker`: 主要的跟踪器类
    - `ExpertActivationRecord`: 激活记录数据结构
    - `ExpertHotColdStats`: Hot-Cold统计信息
  - **使用场景**: 作为独立的专家跟踪工具使用

- **`expert_tracking_launcher.py`** (680行):
  - **功能**: Expert Tracking完整启动器，支持ShareGPT数据集测试
  - **特性**:
    - 完整的专家跟踪测试流程
    - 支持ShareGPT数据集加载和测试
    - 多线程并行测试（支持16-32线程）
    - 自动导出专家分析结果
    - 支持优雅关闭和资源清理
  - **核心功能**:
    - 启动SGLang服务并启用专家跟踪
    - 加载ShareGPT数据集进行测试
    - 计算hot-cold分数和专家分析
    - 导出完整的分析报告
  - **使用场景**: 完整的专家跟踪功能测试和验证

- **`test_expert_tracking.py`** (102行):
  - **功能**: 测试Expert激活统计功能
  - **特性**:
    - 简单的专家跟踪功能测试
    - 服务健康状态检查
    - 基本的API请求测试
    - 提供详细的测试结果反馈
  - **使用场景**: 快速验证专家跟踪功能是否正常工作

#### 模型测试工具

- **`test_qwen_service.py`** (327行):
  - **功能**: Qwen3-235B-A22B模型服务测试程序（多线程版）
  - **特性**:
    - 支持多线程并发测试（默认16线程）
    - 支持多种输入格式（TXT、JSONL、JSON）
    - 提供详细的性能统计和报告
    - 支持请求参数配置（温度、top-p等）
    - 线程安全的Session管理
  - **核心类**:
    - `QwenServiceClient`: 模型服务客户端
    - `TestDataProcessor`: 测试数据处理器
    - `ResultRecorder`: 结果记录器
  - **使用场景**: 大规模模型服务性能测试和验证

- **`load_requests.py`** (136行):
  - **功能**: 测试数据加载工具，支持多种数据格式
  - **支持格式**:
    - ChatGPT格式的JSON文件
    - 纯文本TXT文件
    - ChatGPT释义CSV文件
    - 多轮对话JSONL文件
    - 可配置系统提示的多任务数据
  - **核心函数**:
    - `read_chatGPT()`: 读取ChatGPT格式数据
    - `read_txt()`: 读取文本文件
    - `read_multiturn_chat()`: 读取多轮对话数据
    - `load_jsonl_dataset()`: 加载JSONL数据集
  - **使用场景**: 为模型测试提供多样化的数据源

#### 配置生成工具

- **`gen_expert_fp8_mapping.py`** (100行):
  - **功能**: 从safetensors索引文件生成专家精度映射配置
  - **特性**:
    - 自动解析safetensors索引文件
    - 识别专家相关的权重参数
    - 生成完整的混合精度配置文件
    - 支持自定义精度设置和缩进格式
  - **使用方法**:
    ```bash
    python3 gen_expert_fp8_mapping.py \
        /path/to/model.safetensors.index.json \
        --precision fp8 --indent 4 > mixed_precision_config.yaml
    ```
  - **输出内容**:
    - 混合精度配置模板
    - 专家权重映射规则
    - 加载策略配置
    - 推理和服务器配置
  - **使用场景**: 自动化生成混合精度配置文件

### 5. 删除的过期脚本

在代码整理过程中，以下脚本已被识别为过期或不再使用，已从项目中删除：

- **`analyze_scores.py`** (194行):
  - **原功能**: Coze评分数据分析脚本
  - **删除原因**: 功能已整合到其他工具中，不再需要独立的评分分析功能
  - **替代方案**: 使用专家跟踪系统的统计功能进行数据分析

- **`coze_api_processor.py`** (487行):
  - **原功能**: Coze API结果解析程序
  - **删除原因**: 项目不再依赖Coze API进行评分，转向内置的专家跟踪分析
  - **替代方案**: 使用`expert_tracking_launcher.py`进行专家分析

- **`run_coze_analysis.py`** (67行):
  - **原功能**: Coze API分析程序运行脚本
  - **删除原因**: 与`coze_api_processor.py`配套使用，随着主程序删除而失效
  - **替代方案**: 使用内置的专家跟踪和分析工具

- **`coze_config.json`**:
  - **原功能**: Coze API配置文件
  - **删除原因**: 不再需要Coze API配置
  - **替代方案**: 使用`mixed_precision_config.yaml`进行系统配置

**清理效果**:
- 减少了约750行过期代码
- 简化了项目结构
- 避免了功能重复和混淆
- 提高了代码维护性

### 6. 核心功能扩展

#### 专家激活跟踪

这是DynaQuant的核心创新功能，实现了对MoE模型专家激活的实时监控和分析：

**核心特性**:
- **实时跟踪**: 在模型推理过程中实时记录每个expert的激活情况
- **Hot-Cold分析**: 计算expert的激活热度分数，识别热点和冷点专家
- **多线程安全**: 支持多线程并发环境下的安全激活记录
- **详细统计**: 提供激活次数、token处理量、时间戳等详细统计信息
- **衰减计算**: 支持基于时间的激活历史衰减，突出近期活跃的专家

**技术实现**:
- 使用高效的字典数据结构存储激活信息
- 实现线程安全的记录机制，避免数据竞争
- 提供全局专家跟踪器，支持跨模块的激活统计
- 集成到MoE层的前向传播过程中，实现透明的跟踪

**应用价值**:
- 帮助理解模型对不同任务的处理方式
- 识别模型中的热点专家，用于负载均衡优化
- 分析专家专业化程度，指导模型架构改进
- 为混合精度量化提供数据支持

#### 混合精度量化

基于专家激活跟踪数据，实现智能的混合精度量化策略：

**量化策略**:
- **Hot experts (热度 > 0.8)**: 使用8位量化，保持高精度以维持性能
- **Medium experts (热度 0.5-0.8)**: 使用6位量化，平衡精度和性能
- **Cold experts (热度 < 0.5)**: 使用4位量化，最大化内存节省

**技术特性**:
- **动态调整**: 根据运行时专家激活情况动态调整量化策略
- **格式支持**: 支持FP16、FP8、Int4等多种精度格式
- **GPTQ兼容**: 支持GPTQ量化模型的反量化和处理
- **缓存优化**: 提供权重缓存机制，避免重复加载和转换
- **性能监控**: 提供量化对模型性能影响的分析

**实现细节**:
- 基于专家量化配置档案管理系统
- 支持量化配置的动态更新和热重载
- 提供量化性能影响评估和报告
- 集成到模型加载和推理过程中

#### 兼容性增强

确保DynaQuant与原始SGLang的完全兼容性：

**启动兼容**:
- **原生启动**: 完全兼容SGLang原生的启动方式和命令行参数
- **渐进启用**: 支持渐进式功能启用，可以逐步添加混合精度和专家跟踪功能
- **配置兼容**: 保持与SGLang原生配置文件的兼容性

**功能兼容**:
- **并行支持**: 完全支持TP（张量并行）、DP（数据并行）、EP（专家并行）
- **API兼容**: 保持与原始SGLang API的完全兼容性
- **模型兼容**: 支持SGLang原生支持的所有模型格式和架构

**扩展性**:
- **模块化设计**: 新功能以模块化方式添加，不影响原有功能
- **可选启用**: 所有增强功能都可以选择性启用或禁用
- **向后兼容**: 确保现有SGLang用户的无缝迁移体验

**技术保障**:
- 通过完整的测试套件确保兼容性
- 提供详细的迁移指南和最佳实践
- 支持与SGLang社区版本的同步更新

## 🚀 主要特性

### 1. 混合精度权重加载
- 支持从多个不同精度的权重文件中选择性加载参数
- 支持FP16、FP8、Int4等不同精度格式
- 通过配置文件灵活定义权重映射关系
- 支持权重文件缓存，提高加载效率

### 2. 混合精度推理
- 支持不同精度权重的混合推理
- 自动处理不同精度之间的转换
- 优化内存使用和计算效率

### 3. 专家激活跟踪
- 实时跟踪每个expert的激活次数
- 提供详细的统计信息和可视化
- 支持统计数据的导出和分析
- 计算hot-cold分数，识别热点专家

### 4. 动态量化策略
- 基于expert激活热度进行动态量化
- Hot experts (热度 > 0.8): 8位量化
- Medium experts (热度 0.5-0.8): 6位量化  
- Cold experts (热度 < 0.5): 4位量化

### 5. GPTQ量化支持
- 支持GPTQ-Int4量化模型格式
- 自动检测和反量化GPTQ权重
- 兼容qweight、qzeros、scales等GPTQ组件

### 6. 网络API服务
- 基于SGLang的RESTful API服务
- 支持单次和批量文本生成
- 异步处理，支持并发请求
- 完整的错误处理和日志记录

## 📁 项目结构

```
DynaQuant/
├── sglang-0.4.7/                    # 基于SGLang 0.4.7的修改版本
│   ├── python/sglang/srt/
│   │   ├── model_loader/            # 混合精度加载器模块
│   │   │   ├── enhanced_mixed_precision_loader.py
│   │   │   ├── mixed_precision_quantizer.py
│   │   │   └── sglang_mixed_precision_loader.py
│   │   ├── models/                  # 模型实现
│   │   │   ├── enhanced_expert_tracker.py
│   │   │   ├── moe_tracker.py
│   │   │   └── qwen3_moe.py
│   │   ├── managers/                # 管理层扩展
│   │   │   ├── expert_distribution.py
│   │   │   ├── expert_location.py
│   │   │   └── dynamic_quantization_manager.py
│   │   └── layers/                  # 层实现修改
│   │       ├── moe/enhanced_ep_moe.py
│   │       ├── mixed_precision_epmoe.py
│   │       └── mixed_precision_linear.py
│   ├── launch_sglang_mixed_precision.py    # 启动脚本
│   ├── mixed_precision_config.yaml         # 配置文件
│   └── README_MIXED_PRECISION_MOE.md       # 详细文档
├── enable_expert_tracking.py        # 专家跟踪启用工具
├── enhanced_expert_tracker.py       # 专家跟踪器（独立版本）
├── expert_tracking_launcher.py      # 专家跟踪启动器
├── test_expert_tracking.py          # 专家跟踪测试
├── test_qwen_service.py             # 模型服务测试
├── load_requests.py                 # 测试数据加载工具
├── gen_expert_fp8_mapping.py        # 配置生成工具
├── requirements.txt                 # 依赖包列表
└── README.md                        # 项目说明
```

## 🔧 安装和配置

### 1. 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)
- 8GB+ GPU内存

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置模型

#### 方法一：使用配置生成工具

```bash
# 从safetensors索引文件生成配置
python3 gen_expert_fp8_mapping.py \
    /path/to/model/model.safetensors.index.json \
    --precision fp8 --indent 4 > sglang-0.4.7/mixed_precision_config.yaml
```

#### 方法二：手动配置

编辑 `sglang-0.4.7/mixed_precision_config.yaml` 文件：

```yaml
  mixed_precision:
  fp16_path: "/path/to/fp16/model"
  fp8_path: "/path/to/fp8/model"
  int4_path: "/path/to/int4/model"
    weight_mapping:
      "model.layers.0.mlp.experts.0.gate_proj.weight": "int4"
      "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
      "model.layers.0.mlp.experts.0.down_proj.weight": "int4"

expert_tracking:
  enable_tracking: true
  max_history: 1000
  decay_factor: 0.95

server:
  host: "127.0.0.1"
  port: 8080
```

## 🚀 快速开始

### 1. 启动混合精度服务器

```bash
cd sglang-0.4.7
./start_sglang_mixed_precision.sh \
    -m /path/to/model \
    --enable-mixed-precision \
    -c mixed_precision_config.yaml \
    -t 4 -d 2
```

### 2. 启用专家跟踪

```bash
python3 enable_expert_tracking.py
```

### 3. 运行测试

```bash
# 测试专家跟踪功能
python3 test_expert_tracking.py

# 测试模型服务
python3 test_qwen_service.py --input test_data.txt --workers 16
```

### 4. 运行完整的专家跟踪测试

```bash
python3 expert_tracking_launcher.py --workers 32
```

## 📊 API接口

### 1. 健康检查
```
GET /health
```

### 2. 聊天完成
```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen3-235b-a22b",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

### 3. 专家统计查询
```
GET /expert_stats
```

### 4. 启动专家分布记录
```
POST /start_expert_distribution_record
```

## 🔍 专家激活跟踪功能

### 功能概述

专家激活跟踪功能可以实时跟踪MoE（Mixture of Experts）模型中每个expert的激活情况，帮助分析模型的行为和性能。

### 统计信息

系统提供以下统计信息：

1. **摘要统计**
   - 总激活次数
   - 总token数
   - 总请求数
   - 总层数和专家数

2. **层统计**
   - 每层的总激活次数
   - 每层的专家数量
   - 每层的平均激活率

3. **专家统计**
   - 每个专家的激活次数
   - 每个专家的激活率
   - 每个专家的hot-cold分数

4. **实时监控**
   - 最近的激活记录
   - 激活最多的专家排名
   - 实时性能指标

### 使用方法

#### 1. 启动服务器时启用专家跟踪

专家激活跟踪会在模型加载时自动启用，无需额外配置。

#### 2. 发送请求并查看统计

```python
import requests

# 发送请求
response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={
        "model": "qwen3-235b-a22b",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100
    }
)

# 获取专家统计
stats_response = requests.get("http://127.0.0.1:8080/expert_stats")
stats = stats_response.json()
print(f"总激活次数: {stats['summary']['total_activations']}")
```

#### 3. 运行专门的测试

```bash
# 专家激活跟踪测试
python3 test_expert_tracking.py

# 完整启动器测试
python3 expert_tracking_launcher.py --workers 16
```

## 📈 混合精度量化

### 量化策略

系统支持基于expert激活热度的动态量化策略：

- **Hot experts (热度 > 0.8)**: 8位量化，保持高精度
- **Medium experts (热度 0.5-0.8)**: 6位量化，平衡精度和性能
- **Cold experts (热度 < 0.5)**: 4位量化，最大化压缩

### 配置文件

```yaml
mixed_precision:
  # 不同精度的模型路径
  fp16_path: "/path/to/fp16/model"
  fp8_path: "/path/to/fp8/model"
  int4_path: "/path/to/int4/model"
  
  # 权重映射配置
weight_mapping:
  # 注意力层使用FP16
  "model.layers.0.self_attn.q_proj.weight": "fp16"
  "model.layers.0.self_attn.k_proj.weight": "fp16"
  
  # 专家层使用Int4
  "model.layers.0.mlp.experts.0.gate_proj.weight": "int4"
  "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
```

## 🐛 故障排除

### 1. 常见问题

**问题**: 模块导入失败
```
ImportError: No module named 'sglang.srt.model_loader.enhanced_mixed_precision_loader'
```

**解决方案**: 确保在正确的目录下运行，并检查Python路径设置。

**问题**: 服务器启动失败
```
❌ SGLang服务器启动失败
```

**解决方案**: 检查启动脚本路径和权限，确保模型文件存在。

**问题**: 专家跟踪器不可用
```
⚠️ 专家跟踪器不可用
```

**解决方案**: 确保在启动时正确启用了expert tracking功能。

### 2. 调试技巧

1. **启用详细日志**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查系统状态**:
```bash
python3 enable_expert_tracking.py
```

3. **验证配置**:
```bash
python3 -c "import yaml; print(yaml.safe_load(open('mixed_precision_config.yaml')))"
```

## 📚 参考资料

- [SGLang项目](https://github.com/sgl-project/sglang) - 原始SGLang框架
- [MoE模型论文](https://arxiv.org/abs/1701.06538) - Mixture of Experts模型
- [GPTQ量化论文](https://arxiv.org/abs/2210.17323) - GPTQ量化方法

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于Apache 2.0许可证开源。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者