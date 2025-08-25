# 混合量化加载时序图

## 完整时序流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant Server as sglang.launch_server
    participant Args as ServerArgs
    participant Config as ModelConfig
    participant Loader as DefaultModelLoader
    participant MixedLoader as TrueMixedPrecisionLoader
    participant BaseModel as FP8基础模型
    participant FP16Model as FP16模型
    participant GPTQModel as GPTQ-Int4模型
    participant Linear as MixedPrecisionLinear

    User->>Server: 启动命令
    Server->>Args: 解析命令行参数
    Args->>Config: 创建ModelConfig
    Config->>Loader: 创建DefaultModelLoader
    Loader->>MixedLoader: 创建TrueMixedPrecisionLoader
    
    Note over MixedLoader: 步骤1: 加载基础模型
    MixedLoader->>BaseModel: 加载FP8基础模型
    BaseModel-->>MixedLoader: 返回基础模型权重
    
    Note over MixedLoader: 步骤2: 生成权重映射
    MixedLoader->>MixedLoader: _generate_weight_mapping()
    MixedLoader->>MixedLoader: 识别专家层和非专家层
    
    Note over MixedLoader: 步骤3: 替换指定层权重
    loop 遍历weight_mapping
        MixedLoader->>FP16Model: 加载非专家层FP16权重
        FP16Model-->>MixedLoader: 返回FP16权重
        MixedLoader->>GPTQModel: 加载专家层GPTQ-Int4权重
        GPTQModel-->>MixedLoader: 返回GPTQ-Int4权重
    end
    
    Note over MixedLoader: 步骤4: 替换Linear层
    MixedLoader->>Linear: replace_linear_with_mixed_precision()
    Linear-->>MixedLoader: 替换完成
    
    MixedLoader-->>Loader: 返回加载完成的模型
    Loader-->>Server: 模型加载完成
    Server-->>User: 启动推理服务
```

## 不同精度模型加载顺序详解

### 1. 基础模型加载阶段

```mermaid
graph TD
    A[开始加载] --> B[读取base_model_path]
    B --> C[检查FP8模型路径]
    C --> D{路径存在?}
    D -->|是| E[加载FP8基础模型]
    D -->|否| F[使用第一个可用路径]
    E --> G[加载model.safetensors]
    F --> G
    G --> H[支持索引文件]
    H --> I[基础模型加载完成]
    
    style E fill:#e1f5fe
    style G fill:#e1f5fe
    style I fill:#e1f5fe
```

**加载顺序**: FP8基础模型 → 避免OOM

### 2. 权重映射生成阶段

```mermaid
graph TD
    A[遍历模型权重] --> B[检查权重名称]
    B --> C{包含experts?}
    C -->|是| D{包含mlp?}
    C -->|否| E[标记为非专家层]
    D -->|是| F[标记为专家层]
    D -->|否| E
    E --> G[默认使用FP16]
    F --> H[使用配置文件精度]
    G --> I[生成权重映射]
    H --> I
    I --> J[映射生成完成]
    
    style F fill:#fff3e0
    style G fill:#e8f5e8
    style H fill:#fff3e0
```

**映射策略**: 
- 专家层 → 根据配置文件（GPTQ-Int4）
- 非专家层 → 默认FP16

### 3. 权重文件查找阶段

```mermaid
graph TD
    A[查找权重文件] --> B[检查索引文件]
    B --> C{索引文件存在?}
    C -->|是| D[解析weight_map]
    C -->|否| E[传统文件查找]
    D --> F[根据权重名找文件]
    E --> G[尝试多种文件格式]
    F --> H[返回文件路径]
    G --> H
    H --> I[文件查找完成]
    
    style D fill:#f3e5f5
    style F fill:#f3e5f5
```

**查找顺序**:
1. 优先使用 `model.safetensors.index.json`
2. 回退到传统文件查找方式

### 4. 权重加载和存储阶段

```mermaid
graph TD
    A[加载权重] --> B[根据精度选择路径]
    B --> C{精度类型}
    C -->|fp16| D[加载FP16权重]
    C -->|gptq_int4| E[加载GPTQ-Int4权重]
    C -->|fp8| F[加载FP8权重]
    D --> G[创建CompressedWeight]
    E --> G
    F --> G
    G --> H[存储到compressed_weights]
    H --> I[权重加载完成]
    
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#e1f5fe
```

**加载顺序**:
1. FP16权重（非专家层）
2. GPTQ-Int4权重（专家层）
3. FP8权重（如果配置）

## 具体文件加载顺序

### 基础模型文件加载
```
1. /dcar-vepfs-trans-models/Qwen3-30B-A3B-FP8/
   ├── model.safetensors.index.json (优先)
   ├── model.safetensors
   └── pytorch_model.bin (回退)
```

### 不同精度权重文件加载
```
2. /dcar-vepfs-trans-models/Qwen3-30B-A3B/ (FP16)
   ├── model.safetensors.index.json
   ├── model.safetensors
   └── pytorch_model.bin

3. /dcar-vepfs-trans-models/Qwen3-30B-A3B-GPTQ-Int4/ (GPTQ-Int4)
   ├── model.safetensors.index.json
   ├── model.safetensors
   └── pytorch_model.bin
```

## 内存使用时间线

```mermaid
gantt
    title 混合量化内存使用时间线
    dateFormat X
    axisFormat %s
    
    section 基础模型加载
    FP8基础模型    :0, 5s
    section 权重映射生成
    生成映射       :5s, 6s
    section 权重替换
    FP16权重加载   :6s, 8s
    GPTQ权重加载   :6s, 8s
    section 层替换
    Linear层替换   :8s, 9s
    section 推理服务
    服务启动       :9s, 10s
```

## 关键时间点说明

### T0-T5: 基础模型加载
- **内存使用**: 低（FP8格式）
- **主要操作**: 加载FP8基础模型
- **目的**: 避免OOM，建立模型结构

### T5-T6: 权重映射生成
- **内存使用**: 无额外内存
- **主要操作**: 分析模型结构，生成映射
- **目的**: 确定每个权重的精度策略

### T6-T8: 权重替换
- **内存使用**: 逐步增加
- **主要操作**: 加载不同精度权重
- **目的**: 替换指定层为高精度权重

### T8-T9: 层替换
- **内存使用**: 保持稳定
- **主要操作**: 替换Linear层
- **目的**: 启用动态反量化

### T9+: 推理服务
- **内存使用**: 按需反量化
- **主要操作**: 处理推理请求
- **目的**: 高效推理

## 优化策略总结

### 1. 内存优化
- **基础模型**: FP8格式，减少50%内存
- **压缩存储**: 保持压缩格式，不立即反量化
- **按需处理**: 推理时动态反量化

### 2. 加载优化
- **索引文件**: 快速定位权重文件
- **缓存机制**: 避免重复加载
- **并行加载**: 支持多文件并行加载

### 3. 精度平衡
- **关键层**: 使用FP16保持精度
- **专家层**: 使用Int4节省内存
- **灵活配置**: 根据需求调整精度

## 故障排除时间线

```mermaid
graph TD
    A[加载失败] --> B{检查阶段}
    B -->|基础模型| C[检查base_model_path]
    B -->|权重映射| D[检查weight_mapping]
    B -->|文件查找| E[检查模型路径]
    B -->|层替换| F[检查Linear层]
    
    C --> G[路径不存在]
    D --> H[映射错误]
    E --> I[文件缺失]
    F --> J[替换失败]
    
    G --> K[使用默认路径]
    H --> L[检查配置文件]
    I --> M[检查文件格式]
    J --> N[检查模型结构]
```

这个时序图清晰地展示了混合量化从启动到推理的完整流程，以及不同精度模型的加载顺序和内存使用情况。
