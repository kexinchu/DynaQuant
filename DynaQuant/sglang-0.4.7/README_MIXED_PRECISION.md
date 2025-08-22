# SGLang 混合精度功能

本目录包含了在SGLang 0.4.7版本基础上添加的混合精度功能，支持从不同精度文件中选择性加载权重参数，并进行混合精度推理。

## 主要特性

### 1. 混合精度权重加载
- 支持从多个不同精度的权重文件中选择性加载参数
- 支持FP16、FP8、Int4等不同精度格式
- 通过配置文件灵活定义权重映射关系
- 支持权重文件缓存，提高加载效率

### 2. 与SGLang集成
- 完全保留SGLang的原始API和功能
- 向后兼容，不影响现有部署
- 支持SGLang的所有特性（张量并行、数据并行等）

### 3. 动态配置管理
- 支持运行时更新权重映射配置
- 支持权重的动态重载
- 提供模型和权重信息查询接口

## 文件结构

```
sglang-0.4.7/
├── python/sglang/srt/model_loader/
│   ├── mixed_precision_loader.py     # 混合精度权重加载器
│   └── loader.py                     # 修改后的模型加载器
├── python/sglang/srt/configs/
│   └── model_config.py               # 修改后的模型配置
├── mixed_precision_config.yaml       # 混合精度配置文件
├── launch_mixed_precision_server.py  # 混合精度服务器启动脚本
├── convert_weights_for_sglang.py     # 权重转换工具
├── start_mixed_precision_server.sh   # 启动脚本
└── README_MIXED_PRECISION.md         # 本文档
```

## 快速开始

### 1. 准备权重文件

首先，您需要准备不同精度的权重文件。可以使用提供的转换工具：

```bash
# 转换权重文件
python3 convert_weights_for_sglang.py \
    --model_path /path/to/original/model \
    --output_dir /path/to/mixed_precision_weights \
    --create_config
```

这将创建以下目录结构：
```
/path/to/mixed_precision_weights/
├── fp16/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
├── fp8/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
├── int4/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
└── mixed_precision_config.yaml
```

### 2. 配置混合精度

#### 方法一：手动配置

编辑 `mixed_precision_config.yaml` 文件：

```yaml
mixed_precision:
  # 权重文件路径
  fp16_path: "/path/to/mixed_precision_weights/fp16"
  fp8_path: "/path/to/mixed_precision_weights/fp8"
  int4_path: "/path/to/mixed_precision_weights/int4"
  
  # 权重映射配置
  weight_mapping:
    # 注意力层使用FP16
    "model.layers.0.self_attn.q_proj.weight": "fp16"
    "model.layers.0.self_attn.k_proj.weight": "fp16"
    "model.layers.0.self_attn.v_proj.weight": "fp16"
    "model.layers.0.self_attn.o_proj.weight": "fp16"
    
    # MLP层使用FP8
    "model.layers.0.mlp.gate_proj.weight": "fp8"
    "model.layers.0.mlp.up_proj.weight": "fp8"
    "model.layers.0.mlp.down_proj.weight": "fp8"
    
    # 专家层使用Int4
    "model.layers.0.mlp.experts.0.gate_proj.weight": "int4"
    "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
    "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
    
    # 其他层使用FP16
    "model.layers.0.input_layernorm.weight": "fp16"
    "model.layers.0.post_attention_layernorm.weight": "fp16"
    "model.embed_tokens.weight": "fp16"
    "model.norm.weight": "fp16"
    "lm_head.weight": "fp16"
```

#### 方法二：使用Safetensors索引文件自动生成配置

如果您的模型包含 `model.safetensors.index.json` 文件，可以使用分析工具自动生成配置：

```bash
# 分析模型并生成混合精度配置
python3 analyze_safetensors_index.py \
    --model_path /path/to/your/model \
    --export_config mixed_precision_config.yaml \
    --attention_precision fp16 \
    --mlp_precision fp8 \
    --expert_precision int4
```

这将自动分析模型结构并生成合适的权重映射配置。

### 3. 启动混合精度服务器

使用提供的启动脚本：

```bash
# 启用混合精度
./start_mixed_precision_server.sh \
    --enable-mixed-precision \
    --model-path /path/to/original/model \
    --mixed-precision-config mixed_precision_config.yaml \
    --host 127.0.0.1 \
    --port 8080 \
    --tp-size 1 \
    --dp-size 4
```

或者使用Python脚本：

```bash
python3 launch_mixed_precision_server.py \
    --enable-mixed-precision \
    --mixed-precision-config mixed_precision_config.yaml \
    --model-path /path/to/original/model \
    --tp-size 1 \
    --dp-size 4 \
    --host 127.0.0.1 \
    --port 8080
```

### 4. 使用标准SGLang API

启动后，您可以使用标准的SGLang API进行推理：

```python
import sglang as sgl

# 连接到服务器
sgl.set_default_backend("http://127.0.0.1:8080")

# 创建提示
prompt = "请介绍一下人工智能："

# 生成文本
response = sgl.generate(prompt, max_new_tokens=200, temperature=0.7)
print(response.text)
```

## 配置说明

### 权重映射配置

权重映射配置定义了每个权重参数应该从哪个精度文件中加载：

```yaml
weight_mapping:
  # 格式: "权重名称": "精度类型"
  "model.layers.0.self_attn.q_proj.weight": "fp16"
  "model.layers.0.mlp.gate_proj.weight": "fp8"
  "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
```

支持的精度类型：
- `fp16`: 16位浮点数
- `fp8`: 8位浮点数
- `int4`: 4位整数

### 服务器参数

启动脚本支持以下参数：

- `--model-path`: 原始模型路径
- `--mixed-precision-config`: 混合精度配置文件路径
- `--enable-mixed-precision`: 启用混合精度加载
- `--host`: 服务器主机地址
- `--port`: 服务器端口
- `--tp-size`: 张量并行大小
- `--dp-size`: 数据并行大小
- `--max-running-requests`: 最大运行请求数
- `--max-total-tokens`: 最大总token数
- `--dtype`: 数据类型

## 性能优化

### 1. 内存优化
- 支持不同精度的权重混合使用，减少内存占用
- 权重文件缓存机制，避免重复加载
- 支持权重的动态加载和卸载

### 2. 计算优化
- 混合精度计算，平衡精度和性能
- 异步处理，支持并发请求
- 批量处理，提高吞吐量

### 3. 存储优化
- 支持多种权重文件格式（safetensors、pytorch）
- 选择性加载，只加载需要的权重
- 压缩存储，减少磁盘占用

## 故障排除

### 1. 模型加载失败
- 检查模型路径是否正确
- 确认权重文件是否存在
- 检查CUDA内存是否充足

### 2. 权重映射错误
- 检查权重名称是否正确
- 确认精度路径是否存在
- 验证权重形状是否匹配

### 3. 混合精度加载失败
- 检查配置文件格式是否正确
- 确认所有精度路径都存在
- 查看服务器日志获取详细错误信息

## 与原始SGLang的兼容性

### 1. API兼容性
- 完全保留SGLang的原始API
- 所有现有的客户端代码无需修改
- 支持SGLang的所有功能特性

### 2. 配置兼容性
- 如果不启用混合精度，使用标准的SGLang加载方式
- 所有SGLang参数都保持不变
- 向后兼容现有部署

### 3. 性能兼容性
- 混合精度模式下的性能与原始SGLang相当
- 支持所有SGLang的优化特性
- 保持相同的延迟和吞吐量

## 示例配置

### Qwen3-235B-A22B 混合精度配置

```yaml
mixed_precision:
  fp16_path: "/dcar-vepfs-trans-models/Qwen3-235B-A22B"
  fp8_path: "/dcar-vepfs-trans-models/Qwen3-235B-A22B-FP8"
  int4_path: "/dcar-vepfs-trans-models/Qwen3-235B-A22B-GPTQ-Int4"
  
  weight_mapping:
    # 注意力层使用FP16
    "model.layers.0.self_attn.q_proj.weight": "fp16"
    "model.layers.0.self_attn.k_proj.weight": "fp16"
    "model.layers.0.self_attn.v_proj.weight": "fp16"
    "model.layers.0.self_attn.o_proj.weight": "fp16"
    
    # MLP层使用FP8
    "model.layers.0.mlp.gate_proj.weight": "fp8"
    "model.layers.0.mlp.up_proj.weight": "fp8"
    "model.layers.0.mlp.down_proj.weight": "fp8"
    
    # 专家层使用Int4
    "model.layers.0.mlp.experts.0.gate_proj.weight": "int4"
    "model.layers.0.mlp.experts.0.up_proj.weight": "int4"
    "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
    "model.layers.0.mlp.experts.1.gate_proj.weight": "int4"
    "model.layers.0.mlp.experts.1.up_proj.weight": "int4"
    "model.layers.0.mlp.experts.1.down_proj.weight": "int4"
    
    # 其他层使用FP16
    "model.layers.0.input_layernorm.weight": "fp16"
    "model.layers.0.post_attention_layernorm.weight": "fp16"
    "model.embed_tokens.weight": "fp16"
    "model.norm.weight": "fp16"
    "lm_head.weight": "fp16"
```

## 许可证

本混合精度功能遵循SGLang的原始许可证。
