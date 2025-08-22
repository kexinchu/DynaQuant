# 混合精度Transformer模型部署系统

这是一个基于Transformer的模型部署系统，支持混合精度推理和选择性权重加载。系统可以根据配置文件从不同精度的权重文件中选择性加载参数，支持FP16、FP8、Int4等不同精度的混合使用，并提供专家激活统计功能。

## 主要特性

### 1. 混合精度权重加载
- 支持从多个不同精度的权重文件中选择性加载参数
- 支持FP16、FP8、Int4等不同精度格式
- 通过配置文件灵活定义权重映射关系
- 支持权重文件缓存，提高加载效率

### 2. 混合精度推理
- 支持不同精度权重的混合推理
- 自动处理不同精度之间的转换
- 优化内存使用和计算效率

### 3. 网络API服务
- 基于FastAPI的RESTful API服务
- 支持单次和批量文本生成
- 异步处理，支持并发请求
- 完整的错误处理和日志记录

### 4. 动态配置管理
- 支持运行时更新权重映射配置
- 支持权重的动态重载
- 提供模型和权重信息查询接口

### 5. 专家激活统计
- 实时跟踪每个expert的激活次数
- 提供详细的统计信息和可视化
- 支持统计数据的导出和分析

## 项目结构

```
├── config/
│   └── model_config.yaml          # 模型配置文件
├── src/
│   ├── __init__.py
│   ├── weight_loader.py           # 权重加载器
│   ├── mixed_precision_model.py   # 混合精度模型
│   ├── api_server.py              # API服务器
│   ├── expert_activation_tracker.py # 专家激活统计器
│   └── moe_tracker.py             # MoE跟踪器
├── examples/
│   ├── test_client.py             # 测试客户端
│   └── test_expert_tracking.py    # 专家激活跟踪测试
├── main.py                        # 主程序入口
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明
```

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

#### 方法一：手动配置

编辑 `config/model_config.yaml` 文件，配置模型路径和权重映射：

```yaml
model:
  name: "Qwen3-235B-A22B"
  base_path: "/path/to/your/model"
  
  mixed_precision:
    fp16_path: "/path/to/fp16/weights"
    fp8_path: "/path/to/fp8/weights"
    int4_path: "/path/to/int4/weights"
    
    weight_mapping:
      "model.layers.0.self_attn.q_proj.weight": "fp16"
      "model.layers.0.mlp.experts.0.down_proj.weight": "int4"
      # ... 更多权重映射
```

#### 方法二：使用Safetensors索引文件自动生成配置

如果您的模型包含 `model.safetensors.index.json` 文件，可以使用分析工具自动生成配置：

```bash
# 分析模型并生成混合精度配置
python3 src/safetensors_index_analyzer.py \
    --model_path /path/to/your/model \
    --export_config config/mixed_precision_config.yaml \
    --attention_precision fp16 \
    --mlp_precision fp8 \
    --expert_precision int4
```

这将自动分析模型结构并生成合适的权重映射配置。

### 3. 启动服务器

```bash
python main.py --config config/model_config.yaml --host 127.0.0.1 --port 8080
```

## API接口

### 1. 健康检查
```
GET /
```

### 2. 单次文本生成
```
POST /generate
Content-Type: application/json

{
  "prompt": "输入提示文本",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "do_sample": true
}
```

### 3. 批量文本生成
```
POST /batch_generate
Content-Type: application/json

{
  "prompts": ["提示1", "提示2", "提示3"],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "do_sample": true
}
```

### 4. 获取模型信息
```
GET /model_info
```

### 5. 更新权重映射
```
POST /update_weight_mapping
Content-Type: application/json

{
  "weight_mapping": {
    "model.layers.0.self_attn.q_proj.weight": "fp8"
  }
}
```

### 6. 重新加载权重
```
POST /reload_weights
```

### 7. 获取专家激活统计
```
GET /expert_stats
```

### 8. 获取专家激活统计（带参数）
```
POST /expert_stats
Content-Type: application/json

{
  "top_k": 10,
  "minutes": 5
}
```

### 9. 重置专家激活统计
```
POST /reset_expert_stats
```

### 10. 导出专家激活统计
```
POST /export_expert_stats
```

## 使用示例

### 1. 使用Python客户端

```python
from examples.test_client import MixedPrecisionAPIClient

# 创建客户端
client = MixedPrecisionAPIClient("http://127.0.0.1:8080")

# 健康检查
health = client.health_check()
print(f"服务器状态: {health['status']}")

# 生成文本
result = client.generate_text(
    prompt="请介绍一下人工智能：",
    max_new_tokens=200,
    temperature=0.7
)
print(f"生成文本: {result['generated_text']}")

# 批量生成
prompts = ["什么是机器学习？", "什么是深度学习？"]
results = client.batch_generate(prompts=prompts, max_new_tokens=150)
for prompt, text in zip(results['prompts'], results['generated_texts']):
    print(f"输入: {prompt}")
    print(f"输出: {text}")

# 获取专家激活统计
stats = client.get_expert_stats()
print(f"总激活次数: {stats['summary']['total_activations']}")
print(f"总专家数: {stats['summary']['total_experts']}")

# 获取前10个激活最多的专家
top_experts = stats['top_experts']
for expert in top_experts[:5]:
    print(f"Layer {expert['layer_id']}, Expert {expert['expert_id']}: {expert['activation_count']} 次激活")
```

### 2. 使用curl命令

```bash
# 健康检查
curl http://127.0.0.1:8080/

# 生成文本
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请介绍一下人工智能：",
    "max_new_tokens": 200,
    "temperature": 0.7
  }'

# 获取模型信息
curl http://127.0.0.1:8080/model_info

# 获取专家激活统计
curl http://127.0.0.1:8080/expert_stats

# 获取专家激活统计（带参数）
curl -X POST http://127.0.0.1:8080/expert_stats \
  -H "Content-Type: application/json" \
  -d '{"top_k": 10, "minutes": 5}'

# 重置专家激活统计
curl -X POST http://127.0.0.1:8080/reset_expert_stats

# 导出专家激活统计
curl -X POST http://127.0.0.1:8080/export_expert_stats
```

## 专家激活统计功能

### 功能概述

专家激活统计功能可以实时跟踪MoE（Mixture of Experts）模型中每个expert的激活情况，帮助分析模型的行为和性能。

### 统计信息

系统提供以下统计信息：

1. **摘要统计**
   - 总激活次数
   - 总token数
   - 总请求数
   - 总层数和专家数
   - 运行时间和性能指标

2. **层统计**
   - 每层的总激活次数
   - 每层的专家数量
   - 每层的平均激活率

3. **专家统计**
   - 每个专家的激活次数
   - 每个专家的激活率
   - 每个专家的最后激活时间

4. **实时监控**
   - 最近的激活记录
   - 激活最多的专家排名
   - 实时性能指标

### 使用方法

#### 1. 启动服务器时启用专家跟踪

专家激活跟踪会在模型加载时自动启用，无需额外配置。

#### 2. 发送请求并查看统计

```python
from examples.test_client import MixedPrecisionAPIClient

client = MixedPrecisionAPIClient("http://127.0.0.1:8080")

# 生成一些文本
client.generate_text("请介绍一下人工智能：", max_new_tokens=100)

# 获取专家统计
stats = client.get_expert_stats()
print(f"总激活次数: {stats['summary']['total_activations']}")
```

#### 3. 运行专门的测试

```bash
python examples/test_expert_tracking.py
```

### 统计数据分析

#### 1. 专家利用率分析

通过分析专家激活统计，可以了解：
- 哪些专家被频繁使用
- 哪些专家很少被激活
- 不同层的专家使用模式

#### 2. 性能优化

基于统计信息可以：
- 识别热点专家，进行负载均衡
- 优化专家分配策略
- 调整模型架构

#### 3. 模型行为分析

统计信息有助于：
- 理解模型对不同任务的处理方式
- 分析专家专业化程度
- 评估模型效率

## 配置说明

### 权重映射配置

权重映射配置定义了每个权重参数应该从哪个精度文件中加载：

```yaml
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
```

### 推理配置

```yaml
inference:
  max_seq_length: 4096        # 最大序列长度
  max_batch_size: 32          # 最大批处理大小
  dtype: "bfloat16"           # 计算数据类型
  device_map: "auto"          # 设备映射
  load_in_8bit: false         # 是否加载8位量化
  load_in_4bit: false         # 是否加载4位量化
```

### 服务器配置

```yaml
server:
  host: "127.0.0.1"           # 服务器主机地址
  port: 8080                  # 服务器端口
  max_workers: 4              # 最大工作线程数
```

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

### 3. API请求失败
- 检查服务器是否正常运行
- 确认请求格式是否正确
- 查看服务器日志获取详细错误信息

## 扩展开发

### 1. 添加新的精度格式
在 `weight_loader.py` 中添加新的精度处理函数：

```python
def _process_new_precision_weight(self, weight: torch.Tensor) -> torch.Tensor:
    """处理新的精度格式"""
    # 实现新的精度转换逻辑
    return weight
```

### 2. 添加新的API接口
在 `api_server.py` 中添加新的路由：

```python
@self.app.post("/new_endpoint")
async def new_endpoint(request: NewRequestModel):
    # 实现新的接口逻辑
    pass
```

### 3. 自定义模型架构
继承 `MixedPrecisionTransformerModel` 类，实现自定义的模型逻辑。

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者
