# SGLang增强功能集成

本项目将混合精度权重加载和专家激活跟踪功能集成到SGLang 0.4.7中，提供高效的模型推理和专家行为分析能力。

## 🚀 核心功能

### 1. 混合精度权重加载
- **选择性权重加载**: 根据配置文件选择性加载不同精度的权重
- **GPTQ支持**: 完整的GPTQ-Int4量化模型支持
- **Safetensors兼容**: 支持safetensors索引文件
- **内存优化**: 智能缓存和内存管理

### 2. 专家激活跟踪
- **实时监控**: 跟踪MoE模型中每个专家的激活情况
- **统计分析**: 提供详细的专家使用统计
- **性能分析**: 分析专家利用率和负载分布
- **数据导出**: 支持统计数据的导出和分析

## 📁 文件结构

```
sglang-0.4.7/
├── python/sglang/srt/
│   ├── model_loader/
│   │   ├── enhanced_mixed_precision_loader.py  # 增强的混合精度加载器
│   │   └── mixed_precision_loader.py           # 原始混合精度加载器
│   ├── models/
│   │   └── moe_tracker.py                      # MoE专家跟踪器
│   └── enhanced_model_loader.py                # 增强的模型加载器
├── launch_enhanced_server.py                   # 增强服务器启动脚本
├── test_enhanced_features.py                   # 功能测试脚本
├── start_enhanced_server.sh                    # 启动脚本
├── mixed_precision_config.yaml                 # 混合精度配置文件
└── README_ENHANCED_FEATURES.md                 # 本文档
```

## 🛠️ 安装和配置

### 1. 环境要求
```bash
# Python 3.8+
python3 --version

# 必要的包
pip install torch transformers safetensors pyyaml
```

### 2. 配置文件
创建混合精度配置文件 `mixed_precision_config.yaml`:

```yaml
mixed_precision:
  # 不同精度权重的路径
  fp16_path: "/path/to/fp16/weights"
  fp8_path: "/path/to/fp8/weights"
  int4_path: "/path/to/int4/weights"
  
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

inference:
  max_seq_length: 4096
  max_batch_size: 32
  dtype: "bfloat16"
  device_map: "auto"

server:
  host: "127.0.0.1"
  port: 8080
  max_workers: 4
```

## 🚀 快速开始

### 1. 运行功能测试
```bash
# 测试所有增强功能
./start_enhanced_server.sh --test

# 或者直接运行测试脚本
python3 test_enhanced_features.py
```

### 2. 启动增强服务器
```bash
# 使用启动脚本
./start_enhanced_server.sh -m /path/to/model -c mixed_precision_config.yaml

# 或者直接运行Python脚本
python3 launch_enhanced_server.py \
  --config mixed_precision_config.yaml \
  --model /path/to/model \
  --port 8080 \
  --enable-expert-tracking
```

### 3. 使用API
```python
from sglang.srt.enhanced_model_loader import (
    load_model_with_enhanced_features,
    get_expert_activation_stats,
    reset_expert_activation_stats
)

# 加载模型
stats = load_model_with_enhanced_features(
    model, config_path, enable_expert_tracking=True
)

# 获取专家统计
expert_stats = get_expert_activation_stats()
print(f"专家统计: {expert_stats}")
```

## 📊 专家激活跟踪

### 1. 统计信息类型
- **专家激活次数**: 每个专家被激活的总次数
- **激活时间**: 最后一次激活的时间戳
- **处理token数**: 每个专家处理的token总数
- **层统计**: 每层的专家使用情况
- **热门专家**: 激活次数最多的专家排名

### 2. API接口
```python
# 获取所有专家统计
stats = get_expert_activation_stats()

# 获取特定专家统计
expert_stats = get_expert_activation_stats(layer_id=0, expert_id=1)

# 获取热门专家
top_experts = get_expert_activation_stats()['top_experts']

# 重置统计
reset_expert_activation_stats()

# 导出统计
export_expert_activation_stats("expert_stats.json")
```

### 3. 统计数据结构
```json
{
  "expert_stats": {
    "layer_0_expert_1": {
      "layer_id": 0,
      "expert_id": 1,
      "activation_count": 150,
      "last_activation_time": 1640995200.0,
      "total_tokens_processed": 1500
    }
  },
  "layer_stats": {
    "0": {
      "total_experts": 8,
      "total_activations": 1200,
      "total_tokens": 12000,
      "experts": {
        "0": {"activation_count": 150, "total_tokens_processed": 1500},
        "1": {"activation_count": 120, "total_tokens_processed": 1200}
      }
    }
  },
  "top_experts": [
    {
      "layer_id": 0,
      "expert_id": 1,
      "activation_count": 150,
      "total_tokens_processed": 1500
    }
  ]
}
```

## 🔧 GPTQ支持

### 1. GPTQ权重格式
支持标准的GPTQ量化格式：
- `qweight`: 量化的权重
- `qzeros`: 量化的零点
- `scales`: 缩放因子
- `g_idx`: 分组索引（可选）

### 2. 自动检测
系统会自动检测GPTQ格式并应用相应的反量化算法：
```python
# 自动检测GPTQ格式
if precision == 'int4' and is_gptq_weight(weights, weight_name):
    # 使用GPTQ反量化
    weight = dequantize_gptq_weight(qweight, qzeros, scales, g_idx)
else:
    # 使用标准加载
    weight = weights[weight_name]
```

### 3. 修复的GPTQ反量化算法
最新版本修复了GPTQ反量化中的维度匹配问题：

#### 问题描述
```
ERROR: The size of tensor a (96) must match the size of tensor b (768) at non-singleton dimension 1
qweight: torch.Size([256, 768])
qzeros: torch.Size([16, 96])
scales: torch.Size([16, 768])
```

#### 修复方案
- **正确的维度计算**: 基于实际的group_size计算扩展因子
- **智能维度匹配**: 自动调整scales和zeros的维度以匹配unpacked权重
- **详细的调试信息**: 提供完整的维度计算过程日志

#### 修复文件
- `gptq_dequantizer_fixed.py`: 修复的GPTQ反量化器
- `enhanced_mixed_precision_loader.py`: 集成修复的加载器
- `test_gptq_fix.py`: 修复验证测试

#### 测试验证
```bash
# 运行GPTQ修复测试
python3 test_gptq_fix.py

# 运行简单测试
python3 simple_gptq_test.py
```

## 📈 性能优化

### 1. 内存优化
- **权重缓存**: 智能缓存已加载的权重文件
- **选择性加载**: 只加载需要的权重
- **内存映射**: 支持大文件的懒加载

### 2. 计算优化
- **混合精度**: 平衡精度和性能
- **向量化操作**: 优化的GPTQ反量化
- **并行处理**: 支持并发请求

### 3. 存储优化
- **压缩存储**: 支持多种压缩格式
- **索引文件**: 利用safetensors索引
- **分片加载**: 支持大模型的分片加载

## 🔧 设备问题修复

### 1. 设备不匹配问题
自动检测和修复CUDA/CPU设备不匹配问题：

```python
from fix_device_issues import comprehensive_device_fix

# 综合设备修复
results = comprehensive_device_fix(model, tokenizer, 'cuda')
print(f"修复结果: {results}")
```

### 2. 注意力掩码修复
自动处理pad_token和eos_token相同的情况：

```python
from fix_device_issues import create_proper_attention_mask

# 创建正确的注意力掩码
attention_mask = create_proper_attention_mask(input_ids, tokenizer, 'cuda')
```

### 3. MoE模块设备修复
专门处理MoE模块的设备问题：

```python
from fix_device_issues import fix_moe_device_issues

# 修复MoE模块设备问题
model = fix_moe_device_issues(model, 'cuda')
```

### 4. 设备一致性验证
验证模型所有参数都在正确设备上：

```python
from fix_device_issues import validate_model_device_consistency

# 验证设备一致性
validation = validate_model_device_consistency(model, 'cuda')
if validation['is_consistent']:
    print("✓ 设备一致性检查通过")
else:
    print(f"⚠ 发现设备问题: {validation['issues']}")
```

## 🧪 测试和验证

### 1. 功能测试
```bash
# 运行所有测试
python3 test_enhanced_features.py

# 测试特定功能
python3 -c "
from sglang.srt.model_loader.enhanced_mixed_precision_loader import GPTQDequantizer
import torch
# 测试GPTQ反量化
qweight = torch.randint(0, 16, (256, 768), dtype=torch.int32)
qzeros = torch.randint(0, 16, (16, 96), dtype=torch.int32)
scales = torch.randn(16, 768, dtype=torch.float16)
weight = GPTQDequantizer.dequantize_gptq_weight(qweight, qzeros, scales)
print(f'GPTQ反量化成功，输出形状: {weight.shape}')
"
```

### 2. 性能测试
```python
import time
from sglang.srt.enhanced_model_loader import load_model_with_enhanced_features

# 测试加载时间
start_time = time.time()
stats = load_model_with_enhanced_features(model, config_path)
load_time = time.time() - start_time
print(f"模型加载时间: {load_time:.2f}秒")
```

## 🔍 故障排除

### 1. 常见问题

**Q: 配置文件格式错误**
```bash
# 检查YAML格式
python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

**Q: 模型路径不存在**
```bash
# 检查模型路径
ls -la /path/to/model/
```

**Q: 专家跟踪不工作**
```python
# 检查专家跟踪器
from sglang.srt.model_loader.enhanced_mixed_precision_loader import get_global_expert_tracker
tracker = get_global_expert_tracker()
if tracker:
    print("专家跟踪器已启用")
else:
    print("专家跟踪器未启用")
```

**Q: 设备不匹配错误**
```
WARNING: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**解决方案**:
```bash
# 运行设备修复测试
python3 test_device_fix.py

# 使用设备修复功能
from fix_device_issues import comprehensive_device_fix
results = comprehensive_device_fix(model, tokenizer, 'cuda')
```

**Q: 注意力掩码警告**
```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
```

**解决方案**:
- 系统会自动创建正确的注意力掩码
- 确保tokenizer的pad_token_id和eos_token_id设置正确

### 2. 调试模式
```bash
# 启用详细日志
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH
python3 -u launch_enhanced_server.py --config config.yaml --model /path/to/model
```

## 📚 API参考

### EnhancedModelLoader
```python
class EnhancedModelLoader:
    def __init__(self, config_path: str, enable_expert_tracking: bool = True)
    def load_model(self, model: torch.nn.Module, enable_moe_tracking: bool = True) -> Dict[str, Any]
    def get_expert_tracker(self) -> Optional[ExpertActivationTracker]
    def get_expert_stats(self, layer_id: Optional[int] = None, expert_id: Optional[int] = None) -> Dict
    def get_top_experts(self, top_k: int = 10) -> list
    def get_layer_stats(self) -> Dict
    def reset_expert_stats(self)
    def export_expert_stats(self, file_path: str)
```

### GPTQDequantizer
```python
class GPTQDequantizer:
    @staticmethod
    def dequantize_gptq_weight(qweight: torch.Tensor, qzeros: torch.Tensor, 
                              scales: torch.Tensor, g_idx: Optional[torch.Tensor] = None,
                              bits: int = 4, group_size: int = 128) -> torch.Tensor
```

### ExpertActivationTracker
```python
class ExpertActivationTracker:
    def record_expert_activation(self, layer_id: int, expert_id: int, 
                               tokens_processed: int = 1, request_id: str = None)
    def record_request(self, request_id: str, input_length: int, output_length: int)
    def get_expert_stats(self, layer_id: Optional[int] = None, expert_id: Optional[int] = None) -> Dict
    def get_top_experts(self, top_k: int = 10) -> List[Dict]
    def get_layer_stats(self) -> Dict[int, Dict]
    def reset_stats(self)
    def export_stats(self, file_path: str)
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd sglang-0.4.7

# 安装依赖
pip install -r requirements.txt

# 运行测试
python3 test_enhanced_features.py
```

## 📄 许可证

本项目基于SGLang的许可证，请参考原始项目的许可证文件。

## 🙏 致谢

感谢SGLang团队提供的高性能LLM推理框架，以及开源社区对GPTQ和MoE技术的贡献。
