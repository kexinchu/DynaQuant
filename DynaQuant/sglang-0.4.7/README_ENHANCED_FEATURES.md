# SGLang真正混合精度集成

本项目将**真正的混合精度**功能集成到SGLang 0.4.7的核心架构中，支持多种量化格式共存，保持各自的压缩格式以节省GPU存储，而不是转换为统一格式。

## 🎯 核心优势

### 1. **真正的混合精度**
- **多种量化格式共存**: FP16、FP8、Int4、Int8格式同时存在
- **保持压缩格式**: GPTQ Int4权重保持压缩格式，节省75%存储
- **动态反量化**: 仅在推理时反量化，不预先转换
- **真正的内存节省**: 不是虚假的格式转换，而是真正的压缩存储

### 2. **SGLang深度集成**
- **使用SGLang的API**: 通过`ModelConfig`、`DeviceConfig`、`LoadConfig`等SGLang核心配置
- **利用SGLang优化**: 使用SGLang的高性能推理引擎和内存管理
- **保持API兼容**: 完全兼容SGLang的现有API和功能

## 🚀 核心功能

### 1. 真正的混合精度权重加载
- **多种量化格式共存**: FP16、FP8、Int4、Int8格式同时存在
- **保持压缩格式**: GPTQ Int4权重保持压缩格式，节省75%存储
- **动态反量化**: 仅在推理时反量化，不预先转换
- **权重缓存**: 支持反量化结果缓存，避免重复计算

### 2. 混合精度线性层
- **动态格式处理**: 根据权重格式动态选择处理方式
- **缓存优化**: 支持权重缓存，提高推理效率
- **内存优化**: 真正的内存节省，不是格式转换

### 3. 专家激活跟踪（独立版本）
- **实时监控**: 跟踪MoE模型中每个专家的激活情况
- **统计分析**: 提供详细的专家使用统计
- **性能分析**: 分析专家利用率和负载分布
- **数据导出**: 支持统计数据的导出和分析

## 📁 文件结构

```
sglang-0.4.7/
├── python/sglang/srt/
│   ├── model_loader/
│   │   ├── true_mixed_precision_loader.py      # 真正的混合精度加载器
│   │   ├── sglang_mixed_precision_loader.py    # SGLang集成的混合精度加载器
│   │   ├── enhanced_mixed_precision_loader.py  # 增强的混合精度加载器（独立版本）
│   │   └── loader.py                           # 修改的SGLang加载器（集成混合精度）
│   ├── layers/
│   │   └── mixed_precision_linear.py           # 混合精度线性层
│   ├── models/
│   │   └── moe_tracker.py                      # MoE专家跟踪器
│   └── enhanced_model_loader.py                # 增强的模型加载器（独立版本）
├── launch_sglang_mixed_precision.py            # SGLang集成服务器启动脚本
├── test_true_mixed_precision.py                # 真正混合精度功能测试脚本
├── test_sglang_integration.py                  # SGLang集成测试脚本
├── start_sglang_mixed_precision.sh             # SGLang集成启动脚本
├── mixed_precision_config.yaml            # 真正混合精度配置文件
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

### 1. 运行真正混合精度测试
```bash
# 测试真正混合精度功能
python3 test_true_mixed_precision.py

# 测试SGLang集成功能
python3 test_sglang_integration.py

# 或者使用启动脚本测试
./start_sglang_mixed_precision.sh --help
```

### 2. 启动真正混合精度服务器
```bash
# 使用真正混合精度配置
./start_sglang_mixed_precision.sh -m /path/to/model -c mixed_precision_config.yaml

# 或者直接运行Python脚本
python3 launch_sglang_mixed_precision.py \
  --model /path/to/model \
  --mixed-precision-config mixed_precision_config.yaml \
  --device cuda \
  --dtype auto \
  --test
```

### 3. 使用真正混合精度API
```python
# 使用真正混合精度功能
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import DefaultModelLoader

# 创建SGLang配置
model_config = ModelConfig(
    model_path="/path/to/model",
    mixed_precision_config="mixed_precision_config.yaml",
    dtype="auto",
    trust_remote_code=True
)

device_config = DeviceConfig(device="cuda")
load_config = LoadConfig(load_format=LoadFormat.AUTO)

# 使用SGLang加载器加载模型（自动使用真正混合精度）
loader = DefaultModelLoader(load_config)
model = loader.load_model(
    model_config=model_config,
    device_config=device_config
)

# 获取真正混合精度统计
from sglang.srt.model_loader.true_mixed_precision_loader import get_global_true_mixed_precision_loader
from sglang.srt.layers.mixed_precision_linear import get_mixed_precision_memory_stats

mixed_precision_loader = get_global_true_mixed_precision_loader()
if mixed_precision_loader:
    memory_stats = get_mixed_precision_memory_stats()
    print(f"内存节省: {memory_stats['memory_saved_mb']:.2f}MB")
    print(f"压缩比: {memory_stats['compression_ratio']:.2f}x")
```

## 🔧 SGLang集成架构

### 1. 集成方式
- **继承SGLang基类**: `SGLangMixedPrecisionLoader`继承自`ModelLoader`
- **使用SGLang配置**: 通过`ModelConfig`的`mixed_precision_config`参数
- **集成到加载流程**: 在`DefaultModelLoader.load_model()`中自动检测和使用
- **保持向后兼容**: 不影响SGLang的现有功能

### 2. 核心组件
- **SGLangMixedPrecisionLoader**: 继承SGLang的ModelLoader，支持混合精度
- **SGLangGPTQDequantizer**: 集成到SGLang的GPTQ反量化器
- **MixedPrecisionConfig**: 混合精度配置数据结构
- **全局加载器管理**: 通过全局变量管理混合精度加载器实例

### 3. 工作流程
```
1. 创建ModelConfig，指定mixed_precision_config
2. DefaultModelLoader检测到混合精度配置
3. 自动创建SGLangMixedPrecisionLoader
4. 加载混合精度权重到模型
5. 使用SGLang的推理引擎进行推理
```

## 📊 专家激活跟踪（独立版本）

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
