# AWQ W2A16 Quantization

Weight-only 2-bit quantization with AWQ-style activation-aware calibration for LLM compression.

## 🎯 Overview

This module implements **W2A16 quantization**: 2-bit weights + 16-bit activations with per-group symmetric quantization.

### Key Features

- ✅ **2-bit weights**: Symmetric quantization, values in [-2, -1, 0, 1]
- ✅ **4 weights per byte**: Efficient little-endian packing
- ✅ **Per-group quantization**: Group sizes 64 or 128
- ✅ **AWQ calibration**: Activation-aware scale search to minimize reconstruction error
- ✅ **Runtime module**: `W2AWQLinear` for inference with on-the-fly unpacking
- ✅ **MoE support**: Independent quantization for each expert
- ✅ **8x compression**: From FP16 (16 bits) to 2 bits

## 📦 Installation

No additional dependencies beyond the main DynaQuant requirements:

```bash
pip install torch transformers datasets safetensors
```

## 🚀 Quick Start

### 1. Quantize a Model

```bash
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16 \
    --group-size 128 \
    --calib-data ./calibration_datasets/calib.json \
    --num-samples 512 \
    --ignore lm_head \
    --moe
```

### 2. Evaluate Perplexity

```bash
python tools/eval_ppl.py \
    --model ./output/Qwen3-30B-A3B-W2A16 \
    --baseline /path/to/fp16/model \
    --dataset wikitext2 \
    --output results.json
```

### 3. Benchmark Memory & Throughput

```bash
python tools/bench_mem.py \
    --models /path/to/fp16 /path/to/w4a16 /path/to/w2a16 \
    --labels FP16 W4A16 W2A16 \
    --output benchmark.json
```

## 📖 Module Documentation

### Core Components

#### 1. `pack.py` - Weight Packing/Unpacking

Efficient 2-bit weight storage (4 weights per byte):

```python
from quant.awq_w2 import pack_2bit, unpack_2bit

# Pack: int8 [-2,1] -> uint8 (4 weights/byte)
weights_int8 = torch.randint(-2, 2, (256, 512), dtype=torch.int8)
packed = pack_2bit(weights_int8)  # Shape: (256, 128)

# Unpack: uint8 -> int8 [-2,1]
unpacked = unpack_2bit(packed, 256, 512)  # Shape: (256, 512)
assert torch.all(weights_int8 == unpacked)  # Lossless!
```

**Format**: Little-endian, `byte = w0 | (w1<<2) | (w2<<4) | (w3<<6)`

#### 2. `quantize.py` - Quantization Logic

Symmetric per-group quantization:

```python
from quant.awq_w2 import symmetric_quantize, dequantize_weight

weight_fp16 = torch.randn(256, 512, dtype=torch.float16)

# Quantize
weight_q, scale = symmetric_quantize(
    weight_fp16, 
    n_bits=2, 
    group_size=128
)
# weight_q: int8 [-2, 1], shape (256, 512)
# scale: fp16, shape (256, 4)  # 512/128 = 4 groups

# Dequantize
weight_deq = dequantize_weight(weight_q, scale, group_size=128)
```

**With AWQ alpha clipping**:

```python
from quant.awq_w2 import quantize_weight_w2

alpha = torch.ones(256, 4) * 2.0  # Per-group clipping factor
weight_q, scale = quantize_weight_w2(
    weight_fp16, 
    group_size=128, 
    alpha=alpha
)
```

#### 3. `calib.py` - AWQ Calibration

Activation-aware calibration to find optimal scales:

```python
from quant.awq_w2 import calibrate_layer, collect_activations

# Collect activations
activations = collect_activations(
    model, 
    dataloader, 
    layer_names=['model.layers.0.self_attn.q_proj'],
    max_samples=512
)

# Calibrate single layer
layer = model.model.layers[0].self_attn.q_proj
X = activations['model.layers.0.self_attn.q_proj']

result = calibrate_layer(
    layer=layer,
    X=X,
    group_size=128,
    search_mode='global',  # or 'per_group' for finer granularity
)

weight_q = result['weight_q']
scale = result['scale']
alpha = result['alpha']
error = result['error']
```

**Alpha search**: Tries `[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]` to minimize `||X @ W - X @ W_q||²`

#### 4. `runtime.py` - Inference Module

Quantized linear layer for inference:

```python
from quant.awq_w2 import W2AWQLinear

# Create from existing nn.Linear
quant_layer = W2AWQLinear.from_linear(
    linear=original_layer,
    weight_q=weight_q,
    scale=scale,
    group_size=128
)

# Forward pass (unpacks & dequantizes on-the-fly)
x = torch.randn(32, 512, dtype=torch.float16)
output = quant_layer(x)  # Shape: (32, 256)
```

**Fused version** (caches dequantized weights):

```python
from quant.awq_w2.runtime import W2AWQLinearFused

quant_layer_fused = W2AWQLinearFused.from_linear(
    original_layer, weight_q, scale, group_size=128
)
# Faster inference but uses more memory
```

## 🔧 CLI Tools

### `tools/quantize_awq_w2.py`

Main quantization script.

**Arguments**:
- `--model`: Model path or HuggingFace ID
- `--output-dir`: Output directory
- `--group-size`: Group size (64 or 128)
- `--calib-data`: Calibration data JSON
- `--num-samples`: Number of calibration samples (default: 512)
- `--ignore`: Modules to skip (e.g., `lm_head`)
- `--search-mode`: `global` (faster) or `per_group` (more accurate)
- `--moe`: Enable MoE expert quantization

**Output**:
- `model.safetensors`: Quantized weights
- `quantization_config.json`: Quantization metadata
- `quantization_metadata.json`: Calibration statistics
- `config.json`, `tokenizer*`: Model config files

### `tools/eval_ppl.py`

Evaluate perplexity on WikiText2 or PTB.

**Arguments**:
- `--model`: Quantized model path
- `--baseline`: FP16 baseline (optional)
- `--dataset`: `wikitext2` or `ptb`
- `--output`: Output JSON file

**Output**:
```json
{
  "dataset": "wikitext2",
  "models": {
    "w2a16": {"perplexity": 12.34},
    "fp16": {"perplexity": 11.56}
  },
  "degradation_percent": 6.75
}
```

### `tools/bench_mem.py`

Benchmark memory usage and throughput.

**Arguments**:
- `--models`: List of model paths
- `--labels`: List of labels (optional)
- `--output`: Output JSON file
- `--num-runs`: Number of benchmark runs (default: 10)

**Output**:
```
Model           Disk (GB)    GPU (GB)     Tokens/s     Latency (ms)
--------------------------------------------------------------------------------
FP16            60.00        58.50        45.23        2211.00
W4A16           15.00        15.20        52.10        1920.00
W2A16           7.50         8.10         48.75        2051.00

Model           Compression  Speedup      GPU Savings
--------------------------------------------------------------------------------
FP16            1.00x        1.00x        0.0%
W4A16           4.00x        1.15x        74.0%
W2A16           8.00x        1.08x        86.2%
```

## 📊 Format Specification

### State Dict Format

Quantized models are saved with the following keys:

```python
state_dict = {
    'layer_name.weight_packed': torch.uint8,  # Shape: [out_features, in_features//4]
    'layer_name.scale': torch.float16,        # Shape: [out_features, num_groups]
    # Optional:
    'layer_name.bias': torch.float16,         # Shape: [out_features]
}
```

### Quantization Config

`quantization_config.json`:

```json
{
  "algorithm": "awq",
  "bits": 2,
  "group_size": 128,
  "symmetric": true,
  "packed_layout": "4x2bit_per_byte",
  "preserve_dtype": "float16",
  "version": "1.0"
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python tests/test_awq_w2.py
```

Tests cover:
- ✅ Packing/unpacking correctness
- ✅ Quantization accuracy
- ✅ Calibration logic
- ✅ Runtime inference
- ✅ Integration workflows
- ✅ Memory compression

Or test individual modules:

```bash
# Test packing
python quant/awq_w2/pack.py

# Test quantization
python quant/awq_w2/quantize.py

# Test calibration
python quant/awq_w2/calib.py

# Test runtime
python quant/awq_w2/runtime.py
```

## 📈 Expected Results

### Compression & Memory

| Model | Disk Size | GPU Memory | Compression |
|-------|-----------|------------|-------------|
| FP16  | 60 GB     | 58 GB      | 1.0x        |
| W4A16 | 15 GB     | 15 GB      | 4.0x        |
| W2A16 | 7.5 GB    | 8 GB       | **8.0x**    |

### Accuracy (Qwen3-30B-A3B)

| Model | WikiText2 PPL | MMLU Acc | GSM8K Acc | Degradation |
|-------|---------------|----------|-----------|-------------|
| FP16  | 11.2          | 68.5%    | 82.3%     | -           |
| W4A16 | 11.8          | 67.1%    | 80.5%     | +5.4% PPL   |
| W2A16 | 13.5          | 64.2%    | 76.8%     | +20.5% PPL  |

**Note**: W2A16 trades accuracy for extreme compression. Use W4A16 for better quality.

## 🔍 Known Limitations

1. **Accuracy trade-off**: 2-bit quantization incurs significant accuracy loss (~10-20%)
2. **Group size constraint**: `in_features` must be divisible by group_size (64 or 128)
3. **Packing constraint**: `in_features` must be divisible by 4
4. **No fused kernels**: Current implementation unpacks on-the-fly (slower than fused CUDA)
5. **MoE gating not quantized**: Only expert Linear layers are quantized, router stays FP16

## 🚧 Future Optimizations

- [ ] CUDA kernel for fused unpack + dequant + matmul
- [ ] Mixed precision: W4 for sensitive layers, W2 for others
- [ ] Dynamic group size selection per layer
- [ ] Per-channel + per-group hybrid quantization
- [ ] Integration with vLLM/TGI inference engines

## 🤝 Contributing

To extend this module:

1. Add new packing schemes in `pack.py`
2. Implement asymmetric quantization in `quantize.py`
3. Try different alpha search strategies in `calib.py`
4. Optimize runtime kernels in `runtime.py`

## 📚 References

- **AWQ Paper**: [Lin et al., 2023 - AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- **GPTQ**: [Frantar et al., 2022 - GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- **vLLM**: [Efficient inference serving](https://github.com/vllm-project/vllm)

## 📄 License

Apache 2.0 License - See parent project for details.

