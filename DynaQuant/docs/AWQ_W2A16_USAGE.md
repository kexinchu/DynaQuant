# AWQ W2A16 Usage Guide

Complete guide for using AWQ W2A16 quantization on LLMs, including MoE models.

## 📋 Table of Contents

1. [Installation](#installation)
2. [Preparing Calibration Data](#preparing-calibration-data)
3. [Quantizing Models](#quantizing-models)
4. [Loading and Using Quantized Models](#loading-and-using-quantized-models)
5. [Evaluation](#evaluation)
6. [Benchmarking](#benchmarking)
7. [MoE Models](#moe-models)
8. [Troubleshooting](#troubleshooting)

## Installation

```bash
cd /root/code/DynaQuant/DynaQuant
pip install -r requirements.txt

# Verify installation
python -c "from quant.awq_w2 import W2AWQLinear; print('✓ AWQ W2A16 installed')"
```

## Preparing Calibration Data

### Option 1: Use Existing Data

If you have calibration data in the standard format:

```json
{
  "model_name": "Qwen3-30B-A3B",
  "num_samples": 1024,
  "samples": [
    "This is a sample text for calibration...",
    "Another sample text...",
    ...
  ]
}
```

### Option 2: Generate from Datasets

```python
from datasets import load_dataset
import json

# Load datasets
wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
code = load_dataset('bigcode/the-stack', split='train', streaming=True)

# Collect samples
samples = []

# WikiText samples
for item in wiki['text'][:500]:
    if len(item) > 100:  # Filter short texts
        samples.append(item)

# Code samples (for code-focused models)
for i, item in enumerate(code):
    if i >= 500:
        break
    samples.append(item['content'])

# Save
calib_data = {
    "model_name": "Qwen3-30B-A3B",
    "num_samples": len(samples),
    "samples": samples
}

with open('calibration_datasets/calib_qwen3.json', 'w') as f:
    json.dump(calib_data, f)
```

### Option 3: Multi-Domain Mix (Recommended)

```python
# Mix multiple domains for better calibration
samples = []

# 40% Wiki (general knowledge)
samples.extend(wiki_samples[:400])

# 30% Code (technical)
samples.extend(code_samples[:300])

# 20% Math (reasoning)
samples.extend(math_samples[:200])

# 10% Conversation (dialogue)
samples.extend(chat_samples[:100])
```

## Quantizing Models

### Basic Quantization

```bash
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16 \
    --group-size 128 \
    --calib-data calibration_datasets/calib_qwen3.json \
    --num-samples 512
```

### Advanced Options

#### 1. Per-Group Alpha Search (More Accurate)

```bash
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16-fine \
    --group-size 128 \
    --search-mode per_group \
    --num-samples 1024
```

**Trade-off**: 2-3x slower but typically 1-2% better accuracy.

#### 2. Smaller Group Size (Better Quality)

```bash
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16-gs64 \
    --group-size 64 \
    --num-samples 512
```

**Trade-off**: ~10% larger model size, but lower quantization error.

#### 3. Ignore Specific Layers

```bash
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16-selective \
    --ignore lm_head model.embed_tokens model.norm \
    --num-samples 512
```

**Recommended ignores**:
- `lm_head` - Output layer (keep FP16 for numerical stability)
- `embed_tokens` - Embedding layer
- `norm` - LayerNorm layers

## Loading and Using Quantized Models

### Option 1: Using Transformers (Simple)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('/root/code/DynaQuant/DynaQuant')

from quant.awq_w2.runtime import replace_linear_with_w2awq
import safetensors.torch as safetensors

model_path = './output/Qwen3-30B-A3B-W2A16'

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

# Load quantized weights
state_dict = safetensors.load_file(f'{model_path}/model.safetensors')

# Replace layers
replace_linear_with_w2awq(model, state_dict, group_size=128)

# Use normally
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_ids = tokenizer("Hello, how are you?", return_tensors='pt').input_ids

output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

### Option 2: Manual Replacement

```python
from quant.awq_w2 import W2AWQLinear, unpack_2bit
import torch.nn as nn

# Find a linear layer
layer_name = 'model.layers.0.self_attn.q_proj'
original_layer = get_layer_by_name(model, layer_name)

# Load quantized weights
weight_packed = state_dict[f'{layer_name}.weight_packed']
scale = state_dict[f'{layer_name}.scale']

# Create quantized layer
quant_layer = W2AWQLinear(
    in_features=original_layer.in_features,
    out_features=original_layer.out_features,
    bias=original_layer.bias is not None,
    group_size=128
)

quant_layer.load_weights(weight_packed, scale, bias=original_layer.bias, packed=True)

# Replace
set_layer_by_name(model, layer_name, quant_layer)
```

### Option 3: Custom Inference Loop

```python
import torch

model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        logits = outputs.logits
        # Process logits...
```

## Evaluation

### WikiText2 Perplexity

```bash
# With baseline comparison
python tools/eval_ppl.py \
    --model ./output/Qwen3-30B-A3B-W2A16 \
    --baseline /dev/shm/Qwen3-30B-A3B \
    --dataset wikitext2 \
    --output eval_results.json

# Output:
# ================================================================================
# Comparison Results
# ================================================================================
# FP16 PPL:         11.23
# W2A16 PPL:        13.54
# Degradation:      +20.57%
# ================================================================================
```

### PTB Perplexity

```bash
python tools/eval_ppl.py \
    --model ./output/Qwen3-30B-A3B-W2A16 \
    --dataset ptb \
    --output eval_ptb.json
```

### MMLU / GSM8K (using existing scripts)

```bash
# Use project's existing evaluation script
python scripts/evaluate_model.py \
    --model ./output/Qwen3-30B-A3B-W2A16 \
    --datasets mmlu gsm8k \
    --output mmlu_gsm8k_results.json
```

## Benchmarking

### Memory & Throughput

```bash
python tools/bench_mem.py \
    --models \
        /dev/shm/Qwen3-30B-A3B \
        ./output/Qwen3-30B-A3B-W4A16 \
        ./output/Qwen3-30B-A3B-W2A16 \
    --labels FP16 W4A16 W2A16 \
    --num-runs 20 \
    --output benchmark_comparison.json
```

### Expected Output

```
================================================================================
Memory & Throughput Benchmark
================================================================================
Models: 3
Device: cuda
Runs: 20
================================================================================

Benchmarking: FP16
Path: /dev/shm/Qwen3-30B-A3B
...
[1/3] Measuring disk size...
  Total size: 60.24 GB (4 files)
[2/3] Loading model and measuring memory...
  GPU memory: 58.12 GB
  CPU memory: 2.45 GB
[3/3] Benchmarking inference throughput...
  Tokens/second: 45.67
  Avg latency: 2190.23 ms

...

================================================================================
Comparison Results
================================================================================

Model           Disk (GB)    GPU (GB)     Tokens/s     Latency (ms)
--------------------------------------------------------------------------------
FP16            60.24        58.12        45.67        2190.23
W4A16           15.12        15.34        52.89        1891.45
W2A16           7.56         8.23         48.12        2078.34

Model           Compression  Speedup      GPU Savings
--------------------------------------------------------------------------------
FP16            1.00x        1.00x        0.0%
W4A16           3.99x        1.16x        73.6%
W2A16           7.97x        1.05x        85.8%
================================================================================
```

## MoE Models

For MoE (Mixture of Experts) models like Qwen3-30B-A3B:

### Enable MoE Quantization

```bash
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16-moe \
    --group-size 128 \
    --num-samples 512 \
    --moe  # Enable MoE expert quantization
```

### How It Works

- **Experts**: Each expert's Linear layers are independently quantized
- **Router**: Gating mechanism stays in FP16 (not quantized)
- **Load balancing**: Preserved - quantization doesn't affect routing decisions

### Verify Expert Routing Consistency

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load both models
fp16_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-30B-A3B', torch_dtype=torch.float16, device_map='auto'
)
w2_model = load_quantized_model('./output/Qwen3-30B-A3B-W2A16-moe')

# Tokenize test inputs
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-30B-A3B')
inputs = tokenizer("Test prompt", return_tensors='pt')

# Hook to collect routing decisions
fp16_routes = []
w2_routes = []

def hook_fn_fp16(module, input, output):
    fp16_routes.append(output[1])  # routing weights

def hook_fn_w2(module, input, output):
    w2_routes.append(output[1])

# Register hooks on router layers
# ... (register on both models)

# Run inference
with torch.no_grad():
    fp16_model(**inputs)
    w2_model(**inputs)

# Compare routing consistency
for i, (r1, r2) in enumerate(zip(fp16_routes, w2_routes)):
    top_k_match = (r1.argmax(dim=-1) == r2.argmax(dim=-1)).float().mean()
    print(f"Layer {i}: Top-1 expert match rate: {top_k_match:.2%}")
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

```bash
# 1. Reduce calibration samples
python tools/quantize_awq_w2.py ... --num-samples 256

# 2. Quantize in CPU mode (slower)
python tools/quantize_awq_w2.py ... --device cpu

# 3. Use gradient checkpointing
# (modify quantize_awq_w2.py to enable gradient_checkpointing_enable())
```

### Issue 2: Dimension Mismatch

**Symptom**: `AssertionError: in_features must be divisible by 4`

**Solution**: Pad the layer or skip it:

```python
# Skip layers with incompatible dimensions
python tools/quantize_awq_w2.py ... --ignore lm_head embed_tokens layer_with_odd_dim
```

### Issue 3: Accuracy Degradation Too High

**Symptom**: Perplexity increases by >30%

**Solutions**:

1. **Use smaller group size**:
   ```bash
   --group-size 64
   ```

2. **More calibration samples**:
   ```bash
   --num-samples 1024
   ```

3. **Better calibration data**:
   - Use multi-domain mix
   - Ensure samples are representative

4. **Per-group alpha search**:
   ```bash
   --search-mode per_group
   ```

5. **Hybrid quantization** (some layers W4, others W2):
   ```python
   # Quantize sensitive layers with W4A16
   # Quantize less sensitive layers with W2A16
   # (requires manual scripting)
   ```

### Issue 4: Slow Inference

**Symptom**: Quantized model is slower than FP16

**Cause**: Unpacking overhead

**Solutions**:

1. **Use fused cached layer**:
   ```python
   from quant.awq_w2.runtime import W2AWQLinearFused
   # Replace W2AWQLinear with W2AWQLinearFused
   ```

2. **Pre-unpack weights**:
   ```python
   # Unpack all weights once at load time
   for name, module in model.named_modules():
       if isinstance(module, W2AWQLinear):
           module._weight_cache = module.unpack_and_dequantize()
   ```

3. **Wait for CUDA kernel** (future work):
   - Fused unpack + dequant + matmul kernel
   - Expected 2-3x speedup

### Issue 5: Loading Errors

**Symptom**: `FileNotFoundError: model.safetensors`

**Solution**: Verify quantization completed successfully:

```bash
ls -lh ./output/Qwen3-30B-A3B-W2A16/
# Should see:
# - model.safetensors
# - quantization_config.json
# - quantization_metadata.json
# - config.json
# - tokenizer files
```

If missing, re-run quantization with `--output-dir` specified.

## Best Practices

### 1. Calibration Data Quality

- ✅ Use 512-1024 samples
- ✅ Mix multiple domains (wiki, code, math, chat)
- ✅ Filter out very short texts (< 100 chars)
- ✅ Ensure diversity (different topics, styles)

### 2. Layer Selection

**Always ignore**:
- `lm_head` (output projection)
- Embedding layers
- Very small layers (< 512 dims)

**Consider ignoring** (for better accuracy):
- First few attention layers
- Final attention layers
- Norm layers

### 3. Group Size Selection

- **group_size=128**: Default, good balance
- **group_size=64**: Better accuracy, slightly larger model
- **group_size=256**: Not recommended (too coarse)

### 4. Search Mode

- **global**: 5-10 minutes per model, good for most cases
- **per_group**: 30-60 minutes per model, 1-2% better accuracy

Use `global` first, then try `per_group` if accuracy is critical.

### 5. Validation Workflow

```bash
# 1. Quantize
python tools/quantize_awq_w2.py ...

# 2. Quick sanity check (fast)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./output/MODEL-W2A16')
tokenizer = AutoTokenizer.from_pretrained('./output/MODEL-W2A16')
out = model.generate(tokenizer('Hello', return_tensors='pt').input_ids, max_new_tokens=10)
print(tokenizer.decode(out[0]))
"

# 3. Evaluate perplexity
python tools/eval_ppl.py ...

# 4. Benchmark if acceptable
python tools/bench_mem.py ...
```

## Example: Complete Workflow

```bash
#!/bin/bash
# complete_w2a16_workflow.sh

MODEL="Qwen/Qwen3-30B-A3B"
OUTPUT_DIR="./output/Qwen3-30B-A3B-W2A16"
CALIB_DATA="calibration_datasets/qwen3_calib.json"

# Step 1: Quantize
echo "Step 1: Quantizing model..."
python tools/quantize_awq_w2.py \
    --model $MODEL \
    --output-dir $OUTPUT_DIR \
    --group-size 128 \
    --calib-data $CALIB_DATA \
    --num-samples 512 \
    --ignore lm_head \
    --moe

# Step 2: Evaluate perplexity
echo "Step 2: Evaluating perplexity..."
python tools/eval_ppl.py \
    --model $OUTPUT_DIR \
    --baseline /dev/shm/Qwen3-30B-A3B \
    --dataset wikitext2 \
    --output $OUTPUT_DIR/eval_ppl.json

# Step 3: Benchmark
echo "Step 3: Benchmarking..."
python tools/bench_mem.py \
    --models /dev/shm/Qwen3-30B-A3B $OUTPUT_DIR \
    --labels FP16 W2A16 \
    --output $OUTPUT_DIR/benchmark.json

echo "Done! Check $OUTPUT_DIR for results."
```

Run with:

```bash
bash complete_w2a16_workflow.sh
```

---

For more details, see:
- [Module README](../quant/awq_w2/README.md)
- [Test Suite](../tests/test_awq_w2.py)
- [Main README](../README.md)

