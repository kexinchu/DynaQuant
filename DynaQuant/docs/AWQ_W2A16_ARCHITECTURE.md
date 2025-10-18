# AWQ W2A16 Architecture

Visual architecture and data flow documentation.

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWQ W2A16 System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │ Calibration  │ --> │ Quantization │ --> │   Packing    │   │
│  │   (calib.py) │     │(quantize.py) │     │   (pack.py)  │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                     │                     │          │
│         v                     v                     v          │
│  Activation Stats      Quantized Weights    Packed uint8      │
│  [n_samples, dim]      [out, in] int8       [out, in//4]      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Runtime Inference                     │  │
│  │                     (runtime.py)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │                                                       │
│         v                                                       │
│  W2AWQLinear: unpack -> dequantize -> matmul                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### 1. Quantization Pipeline

```
FP16 Model
    │
    ├─> Load Model & Tokenizer
    │       │
    │       v
    ├─> Load Calibration Data (JSON)
    │       │
    │       └─> [512-1024 text samples]
    │
    ├─> FOR EACH Linear Layer:
    │   │
    │   ├─> Collect Activations (calib.py)
    │   │   ├─> Register forward hook
    │   │   ├─> Run calibration samples
    │   │   └─> X: [n_samples, in_features]
    │   │
    │   ├─> Search Alpha (calib.py)
    │   │   ├─> Try α ∈ [1.0, 1.5, ..., 8.0]
    │   │   ├─> Compute: ||X@W - X@W_q||²
    │   │   └─> α_opt, s_opt: [out_features, n_groups]
    │   │
    │   ├─> Quantize Weights (quantize.py)
    │   │   ├─> W_clipped = clip(W, -α*max, α*max)
    │   │   ├─> s = max(|W_clipped|) per group
    │   │   ├─> W_q = clamp(round(W / s), -2, 1)
    │   │   └─> W_q: [out, in] int8, s: [out, n_groups] fp16
    │   │
    │   └─> Pack Weights (pack.py)
    │       ├─> W_uint8 = W_int8 + 2  # [-2,1] -> [0,3]
    │       ├─> Reshape: [out, in//4, 4]
    │       ├─> byte = w0|(w1<<2)|(w2<<4)|(w3<<6)
    │       └─> W_packed: [out, in//4] uint8
    │
    └─> Save to SafeTensors
        ├─> model.safetensors
        │   ├─> layer.weight_packed: uint8
        │   └─> layer.scale: fp16
        ├─> quantization_config.json
        └─> metadata.json
```

### 2. Inference Pipeline

```
Load Quantized Model
    │
    ├─> Load model structure (transformers)
    ├─> Load quantized weights (safetensors)
    │
    └─> FOR EACH Quantized Layer:
        │
        ├─> Create W2AWQLinear
        │   ├─> weight_packed: [out, in//4] uint8
        │   └─> scale: [out, n_groups] fp16
        │
        └─> Forward Pass:
            │
            ├─> INPUT: x [batch, seq, in] fp16
            │
            ├─> Unpack (pack.py)
            │   ├─> Extract: w0 = byte & 0b11
            │   ├─>         w1 = (byte >> 2) & 0b11
            │   ├─>         w2 = (byte >> 4) & 0b11
            │   ├─>         w3 = (byte >> 6) & 0b11
            │   └─> W_int8: [out, in] int8 [-2,1]
            │
            ├─> Dequantize (quantize.py)
            │   ├─> Reshape: [out, n_groups, group_size]
            │   ├─> Broadcast scale: [out, n_groups, 1]
            │   ├─> W_deq = W_int8.float() * scale
            │   └─> W_deq: [out, in] fp16
            │
            ├─> MatMul
            │   └─> y = x @ W_deq.T + bias
            │
            └─> OUTPUT: y [batch, seq, out] fp16
```

## 🧩 Component Interactions

```
┌────────────────────────────────────────────────────────────┐
│                    Component Diagram                        │
└────────────────────────────────────────────────────────────┘

┌─────────────┐
│  User Input │
└──────┬──────┘
       │
       v
┌─────────────────────┐
│ quantize_awq_w2.py  │ (CLI Entry Point)
└──────┬──────────────┘
       │
       ├──> Load Model (transformers)
       │
       ├──> find_linear_layers() 
       │    └─> Returns: Dict[name, nn.Linear]
       │
       ├──> create_calibration_dataloader()
       │    └─> Returns: DataLoader[calibration samples]
       │
       └──> FOR EACH layer:
            │
            ├──> collect_activations(model, dataloader, [layer_name])
            │    │   [from calib.py]
            │    └─> Returns: X [n_samples, in_features]
            │
            ├──> calibrate_layer(layer, X, group_size, search_mode)
            │    │   [from calib.py]
            │    │
            │    ├─> search_scale_alpha(W, X, group_size)
            │    │   └─> Returns: α_opt, s_opt, error
            │    │
            │    └─> quantize_weight_w2(W, group_size, α_opt)
            │        │   [from quantize.py]
            │        └─> Returns: W_q (int8), s (fp16)
            │
            └──> pack_2bit(W_q)
                 │   [from pack.py]
                 └─> Returns: W_packed (uint8)

Save Results:
├─> safetensors.save_file(state_dict, "model.safetensors")
├─> json.dump(quant_config, "quantization_config.json")
└─> model.config.save_pretrained(output_dir)
```

## 🔢 Data Types & Shapes

### Quantization Stage

```python
# Original weights
W_fp16: torch.Tensor          # [out_features, in_features], dtype=float16

# After quantization
W_q: torch.Tensor             # [out_features, in_features], dtype=int8, range=[-2,1]
scale: torch.Tensor           # [out_features, n_groups], dtype=float16
                              # where n_groups = in_features / group_size

# After packing
W_packed: torch.Tensor        # [out_features, in_features//4], dtype=uint8
                              # each byte stores 4 weights
```

### Runtime Stage

```python
# Stored in module
self.weight_packed: torch.Tensor  # [out_features, in_features//4], uint8
self.scale: torch.Tensor          # [out_features, n_groups], float16
self.bias: torch.Tensor           # [out_features], float16 (optional)

# Forward pass
x_in: torch.Tensor               # [batch, seq_len, in_features], float16
W_unpacked: torch.Tensor         # [out_features, in_features], int8 [-2,1]
W_dequant: torch.Tensor          # [out_features, in_features], float16
y_out: torch.Tensor              # [batch, seq_len, out_features], float16
```

## 🎨 Quantization Schemes Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                  Quantization Bit Budget                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  FP16:    ████████████████  16 bits/weight                    │
│           Memory: 60 GB                                        │
│                                                                │
│  W8A16:   ████████          8 bits/weight                     │
│           Memory: 30 GB (2x compression)                       │
│                                                                │
│  W4A16:   ████              4 bits/weight                     │
│           Memory: 15 GB (4x compression) ✅ Recommended        │
│                                                                │
│  W2A16:   ██                2 bits/weight                     │
│           Memory: 7.5 GB (8x compression) 🆕 This impl        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Accuracy Trade-off:
FP16   ━━━━━━━━━━━━━━━━━━━━━━━━ 100%
W8A16  ━━━━━━━━━━━━━━━━━━━━━━━  99%  (< 1% loss)
W4A16  ━━━━━━━━━━━━━━━━━━━━━    97%  (1-3% loss)
W2A16  ━━━━━━━━━━━━━━━━━       85%  (10-20% loss)
```

## 🔐 Weight Packing Format

### 2-bit Little-Endian Packing

```
Input: 4 weights in range [-2, -1, 0, 1]
Example: w = [-2, 0, 1, -1]

Step 1: Convert to unsigned [0, 3]
w_unsigned = w + 2 = [0, 2, 3, 1]
Binary: 00, 10, 11, 01

Step 2: Pack into byte (little-endian)
byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
     = 00 | (10 << 2) | (11 << 4) | (01 << 6)
     = 00 | 00001000 | 00110000 | 01000000
     = 01111000
     = 0x78

Visualization:
┌─────┬─────┬─────┬─────┐
│ w3  │ w2  │ w1  │ w0  │  Weights
├─────┼─────┼─────┼─────┤
│ 01  │ 11  │ 10  │ 00  │  Binary
└─────┴─────┴─────┴─────┘
   ↓
0b01111000 = 0x78 = 120

Step 3: Unpack
w0 = byte & 0b00000011 = 0b00 = 0
w1 = (byte >> 2) & 0b00000011 = 0b10 = 2
w2 = (byte >> 4) & 0b00000011 = 0b11 = 3
w3 = (byte >> 6) & 0b00000011 = 0b01 = 1

Step 4: Convert back to signed
w = [0, 2, 3, 1] - 2 = [-2, 0, 1, -1] ✓
```

## 🧮 Mathematical Formulation

### Symmetric Quantization

```
Given:
- W ∈ ℝ^(out×in): Original weight matrix
- b = 2: Number of bits
- g = 128: Group size

Per-group quantization:
1. Reshape: W → [out, n_groups, g] where n_groups = in/g

2. Compute scale per group:
   s[i,j] = max(|W[i,j,:]|)  # i=out_channel, j=group

3. Quantize:
   Q_min = -2^(b-1) = -2
   Q_max = 2^(b-1) - 1 = 1
   
   W_q[i,j,k] = clamp(round(W[i,j,k] / s[i,j]), Q_min, Q_max)

4. Dequantize:
   W_deq[i,j,k] = W_q[i,j,k] * s[i,j]

Reconstruction error:
   E = ||W - W_deq||_2^2
```

### AWQ Alpha Clipping

```
Goal: Minimize activation-weighted error

Given:
- X ∈ ℝ^(m×in): Calibration activations (m samples)
- α: Clipping factor

Algorithm:
1. For each α ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}:
   
   a. Clip weights per group:
      W_clip[i,j,k] = clamp(W[i,j,k], -α*s[i,j], α*s[i,j])
   
   b. Quantize clipped weights:
      W_q = Quantize(W_clip)
   
   c. Compute error:
      E(α) = ||X @ W - X @ Dequantize(W_q)||_2^2
   
2. Select α* = argmin_α E(α)

3. Return quantized weights with optimal α*
```

## 📊 Memory Layout

### State Dict Structure

```json
{
  "model.layers.0.self_attn.q_proj.weight_packed": {
    "shape": [4096, 1024],  // [out_features, in_features//4]
    "dtype": "uint8",
    "size_bytes": 4194304    // 4MB
  },
  "model.layers.0.self_attn.q_proj.scale": {
    "shape": [4096, 32],     // [out_features, in_features//group_size]
    "dtype": "float16",
    "size_bytes": 262144      // 256KB
  },
  "model.layers.0.self_attn.q_proj.bias": {
    "shape": [4096],
    "dtype": "float16",
    "size_bytes": 8192        // 8KB
  }
}
```

### Compression Analysis

```
Original (FP16):
- Weights: out × in × 2 bytes = 4096 × 4096 × 2 = 33.6 MB

Quantized (W2A16):
- Packed weights: out × (in/4) × 1 byte = 4096 × 1024 × 1 = 4.2 MB
- Scales: out × (in/group_size) × 2 bytes = 4096 × 32 × 2 = 0.26 MB
- Total: 4.46 MB

Compression ratio: 33.6 / 4.46 ≈ 7.5x ✓

Additional metadata overhead:
- Quantization config: ~1 KB
- Metadata JSON: ~1 KB
- Negligible relative to model size
```

## 🔄 Workflow State Machine

```
┌─────────┐
│  START  │
└────┬────┘
     │
     v
┌────────────────┐      Error: CUDA OOM
│ Load Model     │ ─────────────────────┐
└────┬───────────┘                      │
     │                                  v
     v                          ┌───────────────┐
┌────────────────┐              │ Reduce samples│
│ Load Calib     │              │ or use CPU    │
└────┬───────────┘              └───────┬───────┘
     │                                  │
     v                                  │
┌────────────────┐                      │
│ Collect Acts   │ <────────────────────┘
└────┬───────────┘
     │
     v
┌────────────────┐      Error: Dim mismatch
│ Quantize       │ ─────────────────────┐
└────┬───────────┘                      │
     │                                  v
     v                          ┌───────────────┐
┌────────────────┐              │ Ignore layer  │
│ Pack Weights   │              └───────┬───────┘
└────┬───────────┘                      │
     │                                  │
     v                                  │
┌────────────────┐                      │
│ Save Model     │ <────────────────────┘
└────┬───────────┘
     │
     v
┌────────────────┐      PPL too high
│ Evaluate       │ ─────────────────────┐
└────┬───────────┘                      │
     │                                  v
     v                          ┌───────────────┐
┌────────────────┐              │ Retry with    │
│ Benchmark      │              │ better config │
└────┬───────────┘              └───────┬───────┘
     │                                  │
     v                                  │
┌────────────────┐                      │
│   DEPLOY       │ <────────────────────┘
└────────────────┘
```

## 🎯 Performance Characteristics

### Time Complexity

```
Quantization (per layer):
- Activation collection: O(m × n)  where m=samples, n=in_features
- Alpha search: O(|A| × g × o)  where |A|=7 alphas, g=groups, o=out_features
- Packing: O(o × i)  where i=in_features

Total: O(m×n + |A|×g×o + o×i)
For 4096×4096 layer with 512 samples:
≈ 512×4096 + 7×32×4096 + 4096×4096
≈ 2M + 900K + 16M ≈ 19M operations

Inference (per forward pass):
- Unpack: O(o × i)
- Dequantize: O(o × i)
- MatMul: O(b × s × o × i)  where b=batch, s=seq_len

Total: O(b × s × o × i)
Dominated by matmul, same as FP16
```

### Space Complexity

```
Quantization:
- Activations: O(m × n)
- Quantized weights: O(o × i / 4)
- Scales: O(o × i / g)

Runtime:
- Packed weights: O(o × i / 4)  [permanent]
- Unpacked weights: O(o × i)  [temporary, during forward]
- Activations: O(b × s × max(i, o))

Memory savings: 8x for weights, but activations still FP16
```

---

**Architecture designed for**: Production LLM deployment with extreme memory constraints  
**Optimized for**: 8x compression with acceptable 10-20% accuracy trade-off  
**Target hardware**: NVIDIA GPUs with CUDA compute capability ≥ 7.0

