# MoE-Quant: W2A2 Quantization with EBSS and AGQ

完整的MoE模型极低比特量化方案，实现2-bit权重和2-bit激活（W2A2）量化，并包含：

- **EBSS** (Expert-Balanced Self-Sampling): 专家均衡自回采样
- **AGQ** (Affinity-Guided Quantization): 门控感知量化
- **W2A2 Quantizer**: 2-bit权重和激活量化器（含分布整形）
- **Enhanced Router Guard**: 增强路由守护（高精度累加 + 一致性检测）
- **PTQ + QAT**: 完整的训练后量化和量化感知训练流程

## 特性

### 1. EBSS - 专家均衡采样
- Beam搜索自回采样，确保生成的校准数据覆盖所有专家
- 评分函数：`score = perplexity + (expert_balance / τ)`
- 自动平衡冷热专家，避免校准偏差

### 2. AGQ - 门控感知量化
- 基于token-expert亲和度的加权量化
- 损失函数：`L = Σ c_i ||W x_i - W_hat x_i||^2`
- 近似Hessian：`H = (X * sqrt(c)) (X * sqrt(c))^T`
- 列级逐步量化 + 误差补偿（GPTQ风格）

### 3. W2A2量化器
- **权重量化**: 逐通道/分组2-bit对称量化
- **激活量化**: 前置分布整形（正交旋转/白化）→ 2-bit量化
- **渐进回退**: 通道级从A2→A3→A4（仅对引发topk翻转的热点）
- 误差回填和补偿机制

### 4. 增强路由守护
- 高精度路由计算：INT8输入 + INT32累加 或 FP16路径
- Fused softmax + top-k（确定性tie-break）
- 在线一致性检测：统计topk_match_rate
- 自适应回退：低于阈值时临时升比特

### 5. QAT微调
- 仅对路由前后2-3个线性层插入fake-quant
- 联合损失：`L = L_task + λ·L_consistency + μ·L_margin`
- 可选冻结专家，仅调整路由相邻层
- 1-3 epoch低学习率微调

## 安装

```bash
cd /root/code/DynaQuant/DynaQuant
pip install -r requirements.txt

# 确保安装了以下依赖
pip install torch transformers datasets tqdm numpy
```

## 快速开始

### 1. PTQ（训练后量化）

```bash
# 方式1：使用Shell脚本（推荐）
bash scripts/run_ptq_moe.sh \
  --model /dev/shm/Qwen3-30B-A3B \
  --calib-size 128 \
  --ebss-beam 4 \
  --ebss-tau 1.2 \
  --bit-w 2 \
  --bit-a 2 \
  --group-size 64 \
  --router-mode fp16 \
  --output-dir /dev/shm/Qwen3-30B-A3B-W2A2

# 方式2：直接使用Python
python3 -m moe_quant.runners.run_ptq \
  --model /dev/shm/Qwen3-30B-A3B/ \
  --calib-size 128 \
  --seed-text data/seed_text.txt \
  --output-dir ./output/ptq_qwen_moe
```

**参数说明**：
- `--model`: 模型名称或路径
- `--calib-size`: 校准集大小
- `--ebss-beam-width`: EBSS beam宽度
- `--ebss-tau`: EBSS温度参数
- `--bit-w`: 权重比特数（2, 4, 8）
- `--bit-a`: 激活比特数（2, 4, 8）
- `--group-size`: 分组大小（64, 128）
- `--use-rotation`: 是否使用激活旋转（0/1）
- `--enable-fallback`: 是否启用渐进回退（0/1）
- `--router-mode`: 路由模式（fp16/int8_acc32）
- `--strict-topk`: 严格topk一致性（0/1）

### 2. QAT（量化感知训练）

```bash
# 方式1：使用Shell脚本（推荐）
bash scripts/run_qat_moe.sh \
  --model Qwen/Qwen-MoE-14B \
  --load-ptq ./output/ptq_qwen_moe/quantized_model.pt \
  --epochs 2 \
  --lr 5e-6 \
  --lambda-topk 1.0 \
  --mu-margin 0.2 \
  --output-dir ./output/qat_qwen_moe

# 方式2：直接使用Python
python3 -m moe_quant.qat.run_qat \
  --model Qwen/Qwen-MoE-14B \
  --load-ptq ./output/ptq_qwen_moe/quantized_model.pt \
  --epochs 2 \
  --lr 5e-6 \
  --output-dir ./output/qat_qwen_moe
```

**参数说明**：
- `--load-ptq`: PTQ checkpoint路径
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--batch-size`: 批大小
- `--grad-accum`: 梯度累积步数
- `--lambda-topk`: topk一致性损失权重
- `--mu-margin`: margin损失权重
- `--freeze-experts`: 是否冻结专家（0/1）
- `--train-router-adjacent-only`: 仅训练路由相邻层（0/1）

### 3. 评测

```python
from moe_quant.runners.eval_metrics import create_evaluator

# 创建评估器（需要量化模型和参考模型）
evaluator = create_evaluator(
    model_name="path/to/quantized/model",
    reference_model_name="Qwen/Qwen-MoE-14B"
)

# 准备测试数据
test_texts = ["Your test samples here..."]

# 运行完整评测
metrics = evaluator.evaluate_full(
    test_texts=test_texts,
    output_path="./output/eval_results.json"
)

print(f"Perplexity: {metrics.perplexity:.2f}")
print(f"Top-k Match Rate: {metrics.overall_topk_match_rate:.2%}")
print(f"Latency: {metrics.latency_ms:.2f} ms")
```

## 完整示例

以下是一个完整的PTQ → QAT → 评测流程示例：

```bash
# 1. 生成EBSS校准集并执行PTQ
python3 -m moe_quant.runners.run_ptq \
  --model Qwen/Qwen-MoE-14B \
  --calib-size 256 \
  --ebss-beam-width 4 \
  --ebss-tau 1.2 \
  --bit-w 2 \
  --bit-a 2 \
  --group-size 64 \
  --use-rotation 1 \
  --enable-fallback 1 \
  --router-mode fp16 \
  --strict-topk 1 \
  --seed-text data/seed_text.txt \
  --output-dir ./output/ptq_step1

# 2. QAT微调（可选）
python3 -m moe_quant.qat.run_qat \
  --model Qwen/Qwen-MoE-14B \
  --load-ptq ./output/ptq_step1/quantized_model.pt \
  --epochs 2 \
  --lr 5e-6 \
  --batch-size 1 \
  --grad-accum 8 \
  --lambda-topk 1.0 \
  --mu-margin 0.2 \
  --train-data data/train_samples.txt \
  --output-dir ./output/qat_step2

# 3. 评测（Python脚本）
python3 -c "
from moe_quant.runners.eval_metrics import create_evaluator

evaluator = create_evaluator(
    model_name='./output/qat_step2',
    reference_model_name='Qwen/Qwen-MoE-14B'
)

with open('data/test_samples.txt', 'r') as f:
    test_texts = [line.strip() for line in f if line.strip()]

metrics = evaluator.evaluate_full(
    test_texts=test_texts,
    output_path='./output/eval_results.json'
)

print(f'Results saved to ./output/eval_results.json')
"
```

## 模块说明

### 核心量化模块 (`moe_quant/quant/`)

- **`ebss.py`**: EBSS采样器实现
- **`agq.py`**: AGQ量化器实现
- **`quantizers.py`**: W2A2量化器实现（含激活整形和回退）
- **`router_guard_enhanced.py`**: 增强路由守护

### 模型与数据 (`moe_quant/models/`, `moe_quant/runners/`)

- **`load_moe.py`**: MoE模型统一加载器
- **`collect_calib.py`**: 校准数据收集器
- **`ptq_runner.py`**: PTQ完整流程运行器
- **`eval_metrics.py`**: 评测指标计算器

### 训练模块 (`moe_quant/qat/`)

- **`train_qat.py`**: QAT训练器
- **`run_qat.py`**: QAT命令行入口

### Loss函数 (`moe_quant/losses/`)

- **`routing_losses.py`**: 路由损失函数（一致性、margin、多样性）

## API使用示例

### 1. 使用EBSS生成校准数据

```python
from moe_quant.quant.ebss import create_ebss_sampler
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-MoE-14B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-MoE-14B")

sampler = create_ebss_sampler(
    model=model,
    tokenizer=tokenizer,
    beam_width=4,
    tau=1.2,
    max_tokens=512
)

seed_texts = ["Your seed texts here..."]
ebss_samples = sampler.generate(seed_texts)
```

### 2. 使用AGQ量化线性层

```python
from moe_quant.quant.agq import create_agq_quantizer
import torch.nn as nn

quantizer = create_agq_quantizer(bit_width=2, group_size=64)

# 假设已收集激活和亲和度
layer = nn.Linear(4096, 4096)
inputs = torch.randn(128, 512, 4096)  # [batch, seq, hidden]
affinities = torch.rand(128, 512)      # [batch, seq]

W_quant, scales, stats = quantizer.quantize_linear(
    layer, inputs, affinities
)

print(f"Quantization MSE: {stats['mse']:.6f}")
print(f"Weighted MSE: {stats['weighted_mse']:.6f}")
```

### 3. 使用W2A2量化器

```python
from moe_quant.quant.quantizers import create_w2a2_quantizer

quantizer = create_w2a2_quantizer(
    use_rotation=True,
    use_whitening=True,
    enable_fallback=True
)

# 量化线性层
W_quant, W_absorbed, stats = quantizer.quantize_linear_layer(
    layer, X_calib, layer_id=0
)

print(f"W2A2 MSE: {stats['mse']:.6f}")
print(f"Relative Error: {stats['relative_error']:.4f}")
```

## 输出说明

### PTQ输出 (`output/ptq_*/`)

```
ptq_output/
├── ebss_samples.txt              # EBSS生成的样本
├── calibration_data.pkl          # 校准数据（激活+亲和度）
├── quantization_stats.json       # 量化统计信息
├── router_stats.json             # 路由一致性统计
├── ptq_results.json              # 完整PTQ结果
└── quantized_model.pt            # 量化模型权重
```

### QAT输出 (`output/qat_*/`)

```
qat_output/
├── checkpoint_epoch1.pt          # 训练checkpoint
├── checkpoint_epoch2.pt
├── qat_results.json              # QAT训练结果
└── final_model.pt                # 最终模型
```

### 评测输出

```json
{
  "overall_topk_match_rate": 0.95,
  "per_layer_topk_match_rate": {
    "0": 0.96,
    "1": 0.94,
    ...
  },
  "perplexity": 12.34,
  "latency_ms": 45.67,
  "throughput_tokens_per_sec": 123.45,
  "peak_memory_mb": 8192.0
}
```

## 高级配置

### 自定义量化配置

```python
from moe_quant.quant.quantizers import W2A2Config

config = W2A2Config(
    w_bit=2,                    # 权重比特数
    a_bit=2,                    # 激活比特数
    w_group_size=64,            # 权重分组大小
    a_group_size=64,            # 激活分组大小
    w_symmetric=True,           # 权重对称量化
    a_symmetric=False,          # 激活非对称量化
    use_rotation=True,          # 使用激活旋转
    use_whitening=True,         # 使用激活白化
    rotation_granularity="per_layer",
    enable_fallback=True,       # 启用渐进回退
    fallback_threshold=0.05,    # 回退阈值（topk翻转率）
    fallback_bits=[3, 4]        # 回退比特序列
)
```

### 自定义路由守护配置

```python
from moe_quant.quant.router_guard_enhanced import EnhancedRouterConfig

config = EnhancedRouterConfig(
    mode="fp16",                      # 路由模式
    top_k=2,                          # top-k专家数
    strict_topk=True,                 # 严格一致性
    consistency_threshold=0.95,       # 一致性阈值
    deterministic_tiebreak=True,      # 确定性tie-break
    enable_online_detection=True,     # 在线检测
    detection_window_size=100,        # 检测窗口大小
    enable_fallback=True,             # 启用回退
    fallback_on_first_flip=False     # 首次翻转即回退
)
```

## 性能优化建议

1. **校准数据量**: 建议128-512个样本，过多会降低速度
2. **EBSS beam宽度**: 4-8较为合适，更大beam会增加采样时间
3. **分组大小**: 64或128，较小分组精度更高但计算量更大
4. **激活整形**: 旋转+白化效果最好，但增加计算开销
5. **QAT epoch数**: 1-3个epoch足够，过多可能过拟合

## 已知限制

1. 当前实现主要支持Qwen-MoE架构，其他MoE架构可能需要适配
2. W2A2量化对某些层可能不稳定，建议启用fallback
3. 大模型（>100B）可能需要调整batch size和gradient accumulation
4. INT8 router模式尚未完全优化，建议使用FP16模式

## 引用

如果使用本实现，请引用相关论文：

```bibtex
@article{moequant2024,
  title={MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance},
  year={2024}
}

@article{eaquant2024,
  title={EaQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization},
  year={2024}
}
```

## License

MIT License

## 联系

如有问题，请提交GitHub Issue。

