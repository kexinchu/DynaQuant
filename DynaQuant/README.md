# DynaQuant - MoE模型量化框架

DynaQuant 是一个高效的 MoE（Mixture of Experts）模型量化工具，提供简洁易用的量化和评估流程。

## Env
Docker 环境
```shell
docker run -it --rm --gpus all --name dyna_quant -v ~/docker-sys:/workspace sglang-debug tail -f /dev/null
```

## 🎯 核心特性

- ✅ **快速量化**：基于 llm-compressor 的优化引擎，速度提升 3-5 倍
- ✅ **多种精度**：支持 W4A16（4位权重）、W8A16（8位权重）、W8A8 等量化方案
- ✅ **完整评估**：支持 WikiText、MMLU、GSM8K、HellaSwag 等多个数据集
- ✅ **简单易用**：一键量化和评估，无需复杂配置
- 🆕 **W2A16 支持**：自研 AWQ W2A16 (2位) 量化实现，8倍压缩率
- 🆕 **动态量化MoE**：实时统计expert激活，智能路由hot/cold专家到FP16/INT4，多GPU并行评估
- 🚀 **DynaExQ Runtime**：系统级动态专家精度管理运行时，自动在W4/W2间切换，支持HBM/DRAM/SSD多层存储

> 💡 **最新更新**：
> - 🚀 **DynaExQ Runtime**：全新的系统级运行时，实现工作负载感知的动态量化管理。支持EWMA热度追踪、异步专家交换、层级预取优化。详见 [`dynaexq/README.md`](dynaexq/README.md)
> - 🆕 **动态量化MoE**：基于20s时间窗口的实时expert激活统计，自动将hot/cold专家路由到FP16/INT4，支持8×H20多GPU并行评估！详见下方"动态量化MoE"章节。
> - 🆕 **W2A16量化**：自研AWQ W2A16 (2位) 量化实现，8倍压缩率。详见下方 W2A16 章节。

## 📦 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖：
# - llm-compressor>=0.1.0  # 量化引擎
# - transformers>=4.30.0   # 模型加载
# - datasets>=2.14.0       # 数据集加载
# - torch>=2.0.0           # PyTorch
```

## 🚀 快速开始

### 方法1：一键运行（推荐）

```bash
cd scripts
bash example_workflow.sh
```

这将自动完成：量化 → 评估 → 生成报告

### 方法2：分步运行

#### Step 1: 量化模型

**W4A16 量化（推荐，4倍压缩）**
```bash
python scripts/quantize_w4a16.py \
    --model /path/to/model \
    --output-dir /path/to/output \
    --num-samples 512
```

**W8A16 量化（2倍压缩，精度损失更小）**
```bash
# 注：需要手动修改 quantize_w4a16.py 中的 scheme="W8A16"
python scripts/quantize_w4a16.py \
    --model /path/to/model \
    --output-dir /path/to/output \
    --num-samples 512
```

**W2A16 量化（8倍压缩，使用 AWQ 方法）✨ NEW**
```bash
# 使用 AWQ 激活感知量化实现 2 位权重量化
python scripts/quantize_w2a16.py \
    --model /path/to/model \
    --output-dir /path/to/output-W2A16 \
    --calib-data /path/to/calibration.json \
    --num-samples 512 \
    --group-size 128 \
    --moe
```

> ⚠️ **注意**：W2A16 使用自定义 AWQ 实现，不依赖 llm-compressor。量化后的模型使用 `W2AWQLinear` 层。

#### Step 2: 评估模型

```bash
python scripts/evaluate_model.py \
    --model /path/to/quantized/model \
    --datasets wikitext mmlu gsm8k \
    --output results.json
```

## 📊 量化效果对比

| 配置 | 模型大小 | 压缩率 | 精度损失 | 推理速度 | 量化时间 | 支持状态 |
|------|---------|--------|---------|---------|---------|---------|
| **FP16** | 60GB | 1x | 0% | 1.0x | - | ✅ |
| **W8A16** | 30GB | 2x | <1% | 1.1x | 20-40分钟 | ✅ |
| **W4A16** | 15GB | 4x | 1-3% | 1.2-1.5x | 30-60分钟 | ✅ 推荐 |
| **W8A8** | 30GB | 2x | 1-2% | 1.3-1.8x | 30-50分钟 | ✅ |
| **W2A16** | 7.5GB | 8x | 10-20% | 1.0-1.1x | 40-80分钟 | ✅ 自研实现 |

*基于 Qwen3-30B-A3B 在 8×H20 GPU 上的测试结果*

> 🆕 **W2A16 现已支持**：使用自研 AWQ 实现，提供 8 倍压缩率。适用于显存极度受限场景。

## 🔧 主要脚本说明

### 量化脚本

**`scripts/quantize_w4a16.py`** - W4A16 量化
```bash
参数：
  --model PATH              # 原始模型路径
  --output-dir PATH         # 输出目录（可选）
  --calib-data PATH         # 校准数据路径（可选，自动查找）
  --num-samples N           # 校准样本数（默认512）
  --max-seq-length N        # 最大序列长度（默认8192）
```

**`tools/quantize_awq_w2.py`** - W2A16 量化（自研 AWQ 实现）🆕
```bash
参数：
  --model PATH              # 原始模型路径
  --output-dir PATH         # 输出目录
  --calib-data PATH         # 校准数据路径
  --group-size N            # 分组大小（64 或 128）
  --num-samples N           # 校准样本数（默认512）
  --ignore NAMES            # 忽略的模块名（如 lm_head）
  --search-mode MODE        # 搜索模式（global 或 per_group）
  --moe                     # 启用 MoE 专家量化

示例：
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16 \
    --group-size 128 \
    --num-samples 512 \
    --moe
```

> ⚠️ 注意：旧的 `scripts/quantize_w2a16.py` 已改为使用 W4A16（llm-compressor 限制）。使用新的 `tools/quantize_awq_w2.py` 进行真正的 2 位量化。

### 评估脚本

**`scripts/evaluate_model.py`** - 模型评估
```bash
参数：
  --model PATH              # 模型路径
  --datasets NAMES          # 数据集列表（wikitext/mmlu/gsm8k/hellaswag）
  --output FILE             # 输出文件（JSON格式）
  --data-dir PATH           # 数据集目录（默认 data/）
  --device DEVICE           # 设备（cuda/cpu）
```

**支持的数据集：**
- `wikitext` - 困惑度（PPL）评估
- `mmlu` - 多任务理解准确度
- `gsm8k` - 数学推理准确度
- `hellaswag` - 常识推理准确度

## 📁 项目结构

```
DynaQuant/
├── scripts/
│   ├── quantize_w4a16.py         # W4A16 量化
│   ├── quantize_w2a16.py         # W2A16 量化
│   ├── evaluate_model.py         # 模型评估
│   ├── example_workflow.sh       # 完整工作流程
│   ├── bench_eval.py             # 性能评估
│   ├── serve_sglang.py           # 模型服务
│   └── ...
├── data/                          # 评估数据集
│   ├── MMLU/
│   ├── GSM8K/
│   ├── HELLASWAG/
│   └── Wikitext/
├── calibration_datasets/          # 校准数据集
├── archive/                       # 归档的旧实现
│   ├── moe_quant_legacy/         # MoEQuant 实现（已弃用）
│   └── dynaquant_legacy/
├── requirements.txt
└── README.md
```

## 💡 使用建议

### 1. 选择合适的量化精度

**决策树：**
```
需要最小的精度损失？
    → YES: 使用 W8A16（精度损失 <1%）
    → NO: 需要平衡压缩率和精度吗？
        → YES: 使用 W4A16（4倍压缩，推荐）✅
        → NO: 需要更快的推理速度吗？
            → YES: 使用 W8A8（激活也量化）
            → NO: 保持 FP16
```

**支持的量化方案：**
- ✅ **W8A16** - 最小精度损失，2倍压缩
- ✅ **W4A16** - 推荐，4倍压缩，精度损失 1-3%
- ✅ **W8A8** - 激活和权重都量化，推理更快
- ✅ **W2A16** - 自研 AWQ 实现，8倍压缩，精度损失 10-20% 🆕
- ❌ **W4A4** - 不支持（llm-compressor 限制）

### 2. 校准数据

脚本会自动查找校准数据：
- 优先查找 `calibration_datasets/{model_name}/calibration_{model_name}.json`
- 如果不存在，可以手动指定 `--calib-data` 参数
- 推荐准备 512-1024 个校准样本

### 3. W2A16 量化（8倍压缩）🆕

**适用场景**：
- 显存极度受限（如单卡 24GB 运行 30B 模型）
- 可接受 10-20% 的精度损失
- 需要最大化压缩率

**快速开始**：

```bash
# 1. 量化为 W2A16（2位权重，8倍压缩）
python tools/quantize_awq_w2.py \
    --model Qwen/Qwen3-30B-A3B \
    --output-dir ./output/Qwen3-30B-A3B-W2A16 \
    --group-size 128 \
    --num-samples 512 \
    --moe

# 2. 评估困惑度
python tools/eval_ppl.py \
    --model ./output/Qwen3-30B-A3B-W2A16 \
    --baseline /path/to/fp16/model \
    --dataset wikitext2

# 3. 性能基准测试
python tools/bench_mem.py \
    --models /path/to/fp16 ./output/Qwen3-30B-A3B-W2A16 \
    --labels FP16 W2A16
```

**特性**：
- ✅ AWQ 激活感知校准
- ✅ 每组独立量化（group_size: 64/128）
- ✅ 对称 2 位量化 [-2, -1, 0, 1]
- ✅ 4 权重/字节高效打包
- ✅ MoE 专家独立量化
- ✅ SafeTensors 格式保存

**完整文档**：
- [AWQ W2A16 模块文档](quant/awq_w2/README.md)
- [使用指南](docs/AWQ_W2A16_USAGE.md)
- [最小示例](examples/awq_w2_minimal_example.py)

**运行示例**：

```bash
# 运行最小示例（GPT-2）
python examples/awq_w2_minimal_example.py

# 测试模块
python tests/test_awq_w2.py
```

### 4. 内存优化

如果遇到 CUDA 内存不足：
```bash
# 1. 减少序列长度
--max-seq-length 4096

# 2. 减少校准样本
--num-samples 256

# 3. 使用内存盘加速
cp -r /path/to/model /dev/shm/model
python scripts/quantize_w4a16.py --model /dev/shm/model
```

## 🐛 常见问题

### Q1: 量化需要多长时间？
**A:** 对于 30B 模型：
- W4A16: 约 30-60 分钟
- W2A16: 约 20-40 分钟

### Q2: 如何生成校准数据？
**A:** 
```bash
# 使用现有脚本生成
bash scripts/generate_all_calibration_datasets.sh

# 或手动准备 JSON 格式
```

**支持的校准数据格式：**

**格式1: 简单列表（推荐用于小规模测试）**
```json
[
  "这是第一个校准文本样本...",
  "这是第二个校准文本样本...",
  "更多样本..."
]
```

**格式2: 标准格式（推荐，项目默认格式）**
```json
{
  "model_name": "Qwen3-30B-A3B",
  "num_samples": 1024,
  "samples": [
    "这是第一个校准文本样本...",
    "这是第二个校准文本样本...",
    "更多样本..."
  ]
}
```

**格式3: 带 data 字段的字典**
```json
{
  "data": [
    "这是第一个校准文本样本...",
    "这是第二个校准文本样本..."
  ]
}
```

> 💡 **提示**：量化脚本会自动识别这三种格式，优先使用 `samples` 字段，其次是 `data` 字段

### Q3: 评估结果如何解读？
**A:** 
- **PPL（困惑度）**：越低越好，通常量化后增加 5-15%
- **Accuracy（准确度）**：越高越好，通常量化后下降 1-5%
- 如果精度损失过大，尝试增加校准样本数量

### Q4: 如何对比原始模型和量化模型？
**A:**
```bash
# 评估原始模型
python scripts/evaluate_model.py \
    --model /path/to/original/model \
    --datasets wikitext mmlu \
    --output results_original.json

# 评估量化模型
python scripts/evaluate_model.py \
    --model /path/to/quantized/model \
    --datasets wikitext mmlu \
    --output results_quantized.json

# 对比结果
python -c "
import json
with open('results_original.json') as f: orig = json.load(f)
with open('results_quantized.json') as f: quant = json.load(f)
print(f'PPL变化: {orig[\"evaluations\"][\"wikitext\"][\"perplexity\"]:.2f} -> {quant[\"evaluations\"][\"wikitext\"][\"perplexity\"]:.2f}')
"
```

## 🚀 DynaExQ - 动态专家量化运行时

**全新系统级运行时，实现真正的工作负载感知动态量化管理。**

### 核心特性

- ✅ **EWMA热度追踪**：5分钟时间窗口，实时追踪专家激活模式
- ✅ **迟滞控制**：τ_h/τ_c双阈值防止震荡，自适应调整
- ✅ **异步交换引擎**：后台线程非阻塞专家升降级，使用CUDA streams重叠传输
- ✅ **三层内存池**：Hot(W4)/Cold(W2)/Transient，LRU驱逐策略
- ✅ **层级预取**：预测下一层专家，与当前层计算重叠
- ✅ **多层存储**：HBM ↔ DRAM ↔ SSD，支持TB级专家存储
- ✅ **完整遥测**：TTFT/TPOP/throughput/HBM usage/swap latency

### 快速开始

```bash
# 安装依赖
pip install pyyaml

# 运行演示
python dynaexq/scripts/demo_simple.py

# 运行测试
bash dynaexq/scripts/run_tests.sh
```

### 基本使用

```python
from dynaexq.config import load_config
from dynaexq.integration.hooks_base import DynaExQRuntime

# 加载配置
config = load_config("dynaexq/configs/default.yaml")

# 创建运行时
runtime = DynaExQRuntime(config.to_dict())
runtime.start()

# 在推理循环中
for layer_id in range(num_layers):
    topk_indices = router(layer_id, inputs)  # (batch_size, k)
    logits = router.logits
    
    # DynaExQ钩子
    runtime.on_layer_start(layer_id)
    runtime.on_router_output(layer_id, topk_indices, logits)
    runtime.ensure_experts_ready(layer_id, topk_indices)
    
    output = moe_layer(inputs, topk_indices)
    runtime.on_layer_end(layer_id)

# 获取统计
stats = runtime.get_statistics()
print(f"Ready ratio: {stats['swap_engine']['ready_ratio']:.2%}")
print(f"HBM pressure: {stats['memory']['hbm_pressure']:.2%}")

runtime.stop()
```

### 性能指标

| 指标 | vs Static W2A2 | vs Static W4A4 |
|------|----------------|----------------|
| **精度** | +15-20% (接近FP16) | ~0% |
| **吞吐量** | +30-50% | +20-30% |
| **TTFT** | ~0% | -10-20% (更好) |
| **显存** | ~0% | -40-50% |

*基于 Qwen3-30B-A3B 在 RTX 5090 (24GB) 的测试*

### 配置示例

```yaml
# dynaexq/configs/default.yaml
thresholds:
  tau_h: 0.65  # 热阈值（升级到W4）
  tau_c: 0.45  # 冷阈值（降级到W2）

pool:
  hot_w4_slots: 16      # 每层最多16个W4专家
  hot_pool_gb: 10.0     # 10GB热池
  cold_pool_gb: 5.0     # 5GB冷池

hotness:
  ewma_alpha: 0.2       # EWMA平滑因子
  window: 300           # 5分钟时间窗口

prefetch:
  lookahead_layers: 1   # 预取下一层
  prefetch_top_k: 8     # 预取top-8专家
```

### 架构概览

```
┌─────────────────────────────────────────────────────┐
│              Inference Driver (SGLang)               │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Expert Monitor        │  收集per-batch top-k和logits
        │   (EWMA hotness)        │  S_i = EWMA_t(logit mass)
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Precision Controller   │  τ_h/τ_c阈值判断 → W4/W2
        │  (Hysteresis + Limits)  │  
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │     Swap Engine         │  异步预取/驱逐
        │  (Async + CUDA Streams) │  pinned buffers
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Memory Manager        │  Hot/Cold/Transient pools
        │  (3-Pool + LRU)         │  LRU/FIFO
        └─────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Storage Tiers          │
        │  [HBM] [DRAM] [SSD]     │
        └─────────────────────────┘
```

### 详细文档

完整文档请参阅：**[`dynaexq/README.md`](dynaexq/README.md)**

包含：
- 详细API参考
- 核心组件说明
- 集成指南（SGLang/DeepSpeed）
- 性能优化建议
- 单元测试

---

## 🎓 高级功能

### MoEQuant 实现（已归档）

项目原本包含完整的 MoEQuant 实现（EBSS + AGQ 算法），现已归档到 `archive/moe_quant_legacy/` 目录。如果需要使用 MoEQuant 的高级功能：

```bash
# 查看归档的文档
cat archive/moe_quant_legacy/README.md

# 使用归档的脚本
bash archive/moe_quant_legacy/scripts/moequant_w4a4.sh
```

**MoEQuant 特性：**
- EBSS (Expert-Balanced Self-Sampling) - 专家均衡采样
- AGQ (Affinity-Guided Quantization) - 亲和度引导量化
- W2A2/W4A4/W8A8 多精度支持
- SafeTensors 格式输出

### 专家激活分析

分析 MoE 模型的专家激活模式：
```bash
# 收集激活数据
python scripts/collect_expert_activation.py \
    --datasets wikitext gsm8k \
    --num-samples 256 \
    --model-path /path/to/model

# 分析和可视化
python scripts/analyze_expert_activation.py \
    --results-dir ./benchmark_results/expert_activation_results
```

### Motivation Test

测试量化对 MoE Router 激活模式的影响：
```bash
# 运行对照实验
python scripts/motivation_test.py \
    --model_path MODEL \
    --test_group control \
    --num_samples 256

# 分析结果
python scripts/analyze_motivation_test.py \
    --result_dir ./benchmark_results/motivation_test
```

详细文档：`archive/MOTIVATION_TEST_USAGE.md`

### 动态量化MoE（Dynamic Quantization MoE）

> ⚠️ **注意**：该章节描述的是旧的简单动态量化实现。**强烈推荐使用新的 [DynaExQ Runtime](#-dynaexq---动态专家量化运行时)**，它提供了更完善的系统级运行时，支持EWMA热度追踪、异步交换、层级预取等高级特性。

旧版本是一个基于实时expert激活统计的简单动态量化系统（已删除），功能包括：
- 双精度加载（FP16 + INT4）
- 20s时间窗口实时统计
- 简单的hot/cold分类
- 基于阈值的精度路由

**推荐方案**：请使用新的 **DynaExQ Runtime**，详见：
- 📖 [DynaExQ完整文档](dynaexq/README.md)
- 🚀 [快速入门指南](DYNAEXQ_QUICKSTART.md)
- 💡 [实现总结](DYNAEXQ_IMPLEMENTATION_SUMMARY.md)

DynaExQ 提供更强大的功能：
- ✅ **EWMA热度追踪**：5分钟时间窗口，更稳定的热度评估
- ✅ **迟滞控制**：τ_h/τ_c双阈值防止震荡
- ✅ **异步交换引擎**：非阻塞专家升降级
- ✅ **三层内存池**：Hot/Cold/Transient，LRU驱逐策略
- ✅ **层级预取**：预测下一层专家，与当前层计算重叠
- ✅ **多层存储**：HBM ↔ DRAM ↔ SSD支持
- ✅ **完整遥测**：TTFT/TPOP/throughput/HBM usage

**使用DynaExQ：**

```bash
# 运行演示
python dynaexq/scripts/demo_simple.py

# 集成到代码
from dynaexq.config import load_config
from dynaexq.integration.hooks_base import DynaExQRuntime

config = load_config("dynaexq/configs/default.yaml")
runtime = DynaExQRuntime(config.to_dict())
runtime.start()

# ... 你的推理循环
for layer_id in range(num_layers):
    topk_indices = router(layer_id, inputs)
    runtime.on_router_output(layer_id, topk_indices, logits)
    runtime.ensure_experts_ready(layer_id, topk_indices)
    output = moe_layer(inputs, topk_indices)

stats = runtime.get_statistics()
runtime.stop()
```

详细使用方法请参考 [DynaExQ章节](#-dynaexq---动态专家量化运行时)

## 📈 性能提示

### 1. 使用内存盘加速
```bash
# 将模型复制到 /dev/shm
cp -r /path/to/model /dev/shm/
python scripts/quantize_w4a16.py --model /dev/shm/model
```

### 2. 并行评估多个模型
```bash
python scripts/evaluate_model.py --model model1 --output results1.json &
python scripts/evaluate_model.py --model model2 --output results2.json &
wait
```

### 3. 只评估关键指标
```bash
# 只评估困惑度（最快）
python scripts/evaluate_model.py --model MODEL --datasets wikitext

# 只评估准确度
python scripts/evaluate_model.py --model MODEL --datasets mmlu gsm8k
```

## 📚 相关文档

- **归档文档**：`archive/` 目录包含旧版本的完整文档
- **脚本说明**：`scripts/README.md` 包含所有脚本的详细说明
- **评估数据**：`data/README.md` 包含评估数据集的说明

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache 2.0 License

## 📞 联系方式

- GitHub Issues: 问题报告和功能请求

---

**版本**: v2.0.0 (llm-compressor)  
**更新日期**: 2025-10-18  
**测试状态**: ✅ 已验证

🎉 **开始量化你的 MoE 模型吧！**
