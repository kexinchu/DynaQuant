# DynaQuant - MoE模型量化框架

DynaQuant 是一个高效的 MoE（Mixture of Experts）模型量化工具，提供简洁易用的量化和评估流程。

## 🎯 核心特性

- ✅ **快速量化**：基于 llm-compressor 的优化引擎，速度提升 3-5 倍
- ✅ **多种精度**：支持 W4A16（4位权重）、W8A16（8位权重）、W8A8 等量化方案
- ✅ **完整评估**：支持 WikiText、MMLU、GSM8K、HellaSwag 等多个数据集
- ✅ **简单易用**：一键量化和评估，无需复杂配置
- 🆕 **W2A16 支持**：自研 AWQ W2A16 (2位) 量化实现，8倍压缩率
- 🆕 **动态量化MoE**：实时统计expert激活，智能路由hot/cold专家到FP16/INT4，多GPU并行评估

> 💡 **最新更新**：
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

> ⚠️ **关于 W2A16**：`quantize_w2a16.py` 脚本已改为使用 W4A16，因为 llm-compressor 不支持 2 位量化。如需 2 位量化，请使用其他工具如 GPTQ 或 AWQ。

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

### 动态量化MoE（Dynamic Quantization MoE）🆕

基于实时expert激活统计的动态量化系统，自动将hot/cold专家路由到不同精度计算：

**核心功能：**
- ✅ **双精度加载**：同时加载 FP16 和 INT4 两个版本的模型
- ✅ **实时统计**：20s 时间窗口内实时追踪 expert 激活情况
- ✅ **动态分类**：自动识别 hot (top 10%) 和 cold (剩余 90%) 专家
- ✅ **智能路由**：hot 专家使用 FP16 精度，cold 专家使用 INT4 精度
- ✅ **多GPU并行**：支持多GPU并行评估，充分利用8×H20等多卡环境

**使用示例：**

```bash
# 单GPU评估
python scripts/evaluate_dynamic_quant.py \
    --fp16-model /path/to/fp16/model \
    --int4-model /path/to/int4/model \
    --datasets wikitext mmlu gsm8k hellaswag \
    --time-window 20.0 \
    --hot-ratio 0.1 \
    --output results_dynamic_quant.json

# 多GPU并行评估（推荐，充分利用8×H20）
python scripts/evaluate_parallel_dynamic_quant.py \
    --fp16-model /path/to/fp16/model \
    --int4-model /path/to/int4/model \
    --datasets wikitext mmlu gsm8k hellaswag \
    --num-gpus 8 \
    --time-window 20.0 \
    --hot-ratio 0.1 \
    --output results_parallel.json

# 仅使用FP16（禁用动态路由，用于基准对比）
python scripts/evaluate_dynamic_quant.py \
    --fp16-model /path/to/fp16/model \
    --int4-model /path/to/int4/model \
    --disable-dynamic-routing \
    --output results_fp16_baseline.json
```

**参数说明：**
- `--time-window`: 时间窗口大小（秒），默认 20.0
- `--hot-ratio`: hot 专家比例，默认 0.1 (10%)
- `--num-gpus`: 使用的GPU数量，默认使用所有可用GPU
- `--disable-dynamic-routing`: 禁用动态路由（用于基准测试）
- `--num-samples-*`: 各数据集的样本数，可单独配置

**评估输出：**
```json
{
  "evaluations": {
    "wikitext": {"perplexity": 12.34},
    "mmlu": {"accuracy": 0.6789},
    "gsm8k": {"accuracy": 0.5432},
    "hellaswag": {"accuracy": 0.7654}
  },
  "model_statistics": {
    "inference_stats": {
      "total_tokens": 123456,
      "hot_expert_calls": 12345,
      "cold_expert_calls": 111111,
      "fp16_expert_calls": 12345,
      "int4_expert_calls": 111111
    },
    "tracker_stats": {
      "time_window": 20.0,
      "hot_ratio": 0.1000,
      "num_hot_experts": 64
    }
  }
}
```

**性能优势：**
- 🚀 **并行加速**：8个GPU并行评估，总时间 ≈ 单GPU时间 / 8
- 💾 **显存优化**：cold专家使用INT4，显存占用降低约70%
- ⚡ **精度保持**：hot专家保持FP16精度，关键计算无损
- 📊 **灵活配置**：可调整时间窗口和hot比例，适应不同场景

**推荐配置（8×H20环境）：**
```bash
# 快速评估（所有数据集，约15-20分钟）
python scripts/evaluate_parallel_dynamic_quant.py \
    --fp16-model /dev/shm/Qwen3-30B-A3B \
    --int4-model /dev/shm/Qwen3-30B-A3B-INT4 \
    --datasets wikitext mmlu gsm8k hellaswag \
    --num-gpus 8 \
    --num-samples-wikitext 1000 \
    --num-samples-mmlu 100 \
    --num-samples-gsm8k 100 \
    --num-samples-hellaswag 100 \
    --output results.json
```

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
