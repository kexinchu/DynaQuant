# Expert Activation Analysis - 使用指南

## 概述

本工具用于分析 Qwen3-30B-A3B MoE 模型在不同任务类型和thinking模式下的专家激活情况。支持从本地parquet文件加载数据，提供灵活的命令行参数进行快速测试。

## 数据收集脚本参数说明

### 数据集选择
- `--datasets [wikitext|gsm8k|humaneval ...]`: 选择要测试的数据集（可多选）
- `--all`: 运行所有数据集的所有实验（6组）

### Thinking模式选择
- `--thinking-only`: 只测试thinking开启的情况
- `--no-thinking`: 只测试thinking关闭的情况
- 不指定：测试两种模式

### 配置参数
- `--model-path`: 模型路径（默认：`/dev/shm/Qwen3-30B-A3B`）
- `--num-samples`: 每个数据集的样本数量（默认：256）
- `--output-dir`: 结果输出目录（默认：`./benchmark_results/expert_activation_results`）

### 数据集路径
- `--wikitext-path`: WikiText数据集路径（默认：`./data/WikiText/arz_test-00000-of-00001.parquet`）
- `--gsm8k-path`: GSM8K数据集路径（默认：`./data/GSMK/test-00000-of-00001.parquet`）
- `--humaneval-path`: HumanEval数据集路径（默认：`./data/HumanEval/test-00000-of-00001.parquet`）

## 结果分析

### 1. 分析所有结果

```bash
python scripts/analyze_expert_activation.py
```

### 2. 指定结果目录

```bash
python scripts/analyze_expert_activation.py --results-dir ./my_results
```

### 3. 只显示摘要，不进行对比

```bash
python scripts/analyze_expert_activation.py --summary-only
```

### 4. 对比两个特定实验

```bash
python scripts/analyze_expert_activation.py \
    --compare wikitext_thinking_off wikitext_thinking_on
```

### 5. 分析特定实验

```bash
python scripts/analyze_expert_activation.py \
    --experiments wikitext_thinking_off gsm8k_thinking_off
```

## 分析脚本参数说明

- `--results-dir`: 结果目录路径（默认：`./benchmark_results/expert_activation_results`）
- `--summary-only`: 只显示统计摘要，不进行对比分析
- `--compare EXP1 EXP2`: 对比两个指定的实验
- `--experiments [EXP1 EXP2 ...]`: 指定要分析的实验名称

## 输出结果

### 目录结构
```
benchmark_results/expert_activation_results/
├── wikitext_thinking_off.json
├── wikitext_thinking_on.json
├── gsm8k_thinking_off.json
├── gsm8k_thinking_on.json
├── humaneval_thinking_off.json
└── humaneval_thinking_on.json
```

### JSON格式
每个JSON文件包含一个列表，代表模型的所有MoE层：
```json
[
  {
    "0": 1234,    // Expert 0 被激活 1234 次
    "1": 987,     // Expert 1 被激活 987 次
    ...
  },
  {
    // 第2层的激活统计
    ...
  },
  ...
]
```

## 使用示例

### 示例1：完整的6组实验
```bash
# 数据收集
python scripts/collect_expert_activation.py --all

# 结果分析
python scripts/analyze_expert_activation.py
```

### 示例2：快速测试工作流
```bash
# 快速测试（32个样本）
python scripts/collect_expert_activation.py \
    --datasets wikitext \
    --num-samples 32 \
    --no-thinking \
    --output-dir ./quick_test

# 查看结果
python scripts/analyze_expert_activation.py \
    --results-dir ./quick_test \
    --summary-only
```

### 示例3：对比不同thinking模式
```bash
# 收集数据
python scripts/collect_expert_activation.py --datasets gsm8k

# 对比分析
python scripts/analyze_expert_activation.py \
    --compare gsm8k_thinking_off gsm8k_thinking_on
```

### 示例4：分批测试
```bash
# 第一批：WikiText
python scripts/collect_expert_activation.py --datasets wikitext

# 第二批：GSM8K和HumanEval
python scripts/collect_expert_activation.py --datasets gsm8k humaneval

# 分析所有结果
python scripts/analyze_expert_activation.py
```

## 数据集要求

### Parquet文件格式

#### WikiText
- 列名：`text` 或第一列
- 内容：文本段落
- 要求：长度 > 50字符

#### GSM8K
- 列名：`question` 或第一列
- 内容：数学问题
- 格式：文本

#### HumanEval
- 列名：`prompt` 或第一列
- 内容：Python函数签名和docstring
- 格式：文本

### 自定义数据集路径

如果数据集在不同位置，可以指定路径：
```bash
python scripts/collect_expert_activation.py \
    --datasets wikitext \
    --wikitext-path /path/to/your/wikitext.parquet
```

## 性能优化

### 减少样本数量
```bash
# 从256减少到64个样本，加快4倍
python scripts/collect_expert_activation.py --all --num-samples 64
```

### 分批运行
```bash
# 分别运行每个数据集，避免长时间运行
python scripts/collect_expert_activation.py --datasets wikitext
python scripts/collect_expert_activation.py --datasets gsm8k
python scripts/collect_expert_activation.py --datasets humaneval
```

### 只测试单一模式
```bash
# 只测试thinking关闭的情况，减少一半实验
python scripts/collect_expert_activation.py --all --no-thinking
```

## 常见问题

### Q: 如何查看支持的命令行参数？
```bash
python scripts/collect_expert_activation.py --help
python scripts/analyze_expert_activation.py --help
```

### Q: parquet文件找不到？
确保数据文件路径正确：
```bash
ls -la data/WikiText/
ls -la data/GSMK/
ls -la data/HumanEval/
```

### Q: 显存不够？
1. 减少样本数量：`--num-samples 32`
2. 一次只测试一个数据集：`--datasets wikitext`

### Q: 如何自定义数据处理逻辑？
编辑 `collect_expert_activation.py` 中的 `load_dataset_prompts()` 函数。

## 系统要求

### 硬件
- GPU: 至少40GB显存（推荐A100）
- 内存: 至少32GB RAM
- 存储: 至少10GB可用空间

### 软件
```bash
pip install torch transformers pandas tqdm numpy
```

## 查看帮助

```bash
# 数据收集脚本帮助
python scripts/collect_expert_activation.py --help

# 结果分析脚本帮助
python scripts/analyze_expert_activation.py --help
```

## 完整示例命令

```bash
# === 基本用法 ===
# 运行所有实验
python scripts/collect_expert_activation.py --all

# 分析结果
python scripts/analyze_expert_activation.py

# === 快速测试 ===
# 单个数据集，少量样本
python scripts/collect_expert_activation.py \
    --datasets wikitext \
    --num-samples 32 \
    --no-thinking

# === 自定义配置 ===
# 完整自定义
python scripts/collect_expert_activation.py \
    --datasets wikitext gsm8k \
    --num-samples 128 \
    --model-path /my/model/path \
    --output-dir ./my_results \
    --wikitext-path ./my_data/wiki.parquet \
    --gsm8k-path ./my_data/gsm8k.parquet

# === 结果分析 ===
# 分析自定义目录
python scripts/analyze_expert_activation.py \
    --results-dir ./my_results

# 只对比两个实验
python scripts/analyze_expert_activation.py \
    --compare wikitext_thinking_off gsm8k_thinking_off
```

## 画分析图
### 基本用法

```bash
# 生成单个实验的可视化图表
python3 scripts/analyze_expert_activation.py --experiments wikitext_thinking_off --plot

# 生成所有实验的可视化图表
python3 scripts/analyze_expert_activation.py --plot

# 只生成图表，不显示文本统计
python3 scripts/analyze_expert_activation.py --plot-only
```

### 高级用法

```bash
# 生成合并可视化（多个实验对比）
python3 scripts/analyze_expert_activation.py --experiments wikitext_thinking_off wikitext_thinking_on --merged-plot

# 指定输出目录
python3 scripts/analyze_expert_activation.py --plot --plot-dir ./my_plots

# 指定结果目录
python3 scripts/analyze_expert_activation.py --results-dir ./my_results --plot
```

### 可视化相关参数
- `--plot`: 生成可视化图表（热力图和单层对比图）
- `--plot-only`: 只生成图表，不显示文本统计
- `--plot-dir`: 图表输出目录（默认: plots）
- `--merged-plot`: 生成合并可视化（多个实验对比）

### 其他参数
- `--results-dir`: 结果目录路径
- `--experiments`: 指定要分析的实验名称
- `--summary-only`: 只显示统计摘要

## 使用示例

### 示例1: 快速查看单个实验
```bash
python3 scripts/analyze_expert_activation.py \
    --experiments wikitext_thinking_off \
    --plot-only \
    --plot-dir ./quick_plots
```