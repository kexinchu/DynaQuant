# Scripts 目录说明

本目录包含 DynaQuant 项目的主要脚本。

## 量化脚本

### quantize_w4a16.py
W4A16 量化脚本（4位权重 + 16位激活）

**使用方法：**
```bash
python quantize_w4a16.py \
    --model /dev/shm/Qwen3-30B-A3B \
    --output-dir /dev/shm/Qwen3-30B-A3B-W4A16 \
    --calib-data calibration_datasets/Qwen3-30B-A3B/calibration_Qwen3-30B-A3B.json \
    --num-samples 512 \
    --max-seq-length 8192
```

**特点：**
- 4倍压缩率
- 精度损失小（1-3%）
- 推荐用于生产环境

### quantize_w2a16.py
W2A16 量化脚本（2位权重 + 16位激活）

**使用方法：**
```bash
python quantize_w2a16.py \
    --model /dev/shm/Qwen3-30B-A3B \
    --output-dir /dev/shm/Qwen3-30B-A3B-W2A16 \
    --calib-data calibration_datasets/Qwen3-30B-A3B/calibration_Qwen3-30B-A3B.json \
    --num-samples 512
```

**特点：**
- 8倍压缩率
- 精度损失较大（3-8%）
- 适用于资源受限场景

---

## 评估脚本

### evaluate_model.py
模型评估脚本，支持多个数据集

**使用方法：**
```bash
python evaluate_model.py \
    --model /dev/shm/Qwen3-30B-A3B-W4A16 \
    --datasets wikitext mmlu gsm8k hellaswag \
    --output eval_results.json \
    --device cuda \
    --data-dir ../data
```

**支持的数据集：**
- `wikitext` - WikiText-2 困惑度评估
- `mmlu` - MMLU 多任务准确度
- `gsm8k` - GSM8K 数学推理
- `hellaswag` - HellaSwag 常识推理

**输出格式：**
```json
{
  "model": "/path/to/model",
  "evaluations": {
    "wikitext": {
      "perplexity": 12.34,
      "total_tokens": 245837
    },
    "mmlu": {
      "overall_accuracy": 0.7845,
      "total_correct": 784,
      "total_questions": 1000
    }
  }
}
```

---

## 其他脚本

### bench_eval.py
性能基准测试脚本

### serve_sglang.py
使用 SGLang 提供模型服务

### test_quantized_models.py
量化模型测试脚本

### motivation_test.py
动机测试和分析

### analyze_motivation_test.py
动机测试结果分析

### collect_expert_activation.py
专家激活收集（用于分析）

### analyze_expert_activation.py
专家激活分析（用于分析）

### calibrate.py
校准数据生成（如果需要）

### generate_all_calibration_datasets.sh
批量生成校准数据集

---

## 快速开始

### 1. 量化一个模型

```bash
# W4A16 量化（推荐）
python quantize_w4a16.py \
    --model /path/to/model \
    --num-samples 512
```

### 2. 评估量化效果

```bash
# 评估所有支持的数据集
python evaluate_model.py \
    --model /path/to/quantized/model \
    --datasets wikitext mmlu gsm8k hellaswag
```

### 3. 比较原始模型和量化模型

```bash
# 评估原始模型
python evaluate_model.py \
    --model /path/to/original/model \
    --output results_original.json

# 评估量化模型
python evaluate_model.py \
    --model /path/to/quantized/model \
    --output results_quantized.json

# 比较结果（使用任意JSON比较工具）
```

---

## 注意事项

1. **GPU 内存**：量化 30B 模型需要约 60-80GB GPU 内存
2. **校准数据**：需要准备 JSON 格式的校准数据（文本列表）
3. **评估数据**：评估脚本会自动从 `data/` 目录加载数据集
4. **时间估算**：
   - 量化：30-60分钟（30B模型，8×H20 GPU）
   - 评估：10-30分钟（取决于数据集大小）

---

## 依赖

确保已安装所有依赖：

```bash
pip install -r ../requirements.txt
```

主要依赖：
- `llm-compressor>=0.1.0`
- `transformers>=4.35.0`
- `torch>=2.0.0`
- `datasets>=2.14.0`
- `pandas>=1.5.0`

---

## 故障排除

### 问题1: 找不到校准数据

**解决方案**：
```bash
# 确保校准数据文件存在
ls -lh ../calibration_datasets/

# 或使用 --calib-data 参数显式指定路径
python quantize_w4a16.py \
    --model /path/to/model \
    --calib-data /path/to/calibration.json
```

### 问题2: CUDA 内存不足

**解决方案**：
```bash
# 减少序列长度
python quantize_w4a16.py \
    --model /path/to/model \
    --max-seq-length 4096  # 默认8192

# 减少校准样本数
python quantize_w4a16.py \
    --model /path/to/model \
    --num-samples 256  # 默认512
```

### 问题3: 评估数据集找不到

**解决方案**：
```bash
# 确保数据集目录存在
ls -lh ../data/

# 或使用 --data-dir 参数
python evaluate_model.py \
    --model /path/to/model \
    --data-dir /path/to/data
```

---

## 更多信息

详细文档请参考：
- [README_LLM_COMPRESSOR.md](../README_LLM_COMPRESSOR.md) - 完整使用指南
- [README.md](../README.md) - 项目主README
- [CLEANUP_LIST.md](../CLEANUP_LIST.md) - 代码清理记录

