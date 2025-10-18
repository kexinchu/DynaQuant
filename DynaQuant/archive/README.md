# 归档的旧实现

本目录包含项目早期版本的实现，已被新的 llm-compressor 方案替代。

## 📦 归档内容

### moe_quant_legacy/
完整的 MoEQuant 量化实现，包括：
- **EBSS (Expert-Balanced Self-Sampling)** - 专家均衡采样算法
- **AGQ (Affinity-Guided Quantization)** - 亲和度引导量化算法
- **W2A2/W4A4/W8A8 量化** - 多精度量化支持
- **SafeTensors 格式输出** - HuggingFace 兼容格式

**量化脚本：**
```bash
# W4A4 量化
bash archive/moe_quant_legacy/scripts/moequant_w4a4.sh \
    --model /path/to/model \
    --seed-text calibration.txt

# W2A2 量化（极限压缩）
bash archive/moe_quant_legacy/scripts/moequant_w2a2.sh \
    --model /path/to/model \
    --seed-text calibration.txt
```

### dynaquant_legacy/
旧的 DynaQuant 模块实现

## 🔄 为什么归档？

从 MoEQuant 迁移到 llm-compressor 带来的改进：
- ✅ **速度提升 3-5 倍**：量化时间从 2-3 小时降至 30-60 分钟
- ✅ **代码简化 15 倍**：从 3000+ 行降至 <200 行
- ✅ **维护成本降低**：基于成熟的第三方库，持续更新
- ✅ **更简单的 API**：无需复杂配置，一键量化

## 📚 旧版文档

如果需要使用 MoEQuant 的高级功能，请参考归档目录中的文档：
- `moe_quant_legacy/README.md` - 完整的 MoEQuant 文档
- `moe_quant_legacy/EBSS_GUIDE.md` - EBSS 算法详细说明
- 各种配置和使用示例

## 🔙 如何恢复旧实现

如果需要回退到旧的 MoEQuant 实现：

```bash
# 1. 恢复模块
mv archive/moe_quant_legacy moe_quant
mv archive/dynaquant_legacy dynaquant

# 2. 安装旧依赖
pip install -r archive/requirements_old.txt

# 3. 使用旧脚本
bash moe_quant/scripts/moequant_w4a4.sh ...
```

## ⚠️ 注意事项

- 归档的代码不再维护
- 推荐使用新的 llm-compressor 方案（见项目根目录 README.md）
- 旧代码仅供参考和特殊需求使用

---

**归档日期**: 2025-10-18  
**替代方案**: llm-compressor (见 README.md)

