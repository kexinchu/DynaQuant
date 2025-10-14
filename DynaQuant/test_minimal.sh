#!/bin/bash
# 最小可复现测试脚本

set -e

echo "==== MoE-Quant 最小可复现测试 ===="

# 1. 测试组件
echo "1. 测试组件..."
python3 test_moe_quant.py

# 2. 模拟PTQ（无真实模型）
echo ""
echo "2. 测试PTQ流程（组件级）..."
python3 -c "
from moe_quant.quant.ebss import EBSSConfig
from moe_quant.quant.agq import AGQConfig
from moe_quant.quant.quantizers import W2A2Config

print('✓ 配置对象创建成功')
print(f'  EBSS: beam={EBSSConfig().beam_width}, tau={EBSSConfig().tau}')
print(f'  AGQ: bit={AGQConfig().bit_width}, group={AGQConfig().group_size}')
print(f'  W2A2: w_bit={W2A2Config().w_bit}, a_bit={W2A2Config().a_bit}')
"

echo ""
echo "3. 测试Loss函数..."
python3 -c "
import torch
from moe_quant.losses.routing_losses import combined_routing_loss

logits_fp = torch.randn(2, 10, 8)
logits_q = logits_fp + torch.randn_like(logits_fp) * 0.1
loss, loss_dict = combined_routing_loss(logits_q, logits_fp, k=2)

print('✓ Loss计算成功')
print(f'  Total: {loss_dict[\"total\"]:.4f}')
print(f'  Consistency: {loss_dict[\"consistency\"]:.4f}')
print(f'  Margin: {loss_dict[\"margin\"]:.4f}')
"

echo ""
echo "==== 所有测试通过！ ===="
echo "下一步: 使用真实MoE模型运行PTQ"
echo "  bash scripts/run_ptq_moe.sh --model YOUR_MODEL_NAME"

