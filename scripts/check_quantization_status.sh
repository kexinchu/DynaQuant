#!/bin/bash
# Check the status of quantization processes

echo "=== Quantization Process Status ==="
ps aux | grep quantize_with_autoround | grep -v grep

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== DeepSeek-V2-Lite Quantization Log (last 20 lines) ==="
tail -20 scripts/quantize_deepseek.log

echo ""
echo "=== GPT-OSS-20B Quantization Log (last 20 lines) ==="
tail -20 scripts/quantize_gpt_oss.log

echo ""
echo "=== Output Directories ==="
if [ -d "../Models/DeepSeek-V2-Lite-mixed-AutoRound" ]; then
    echo "DeepSeek-V2-Lite-mixed-AutoRound: $(du -sh ../Models/DeepSeek-V2-Lite-mixed-AutoRound 2>/dev/null | cut -f1)"
    echo "  Files: $(find ../Models/DeepSeek-V2-Lite-mixed-AutoRound -type f 2>/dev/null | wc -l)"
else
    echo "DeepSeek-V2-Lite-mixed-AutoRound: Not created yet"
fi

if [ -d "../Models/gpt-oss-20b-mixed-AutoRound" ]; then
    echo "gpt-oss-20b-mixed-AutoRound: $(du -sh ../Models/gpt-oss-20b-mixed-AutoRound 2>/dev/null | cut -f1)"
    echo "  Files: $(find ../Models/gpt-oss-20b-mixed-AutoRound -type f 2>/dev/null | wc -l)"
else
    echo "gpt-oss-20b-mixed-AutoRound: Not created yet"
fi

