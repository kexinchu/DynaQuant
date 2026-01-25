#!/bin/bash
# Check status of expert activation extraction tasks

echo "=========================================="
echo "Expert Activation Extraction Status"
echo "=========================================="
echo ""

echo "=== Running Processes ==="
ps aux | grep "summarize_expert_activation" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    model=$(echo $line | grep -oP 'Models/\K[^ ]+' || echo "unknown")
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    time=$(echo $line | awk '{print $10}')
    echo "PID: $pid | Model: $model | CPU: ${cpu}% | MEM: ${mem}% | TIME: $time"
done

echo ""
echo "=== Output Files ==="
ls -lh activations/activation_*_mmlu_pro.json 2>/dev/null | awk '{print $9, "(" $5 ")"}' || echo "No output files yet"

echo ""
echo "=== Recent Log Activity ==="
for log in scripts/extract_*.log; do
    if [ -f "$log" ]; then
        echo "--- $(basename $log) ---"
        tail -5 "$log" | grep -E "(INFO|Processed|Saved|Error)" | tail -3 || echo "No recent activity"
        echo ""
    fi
done

echo "=========================================="


