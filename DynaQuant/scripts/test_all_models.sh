#!/bin/bash
#
# Test all quantized models (W2A2 and W4A4)
#

set -e

MODEL_BASE="/dev/shm/Qwen3-30B-A3B"
QUANTIZED_BASE="/dev/shm/quantized_models"

echo "========================================"
echo "Testing Quantized Models"
echo "========================================"

# Test prompts
TEST_PROMPTS="test_prompts.txt"

# Create test prompts if not exists
if [ ! -f "$TEST_PROMPTS" ]; then
    cat > "$TEST_PROMPTS" << 'EOF'
Explain the concept of artificial intelligence in simple terms.
What are the main differences between machine learning and deep learning?
How does a neural network learn?
Describe the applications of natural language processing.
What is the future of quantum computing?
EOF
fi

# Test W2A2 model
echo ""
echo "Testing W2A2 Model..."
python3 scripts/test_quantized_models.py \
    --model "$MODEL_BASE" \
    --quantized-weights "${QUANTIZED_BASE}/Qwen3-30B-A3B-W2A2/quantized_model_full.pt" \
    --test-prompts "$TEST_PROMPTS" \
    --output "${QUANTIZED_BASE}/Qwen3-30B-A3B-W2A2/test_results.json" \
    --num-samples 5

# Test W4A4 model
echo ""
echo "Testing W4A4 Model..."
python3 scripts/test_quantized_models.py \
    --model "$MODEL_BASE" \
    --quantized-weights "${QUANTIZED_BASE}/Qwen3-30B-A3B-W4A4/quantized_model_full.pt" \
    --test-prompts "$TEST_PROMPTS" \
    --output "${QUANTIZED_BASE}/Qwen3-30B-A3B-W4A4/test_results.json" \
    --num-samples 5

# Generate comparison report
python3 - << 'PYTHON_SCRIPT'
import json
from pathlib import Path

quantized_base = Path("/dev/shm/quantized_models")

print("\n" + "="*80)
print("Model Comparison Report")
print("="*80)

models = [
    "Qwen3-30B-A3B-W2A2",
    "Qwen3-30B-A3B-W4A4"
]

for model_name in models:
    result_file = quantized_base / model_name / "test_results.json"
    
    if result_file.exists():
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n{model_name}:")
        print(f"  Average Latency: {results.get('avg_latency_ms', 'N/A'):.2f} ms")
        print(f"  Throughput: {results.get('throughput_tokens_per_sec', 'N/A'):.2f} tokens/s")
        print(f"  Peak Memory: {results.get('peak_memory_gb', 'N/A'):.2f} GB")
        
        if 'sample_outputs' in results:
            print(f"  Samples generated: {len(results['sample_outputs'])}")

print("\n" + "="*80)
print("\nTest complete! Check individual result files for details:")
for model_name in models:
    print(f"  {quantized_base / model_name / 'test_results.json'}")
print("="*80)

PYTHON_SCRIPT

echo ""
echo "All tests completed!"

