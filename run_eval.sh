#!/usr/bin/env bash
# ============================================================
# run_eval.sh — Qwen3-30B-A3B Comprehensive Precision Benchmark
#
# Execution plan:
#   Phase 1: FP16 baseline    — CUDA 0,1 (vLLM tp=2), sequential
#   Phase 2: INT4 + mixed     — GPU 0 and GPU 1 in PARALLEL
#             GPU 0: int4 (vLLM tp=1), then mixed_10, mixed_20 (HF)
#             GPU 1: mixed_30, mixed_40 (HF)
#   Phase 3: Merge results into one JSON
#
#   INT4 uses vLLM + GPTQ backend (3-5x faster than HF Transformers).
#   Mixed configs use HF Transformers + ExpertSwapper (vLLM can't do
#   post-load model surgery required for expert promotion).
#
# Usage:
#   bash run_eval.sh          # full run  (n=200, ~8–12 h)
#   bash run_eval.sh sanity   # sanity    (n=1,   ~30 min)
#   bash run_eval.sh int4     # INT4 only (n=200, single GPU)
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
FP16_PATH="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507"
INT4_PATH="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
VENV="/home/kec23008/auto-paper-reading/venv/bin/activate"
SCRIPT="scripts/eval_comprehensive.py"
RESULTS_DIR="results"

# ── HF token (for GPQA) ──────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN must be set in the environment" >&2; exit 2; fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── Mode: sanity (1 sample) vs full (200 samples) vs int4-only ───────────────
MODE="${1:-full}"
if [[ "$MODE" == "sanity" ]]; then
    N_SAMPLES=1
    N_AIME=1
    N_CALIB=4
    BATCH_SIZE=1
    MAX_NEW_TOKENS=4096
    TAG="sanity"
    echo ">>> SANITY mode (n=1 per benchmark)"
elif [[ "$MODE" == "int4" ]]; then
    N_SAMPLES=200
    N_AIME=30
    N_CALIB=16
    BATCH_SIZE=32
    MAX_NEW_TOKENS=4096
    TAG="full"
    echo ">>> INT4-ONLY mode (n=200, vLLM + GPTQ, GPU 0)"
else
    N_SAMPLES=200
    N_AIME=30
    N_CALIB=16
    BATCH_SIZE=32
    MAX_NEW_TOKENS=4096
    TAG="full"
    echo ">>> FULL mode (n=200 per benchmark, n_aime=30)"
fi

BENCHMARKS="mmlu_pro,gpqa,aime,gsm8k,humaneval"
MAX_MEMORY="44GiB"
LOG_DIR="results/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
source "$VENV"

START_TIME=$(date +%s)
echo "Started at: $(date)"
echo ""

# ── INT4-only mode ────────────────────────────────────────────────────────────
if [[ "$MODE" == "int4" ]]; then
    echo "============================================================"
    echo "INT4 only: vLLM + GPTQ (CUDA_VISIBLE_DEVICES=0, tp=1)"
    echo "============================================================"

    INT4_OUT="${RESULTS_DIR}/${TAG}_int4.json"
    INT4_LOG="${LOG_DIR}/${TAG}_int4.log"

    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
        --fp16-path      "$FP16_PATH" \
        --int4-path      "$INT4_PATH" \
        --configs        int4 \
        --benchmarks     "$BENCHMARKS" \
        --n-samples      "$N_SAMPLES" \
        --n-aime         "$N_AIME" \
        --n-calib        "$N_CALIB" \
        --batch-size     "$BATCH_SIZE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --tp             1 \
        --output         "$INT4_OUT" \
        2>&1 | tee "$INT4_LOG"

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo ""
    echo "INT4 done at $(date)"
    echo "Total elapsed: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
    echo "Results: $INT4_OUT"
    exit 0
fi

# ── Phase 1: FP16 baseline (GPU 0,1, vLLM tp=2) ──────────────────────────────
echo "============================================================"
echo "Phase 1: FP16 baseline (CUDA_VISIBLE_DEVICES=0,1, vLLM tp=2)"
echo "============================================================"

# FP16_OUT="${RESULTS_DIR}/${TAG}_fp16.json"
# FP16_LOG="${LOG_DIR}/${TAG}_fp16.log"

# CUDA_VISIBLE_DEVICES=0,1 python "$SCRIPT" \
#     --fp16-path      "$FP16_PATH" \
#     --int4-path      "$INT4_PATH" \
#     --configs        fp16 \
#     --benchmarks     "$BENCHMARKS" \
#     --n-samples      "$N_SAMPLES" \
#     --n-aime         "$N_AIME" \
#     --n-calib        "$N_CALIB" \
#     --batch-size     "$BATCH_SIZE" \
#     --max-new-tokens "$MAX_NEW_TOKENS" \
#     --output         "$FP16_OUT" \
#     2>&1 | tee "$FP16_LOG"

# echo ""
# echo "Phase 1 done → $FP16_OUT"
# echo ""

# ── Phase 2: INT4 (vLLM) + mixed (HF) — GPU 0 and GPU 1 in parallel ─────────
echo "============================================================"
echo "Phase 2: INT4/Mixed (GPU 0 ∥ GPU 1)"
echo "  GPU 0: int4 (vLLM+GPTQ), then mixed_10, mixed_20 (HF)"
echo "  GPU 1: mixed_30, mixed_40 (HF)"
echo "============================================================"

GPU0_OUT="${RESULTS_DIR}/${TAG}_int4_mixed10_20.json"
GPU1_OUT="${RESULTS_DIR}/${TAG}_mixed30_40.json"
GPU0_LOG="${LOG_DIR}/${TAG}_gpu0.log"
GPU1_LOG="${LOG_DIR}/${TAG}_gpu1.log"

# GPU 0: mixed_10 + mixed_20 (HF)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
    --fp16-path      "$FP16_PATH" \
    --int4-path      "$INT4_PATH" \
    --configs        mixed_10,mixed_20 \
    --benchmarks     "$BENCHMARKS" \
    --n-samples      "$N_SAMPLES" \
    --n-aime         "$N_AIME" \
    --n-calib        "$N_CALIB" \
    --batch-size     "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-memory     "$MAX_MEMORY" \
    --tp             1 \
    --output         "$GPU0_OUT" \
    2>&1 | tee "$GPU0_LOG" &
PID_GPU0=$!

# GPU 1: mixed_30 + mixed_40 (HF)
# CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" \
#     --fp16-path      "$FP16_PATH" \
#     --int4-path      "$INT4_PATH" \
#     --configs        mixed_30,mixed_40 \
#     --benchmarks     "$BENCHMARKS" \
#     --n-samples      "$N_SAMPLES" \
#     --n-aime         "$N_AIME" \
#     --n-calib        "$N_CALIB" \
#     --batch-size     "$BATCH_SIZE" \
#     --max-new-tokens "$MAX_NEW_TOKENS" \
#     --max-memory     "$MAX_MEMORY" \
#     --output         "$GPU1_OUT" \
#     2>&1 | tee "$GPU1_LOG" &
# PID_GPU1=$!

echo "GPU 0 job PID: $PID_GPU0  (log: $GPU0_LOG)"
# echo "GPU 1 job PID: $PID_GPU1  (log: $GPU1_LOG)"
echo ""

# Wait for both to finish, capture exit codes
GPU0_EXIT=0; GPU1_EXIT=0
wait $PID_GPU0 || GPU0_EXIT=$?
echo "GPU 0 job finished (exit=$GPU0_EXIT)"
# wait $PID_GPU1 || GPU1_EXIT=$?
# echo "GPU 1 job finished (exit=$GPU1_EXIT)"

# if [[ $GPU0_EXIT -ne 0 || $GPU1_EXIT -ne 0 ]]; then
#     echo ""
#     echo "ERROR: one or more Phase 2 jobs failed."
#     echo "  GPU0 exit=$GPU0_EXIT  GPU1 exit=$GPU1_EXIT"
#     echo "Check logs: $GPU0_LOG  $GPU1_LOG"
#     exit 1
# fi

echo ""

# ── Phase 3: Merge all results into one JSON ──────────────────────────────────
echo "============================================================"
echo "Phase 3: Merging results"
echo "============================================================"

MERGED_OUT="${RESULTS_DIR}/${TAG}_comprehensive_eval.json"

python3 - <<PYEOF
import json, time
from pathlib import Path

files = [
    "${FP16_OUT}",
    "${GPU0_OUT}",
    "${GPU1_OUT}",
]

merged = {}
for fpath in files:
    p = Path(fpath)
    if not p.exists():
        print(f"  [warn] {fpath} not found, skipping")
        continue
    data = json.loads(p.read_text())
    for cfg, result in data.get("results", {}).items():
        merged[cfg] = result
    print(f"  Loaded {len(data.get('results', {}))} config(s) from {fpath}")

# Print summary table
benchmarks = ["mmlu_pro", "gpqa", "aime", "gsm8k", "humaneval"]
cols = ["Config", "MMLU-Pro", "GPQA", "AIME", "GSM8K", "HumanEval", "GPU(GB)", "#Promoted"]
widths = [14, 9, 9, 9, 9, 11, 9, 10]
header = "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
print("\n" + "=" * sum(widths))
print(header)
print("-" * sum(widths))
for cfg in ["fp16", "int4", "mixed_10", "mixed_20", "mixed_30", "mixed_40"]:
    if cfg not in merged:
        continue
    r = merged[cfg]
    bk = r.get("benchmarks", {})
    row = [cfg]
    for b in benchmarks:
        v = bk.get(b, {}).get("accuracy_pct", "—")
        row.append(f"{v}%" if v != "—" else "—")
    row.append(str(r.get("gpu_mem_gb", "—")))
    row.append(str(r.get("n_promoted", "—")))
    print("".join(f"{c:<{w}}" for c, w in zip(row, widths)))
print("=" * sum(widths))

output = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": merged}
Path("${MERGED_OUT}").write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"\nMerged {len(merged)} configs → ${MERGED_OUT}")
PYEOF

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "All done at $(date)"
echo "Total elapsed: $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo "Final results: $MERGED_OUT"
