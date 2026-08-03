#!/usr/bin/env bash
# Continue the formal Qwen3-30B performance grid without touching the
# manuscript or dirtying the clean measurement worktree.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
static_model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
dynamic_model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned"
initial_map="/dev/shm/dynaexq-formal-bbdaa87/qwen30b_initial_map.json"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
status_file="${artifact_root}/qwen30b_queue.status"

mkdir -p "${artifact_root}"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

wait_for_session_end() {
    session_name=$1
    while tmux has-session -t "${session_name}" 2>/dev/null; do
        sleep 30
    done
}

wait_for_gpu_idle() {
    while true; do
        sample=$(nvidia-smi -i 0 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits)
        used=${sample%%,*}
        util=${sample##*,}
        used=${used//[[:space:]]/}
        util=${util//[[:space:]]/}
        if [[ "${used}" -le 1024 && "${util}" -eq 0 ]]; then
            return
        fi
        record_status "WAIT_GPU used_mib=${used} utilization_pct=${util}"
        sleep 30
    done
}

require_clean_formal_tree() {
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
}

validate_performance_artifact() {
    artifact=$1
    model=$2
    method=$3
    batch=$4
    FORMAL_ROOT="${formal_root}" \
    ARTIFACT="${artifact}" \
    PAPER_MODEL="${model}" \
    PAPER_METHOD="${method}" \
    PAPER_BATCH="${batch}" \
    python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.audit_paper_results import (  # noqa: E402
    _validate_performance_benchmark,
    validate_dynamic_runtime,
)

path = Path(os.environ["ARTIFACT"])
data = json.loads(path.read_text(encoding="utf-8"))
model = os.environ["PAPER_MODEL"]
method = os.environ["PAPER_METHOD"]
batch = int(os.environ["PAPER_BATCH"])
problems = _validate_performance_benchmark(str(path), data.get("benchmark"))
if data.get("paper_model") != model:
    problems.append("paper model mismatch")
if data.get("paper_method") != method:
    problems.append("paper method mismatch")
benchmark = data.get("benchmark", {})
if benchmark.get("batch_size") != batch:
    problems.append("batch mismatch")
if benchmark.get("input_tokens") != 2048:
    problems.append("input length mismatch")
if benchmark.get("output_tokens_per_sequence") != 256:
    problems.append("output length mismatch")
git = data.get("environment", {}).get("git", {})
if not git.get("commit") or git.get("dirty") is not False:
    problems.append("dirty or missing git provenance")
if method == "dynaexq":
    problems.extend(validate_dynamic_runtime(str(path), data))
if problems:
    raise SystemExit("; ".join(problems))
PY
}

run_static_smoke() {
    batch=$1
    output_tokens=$2
    output="${artifact_root}/qwen30b_static_int4_bs${batch}_${output_tokens}_smoke.json"
    log="${artifact_root}/qwen30b_static_int4_bs${batch}_${output_tokens}_smoke.log"
    record_status "START static_smoke bs=${batch} output_tokens=${output_tokens}"
    wait_for_gpu_idle
    require_clean_formal_tree
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
            --model "${static_model}" \
            --paper-model qwen30b \
            --method quantized_checkpoint \
            --quantization int4 \
            --autoround-backend triton \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length "${output_tokens}" \
            --n-warmup 0 \
            --n-repeats 1 \
            --device-map cuda:0 \
            --output "${output}"
    ) >"${log}" 2>&1
    record_status "PASS static_smoke bs=${batch} output_tokens=${output_tokens}"
}

run_static_formal() {
    batch=$1
    artifact="${artifact_root}/qwen30b_static_int4_bs${batch}.json"
    log="${artifact_root}/qwen30b_static_int4_bs${batch}.log"
    record_status "START static_formal bs=${batch}"
    wait_for_gpu_idle
    require_clean_formal_tree
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
            --paper-protocol \
            --model "${static_model}" \
            --paper-model qwen30b \
            --method quantized_checkpoint \
            --quantization int4 \
            --autoround-backend triton \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length 256 \
            --n-warmup 5 \
            --n-repeats 100 \
            --device-map cuda:0 \
            --hash-model-files \
            --output "${artifact}"
    ) >"${log}" 2>&1
    validate_performance_artifact "${artifact}" qwen30b static_ptq "${batch}"
    record_status "PASS static_formal bs=${batch}"
}

run_dynamic_smoke() {
    batch=$1
    artifact="${artifact_root}/qwen30b_dynaexq_bs${batch}_256_smoke.json"
    log="${artifact_root}/qwen30b_dynaexq_bs${batch}_256_smoke.log"
    record_status "START dynaexq_smoke bs=${batch}"
    wait_for_gpu_idle
    require_clean_formal_tree
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
            --config dynaexq/configs/qwen30b.yaml \
            --model-path "${dynamic_model}" \
            --output "${artifact}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu 0 \
            --initial-map "${initial_map}" \
            perf \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length 256 \
            --n-warmup 0 \
            --n-repeats 1
    ) >"${log}" 2>&1
    record_status "PASS dynaexq_smoke bs=${batch}"
}

run_dynamic_formal() {
    batch=$1
    artifact="${artifact_root}/qwen30b_dynaexq_bs${batch}.json"
    log="${artifact_root}/qwen30b_dynaexq_bs${batch}.log"
    record_status "START dynaexq_formal bs=${batch}"
    wait_for_gpu_idle
    require_clean_formal_tree
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
            --config dynaexq/configs/qwen30b.yaml \
            --model-path "${dynamic_model}" \
            --output "${artifact}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu 0 \
            --initial-map "${initial_map}" \
            perf \
            --paper-protocol \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length 256 \
            --n-warmup 5 \
            --n-repeats 100
    ) >"${log}" 2>&1
    validate_performance_artifact "${artifact}" qwen30b dynaexq "${batch}"
    record_status "PASS dynaexq_formal bs=${batch}"
}

record_status "QUEUE_STARTED"
record_status "WAIT existing_batch4_chain"
wait_for_session_end dynaexq_after_static_b4
validate_performance_artifact \
    "${artifact_root}/qwen30b_static_int4_bs4.json" \
    qwen30b static_ptq 4
validate_performance_artifact \
    "${artifact_root}/qwen30b_dynaexq_bs4.json" \
    qwen30b dynaexq 4
record_status "PASS existing_batch4_chain"

for batch in 8 16 32; do
    run_static_smoke "${batch}" 8
    run_static_smoke "${batch}" 256
    run_static_formal "${batch}"
    run_dynamic_smoke "${batch}"
    run_dynamic_formal "${batch}"
done

record_status "QUEUE_COMPLETE"
