#!/usr/bin/env bash
# Backfill formal native-performance points that predate the current clean
# experiment commit. This supervisor waits for the main static queue to finish
# before claiming GPU 0, so it cannot race a watchdog restart.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
main_status="${artifact_root}/native_static_gpu0.status"
status_file="${artifact_root}/native_static_backfill_gpu0.status"
output="${artifact_root}/qwen30b_static_int4_bs1.json"
log="${artifact_root}/qwen30b_static_int4_bs1.log"
model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

validate_artifact() {
    ARTIFACT="${output}" FORMAL_ROOT="${formal_root}" \
    EXPECTED_COMMIT="${expected_commit}" python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.audit_paper_results import _validate_performance_benchmark  # noqa: E402

path = Path(os.environ["ARTIFACT"])
data = json.loads(path.read_text(encoding="utf-8"))
problems = _validate_performance_benchmark(str(path), data.get("benchmark"))
benchmark = data.get("benchmark", {})
expected = {
    "paper_model": "qwen30b",
    "paper_method": "static_ptq",
}
for key, value in expected.items():
    if data.get(key) != value:
        problems.append(f"{key} mismatch")
if benchmark.get("batch_size") != 1:
    problems.append("batch mismatch")
if benchmark.get("input_tokens") != 2048:
    problems.append("input length mismatch")
if benchmark.get("output_tokens_per_sequence") != 256:
    problems.append("output length mismatch")
git = data.get("environment", {}).get("git", {})
if git.get("commit") != os.environ["EXPECTED_COMMIT"]:
    problems.append("experiment commit mismatch")
if git.get("dirty") is not False:
    problems.append("dirty git provenance")
if problems:
    raise SystemExit("; ".join(problems))
PY
}

wait_for_main_queue() {
    while true; do
        if [[ -s "${main_status}" ]] &&
           tail -n 1 "${main_status}" | grep -q 'QUEUE_COMPLETE lane=static physical_gpu=0$'; then
            return
        fi
        sleep 60
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

record_status "BACKFILL_SUPERVISOR_STARTED"
if [[ -s "${output}" ]]; then
    validate_artifact
    record_status "SKIP_VALID qwen30b static_formal bs=1"
    exit 0
fi

wait_for_main_queue
record_status "MAIN_QUEUE_COMPLETE"
wait_for_gpu_idle
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
[[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
record_status "START qwen30b static_formal bs=1"
(
    cd "${formal_root}"
    CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
        --model "${model}" \
        --paper-model qwen30b \
        --method quantized_checkpoint \
        --quantization int4 \
        --autoround-backend triton \
        --batch-size 1 \
        --input-length 2048 \
        --output-length 256 \
        --device-map cuda:0 \
        --paper-protocol \
        --n-warmup 5 \
        --n-repeats 100 \
        --hash-model-files \
        --output "${output}"
) >"${log}" 2>&1
validate_artifact
record_status "PASS qwen30b static_formal bs=1"
record_status "BACKFILL_COMPLETE"
