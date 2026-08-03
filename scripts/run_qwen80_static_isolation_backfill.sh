#!/usr/bin/env bash
# Replace the Qwen80 static bs=1 point after all main lanes and the existing
# Qwen30 isolation backfill finish. The original point overlapped the
# CPU-intensive Qwen80 dynamic checkpoint reconstruction, so it is retained as
# a diagnostic artifact and this run is written to a distinct output path.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
predecessor_status="${artifact_root}/native_isolation_backfill_gpu0.status"
status_file="${artifact_root}/native_qwen80_isolation_backfill_gpu0.status"
model="/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int2-from-int4-formal"
output="${artifact_root}/qwen80b_static_int2_bs1_isolated_rerun.json"
log="${artifact_root}/qwen80b_static_int2_bs1_isolated_rerun.log"

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
    /home/kec23008/miniconda3/bin/python \
        /home/kec23008/DynaQuant/scripts/summarize_performance_artifact.py \
        "$1"
}

wait_for_predecessor() {
    while true; do
        if [[ -s "${predecessor_status}" ]] &&
           tail -n 1 "${predecessor_status}" |
               grep -q 'ISOLATION_BACKFILL_COMPLETE$'; then
            return
        fi
        sleep 60
    done
}

gpu0_idle() {
    sample=$(nvidia-smi -i 0 \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits)
    used=${sample%%,*}
    util=${sample##*,}
    used=${used//[[:space:]]/}
    util=${util//[[:space:]]/}
    [[ "${used}" -le 1024 && "${util}" -eq 0 ]]
}

wait_for_isolation() {
    consecutive=0
    while ((consecutive < 3)); do
        idle_pct=$(vmstat 1 2 | tail -n 1 | awk '{print $15}')
        if gpu0_idle && [[ "${idle_pct}" -ge 80 ]]; then
            consecutive=$((consecutive + 1))
            record_status \
                "ISOLATION_SAMPLE pass=${consecutive}/3 cpu_idle_pct=${idle_pct}"
        else
            consecutive=0
            record_status "WAIT_ISOLATION cpu_idle_pct=${idle_pct}"
        fi
        if ((consecutive < 3)); then
            sleep 30
        fi
    done
}

record_status "QWEN80_ISOLATION_BACKFILL_SUPERVISOR_STARTED"
if [[ -s "${output}" ]]; then
    validate_artifact "${output}"
    record_status "SKIP_VALID qwen80b static_formal bs=1 isolated=true"
    exit 0
fi

wait_for_predecessor
record_status "QWEN30_ISOLATION_BACKFILL_COMPLETE"
wait_for_isolation
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
[[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
record_status "START qwen80b static_formal bs=1 isolated=true"
(
    cd "${formal_root}"
    CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
        --model "${model}" \
        --paper-model qwen80b \
        --method quantized_checkpoint \
        --quantization int2 \
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
validate_artifact "${output}"
record_status "PASS qwen80b static_formal bs=1 isolated=true"
record_status "QWEN80_ISOLATION_BACKFILL_COMPLETE"
