#!/usr/bin/env bash
# Run clean, low-CPU-contention replacements for Qwen30 static points after
# both native performance lanes have fully completed. Until then this process
# is a sleeping supervisor and consumes no GPU memory.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
static_status="${artifact_root}/native_static_gpu0.status"
dynamic_status="${artifact_root}/native_dynamic_gpu1.status"
status_file="${artifact_root}/native_isolation_backfill_gpu0.status"
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

queue_complete() {
    status=$1
    lane=$2
    [[ -s "${status}" ]] &&
        tail -n 1 "${status}" | grep -q \
            "QUEUE_COMPLETE lane=${lane} physical_gpu=$([[ "${lane}" == static ]] && echo 0 || echo 1)$"
}

wait_for_queues() {
    while ! queue_complete "${static_status}" static ||
          ! queue_complete "${dynamic_status}" dynamic; do
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

validate_artifact() {
    artifact=$1
    /home/kec23008/miniconda3/bin/python \
        /home/kec23008/DynaQuant/scripts/summarize_performance_artifact.py \
        "${artifact}"
}

run_point() {
    batch=$1
    output="${artifact_root}/qwen30b_static_int4_bs${batch}_isolated_rerun.json"
    log="${artifact_root}/qwen30b_static_int4_bs${batch}_isolated_rerun.log"
    if [[ -s "${output}" ]]; then
        validate_artifact "${output}"
        record_status "SKIP_VALID qwen30b static_formal bs=${batch}"
        return
    fi
    wait_for_isolation
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
    record_status "START qwen30b static_formal bs=${batch} isolated=true"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
            --model "${model}" \
            --paper-model qwen30b \
            --method quantized_checkpoint \
            --quantization int4 \
            --autoround-backend triton \
            --batch-size "${batch}" \
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
    record_status "PASS qwen30b static_formal bs=${batch} isolated=true"
}

record_status "ISOLATION_BACKFILL_SUPERVISOR_STARTED"
wait_for_queues
record_status "MAIN_QUEUES_COMPLETE"
run_point 1
run_point 32
record_status "ISOLATION_BACKFILL_COMPLETE"
