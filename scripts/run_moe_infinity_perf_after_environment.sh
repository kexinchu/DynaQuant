#!/usr/bin/env bash
# Run all six official MoE-Infinity points serially after the environment and
# isolated NVMe cache are ready. Serial execution avoids CPU, memory, and SSD
# contention in the latency measurements.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-formal-ee5283b"
ready_file="${artifact_root}/moe_infinity_environment.ready"
status_file="${artifact_root}/moe_infinity_performance_gpu0.status"
formal_root="/home/kec23008/DynaQuant-moe-formal"
official_repo="/home/kec23008/third_party/MoE-Infinity-ba56518"
venv_dir="/home/kec23008/.venvs/moe-infinity-ba56518"
model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned"
cache_dir="/mnt/oldroot/data/dynaexq-moe-infinity-cache/gpu0"
validator="/home/kec23008/DynaQuant/scripts/validate_moe_infinity_artifact.py"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

wait_for_environment() {
    while [[ ! -s "${ready_file}" && ! -e "${ready_file}" ]]; do
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

wait_for_gpu0_idle() {
    consecutive=0
    while ((consecutive < 3)); do
        if gpu0_idle; then
            consecutive=$((consecutive + 1))
            record_status "GPU0_IDLE_SAMPLE pass=${consecutive}/3"
        else
            consecutive=0
            record_status "WAIT_GPU0_IDLE"
        fi
        if ((consecutive < 3)); then
            sleep 30
        fi
    done
}

validate_artifact() {
    batch=$1
    output=$2
    "${venv_dir}/bin/python" "${validator}" \
        --formal-root "${formal_root}" \
        --artifact "${output}" \
        --batch-size "${batch}"
}

commit_artifact() {
    batch=$1
    output=$2
    relative=${output#"${formal_root}/"}
    git -C "${formal_root}" add -- "${relative}"
    git -C "${formal_root}" commit \
        -m "Record MoE-Infinity batch ${batch} result"
}

run_point() {
    batch=$1
    output="${formal_root}/results/paper/qwen30b_moe_infinity_bs${batch}.json"
    log="${artifact_root}/qwen30b_moe_infinity_bs${batch}.log"
    if [[ -s "${output}" ]]; then
        validate_artifact "${batch}" "${output}"
        if [[ -n "$(git -C "${formal_root}" status --short -- "${output}")" ]]; then
            commit_artifact "${batch}" "${output}"
        fi
        record_status "SKIP_VALID qwen30b moe_infinity bs=${batch}"
        return
    fi
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    wait_for_gpu0_idle
    record_status "START qwen30b moe_infinity bs=${batch}"
    (
        cd "${formal_root}"
        PATH="${venv_dir}/bin:${PATH}" \
        CUDA_VISIBLE_DEVICES=0 \
        bash scripts/reproduce_paper.sh moe-infinity \
            --model "${model}" \
            --repo "${official_repo}" \
            --offload-dir "${cache_dir}" \
            --batch-size "${batch}" \
            --output "${output}"
    ) >"${log}" 2>&1
    validate_artifact "${batch}" "${output}"
    commit_artifact "${batch}" "${output}"
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    record_status "PASS qwen30b moe_infinity bs=${batch}"
}

record_status "MOE_INFINITY_PERFORMANCE_SUPERVISOR_STARTED"
wait_for_environment
record_status "MOE_INFINITY_ENVIRONMENT_READY"
for batch in 1 2 4 8 16 32; do
    run_point "${batch}"
done
record_status "MOE_INFINITY_PERFORMANCE_COMPLETE"
