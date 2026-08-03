#!/usr/bin/env bash
# Collect the performance-only Qwen3-Next-80B batch-32 telemetry used by the
# manuscript runtime-overhead table. No quality benchmark is run here.

set -euo pipefail

formal_root="/dev/shm/dynaexq-table-ee5283b"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
archive_root="/home/kec23008/DynaQuant-experiment-artifacts/ee5283b"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
physical_gpu=1
output="${artifact_root}/qwen80b_dynaexq_bs32.json"
log="${artifact_root}/qwen80b_dynaexq_bs32.log"
status="${artifact_root}/qwen80b_dynaexq_bs32_table.status"
lock="${artifact_root}/qwen80b_dynaexq_bs32_table.lock"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

exec 9>"${lock}"
flock -n 9

if [[ -s "${output}" ]]; then
    python /home/kec23008/DynaQuant/scripts/summarize_performance_artifact.py \
        "${output}"
    record_status "SKIP_VALID artifact=${output}"
    exit 0
fi

[[ -d "${formal_root}" ]]
[[ -d "/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound" ]]
[[ -s "/dev/shm/dynaexq-formal-bbdaa87/qwen80b_initial_map.json" ]]
[[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]

while true; do
    sample=$(nvidia-smi -i "${physical_gpu}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits)
    used=${sample%%,*}
    util=${sample##*,}
    used=${used//[[:space:]]/}
    util=${util//[[:space:]]/}
    if [[ "${used}" -le 1024 && "${util}" -eq 0 ]]; then
        break
    fi
    record_status "WAIT_GPU used_mib=${used} utilization_pct=${util}"
    sleep 30
done

if [[ -s "${log}" ]]; then
    mv -- "${log}" "${log%.log}.attempt-$(date +%Y%m%dT%H%M%S).log"
fi

record_status "START model=qwen80b method=dynaexq bs=32 repeats=100 gpu=${physical_gpu}"
(
    cd "${formal_root}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" bash scripts/reproduce_paper.sh dynamic \
        --config dynaexq/configs/qwen3_next_80b.yaml \
        --model-path /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound \
        --output "${output}" \
        --device cuda:0 \
        --hash-model-files \
        --wait-for-idle-physical-gpu "${physical_gpu}" \
        --initial-map /dev/shm/dynaexq-formal-bbdaa87/qwen80b_initial_map.json \
        perf \
        --batch-size 32 \
        --input-length 2048 \
        --output-length 256 \
        --paper-protocol \
        --n-warmup 5 \
        --n-repeats 100
) >"${log}" 2>&1

python /home/kec23008/DynaQuant/scripts/summarize_performance_artifact.py \
    "${output}"
mkdir -p "${archive_root}"
cp --preserve=timestamps -- "${output}" "${archive_root}/"
record_status "PASS artifact=${output}"
