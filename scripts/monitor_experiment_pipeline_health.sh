#!/usr/bin/env bash
# Read-only unattended health monitor for the complete experiment pipeline.
# It records stalls but never kills, restarts, or otherwise mutates a run.

set -euo pipefail

native_root="/dev/shm/dynaexq-formal-ee5283b"
mechanism_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/health"
health_log="${native_root}/complete_pipeline_health.log"
final_status="${native_root}/final_data_audit.status"
poll_seconds=300
stall_threshold=3

mkdir -p "${persistent_root}"
declare -A last_ticks=()
declare -A stale_samples=()

record() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${health_log}"
}

pipeline_complete() {
    [[ -s "${final_status}" ]] &&
        tail -n 1 "${final_status}" |
            grep -q 'FINAL_DATA_AUDIT_COMPLETE'
}

sample_processes() {
    mapfile -t pids < <(
        pgrep -f 'python .*((dynaexq\.experiments\.(eval_dynamic|eval_perf|eval_quality))|(benchmark_moe_infinity|collect_activation_density|collect_routing_active_set_trace|benchmark_blocking_offload))' || true
    )
    if [[ "${#pids[@]}" -eq 0 ]]; then
        record "PROCESS_SAMPLE active=0"
        return
    fi
    for pid in "${pids[@]}"; do
        [[ -r "/proc/${pid}/stat" ]] || continue
        ticks=$(awk '{print $14+$15}' "/proc/${pid}/stat")
        state=$(awk '{print $3}' "/proc/${pid}/stat")
        previous=${last_ticks[${pid}]:--1}
        stale=${stale_samples[${pid}]:-0}
        if [[ "${ticks}" -gt "${previous}" ]]; then
            stale=0
        else
            stale=$((stale + 1))
        fi
        last_ticks[${pid}]="${ticks}"
        stale_samples[${pid}]="${stale}"
        record "PROCESS_SAMPLE pid=${pid} state=${state} cpu_ticks=${ticks} stale_samples=${stale}"
        if [[ "${stale}" -ge "${stall_threshold}" ]]; then
            record "STALL_ALERT pid=${pid} stale_samples=${stale}"
        fi
    done
}

sample_pipeline() {
    gpu=$(nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits | tr '\n' ';')
    native_json=$(find "${native_root}" -maxdepth 1 -type f -name '*.json' | wc -l)
    mechanism_json=$(find "${mechanism_root}" -maxdepth 1 -type f -name '*.json' | wc -l)
    failures=0
    for status in "${native_root}"/*.status "${mechanism_root}"/*.status; do
        [[ -f "${status}" ]] || continue
        if tail -n 1 "${status}" | rg -q 'FAILED|GAVE_UP'; then
            failures=$((failures + 1))
        fi
    done
    record "PIPELINE_SAMPLE gpu=${gpu} native_json=${native_json} mechanism_json=${mechanism_json} failure_status_files=${failures}"
    sample_processes
    cp -- "${health_log}" "${persistent_root}/complete_pipeline_health.log"
}

record "HEALTH_MONITOR_STARTED poll_seconds=${poll_seconds} stall_threshold=${stall_threshold}"
while ! pipeline_complete; do
    sample_pipeline
    sleep "${poll_seconds}"
done
sample_pipeline
record "HEALTH_MONITOR_COMPLETE"
cp -- "${health_log}" "${persistent_root}/complete_pipeline_health.log"
