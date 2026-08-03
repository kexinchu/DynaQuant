#!/usr/bin/env bash
# Keep the two idempotent native-performance lanes alive during unattended runs.
# A completed lane is never restarted; a failing lane is retried only a bounded
# number of times so a deterministic error cannot create a restart loop.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-formal-ee5283b"
queue_script="/home/kec23008/DynaQuant/scripts/run_dual_native_perf_lane.sh"
watch_log="${artifact_root}/native_queue_watchdog.log"
poll_seconds=60
retry_delay_seconds=300
max_restarts=3

declare -A session=(
    [static]="dynaexq_static_lane_gpu0"
    [dynamic]="dynaexq_dynamic_lane_gpu1"
)
declare -A gpu=(
    [static]="0"
    [dynamic]="1"
)
declare -A restarts=(
    [static]="0"
    [dynamic]="0"
)
declare -A retry_after=(
    [static]="0"
    [dynamic]="0"
)

record() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${watch_log}"
}

lane_complete() {
    local lane=$1
    local status_file="${artifact_root}/native_${lane}_gpu${gpu[${lane}]}.status"
    [[ -f "${status_file}" ]] && grep -q "QUEUE_COMPLETE lane=${lane}" "${status_file}"
}

start_lane() {
    local lane=$1
    local physical_gpu=${gpu[${lane}]}
    local tmux_session=${session[${lane}]}
    local resume_log="${artifact_root}/native_${lane}_gpu${physical_gpu}_watchdog_resume.log"

    tmux new-session -d -s "${tmux_session}" \
        "bash -c 'bash ${queue_script} ${lane} ${physical_gpu} >> ${resume_log} 2>&1'"
    restarts[${lane}]=$((restarts[${lane}] + 1))
    retry_after[${lane}]=$(( $(date +%s) + retry_delay_seconds ))
    record "RESTART lane=${lane} physical_gpu=${physical_gpu} attempt=${restarts[${lane}]}"
}

record "WATCHDOG_STARTED poll_seconds=${poll_seconds} max_restarts=${max_restarts}"
while true; do
    all_complete=1
    now=$(date +%s)
    for lane in static dynamic; do
        if lane_complete "${lane}"; then
            continue
        fi
        all_complete=0
        if tmux has-session -t "${session[${lane}]}" 2>/dev/null; then
            continue
        fi
        if (( restarts[${lane}] >= max_restarts )); then
            record "GAVE_UP lane=${lane} attempts=${restarts[${lane}]}"
            continue
        fi
        if (( now < retry_after[${lane}] )); then
            continue
        fi
        start_lane "${lane}"
    done
    if (( all_complete )); then
        record "WATCHDOG_COMPLETE"
        exit 0
    fi
    sleep "${poll_seconds}"
done
