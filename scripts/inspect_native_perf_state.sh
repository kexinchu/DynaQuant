#!/usr/bin/env bash
# Read-only snapshot/polling helper for the active two-GPU native benchmark
# queues. It does not create, stop, restart, or otherwise mutate experiments.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-formal-ee5283b"
count=${1:-1}
interval=${2:-0}

if ! [[ "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "count must be a positive integer" >&2
    exit 2
fi
if ! [[ "${interval}" =~ ^[0-9]+$ ]]; then
    echo "interval must be a non-negative integer" >&2
    exit 2
fi

for ((sample = 1; sample <= count; sample++)); do
    date '+%F %T'
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    mapfile -t benchmark_pids < <(
        pgrep -f 'dynaexq.experiments.eval_(perf|dynamic)' || true
    )
    if ((${#benchmark_pids[@]} > 0)); then
        pid_list=$(IFS=,; echo "${benchmark_pids[*]}")
        ps -o pid=,etime=,%cpu=,rss=,stat=,cmd= -p "${pid_list}" --cols 220
    fi
    for result in \
        "${artifact_root}/qwen30b_static_int4_bs32.json" \
        "${artifact_root}/qwen80b_static_int2_bs1.json" \
        "${artifact_root}/qwen80b_static_int2_bs1_isolated_rerun.json" \
        "${artifact_root}/qwen80b_static_int2_bs2.json" \
        "${artifact_root}/qwen80b_static_int2_bs2_isolated_rerun.json" \
        "${artifact_root}/qwen80b_static_int2_bs4.json" \
        "${artifact_root}/qwen80b_static_int2_bs4_isolated_rerun.json" \
        "${artifact_root}/qwen80b_dynaexq_bs2.json"; do
        if [[ -s "${result}" ]]; then
            stat -c 'RESULT %y %s %n' "${result}"
        fi
    done
    find "${artifact_root}" -maxdepth 1 -type f -name '*.json' -mmin -10 \
        -printf 'RECENT_RESULT %TY-%Tm-%Td %TH:%TM:%TS %s %f\n' \
        | sort
    echo 'STATIC_STATUS'
    tail -n 4 "${artifact_root}/native_static_gpu0.status"
    echo 'DYNAMIC_STATUS'
    tail -n 4 "${artifact_root}/native_dynamic_gpu1.status"
    if [[ -s "${artifact_root}/native_static_backfill_gpu0.status" ]]; then
        echo 'STATIC_BACKFILL_STATUS'
        tail -n 2 "${artifact_root}/native_static_backfill_gpu0.status"
    fi
    if [[ -s "${artifact_root}/native_isolation_backfill_gpu0.status" ]]; then
        echo 'ISOLATION_BACKFILL_STATUS'
        tail -n 2 "${artifact_root}/native_isolation_backfill_gpu0.status"
    fi
    if [[ -s "${artifact_root}/native_qwen80_isolation_backfill_gpu0.status" ]]; then
        echo 'QWEN80_ISOLATION_BACKFILL_STATUS'
        tail -n 2 \
            "${artifact_root}/native_qwen80_isolation_backfill_gpu0.status"
    fi
    if [[ -s "${artifact_root}/native_qwen80_bs2_isolation_backfill_gpu0.status" ]]; then
        echo 'QWEN80_BS2_ISOLATION_BACKFILL_STATUS'
        tail -n 2 \
            "${artifact_root}/native_qwen80_bs2_isolation_backfill_gpu0.status"
    fi
    if [[ -s "${artifact_root}/native_qwen80_bs4_isolation_backfill_gpu0.status" ]]; then
        echo 'QWEN80_BS4_ISOLATION_BACKFILL_STATUS'
        tail -n 2 \
            "${artifact_root}/native_qwen80_bs4_isolation_backfill_gpu0.status"
    fi
    if ((sample < count && interval > 0)); then
        sleep "${interval}"
    fi
done
