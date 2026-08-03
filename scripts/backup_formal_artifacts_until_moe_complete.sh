#!/usr/bin/env bash
# Persist tmpfs logs, status files, and validation summaries through the final
# official MoE-Infinity performance point. Result JSON files from that queue
# are also committed in its dedicated formal worktree.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-formal-ee5283b"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/ee5283b"
final_status="${artifact_root}/moe_infinity_performance_gpu0.status"

mkdir -p "${persistent_root}"

moe_complete() {
    [[ -s "${final_status}" ]] &&
        tail -n 1 "${final_status}" |
            grep -q 'MOE_INFINITY_PERFORMANCE_COMPLETE$'
}

while ! moe_complete; do
    rsync -a "${artifact_root}/" "${persistent_root}/"
    sleep 300
done

rsync -a "${artifact_root}/" "${persistent_root}/"
