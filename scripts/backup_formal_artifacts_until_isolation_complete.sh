#!/usr/bin/env bash
# Persist the tmpfs experiment artifacts through the final Qwen80 isolation
# backfill. The original backup session exits with the two main GPU lanes,
# before the chained isolation reruns begin.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-formal-ee5283b"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/ee5283b"
final_status="${artifact_root}/native_qwen80_bs4_isolation_backfill_gpu0.status"

mkdir -p "${persistent_root}"

isolation_complete() {
    [[ -s "${final_status}" ]] &&
        tail -n 1 "${final_status}" |
            grep -q 'QWEN80_BS4_ISOLATION_BACKFILL_COMPLETE$'
}

while ! isolation_complete; do
    rsync -a "${artifact_root}/" "${persistent_root}/"
    sleep 300
done

rsync -a "${artifact_root}/" "${persistent_root}/"
