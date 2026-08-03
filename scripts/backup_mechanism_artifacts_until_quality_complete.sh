#!/usr/bin/env bash
# Persist mechanism logs and JSON artifacts while the two quality lanes and
# their consolidation supervisor are active.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
final_status="${artifact_root}/quality_consolidation.status"

mkdir -p "${persistent_root}"

quality_complete() {
    [[ -s "${final_status}" ]] &&
        tail -n 1 "${final_status}" |
            grep -q 'QUALITY_CONSOLIDATION_COMPLETE$'
}

while ! quality_complete; do
    rsync -a "${artifact_root}/" "${persistent_root}/"
    sleep 300
done

rsync -a "${artifact_root}/" "${persistent_root}/"
