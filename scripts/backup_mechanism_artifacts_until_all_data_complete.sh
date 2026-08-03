#!/usr/bin/env bash
# Persist all mechanism-stage logs and JSON files through the final sensitive
# result consolidation.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
final_status="${artifact_root}/sensitive_consolidation.status"

mkdir -p "${persistent_root}"

all_data_complete() {
    [[ -s "${final_status}" ]] &&
        tail -n 1 "${final_status}" |
            grep -q 'SENSITIVE_CONSOLIDATION_COMPLETE$'
}

while ! all_data_complete; do
    rsync -a "${artifact_root}/" "${persistent_root}/"
    sleep 300
done

rsync -a "${artifact_root}/" "${persistent_root}/"
