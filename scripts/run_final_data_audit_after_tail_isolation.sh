#!/usr/bin/env bash
# Run the 83-claim non-figure data audit after the final supplemental isolation
# point. This supervisor never writes manuscript or figure files.

set -euo pipefail

native_root="/dev/shm/dynaexq-formal-ee5283b"
mechanism_root="/home/kec23008/DynaQuant-phi-formal/results/paper"
moe_root="/home/kec23008/DynaQuant-moe-formal/results/paper"
formal_root="/home/kec23008/DynaQuant-phi-formal"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/final-audit"
prerequisite="${native_root}/qwen30b_static_tail_isolation.status"
status_file="${native_root}/final_data_audit.status"
report="${persistent_root}/all_experiment_data_audit.json"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]} report=${report}"
    exit "${exit_code}"
}
trap fail_status ERR

while true; do
    if [[ -s "${prerequisite}" ]] &&
       tail -n 1 "${prerequisite}" |
           grep -q 'QWEN30_TAIL_ISOLATION_COMPLETE$'; then
        break
    fi
    sleep 60
done

record_status "FINAL_DATA_AUDIT_STARTED"
mkdir -p "${persistent_root}"
python /home/kec23008/DynaQuant/scripts/audit_all_experiment_data.py \
    --native-root "${native_root}" \
    --mechanism-root "${mechanism_root}" \
    --moe-root "${moe_root}" \
    --formal-root "${formal_root}" \
    --output "${report}"
record_status "FINAL_DATA_AUDIT_COMPLETE report=${report}"
