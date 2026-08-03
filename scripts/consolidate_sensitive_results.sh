#!/usr/bin/env bash
# Commit the 24 raw latency-bearing mechanism artifacts after the isolated
# serial queue completes. Only experiment JSON files are written.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-phi-formal"
artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
queue_status="${artifact_root}/sensitive_queue_gpu0.status"
status_file="${artifact_root}/sensitive_consolidation.status"
ready_file="${artifact_root}/sensitive_consolidation.complete"
validator="/home/kec23008/DynaQuant/scripts/validate_sensitive_artifact.py"

mkdir -p "${artifact_root}" "${persistent_root}"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    rsync -a "${artifact_root}/" "${persistent_root}/"
    exit "${exit_code}"
}
trap fail_status ERR

wait_for_queue() {
    while true; do
        if [[ -s "${queue_status}" ]] &&
           tail -n 1 "${queue_status}" |
               grep -q 'SENSITIVE_QUEUE_COMPLETE'; then
            return
        fi
        sleep 60
    done
}

queue_commit() {
    sed -n 's/.* expected_commit=\([0-9a-f]\{40\}\).*/\1/p' \
        "${queue_status}" | tail -n 1
}

validate_and_copy() {
    name=$1
    kind=$2
    model=$3
    expected_commit=$4
    shift 4
    python "${validator}" \
        --artifact "${artifact_root}/${name}" \
        --kind "${kind}" \
        --paper-model "${model}" \
        --expected-commit "${expected_commit}" "$@"
    destination="${formal_root}/results/paper/${name}"
    if [[ -e "${destination}" ]]; then
        cmp -s -- "${artifact_root}/${name}" "${destination}"
    else
        cp -- "${artifact_root}/${name}" "${destination}"
    fi
}

record_status "SENSITIVE_CONSOLIDATION_SUPERVISOR_STARTED"
wait_for_queue
expected_commit=$(queue_commit)
[[ -n "${expected_commit}" ]]
[[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
mkdir -p "${formal_root}/results/paper"

for model in qwen30b qwen80b; do
    for mode in full static blocking no_hysteresis; do
        validate_and_copy "${model}_ablation_${mode}.json" \
            ablation "${model}" "${expected_commit}" --ablation-config "${mode}"
    done
    validate_and_copy "${model}_runtime_overhead.json" \
        overhead "${model}" "${expected_commit}"
    for ratio in 0 5 10 15 20 25 30; do
        validate_and_copy "${model}_budget_ratio${ratio}.json" \
            sensitivity "${model}" "${expected_commit}" --ratio "${ratio}"
    done
done

git -C "${formal_root}" add -- results/paper
if ! git -C "${formal_root}" diff --cached --quiet; then
    git -C "${formal_root}" commit \
        -m "Record formal ablation and sensitivity data"
fi
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
touch "${ready_file}"
rsync -a "${artifact_root}/" "${persistent_root}/"
record_status "SENSITIVE_CONSOLIDATION_COMPLETE"
