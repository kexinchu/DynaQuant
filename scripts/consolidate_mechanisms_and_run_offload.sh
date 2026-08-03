#!/usr/bin/env bash
# Consolidate raw mechanism data after both deterministic GPU lanes complete,
# build the two perplexity data curves, and run blocking-offload measurements
# serially with both GPUs otherwise idle. Manuscript and figure files are not
# modified.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-phi-formal"
artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
status_file="${artifact_root}/mechanism_consolidation.status"
ready_file="${artifact_root}/mechanism_consolidation.complete"
lane0_status="${artifact_root}/mechanism_q30_gpu0.status"
lane1_status="${artifact_root}/mechanism_q80_phi_gpu1.status"
raw_validator="/home/kec23008/DynaQuant/scripts/validate_raw_mechanism_artifact.py"

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

lane_complete() {
    status=$1
    lane=$2
    [[ -s "${status}" ]] &&
        tail -n 1 "${status}" |
            grep -q "MECHANISM_LANE_COMPLETE lane=${lane}"
}

wait_for_lanes() {
    while ! lane_complete "${lane0_status}" q30 ||
          ! lane_complete "${lane1_status}" q80_phi; do
        sleep 60
    done
}

lane_commit() {
    status=$1
    sed -n 's/.* expected_commit=\([0-9a-f]\{40\}\).*/\1/p' "${status}" |
        tail -n 1
}

require_clean_tree() {
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
}

commit_if_needed() {
    message=$1
    shift
    git -C "${formal_root}" add -- "$@"
    if ! git -C "${formal_root}" diff --cached --quiet; then
        git -C "${formal_root}" commit -m "${message}"
    fi
}

validate_raw() {
    name=$1
    kind=$2
    model=$3
    expected_commit=$4
    shift 4
    python "${raw_validator}" \
        --artifact "${artifact_root}/${name}" \
        --kind "${kind}" \
        --paper-model "${model}" \
        --expected-commit "${expected_commit}" "$@"
}

copy_raw() {
    name=$1
    kind=$2
    model=$3
    expected_commit=$4
    shift 4
    validate_raw "${name}" "${kind}" "${model}" "${expected_commit}" "$@"
    destination="${formal_root}/results/paper/${name}"
    if [[ -e "${destination}" ]]; then
        cmp -s -- "${artifact_root}/${name}" "${destination}"
    else
        cp -- "${artifact_root}/${name}" "${destination}"
    fi
}

validate_claim() {
    group=$1
    artifact=$2
    claim=$3
    FORMAL_ROOT="${formal_root}" ARTIFACT="${artifact}" GROUP="${group}" \
        CLAIM="${claim}" python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.audit_paper_results import validate_manifest_artifact

artifact = Path(os.environ["ARTIFACT"])
problems = validate_manifest_artifact(
    os.environ["GROUP"],
    str(artifact),
    json.loads(artifact.read_text(encoding="utf-8")),
    os.environ["CLAIM"],
)
if problems:
    raise SystemExit("; ".join(problems))
PY
}

build_curve() {
    model=$1
    output="${formal_root}/results/paper/${model}_perplexity_curve.json"
    if [[ ! -s "${output}" ]]; then
        require_clean_tree
        args=()
        for ratio in 0 15 30 45 60 75 90 100; do
            args+=(--point "results/paper/${model}_perplexity_ratio${ratio}.json")
        done
        (
            cd "${formal_root}"
            bash scripts/reproduce_paper.sh build-ppl-curve \
                --paper-model "${model}" "${args[@]}" --output "${output}"
        ) >"${artifact_root}/${model}_perplexity_curve.log" 2>&1
    fi
    validate_claim perplexity_curve "${output}" "perplexity_curve:${model}"
    commit_if_needed "Record ${model} perplexity curve data" \
        "results/paper/${model}_perplexity_curve.json"
    require_clean_tree
    record_status "PASS kind=perplexity_curve model=${model}"
}

all_gpus_idle() {
    while IFS=',' read -r used util; do
        used=${used//[[:space:]]/}
        util=${util//[[:space:]]/}
        if [[ "${used}" -gt 1024 || "${util}" -ne 0 ]]; then
            return 1
        fi
    done < <(
        nvidia-smi --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )
}

wait_for_isolated_machine() {
    consecutive=0
    while ((consecutive < 3)); do
        idle_pct=$(vmstat 1 2 | tail -n 1 | awk '{print $15}')
        if all_gpus_idle && [[ "${idle_pct}" -ge 80 ]]; then
            consecutive=$((consecutive + 1))
            record_status "ISOLATION_SAMPLE pass=${consecutive}/3 cpu_idle_pct=${idle_pct}"
        else
            consecutive=0
            record_status "WAIT_ISOLATION cpu_idle_pct=${idle_pct}"
        fi
        if ((consecutive < 3)); then
            sleep 30
        fi
    done
}

validate_trace_in_tree() {
    model=$1
    trace=$2
    FORMAL_ROOT="${formal_root}" PAPER_MODEL="${model}" TRACE="${trace}" \
        python - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.benchmark_blocking_offload import load_trace

load_trace(Path(os.environ["TRACE"]), paper_model=os.environ["PAPER_MODEL"])
PY
}

run_offload() {
    model=$1
    trace="${formal_root}/results/paper/${model}_routing_active_set_trace.json"
    output="${formal_root}/results/paper/${model}_offload_waiting.json"
    log="${artifact_root}/${model}_offload_waiting.log"
    validate_trace_in_tree "${model}" "${trace}"
    if [[ ! -s "${output}" ]]; then
        require_clean_tree
        wait_for_isolated_machine
        record_status "START kind=offload_waiting model=${model} gpu=0"
        (
            cd "${formal_root}"
            CUDA_VISIBLE_DEVICES=0 \
                bash scripts/reproduce_paper.sh offload-waiting \
                    --paper-model "${model}" \
                    --trace "results/paper/${model}_routing_active_set_trace.json" \
                    --device cuda:0 \
                    --output "${output}"
        ) >"${log}" 2>&1
    fi
    validate_claim offload_waiting "${output}" "offload_waiting:${model}"
    commit_if_needed "Record ${model} blocking-offload data" \
        "results/paper/${model}_offload_waiting.json"
    require_clean_tree
    rsync -a "${artifact_root}/" "${persistent_root}/"
    record_status "PASS kind=offload_waiting model=${model}"
}

record_status "MECHANISM_CONSOLIDATION_SUPERVISOR_STARTED"
wait_for_lanes
q30_commit=$(lane_commit "${lane0_status}")
q80_phi_commit=$(lane_commit "${lane1_status}")
[[ -n "${q30_commit}" && "${q30_commit}" == "${q80_phi_commit}" ]]
require_clean_tree
record_status "MECHANISM_LANES_COMPLETE source_commit=${q30_commit}"
mkdir -p "${formal_root}/results/paper"

copy_raw qwen30b_routing_active_set_trace.json routing_trace qwen30b "${q30_commit}"
copy_raw qwen30b_routing_hotset.json routing_hotset qwen30b "${q30_commit}"
copy_raw qwen30b_activation_density.json activation qwen30b "${q30_commit}"
copy_raw qwen80b_routing_active_set_trace.json routing_trace qwen80b "${q30_commit}"
copy_raw qwen80b_activation_density.json activation qwen80b "${q30_commit}"
copy_raw phi35_routing_active_set_trace.json routing_trace phi35 "${q30_commit}"
copy_raw phi35_activation_density.json activation phi35 "${q30_commit}"
for model in qwen30b qwen80b; do
    for ratio in 0 15 30 45 60 75 90 100; do
        copy_raw "${model}_perplexity_ratio${ratio}.json" \
            perplexity_point "${model}" "${q30_commit}" --ratio "${ratio}"
    done
done
commit_if_needed "Record raw mechanism experiment data" results/paper
require_clean_tree
record_status "RAW_MECHANISM_ARTIFACTS_COMMITTED"

build_curve qwen30b
build_curve qwen80b
run_offload qwen30b
run_offload qwen80b
run_offload phi35

touch "${ready_file}"
rsync -a "${artifact_root}/" "${persistent_root}/"
record_status "MECHANISM_CONSOLIDATION_COMPLETE"
