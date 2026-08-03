#!/usr/bin/env bash
# Commit the nine immutable quality artifacts after both GPU lanes finish, then
# derive and commit the three paired-significance artifacts. This worktree is
# dedicated to experiment evidence; manuscript sources and figures are not
# touched.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-phi-formal"
artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
expected_commit="a1dd7362a595560af910002a7e2de63907a6ea23"
status_file="${artifact_root}/quality_consolidation.status"
ready_file="${artifact_root}/quality_consolidation.complete"
validator="/home/kec23008/DynaQuant/scripts/validate_quality_artifact.py"
lane0_status="${artifact_root}/quality_q30_phi_gpu0.status"
lane1_status="${artifact_root}/quality_q80_gpu1.status"

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
        tail -n 1 "${status}" | grep -q "QUALITY_LANE_COMPLETE lane=${lane}"
}

wait_for_lanes() {
    while ! lane_complete "${lane0_status}" q30_phi ||
          ! lane_complete "${lane1_status}" q80; do
        sleep 60
    done
}

validate_quality() {
    name=$1
    model=$2
    method=$3
    python "${validator}" \
        --formal-root "${formal_root}" \
        --artifact "${artifact_root}/${name}" \
        --paper-model "${model}" \
        --paper-method "${method}" \
        --expected-commit "${expected_commit}"
}

copy_quality() {
    name=$1
    model=$2
    method=$3
    validate_quality "${name}" "${model}" "${method}"
    cp -- "${artifact_root}/${name}" "${formal_root}/results/paper/${name}"
}

commit_if_needed() {
    message=$1
    shift
    git -C "${formal_root}" add -- "$@"
    if ! git -C "${formal_root}" diff --cached --quiet; then
        git -C "${formal_root}" commit -m "${message}"
    fi
}

build_significance() {
    model=$1
    left_name=$2
    right_name=$3
    output_name="${model}_static_ptq_vs_dynaexq_significance.json"
    (
        cd "${formal_root}"
        python scripts/compare_quality_artifacts.py \
            --paper-model "${model}" \
            --left "results/paper/${left_name}" \
            --right "results/paper/${right_name}" \
            --output "results/paper/${output_name}"
    )
    FORMAL_ROOT="${formal_root}" ARTIFACT="${formal_root}/results/paper/${output_name}" \
        PAPER_MODEL="${model}" python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.audit_paper_results import validate_manifest_artifact

artifact = Path(os.environ["ARTIFACT"])
model = os.environ["PAPER_MODEL"]
data = json.loads(artifact.read_text(encoding="utf-8"))
problems = validate_manifest_artifact(
    "quality_significance",
    str(artifact),
    data,
    f"quality_significance:{model}:static_ptq_vs_dynaexq",
)
if problems:
    raise SystemExit("; ".join(problems))
PY
    commit_if_needed "Record ${model} paired quality significance" \
        "results/paper/${output_name}"
}

record_status \
    "QUALITY_CONSOLIDATION_SUPERVISOR_STARTED expected_commit=${expected_commit}"
wait_for_lanes
record_status "QUALITY_LANES_COMPLETE"
[[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
mkdir -p "${formal_root}/results/paper"

copy_quality qwen30b_fp16_quality.json qwen30b reference_fp16
copy_quality qwen30b_int4_quality.json qwen30b static_int4
copy_quality qwen30b_dynaexq_quality.json qwen30b dynaexq
copy_quality qwen80b_int2_quality.json qwen80b static_int2
copy_quality qwen80b_int4_quality.json qwen80b static_int4
copy_quality qwen80b_dynaexq_quality.json qwen80b dynaexq
copy_quality phi35_fp16_quality.json phi35 reference_fp16
copy_quality phi35_int4_quality.json phi35 static_int4
copy_quality phi35_dynaexq_quality.json phi35 dynaexq
commit_if_needed "Record formal three-model quality results" results/paper
[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
record_status "QUALITY_ARTIFACTS_COMMITTED"

build_significance qwen30b \
    qwen30b_int4_quality.json qwen30b_dynaexq_quality.json
build_significance qwen80b \
    qwen80b_int2_quality.json qwen80b_dynaexq_quality.json
build_significance phi35 \
    phi35_int4_quality.json phi35_dynaexq_quality.json

[[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
touch "${ready_file}"
rsync -a "${artifact_root}/" "${persistent_root}/"
record_status "QUALITY_CONSOLIDATION_COMPLETE"
