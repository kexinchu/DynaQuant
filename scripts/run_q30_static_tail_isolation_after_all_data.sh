#!/usr/bin/env bash
# Re-measure the two Qwen3-30B static points whose main-run p99/mean ratios
# show strong shared-machine tail contamination. This supplemental queue runs
# only after every other data experiment and with both GPUs plus CPU idle.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/ee5283b"
prerequisite="/dev/shm/dynaexq-mechanism-c7a9999/sensitive_consolidation.complete"
status_file="${artifact_root}/qwen30b_static_tail_isolation.status"
model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"

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

wait_for_prerequisite() {
    while [[ ! -e "${prerequisite}" ]]; do
        sleep 60
    done
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

wait_for_isolation() {
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

validate_artifact() {
    output=$1
    batch=$2
    FORMAL_ROOT="${formal_root}" ARTIFACT="${output}" PAPER_BATCH="${batch}" \
        python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.audit_paper_results import validate_manifest_artifact

artifact = Path(os.environ["ARTIFACT"])
batch = int(os.environ["PAPER_BATCH"])
data = json.loads(artifact.read_text(encoding="utf-8"))
problems = validate_manifest_artifact(
    "performance",
    str(artifact),
    data,
    f"performance:qwen30b:static_ptq:bs{batch}",
)
if problems:
    raise SystemExit("; ".join(problems))
if data.get("environment", {}).get("git", {}).get("commit") != (
    "ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
):
    raise SystemExit("unexpected formal source commit")
PY
}

run_point() {
    batch=$1
    output="${artifact_root}/qwen30b_static_int4_bs${batch}_isolated_tail_rerun.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        validate_artifact "${output}" "${batch}"
        record_status "SKIP_VALID qwen30b static bs=${batch} isolated_tail=true"
        return
    fi
    wait_for_isolation
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
    record_status "START qwen30b static bs=${batch} isolated_tail=true gpu=0"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
            --model "${model}" \
            --paper-model qwen30b \
            --method quantized_checkpoint \
            --quantization int4 \
            --autoround-backend triton \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length 256 \
            --device-map cuda:0 \
            --paper-protocol \
            --n-warmup 5 \
            --n-repeats 100 \
            --hash-model-files \
            --output "${output}"
    ) >"${log}" 2>&1
    validate_artifact "${output}" "${batch}"
    rsync -a "${artifact_root}/" "${persistent_root}/"
    record_status "PASS qwen30b static bs=${batch} isolated_tail=true"
}

record_status "QWEN30_TAIL_ISOLATION_SUPERVISOR_STARTED"
wait_for_prerequisite
record_status "ALL_PRIMARY_DATA_COMPLETE"
run_point 4
run_point 8
record_status "QWEN30_TAIL_ISOLATION_COMPLETE"
rsync -a "${artifact_root}/" "${persistent_root}/"
