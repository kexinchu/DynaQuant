#!/usr/bin/env bash
# Run the Qwen3-Next and Phi-3.5 formal performance grids after the
# Qwen3-30B queue has completed. This operational script lives outside the
# clean formal worktree used for provenance capture.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
status_file="${artifact_root}/remaining_native_perf_queue.status"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

wait_for_session_end() {
    session_name=$1
    while tmux has-session -t "${session_name}" 2>/dev/null; do
        sleep 30
    done
}

wait_for_gpu_idle() {
    while true; do
        sample=$(nvidia-smi -i 0 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits)
        used=${sample%%,*}
        util=${sample##*,}
        used=${used//[[:space:]]/}
        util=${util//[[:space:]]/}
        if [[ "${used}" -le 1024 && "${util}" -eq 0 ]]; then
            return
        fi
        record_status "WAIT_GPU used_mib=${used} utilization_pct=${util}"
        sleep 30
    done
}

require_clean_formal_tree() {
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
}

validate_performance_artifact() {
    artifact=$1
    model=$2
    method=$3
    batch=$4
    FORMAL_ROOT="${formal_root}" ARTIFACT="${artifact}" \
    PAPER_MODEL="${model}" PAPER_METHOD="${method}" PAPER_BATCH="${batch}" \
    python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FORMAL_ROOT"])
sys.path.insert(0, str(root))
from scripts.audit_paper_results import (  # noqa: E402
    _validate_performance_benchmark,
    validate_dynamic_runtime,
)

path = Path(os.environ["ARTIFACT"])
data = json.loads(path.read_text(encoding="utf-8"))
model = os.environ["PAPER_MODEL"]
method = os.environ["PAPER_METHOD"]
batch = int(os.environ["PAPER_BATCH"])
problems = _validate_performance_benchmark(str(path), data.get("benchmark"))
if data.get("paper_model") != model:
    problems.append("paper model mismatch")
if data.get("paper_method") != method:
    problems.append("paper method mismatch")
benchmark = data.get("benchmark", {})
if benchmark.get("batch_size") != batch:
    problems.append("batch mismatch")
if benchmark.get("input_tokens") != 2048:
    problems.append("input length mismatch")
if benchmark.get("output_tokens_per_sequence") != 256:
    problems.append("output length mismatch")
git = data.get("environment", {}).get("git", {})
if not git.get("commit") or git.get("dirty") is not False:
    problems.append("dirty or missing git provenance")
if method == "dynaexq":
    problems.extend(validate_dynamic_runtime(str(path), data))
if problems:
    raise SystemExit("; ".join(problems))
PY
}

configure_model() {
    model_key=$1
    case "${model_key}" in
        qwen80b)
            static_model="/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int2-from-int4-formal"
            static_quant="int2"
            dynamic_model="/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound"
            dynamic_config="dynaexq/configs/qwen3_next_80b.yaml"
            initial_map="/dev/shm/dynaexq-formal-bbdaa87/qwen80b_initial_map.json"
            ;;
        phi35)
            static_model="/home/kec23008/Models/Phi-3.5-MoE-instruct-W4A16-AutoRound-formal"
            static_quant="int4"
            dynamic_model="/dev/shm/dynaexq-models/phi35-moe-bf16"
            dynamic_config="dynaexq/configs/phi35_moe.yaml"
            initial_map="/dev/shm/dynaexq-formal-bbdaa87/phi35_initial_map.json"
            ;;
        *)
            return 2
            ;;
    esac
    [[ -d "${static_model}" ]]
    [[ -d "${dynamic_model}" ]]
    [[ -s "${initial_map}" ]]
}

run_static() {
    model_key=$1
    batch=$2
    output_tokens=$3
    mode=$4
    stem="${artifact_root}/${model_key}_static_${static_quant}_bs${batch}"
    if [[ "${mode}" == "smoke" ]]; then
        output="${stem}_${output_tokens}_smoke.json"
        log="${stem}_${output_tokens}_smoke.log"
        protocol_args=(--n-warmup 0 --n-repeats 1)
    else
        output="${stem}.json"
        log="${stem}.log"
        protocol_args=(--paper-protocol --n-warmup 5 --n-repeats 100 --hash-model-files)
    fi
    record_status "START ${model_key} static_${mode} bs=${batch} output_tokens=${output_tokens}"
    wait_for_gpu_idle
    require_clean_formal_tree
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh perf \
            --model "${static_model}" \
            --paper-model "${model_key}" \
            --method quantized_checkpoint \
            --quantization "${static_quant}" \
            --autoround-backend triton \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length "${output_tokens}" \
            --device-map cuda:0 \
            "${protocol_args[@]}" \
            --output "${output}"
    ) >"${log}" 2>&1
    if [[ "${mode}" == "formal" ]]; then
        validate_performance_artifact "${output}" "${model_key}" static_ptq "${batch}"
    fi
    record_status "PASS ${model_key} static_${mode} bs=${batch} output_tokens=${output_tokens}"
}

run_dynamic() {
    model_key=$1
    batch=$2
    mode=$3
    stem="${artifact_root}/${model_key}_dynaexq_bs${batch}"
    if [[ "${mode}" == "smoke" ]]; then
        output="${stem}_256_smoke.json"
        log="${stem}_256_smoke.log"
        protocol_args=(--n-warmup 0 --n-repeats 1)
    else
        output="${stem}.json"
        log="${stem}.log"
        protocol_args=(--paper-protocol --n-warmup 5 --n-repeats 100)
    fi
    record_status "START ${model_key} dynaexq_${mode} bs=${batch}"
    wait_for_gpu_idle
    require_clean_formal_tree
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
            --config "${dynamic_config}" \
            --model-path "${dynamic_model}" \
            --output "${output}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu 0 \
            --initial-map "${initial_map}" \
            perf \
            --batch-size "${batch}" \
            --input-length 2048 \
            --output-length 256 \
            "${protocol_args[@]}"
    ) >"${log}" 2>&1
    if [[ "${mode}" == "formal" ]]; then
        validate_performance_artifact "${output}" "${model_key}" dynaexq "${batch}"
    fi
    record_status "PASS ${model_key} dynaexq_${mode} bs=${batch}"
}

record_status "QUEUE_STARTED"
record_status "WAIT qwen30b_queue"
wait_for_session_end dynaexq_q30_queue
grep -q 'QUEUE_COMPLETE$' "${artifact_root}/qwen30b_queue.status"
record_status "PASS qwen30b_queue"

for model_key in qwen80b phi35; do
    configure_model "${model_key}"
    for batch in 1 2 4 8 16 32; do
        run_static "${model_key}" "${batch}" 8 smoke
        run_static "${model_key}" "${batch}" 256 smoke
        run_static "${model_key}" "${batch}" 256 formal
        run_dynamic "${model_key}" "${batch}" smoke
        run_dynamic "${model_key}" "${batch}" formal
    done
done

record_status "QUEUE_COMPLETE"
