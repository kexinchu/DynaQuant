#!/usr/bin/env bash
# Run one half of the two-GPU native performance queue. Lane "static" uses
# one physical GPU for static PTQ; lane "dynamic" uses the other for DynaExQ.
# The formal worktree remains clean throughout provenance capture.

set -euo pipefail

lane=${1:?usage: run_dual_native_perf_lane.sh static|dynamic physical_gpu}
physical_gpu=${2:?usage: run_dual_native_perf_lane.sh static|dynamic physical_gpu}
if [[ "${lane}" != "static" && "${lane}" != "dynamic" ]]; then
    exit 2
fi
if [[ "${physical_gpu}" != "0" && "${physical_gpu}" != "1" ]]; then
    exit 2
fi

formal_root="/home/kec23008/DynaQuant-formal"
artifact_root="/dev/shm/dynaexq-formal-ee5283b"
expected_commit="ee5283bfacf12428b5a6fcff284ddb4eb28a9cb9"
status_file="${artifact_root}/native_${lane}_gpu${physical_gpu}.status"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

signal_status() {
    signal_name=$1
    exit_code=$2
    trap - ERR HUP INT TERM
    record_status "FAILED signal=${signal_name} exit=${exit_code}"
    exit "${exit_code}"
}
trap 'signal_status HUP 129' HUP
trap 'signal_status INT 130' INT
trap 'signal_status TERM 143' TERM

archive_existing_log() {
    log_path=$1
    if [[ -s "${log_path}" ]]; then
        timestamp=$(date +%Y%m%dT%H%M%S)
        archived="${log_path%.log}.attempt-${timestamp}.log"
        mv -- "${log_path}" "${archived}"
        record_status "ARCHIVE_LOG source=${log_path} archived=${archived}"
    fi
}

wait_for_session_end() {
    session_name=$1
    while tmux has-session -t "${session_name}" 2>/dev/null; do
        sleep 30
    done
}

wait_for_gpu_idle() {
    while true; do
        sample=$(nvidia-smi -i "${physical_gpu}" \
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

validate_smoke_artifact() {
    artifact=$1
    batch=$2
    output_tokens=$3
    ARTIFACT="${artifact}" PAPER_BATCH="${batch}" OUTPUT_TOKENS="${output_tokens}" \
    python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["ARTIFACT"])
data = json.loads(path.read_text(encoding="utf-8"))
benchmark = data.get("benchmark", {})
problems = []
if benchmark.get("batch_size") != int(os.environ["PAPER_BATCH"]):
    problems.append("batch mismatch")
if benchmark.get("input_tokens") != 2048:
    problems.append("input length mismatch")
if benchmark.get("output_tokens_per_sequence") != int(os.environ["OUTPUT_TOKENS"]):
    problems.append("output length mismatch")
if benchmark.get("measured_iterations") != 1:
    problems.append("smoke iteration mismatch")
if len(benchmark.get("samples", [])) != 1:
    problems.append("smoke sample count mismatch")
if problems:
    raise SystemExit("; ".join(problems))
PY
}

configure_model() {
    model_key=$1
    case "${model_key}" in
        qwen30b)
            static_model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
            static_quant="int4"
            dynamic_model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned"
            dynamic_config="dynaexq/configs/qwen30b.yaml"
            initial_map="/dev/shm/dynaexq-formal-bbdaa87/qwen30b_initial_map.json"
            ;;
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
    if [[ "${lane}" == "static" ]]; then
        [[ -d "${static_model}" ]]
    else
        [[ -d "${dynamic_model}" ]]
        [[ -s "${initial_map}" ]]
    fi
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
    if [[ -s "${output}" ]]; then
        if [[ "${mode}" == "formal" ]]; then
            validate_performance_artifact "${output}" "${model_key}" static_ptq "${batch}"
        else
            validate_smoke_artifact "${output}" "${batch}" "${output_tokens}"
        fi
        record_status "SKIP_VALID ${model_key} static_${mode} bs=${batch} output_tokens=${output_tokens}"
        return
    fi
    record_status "START ${model_key} static_${mode} bs=${batch} output_tokens=${output_tokens}"
    wait_for_gpu_idle
    require_clean_formal_tree
    archive_existing_log "${log}"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" bash scripts/reproduce_paper.sh perf \
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
    if [[ -s "${output}" ]]; then
        if [[ "${mode}" == "formal" ]]; then
            validate_performance_artifact "${output}" "${model_key}" dynaexq "${batch}"
        else
            validate_smoke_artifact "${output}" "${batch}" 256
        fi
        record_status "SKIP_VALID ${model_key} dynaexq_${mode} bs=${batch}"
        return
    fi
    record_status "START ${model_key} dynaexq_${mode} bs=${batch}"
    wait_for_gpu_idle
    require_clean_formal_tree
    archive_existing_log "${log}"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" bash scripts/reproduce_paper.sh dynamic \
            --config "${dynamic_config}" \
            --model-path "${dynamic_model}" \
            --output "${output}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu "${physical_gpu}" \
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

record_status "QUEUE_STARTED lane=${lane} physical_gpu=${physical_gpu}"
if [[ "${lane}" == "static" ]]; then
    record_status "WAIT existing qwen30b static bs4"
    wait_for_session_end dynaexq_q30_static_b4
    validate_performance_artifact \
        "${artifact_root}/qwen30b_static_int4_bs4.json" qwen30b static_ptq 4
    record_status "PASS existing qwen30b static bs4"
else
    record_status "WAIT existing qwen30b dynaexq bs4"
    wait_for_session_end dynaexq_q30_dynamic_b4_gpu1
    validate_performance_artifact \
        "${artifact_root}/qwen30b_dynaexq_bs4.json" qwen30b dynaexq 4
    record_status "PASS existing qwen30b dynaexq bs4"
fi

configure_model qwen30b
for batch in 8 16 32; do
    if [[ "${lane}" == "static" ]]; then
        run_static qwen30b "${batch}" 8 smoke
        run_static qwen30b "${batch}" 256 smoke
        run_static qwen30b "${batch}" 256 formal
    else
        run_dynamic qwen30b "${batch}" smoke
        run_dynamic qwen30b "${batch}" formal
    fi
done

for model_key in qwen80b phi35; do
    configure_model "${model_key}"
    for batch in 1 2 4 8 16 32; do
        if [[ "${lane}" == "static" ]]; then
            run_static "${model_key}" "${batch}" 8 smoke
            run_static "${model_key}" "${batch}" 256 smoke
            run_static "${model_key}" "${batch}" 256 formal
        else
            run_dynamic "${model_key}" "${batch}" smoke
            run_dynamic "${model_key}" "${batch}" formal
        fi
    done
done

record_status "QUEUE_COMPLETE lane=${lane} physical_gpu=${physical_gpu}"
