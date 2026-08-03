#!/usr/bin/env bash
# Run quality-only experiments on one GPU after all latency-sensitive native
# and MoE-Infinity measurements. The lanes run concurrently for all one-GPU
# points. Oversized Qwen3-30B and Phi-3.5 FP16 references wait for q80
# completion and then use both GPUs serially.

set -euo pipefail

lane=${1:?usage: run_formal_quality_lane_after_moe.sh q30_phi|q80 physical_gpu}
physical_gpu=${2:?usage: run_formal_quality_lane_after_moe.sh q30_phi|q80 physical_gpu}
if [[ "${lane}" != "q30_phi" && "${lane}" != "q80" ]]; then
    exit 2
fi
if [[ "${physical_gpu}" != "0" && "${physical_gpu}" != "1" ]]; then
    exit 2
fi
if [[ "${lane}:${physical_gpu}" != "q30_phi:0" &&
      "${lane}:${physical_gpu}" != "q80:1" ]]; then
    exit 2
fi

formal_root="/home/kec23008/DynaQuant-phi-formal"
source_artifact_root="/dev/shm/dynaexq-formal-ee5283b"
artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
expected_commit="a1dd7362a595560af910002a7e2de63907a6ea23"
prerequisite_status="${source_artifact_root}/moe_infinity_performance_gpu0.status"
status_file="${artifact_root}/quality_${lane}_gpu${physical_gpu}.status"
q80_status="${artifact_root}/quality_q80_gpu1.status"
validator="/home/kec23008/DynaQuant/scripts/validate_quality_artifact.py"
benchmarks="mmlu_pro,gpqa,aime25,gsm8k,humaneval"

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

wait_for_prerequisite() {
    while true; do
        if [[ -s "${prerequisite_status}" ]] &&
           tail -n 1 "${prerequisite_status}" |
               grep -q 'MOE_INFINITY_PERFORMANCE_COMPLETE$'; then
            return
        fi
        sleep 60
    done
}

wait_for_gpu_idle() {
    consecutive=0
    while ((consecutive < 3)); do
        sample=$(nvidia-smi -i "${physical_gpu}" \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits)
        used=${sample%%,*}
        util=${sample##*,}
        used=${used//[[:space:]]/}
        util=${util//[[:space:]]/}
        if [[ "${used}" -le 1024 && "${util}" -eq 0 ]]; then
            consecutive=$((consecutive + 1))
        else
            consecutive=0
        fi
        if ((consecutive < 3)); then
            sleep 30
        fi
    done
}

wait_for_q80_lane_complete() {
    while true; do
        if [[ -s "${q80_status}" ]] &&
           tail -n 1 "${q80_status}" |
               grep -q 'QUALITY_LANE_COMPLETE lane=q80 gpu=1$'; then
            return
        fi
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

wait_for_all_gpus_idle() {
    consecutive=0
    while ((consecutive < 3)); do
        if all_gpus_idle; then
            consecutive=$((consecutive + 1))
        else
            consecutive=0
        fi
        if ((consecutive < 3)); then
            sleep 30
        fi
    done
}

require_clean_tree() {
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
}

validate_artifact() {
    output=$1
    model=$2
    method=$3
    python "${validator}" \
        --formal-root "${formal_root}" \
        --artifact "${output}" \
        --paper-model "${model}" \
        --paper-method "${method}" \
        --expected-commit "${expected_commit}"
}

run_static_quality() {
    model=$1
    method=$2
    checkpoint=$3
    quantization=${4:-}
    output="${artifact_root}/${model}_${method#static_}_quality.json"
    if [[ "${method}" == "reference_fp16" ]]; then
        output="${artifact_root}/${model}_fp16_quality.json"
    fi
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        validate_artifact "${output}" "${model}" "${method}"
        record_status "SKIP_VALID model=${model} method=${method}"
        return
    fi
    visible_devices="${physical_gpu}"
    two_gpu_reference=0
    if [[ "${method}" == "reference_fp16" &&
          ("${model}" == "qwen30b" || "${model}" == "phi35") ]]; then
        two_gpu_reference=1
    fi
    if ((two_gpu_reference)); then
        record_status \
            "WAIT q80 lane before model=${model} FP16 two-GPU run"
        wait_for_q80_lane_complete
        wait_for_all_gpus_idle
        visible_devices="0,1"
    else
        wait_for_gpu_idle
    fi
    require_clean_tree
    record_status "START model=${model} method=${method}"
    args=(
        --model "${checkpoint}"
        --paper-model "${model}"
        --benchmarks "${benchmarks}"
        --output "${output}"
        --device cuda:0
        --paper-protocol
        --allow-code-execution
        --hash-model-files
    )
    if [[ "${method}" == "reference_fp16" ]]; then
        args+=(--method reference_fp16)
    else
        args+=(
            --method quantized_checkpoint
            --quantization "${quantization}"
            --autoround-backend triton
        )
    fi
    if ((two_gpu_reference)); then
        args+=(--device-map auto --require-cuda-device-count 2)
    fi
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${visible_devices}" \
            bash scripts/reproduce_paper.sh quality "${args[@]}"
    ) >"${log}" 2>&1
    validate_artifact "${output}" "${model}" "${method}"
    rsync -a "${artifact_root}/" "${persistent_root}/"
    record_status \
        "PASS model=${model} method=${method} visible_devices=${visible_devices}"
}

run_dynamic_quality() {
    model=$1
    config=$2
    checkpoint=$3
    initial_map=$4
    output="${artifact_root}/${model}_dynaexq_quality.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        validate_artifact "${output}" "${model}" dynaexq
        record_status "SKIP_VALID model=${model} method=dynaexq"
        return
    fi
    wait_for_gpu_idle
    require_clean_tree
    record_status "START model=${model} method=dynaexq"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
            bash scripts/reproduce_paper.sh dynamic \
                --config "${config}" \
                --model-path "${checkpoint}" \
                --output "${output}" \
                --device cuda:0 \
                --hash-model-files \
                --wait-for-idle-physical-gpu "${physical_gpu}" \
                --initial-map "${initial_map}" \
                quality \
                --benchmarks "${benchmarks}" \
                --paper-protocol \
                --allow-code-execution
    ) >"${log}" 2>&1
    validate_artifact "${output}" "${model}" dynaexq
    rsync -a "${artifact_root}/" "${persistent_root}/"
    record_status "PASS model=${model} method=dynaexq"
}

record_status \
    "QUALITY_LANE_SUPERVISOR_STARTED lane=${lane} gpu=${physical_gpu} expected_commit=${expected_commit}"
wait_for_prerequisite
record_status "MOE_INFINITY_COMPLETE"
require_clean_tree

if [[ "${lane}" == "q30_phi" ]]; then
    run_static_quality qwen30b static_int4 \
        /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound int4
    run_dynamic_quality qwen30b dynaexq/configs/qwen30b.yaml \
        /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned \
        /dev/shm/dynaexq-formal-bbdaa87/qwen30b_initial_map.json
    run_static_quality phi35 static_int4 \
        /home/kec23008/Models/Phi-3.5-MoE-instruct-W4A16-AutoRound-formal int4
    run_dynamic_quality phi35 dynaexq/configs/phi35_moe.yaml \
        /dev/shm/dynaexq-models/phi35-moe-bf16 \
        /dev/shm/dynaexq-formal-bbdaa87/phi35_initial_map.json
    run_static_quality qwen30b reference_fp16 \
        /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned
    run_static_quality phi35 reference_fp16 \
        /dev/shm/dynaexq-models/phi35-moe-bf16
else
    run_static_quality qwen80b static_int2 \
        /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int2-from-int4-formal int2
    run_static_quality qwen80b static_int4 \
        /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound int4
    run_dynamic_quality qwen80b dynaexq/configs/qwen3_next_80b.yaml \
        /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound \
        /dev/shm/dynaexq-formal-bbdaa87/qwen80b_initial_map.json
fi

rsync -a "${artifact_root}/" "${persistent_root}/"
record_status "QUALITY_LANE_COMPLETE lane=${lane} gpu=${physical_gpu}"
