#!/usr/bin/env bash
# Run deterministic mechanism-data collection on two GPUs after quality
# consolidation. No result here is a cross-GPU latency measurement.

set -euo pipefail

lane=${1:?usage: run_formal_mechanism_lane_after_quality.sh q30|q80_phi gpu}
physical_gpu=${2:?usage: run_formal_mechanism_lane_after_quality.sh q30|q80_phi gpu}
if [[ "${lane}" != "q30" && "${lane}" != "q80_phi" ]]; then
    exit 2
fi
if [[ "${physical_gpu}" != "0" && "${physical_gpu}" != "1" ]]; then
    exit 2
fi

formal_root="/home/kec23008/DynaQuant-phi-formal"
artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
quality_ready="${artifact_root}/quality_consolidation.complete"
status_file="${artifact_root}/mechanism_${lane}_gpu${physical_gpu}.status"
validator="/home/kec23008/DynaQuant/scripts/validate_raw_mechanism_artifact.py"
prompts="calibration_datasets/formal/wikitext103_train_256x2048.jsonl"
expected_commit=""

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

wait_for_quality() {
    while [[ ! -e "${quality_ready}" ]]; do
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

require_clean_tree() {
    [[ -n "${expected_commit}" ]]
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
}

validate_raw() {
    output=$1
    kind=$2
    model=$3
    shift 3
    python "${validator}" \
        --artifact "${output}" \
        --kind "${kind}" \
        --paper-model "${model}" \
        --expected-commit "${expected_commit}" "$@"
}

finish_point() {
    output=$1
    kind=$2
    model=$3
    shift 3
    validate_raw "${output}" "${kind}" "${model}" "$@"
    rsync -a "${artifact_root}/" "${persistent_root}/"
    record_status "PASS kind=${kind} model=${model} output=$(basename "${output}")"
}

run_activation() {
    model=$1
    checkpoint=$2
    output="${artifact_root}/${model}_activation_density.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        finish_point "${output}" activation "${model}"
        return
    fi
    wait_for_gpu_idle
    require_clean_tree
    record_status "START kind=activation model=${model}"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
            bash scripts/reproduce_paper.sh activation-density \
                --paper-model "${model}" \
                --model-path "${checkpoint}" \
                --prompts "${prompts}" \
                --output "${output}" \
                --device cuda:0 \
                --hash-model-files
    ) >"${log}" 2>&1
    finish_point "${output}" activation "${model}"
}

run_routing_trace() {
    model=$1
    checkpoint=$2
    output="${artifact_root}/${model}_routing_active_set_trace.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        finish_point "${output}" routing_trace "${model}"
        return
    fi
    wait_for_gpu_idle
    require_clean_tree
    record_status "START kind=routing_trace model=${model}"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
            bash scripts/reproduce_paper.sh routing-trace \
                --paper-model "${model}" \
                --model-path "${checkpoint}" \
                --prompts "${prompts}" \
                --output "${output}" \
                --device cuda:0 \
                --hash-model-files
    ) >"${log}" 2>&1
    finish_point "${output}" routing_trace "${model}"
}

run_perplexity_points() {
    model=$1
    config=$2
    checkpoint=$3
    initial_map=$4
    for ratio in 0 15 30 45 60 75 90 100; do
        output="${artifact_root}/${model}_perplexity_ratio${ratio}.json"
        log="${output%.json}.log"
        if [[ -s "${output}" ]]; then
            finish_point "${output}" perplexity_point "${model}" --ratio "${ratio}"
            continue
        fi
        wait_for_gpu_idle
        require_clean_tree
        record_status "START kind=perplexity_point model=${model} ratio=${ratio}"
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
                    perplexity-point --low-ratio-pct "${ratio}"
        ) >"${log}" 2>&1
        finish_point "${output}" perplexity_point "${model}" --ratio "${ratio}"
    done
}

run_routing_hotset() {
    output="${artifact_root}/qwen30b_routing_hotset.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        finish_point "${output}" routing_hotset qwen30b
        return
    fi
    wait_for_gpu_idle
    require_clean_tree
    record_status "START kind=routing_hotset model=qwen30b"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES="${physical_gpu}" \
            bash scripts/reproduce_paper.sh dynamic \
                --config dynaexq/configs/qwen30b.yaml \
                --model-path /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned \
                --output "${output}" \
                --device cuda:0 \
                --hash-model-files \
                --wait-for-idle-physical-gpu "${physical_gpu}" \
                --initial-map /dev/shm/dynaexq-formal-bbdaa87/qwen30b_initial_map.json \
                routing-hotset --allow-code-execution
    ) >"${log}" 2>&1
    finish_point "${output}" routing_hotset qwen30b
}

record_status "MECHANISM_LANE_SUPERVISOR_STARTED lane=${lane} gpu=${physical_gpu}"
wait_for_quality
expected_commit=$(git -C "${formal_root}" rev-parse HEAD)
require_clean_tree
record_status "QUALITY_COMPLETE expected_commit=${expected_commit}"

if [[ "${lane}" == "q30" ]]; then
    run_routing_trace qwen30b \
        /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound
    run_perplexity_points qwen30b dynaexq/configs/qwen30b.yaml \
        /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned \
        /dev/shm/dynaexq-formal-bbdaa87/qwen30b_initial_map.json
    run_routing_hotset
    run_activation qwen30b \
        /home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound
else
    run_routing_trace qwen80b \
        /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound
    run_routing_trace phi35 \
        /home/kec23008/Models/Phi-3.5-MoE-instruct-W4A16-AutoRound-formal
    run_perplexity_points qwen80b dynaexq/configs/qwen3_next_80b.yaml \
        /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound \
        /dev/shm/dynaexq-formal-bbdaa87/qwen80b_initial_map.json
    run_activation qwen80b \
        /home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound
    run_activation phi35 \
        /home/kec23008/Models/Phi-3.5-MoE-instruct-W4A16-AutoRound-formal
fi

rsync -a "${artifact_root}/" "${persistent_root}/"
record_status "MECHANISM_LANE_COMPLETE lane=${lane} gpu=${physical_gpu} expected_commit=${expected_commit}"
