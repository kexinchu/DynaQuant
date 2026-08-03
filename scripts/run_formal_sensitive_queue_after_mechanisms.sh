#!/usr/bin/env bash
# Run all latency-bearing mechanism experiments serially after deterministic
# mechanism collection and blocking-offload measurements complete.

set -euo pipefail

formal_root="/home/kec23008/DynaQuant-phi-formal"
artifact_root="/dev/shm/dynaexq-mechanism-c7a9999"
persistent_root="/home/kec23008/DynaQuant-experiment-artifacts/c7a9999"
mechanism_ready="${artifact_root}/mechanism_consolidation.complete"
status_file="${artifact_root}/sensitive_queue_gpu0.status"
ready_file="${artifact_root}/sensitive_queue.complete"
validator="/home/kec23008/DynaQuant/scripts/validate_sensitive_artifact.py"
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

wait_for_mechanisms() {
    while [[ ! -e "${mechanism_ready}" ]]; do
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

require_clean_tree() {
    [[ -n "${expected_commit}" ]]
    [[ -z "$(git -C "${formal_root}" status --porcelain)" ]]
    [[ "$(git -C "${formal_root}" rev-parse HEAD)" == "${expected_commit}" ]]
}

configure_model() {
    model=$1
    case "${model}" in
        qwen30b)
            config=dynaexq/configs/qwen30b.yaml
            checkpoint=/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-BF16-pinned
            initial_map=/dev/shm/dynaexq-formal-bbdaa87/qwen30b_initial_map.json
            ;;
        qwen80b)
            config=dynaexq/configs/qwen3_next_80b.yaml
            checkpoint=/home/kec23008/Models/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound
            initial_map=/dev/shm/dynaexq-formal-bbdaa87/qwen80b_initial_map.json
            ;;
        *)
            return 2
            ;;
    esac
}

validate_artifact() {
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
    validate_artifact "${output}" "${kind}" "${model}" "$@"
    rsync -a "${artifact_root}/" "${persistent_root}/"
    record_status "PASS kind=${kind} model=${model} output=$(basename "${output}")"
}

run_ablation() {
    model=$1
    mode=$2
    configure_model "${model}"
    output="${artifact_root}/${model}_ablation_${mode}.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        finish_point "${output}" ablation "${model}" --ablation-config "${mode}"
        return
    fi
    wait_for_isolated_machine
    require_clean_tree
    record_status "START kind=ablation model=${model} config=${mode} gpu=0"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
            --config "${config}" \
            --model-path "${checkpoint}" \
            --output "${output}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu 0 \
            --initial-map "${initial_map}" \
            ablation --ablation-config "${mode}" --allow-code-execution
    ) >"${log}" 2>&1
    finish_point "${output}" ablation "${model}" --ablation-config "${mode}"
}

run_sensitivity() {
    model=$1
    ratio=$2
    configure_model "${model}"
    output="${artifact_root}/${model}_budget_ratio${ratio}.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        finish_point "${output}" sensitivity "${model}" --ratio "${ratio}"
        return
    fi
    wait_for_isolated_machine
    require_clean_tree
    record_status "START kind=sensitivity model=${model} ratio=${ratio} gpu=0"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
            --config "${config}" \
            --model-path "${checkpoint}" \
            --output "${output}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu 0 \
            --initial-map "${initial_map}" \
            sensitivity --hi-ratio-pct "${ratio}" --allow-code-execution
    ) >"${log}" 2>&1
    finish_point "${output}" sensitivity "${model}" --ratio "${ratio}"
}

run_overhead() {
    model=$1
    configure_model "${model}"
    output="${artifact_root}/${model}_runtime_overhead.json"
    log="${output%.json}.log"
    if [[ -s "${output}" ]]; then
        finish_point "${output}" overhead "${model}"
        return
    fi
    wait_for_isolated_machine
    require_clean_tree
    record_status "START kind=overhead model=${model} gpu=0"
    (
        cd "${formal_root}"
        CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh dynamic \
            --config "${config}" \
            --model-path "${checkpoint}" \
            --output "${output}" \
            --device cuda:0 \
            --hash-model-files \
            --wait-for-idle-physical-gpu 0 \
            --initial-map "${initial_map}" \
            overhead --allow-code-execution
    ) >"${log}" 2>&1
    finish_point "${output}" overhead "${model}"
}

record_status "SENSITIVE_QUEUE_SUPERVISOR_STARTED gpu=0"
wait_for_mechanisms
expected_commit=$(git -C "${formal_root}" rev-parse HEAD)
require_clean_tree
record_status "MECHANISMS_COMPLETE expected_commit=${expected_commit}"

for model in qwen30b qwen80b; do
    for mode in full static blocking no_hysteresis; do
        run_ablation "${model}" "${mode}"
    done
    run_overhead "${model}"
    for ratio in 0 5 10 15 20 25 30; do
        run_sensitivity "${model}" "${ratio}"
    done
done

touch "${ready_file}"
rsync -a "${artifact_root}/" "${persistent_root}/"
record_status "SENSITIVE_QUEUE_COMPLETE expected_commit=${expected_commit}"
