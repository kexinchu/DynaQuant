#!/usr/bin/env bash
# Guarded, resumable diagnostic for estimating warmed DynaByte generation cost.
# This artifact is not part of the paper's formal 152-artifact matrix.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

model_path="${QWEN30B_MODEL_PATH:-/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507}"
output_path="${QWEN30B_PROBE_OUTPUT:-results/evaluation/qwen30b/diagnostic_joint_steady_bs1_in128_out4.json}"
probe_output_length="${QWEN30B_PROBE_OUTPUT_LENGTH:-4}"
probe_warmups="${QWEN30B_PROBE_WARMUPS:-2}"
probe_repeats="${QWEN30B_PROBE_REPEATS:-3}"
calibration_map="results/evaluation/qwen30b/calibration_wikitext103_256x2048.json"
log_prefix="[gpu1-probe]"
formal_pid_path="results/evaluation/qwen30b/formal_matrix_gpu1_source_8277f77.pid"
formal_log_path="results/evaluation/qwen30b/formal_matrix_gpu1_source_8277f77.log"
resume_formal_after_probe="${QWEN30B_RESUME_FORMAL_AFTER_PROBE:-0}"

resume_formal_lane() {
    if [[ "${resume_formal_after_probe}" != "1" ]]; then
        echo "${log_prefix} formal GPU1 lane remains paused pending probe review"
        return 0
    fi
    if [[ -s "${formal_pid_path}" ]]; then
        local recorded_pid
        recorded_pid="$(<"${formal_pid_path}")"
        if [[ "${recorded_pid}" =~ ^[0-9]+$ ]] \
            && kill -0 "${recorded_pid}" 2>/dev/null
        then
            echo "${log_prefix} formal GPU1 runner already active as ${recorded_pid}"
            return 0
        fi
    fi
    setsid nohup bash scripts/run_qwen30b_gpu1_parallel.sh \
        >> "${formal_log_path}" 2>&1 < /dev/null &
    local formal_pid=$!
    printf '%s\n' "${formal_pid}" > "${formal_pid_path}"
    echo "${log_prefix} resumed formal GPU1 runner as ${formal_pid}"
}

wait_for_idle() {
    local poll_seconds="${1:-10}"
    local consecutive=0
    while (( consecutive < 3 )); do
        local sample used util
        sample="$(nvidia-smi --id=1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
        IFS=',' read -r used util <<< "${sample}"
        used="${used//[[:space:]]/}"
        util="${util//[[:space:]]/}"
        if (( used <= 1024 && util == 0 )); then
            consecutive=$((consecutive + 1))
            echo "${log_prefix} candidate idle ${consecutive}/3"
        else
            consecutive=0
            echo "${log_prefix} busy memory_mib=${used} util_pct=${util}"
        fi
        (( consecutive == 3 )) || sleep "${poll_seconds}"
    done
}

foreign_pids() {
    local own_pid="$1"
    nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r pid; do
            pid="${pid//[[:space:]]/}"
            if [[ "${pid}" =~ ^[0-9]+$ ]] && [[ "${pid}" != "${own_pid}" ]]; then
                echo "${pid}"
            fi
        done
}

own_memory_mib() {
    local own_pid="$1"
    nvidia-smi --id=1 \
        --query-compute-apps=pid,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v target="${own_pid}" '
            {
                pid=$1; memory=$2
                gsub(/[[:space:]]/, "", pid)
                gsub(/[[:space:]]/, "", memory)
                if (pid == target) { print memory; found=1; exit }
            }
            END { if (!found) print 0 }
        '
}

wait_for_calibration() {
    until [[ -s "${calibration_map}" ]] \
        && python scripts/validate_qwen30b_artifact.py "${calibration_map}"
    do
        echo "${log_prefix} waiting for validated calibration map"
        sleep 30
    done
}

wait_for_calibration

if [[ -s "${output_path}" ]] \
    && python scripts/validate_qwen30b_artifact.py "${output_path}"
then
    echo "${log_prefix} keeping validated ${output_path}"
    resume_formal_lane
    exit 0
fi

while true; do
    rm -f "${output_path}" "${output_path}.failed"
    set +e
    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m dynaexq.experiments.eval_dynamic \
        --model-path "${model_path}" \
        --output "${output_path}" \
        --device cuda:0 \
        --wait-for-idle-physical-gpu 1 \
        --config dynaexq/configs/qwen30b_joint_a6000.yaml \
        --policy joint --hash-model-files --seed 42 \
        --initial-map "${calibration_map}" \
        perf --batch-size 1 --input-length 128 \
        --output-length "${probe_output_length}" \
        --n-warmup "${probe_warmups}" --n-repeats "${probe_repeats}" &
    eval_pid=$!
    contended=0
    while kill -0 "${eval_pid}" 2>/dev/null; do
        pids="$(foreign_pids "${eval_pid}")"
        if [[ -n "${pids}" ]]; then
            own_mib="$(own_memory_mib "${eval_pid}")"
            if [[ "${own_mib}" =~ ^[0-9]+$ ]] && (( own_mib <= 1024 )); then
                # The child gates itself immediately before device allocation.
                sleep 2
                continue
            fi
            contended=1
            echo "${log_prefix} foreign PID(s) ${pids//$'\n'/,} after arena initialization; retrying"
            kill -TERM "${eval_pid}" 2>/dev/null
            break
        fi
        sleep 2
    done
    wait "${eval_pid}"
    status=$?
    set -e
    if (( contended != 0 )); then
        if [[ -e "${output_path}" ]]; then
            contended_path="${output_path}.contended.$(date -u +%Y%m%dT%H%M%SZ)"
            mv "${output_path}" "${contended_path}"
            echo "${log_prefix} archived contaminated output as ${contended_path}"
        fi
        rm -f "${output_path}.failed"
        continue
    fi
    if (( status != 0 )); then
        printf 'exit_status=%d\noutput=%s\n' "${status}" "${output_path}" \
            > "${output_path}.failed"
        exit "${status}"
    fi
    python scripts/validate_qwen30b_artifact.py "${output_path}"
    echo "${log_prefix} complete"
    resume_formal_lane
    exit 0
done
