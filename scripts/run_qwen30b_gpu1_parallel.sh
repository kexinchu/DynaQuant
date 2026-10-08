#!/usr/bin/env bash
# Parallel, disjoint Qwen3-30B lane for physical GPU 1.
# The GPU0 runner owns calibration, component policies, and final derivation.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

model_path="${QWEN30B_MODEL_PATH:-/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507}"
output_root="results/evaluation/qwen30b"
calibration_map="${output_root}/calibration_wikitext103_256x2048.json"
mkdir -p "${output_root}"

wait_for_calibration() {
    until [[ -s "${calibration_map}" ]] \
        && python scripts/validate_qwen30b_artifact.py "${calibration_map}"
    do
        echo "[gpu1] waiting for validated calibration map"
        sleep 30
    done
}

wait_for_gpu1_idle() {
    local poll_seconds="${1:-10}"
    local consecutive=0
    while (( consecutive < 3 )); do
        local sample used_memory_mib utilization_pct
        if ! sample="$(nvidia-smi --id=1 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits)"; then
            consecutive=0
            sleep "${poll_seconds}"
            continue
        fi
        IFS=',' read -r used_memory_mib utilization_pct <<< "${sample}"
        used_memory_mib="${used_memory_mib//[[:space:]]/}"
        utilization_pct="${utilization_pct//[[:space:]]/}"
        if (( used_memory_mib <= 1024 && utilization_pct == 0 )); then
            consecutive=$((consecutive + 1))
            echo "[gpu1-gate] candidate idle ${consecutive}/3"
        else
            consecutive=0
            echo "[gpu1-gate] busy memory_mib=${used_memory_mib} util_pct=${utilization_pct}"
        fi
        if (( consecutive < 3 )); then
            sleep "${poll_seconds}"
        fi
    done
}

foreign_gpu1_pids() {
    local own_pid="$1"
    nvidia-smi --id=1 --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r gpu_pid; do
            gpu_pid="${gpu_pid//[[:space:]]/}"
            if [[ "${gpu_pid}" =~ ^[0-9]+$ ]] && [[ "${gpu_pid}" != "${own_pid}" ]]; then
                echo "${gpu_pid}"
            fi
        done
}

own_gpu1_memory_mib() {
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

run_artifact() (
    local output_path="$1"
    shift
    # The two resumable queues have fallback overlap near their tails.  The
    # lock covers validation and atomic publication, not just the evaluator,
    # so a waiter always observes the winner's fully validated artifact.
    exec 9>"${output_path}.lock"
    flock 9
    if [[ -s "${output_path}" ]] \
        && python scripts/validate_qwen30b_artifact.py "${output_path}"
    then
        echo "[gpu1-resume] keeping ${output_path}"
        return 0
    fi
    if [[ -s "${output_path}" ]]; then
        mv "${output_path}" "${output_path}.invalid.gpu1"
    fi
    local failure_path="${output_path}.failed"
    rm -f "${failure_path}"
    while true; do
        set +e
        CUDA_VISIBLE_DEVICES=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -m dynaexq.experiments.eval_dynamic \
            --model-path "${model_path}" \
            --output "${output_path}" \
            --device cuda:0 \
            --wait-for-idle-physical-gpu 1 \
            "$@" &
        local eval_pid=$!
        local contended=0
        while kill -0 "${eval_pid}" 2>/dev/null; do
            local foreign_pids
            foreign_pids="$(foreign_gpu1_pids "${eval_pid}")"
            if [[ -n "${foreign_pids}" ]]; then
                local own_mib
                own_mib="$(own_gpu1_memory_mib "${eval_pid}")"
                if [[ "${own_mib}" =~ ^[0-9]+$ ]] && (( own_mib <= 1024 )); then
                    # eval_dynamic gates internally after host-cache creation.
                    sleep 2
                    continue
                fi
                contended=1
                echo "[gpu1-watch] foreign PID(s) ${foreign_pids//$'\n'/,} after arena initialization; retrying"
                kill -TERM "${eval_pid}" 2>/dev/null
                break
            fi
            sleep 2
        done
        wait "${eval_pid}"
        local status=$?
        set -e
        if (( contended != 0 )); then
            [[ ! -e "${output_path}" ]] || mv \
                "${output_path}" \
                "${output_path}.contended.gpu1.$(date -u +%Y%m%dT%H%M%SZ)"
            continue
        fi
        if (( status == 143 )); then
            # Treat the scheduled local GPU service's SIGTERM as an external
            # interruption.  The next attempt waits at the physical-GPU gate
            # and resumes the same matrix point when the service releases it.
            echo "[gpu1-interrupted] ${output_path} received SIGTERM; retrying"
            rm -f "${failure_path}"
            continue
        fi
        if (( status != 0 )); then
            printf 'exit_status=%d\noutput=%s\ngpu=1\n' \
                "${status}" "${output_path}" > "${failure_path}"
            echo "[gpu1-failed] ${output_path} (exit ${status})"
            return 0
        fi
        if python scripts/validate_qwen30b_artifact.py "${output_path}"; then
            return 0
        fi
        mv "${output_path}" "${output_path}.invalid.gpu1"
        printf 'exit_status=0\noutput=%s\nvalidation=failed\ngpu=1\n' \
            "${output_path}" > "${failure_path}"
        return 0
    done
)

wait_for_calibration

# Work backward from the GPU0 lane. This keeps the two runners disjoint for
# the longest interval; the GPU0 runner will validate and skip completed files.
for seed in 46 45 44 43 42; do
    for batch_size in 32 16 8 4 2 1; do
        run_artifact "${output_root}/joint_perf_seed${seed}_bs${batch_size}.json" \
            --config dynaexq/configs/qwen30b_joint_a6000.yaml \
            --policy joint --hash-model-files --seed "${seed}" \
            --initial-map "${calibration_map}" \
            perf --batch-size "${batch_size}" --input-length 2048 \
            --output-length 256 --n-warmup 5 --n-repeats 100 --paper-protocol
    done
done

run_artifact "${output_root}/overhead_seed42.json" \
    --config dynaexq/configs/qwen30b_joint_a6000.yaml \
    --policy joint --hash-model-files --seed 42 \
    --initial-map "${calibration_map}" overhead --allow-code-execution

for hi_ratio_pct in 30 25 20 15 10 5 0; do
    run_artifact "${output_root}/sensitivity_hi${hi_ratio_pct}_seed42.json" \
        --config dynaexq/configs/qwen30b_fidelity_a6000.yaml \
        --policy fidelity --hash-model-files --seed 42 \
        --initial-map "${calibration_map}" \
        sensitivity --hi-ratio-pct "${hi_ratio_pct}" --allow-code-execution
done

for ablation_config in \
    single_timescale slack_only ind_value no_tenure no_hysteresis blocking static full
do
    run_artifact "${output_root}/ablation_${ablation_config}_seed42.json" \
        --config dynaexq/configs/qwen30b_joint_a6000.yaml \
        --policy joint --hash-model-files --seed 42 \
        --initial-map "${calibration_map}" \
        ablation --ablation-config "${ablation_config}" --allow-code-execution
done

run_artifact "${output_root}/joint_quality_seed42.json" \
    --config dynaexq/configs/qwen30b_joint_a6000.yaml \
    --policy joint --hash-model-files --seed 42 \
    --initial-map "${calibration_map}" \
    quality --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
    --paper-protocol --allow-code-execution

for seed in 46 45 44 43 42; do
    for regime in mixed decode prefill; do
        run_artifact "${output_root}/joint_trace_${regime}_seed${seed}.json" \
            --config dynaexq/configs/qwen30b_joint_a6000.yaml \
            --policy joint --hash-model-files --seed "${seed}" \
            --initial-map "${calibration_map}" \
            trace-perf --trace ShareGPT_V3_unfiltered_cleaned_split.json \
            --regime "${regime}" --requests 100 --n-warmup 5 \
            --max-input-tokens 2048 --max-output-tokens 256
    done
done

echo "[gpu1] assigned parallel lane complete"
