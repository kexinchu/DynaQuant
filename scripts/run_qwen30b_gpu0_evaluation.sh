#!/usr/bin/env bash
# Resumable Qwen3-30B lane for the local RTX A6000 (physical GPU 0).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

model_path="${QWEN30B_MODEL_PATH:-/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507}"
output_root="results/evaluation/qwen30b"
calibration_map="${output_root}/calibration_wikitext103_256x2048.json"
mkdir -p "${output_root}"

wait_for_gpu0_idle() {
    local consecutive=0
    local used_memory_mib
    local utilization_pct
    while (( consecutive < 3 )); do
        local sample
        if ! sample="$(nvidia-smi --id=0 \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits)"; then
            echo "[gpu-gate] unable to query physical GPU 0; retrying"
            consecutive=0
            sleep 30
            continue
        fi
        IFS=',' read -r used_memory_mib utilization_pct <<< "${sample}"
        used_memory_mib="${used_memory_mib//[[:space:]]/}"
        utilization_pct="${utilization_pct//[[:space:]]/}"
        if (( used_memory_mib <= 1024 && utilization_pct == 0 )); then
            consecutive=$((consecutive + 1))
            echo "[gpu-gate] candidate idle ${consecutive}/3"
        else
            consecutive=0
            echo "[gpu-gate] busy memory_mib=${used_memory_mib} util_pct=${utilization_pct}"
        fi
        if (( consecutive < 3 )); then
            sleep 30
        fi
    done
}

foreign_gpu0_pids() {
    local own_pid="$1"
    local gpu_pid
    nvidia-smi --id=0 --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r gpu_pid; do
            gpu_pid="${gpu_pid//[[:space:]]/}"
            if [[ "${gpu_pid}" =~ ^[0-9]+$ ]] && [[ "${gpu_pid}" != "${own_pid}" ]]; then
                echo "${gpu_pid}"
            fi
        done
}

run_artifact() {
    local output_path="$1"
    shift
    if [[ -s "${output_path}" ]]; then
        if python scripts/validate_qwen30b_artifact.py "${output_path}"; then
            echo "[resume] keeping ${output_path}"
            return 0
        fi
        local invalid_path="${output_path}.invalid"
        mv "${output_path}" "${invalid_path}"
        echo "[resume] moved invalid artifact to ${invalid_path}"
    fi
    local failure_path="${output_path}.failed"
    rm -f "${failure_path}"
    while true; do
        wait_for_gpu0_idle
        set +e
        CUDA_VISIBLE_DEVICES=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python -m dynaexq.experiments.eval_dynamic \
            --model-path "${model_path}" \
            --output "${output_path}" \
            --device cuda:0 \
            "$@" &
        local eval_pid=$!
        local contended=0
        while kill -0 "${eval_pid}" 2>/dev/null; do
            local foreign_pids
            foreign_pids="$(foreign_gpu0_pids "${eval_pid}")"
            if [[ -n "${foreign_pids}" ]]; then
                contended=1
                echo "[gpu-watch] foreign GPU0 PID(s) ${foreign_pids//$'\n'/,}; retrying ${output_path}"
                kill -TERM "${eval_pid}" 2>/dev/null
                break
            fi
            sleep 2
        done
        wait "${eval_pid}"
        local status=$?
        set -e
        if (( contended != 0 )); then
            if [[ -e "${output_path}" ]]; then
                local contended_path
                contended_path="${output_path}.contended.$(date -u +%Y%m%dT%H%M%SZ)"
                mv "${output_path}" "${contended_path}"
                echo "[gpu-watch] archived contaminated output as ${contended_path}"
            fi
            rm -f "${failure_path}"
            continue
        fi
        if (( status != 0 )); then
            printf 'exit_status=%d\noutput=%s\n' "${status}" "${output_path}" \
                > "${failure_path}"
            echo "[failed] ${output_path} (exit ${status}); continuing matrix"
            return 0
        fi
        if python scripts/validate_qwen30b_artifact.py "${output_path}"; then
            return 0
        fi
        local invalid_path="${output_path}.invalid"
        mv "${output_path}" "${invalid_path}"
        printf 'exit_status=0\noutput=%s\nvalidation=failed\n' "${output_path}" \
            > "${failure_path}"
        echo "[failed] ${output_path} failed post-run validation; continuing matrix"
        return 0
    done
}

# One independent calibration map is shared by every policy run.
run_artifact "${calibration_map}" \
    --config dynaexq/configs/qwen30b.yaml \
    --hash-model-files \
    calibrate \
    --prompts calibration_datasets/formal/wikitext103_train_256x2048.jsonl \
    --max-prompts 256 \
    --max-input-tokens 2048

# Short end-to-end gates catch CUDA dispatch, partial-residency, and
# first-use-accounting failures before the expensive paper protocol starts.
run_artifact "${output_root}/joint_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_joint_a6000.yaml \
    --policy joint \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

run_artifact "${output_root}/residency_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_residency_a6000.yaml \
    --policy residency \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

run_artifact "${output_root}/fidelity_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_fidelity_a6000.yaml \
    --policy fidelity \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

run_artifact "${output_root}/uniform_low_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_uniform_low_a6000.yaml \
    --policy uniform_low \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

run_artifact "${output_root}/static_mixed_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_static_mixed_a6000.yaml \
    --policy static_mixed \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

run_artifact "${output_root}/lookahead_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_lookahead_a6000.yaml \
    --policy lookahead \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

# Fidelity reference: every expert executes in FP16. Partial residency changes
# placement only; both logical tiers map to the same deduplicated FP16 host
# representation, so demand-loaded experts remain FP16 as well.
run_artifact "${output_root}/all_high_reference_runtime_smoke.json" \
    --config dynaexq/configs/qwen30b_all_high_reference_a6000.yaml \
    --policy all_high_reference \
    --hash-model-files \
    --initial-map "${calibration_map}" \
    perf --batch-size 1 --input-length 1 --output-length 2 \
    --n-warmup 0 --n-repeats 1

run_artifact "${output_root}/all_high_reference_quality_seed42.json" \
    --config dynaexq/configs/qwen30b_all_high_reference_a6000.yaml \
    --policy all_high_reference \
    --hash-model-files --seed 42 \
    --initial-map "${calibration_map}" \
    quality --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
    --paper-protocol --allow-code-execution

# Quality uses one fixed task set. Request-level latency uses all five seeds
# for every policy so comparisons against the joint policy are paired on the
# exact same length-stratified ShareGPT requests.
for policy in uniform_low static_mixed residency lookahead fidelity; do
    config_path="dynaexq/configs/qwen30b_${policy}_a6000.yaml"
    run_artifact "${output_root}/${policy}_quality_seed42.json" \
        --config "${config_path}" --policy "${policy}" \
        --hash-model-files --seed 42 \
        --initial-map "${calibration_map}" \
        quality --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
        --paper-protocol --allow-code-execution
    for seed in 42 43 44 45 46; do
        for regime in prefill decode mixed; do
            run_artifact "${output_root}/${policy}_trace_${regime}_seed${seed}.json" \
                --config "${config_path}" --policy "${policy}" \
                --hash-model-files --seed "${seed}" \
                --initial-map "${calibration_map}" \
                trace-perf \
                --trace ShareGPT_V3_unfiltered_cleaned_split.json \
                --regime "${regime}" --requests 100 --n-warmup 5 \
                --max-input-tokens 2048 --max-output-tokens 256
        done
    done
done

# Primary request-level latency evidence uses the verified ShareGPT trace.
for seed in 42 43 44 45 46; do
    for regime in prefill decode mixed; do
        run_artifact "${output_root}/joint_trace_${regime}_seed${seed}.json" \
            --config dynaexq/configs/qwen30b_joint_a6000.yaml \
            --policy joint \
            --hash-model-files \
            --seed "${seed}" \
            --initial-map "${calibration_map}" \
            trace-perf \
            --trace ShareGPT_V3_unfiltered_cleaned_split.json \
            --regime "${regime}" --requests 100 --n-warmup 5 \
            --max-input-tokens 2048 --max-output-tokens 256
    done
done

# Qwen3-30B quality and routing evidence. HumanEval execution is an explicit
# opt-in in the evaluator and is intentionally enabled only in this formal lane.
run_artifact "${output_root}/joint_quality_seed42.json" \
    --config dynaexq/configs/qwen30b_joint_a6000.yaml \
    --policy joint \
    --hash-model-files \
    --seed 42 \
    --initial-map "${calibration_map}" \
    quality --benchmarks wikitext,mmlu_pro,gpqa,aime25,gsm8k,humaneval \
    --paper-protocol --allow-code-execution

run_artifact "${output_root}/routing_hotset.json" \
    --config dynaexq/configs/qwen30b.yaml \
    --policy uniform_low \
    --hash-model-files \
    routing-hotset --allow-code-execution

# Mechanism, fidelity-sensitivity, and controller-cost experiments use the
# same checkpoint, arena, initial ranking, and formal quality protocol.
for ablation_config in \
    full static blocking no_hysteresis no_tenure ind_value slack_only single_timescale
do
    run_artifact "${output_root}/ablation_${ablation_config}_seed42.json" \
        --config dynaexq/configs/qwen30b_joint_a6000.yaml \
        --policy joint \
        --hash-model-files --seed 42 \
        --initial-map "${calibration_map}" \
        ablation --ablation-config "${ablation_config}" \
        --allow-code-execution
done

for hi_ratio_pct in 0 5 10 15 20 25 30; do
    run_artifact "${output_root}/sensitivity_hi${hi_ratio_pct}_seed42.json" \
        --config dynaexq/configs/qwen30b_fidelity_a6000.yaml \
        --policy fidelity \
        --hash-model-files --seed 42 \
        --initial-map "${calibration_map}" \
        sensitivity --hi-ratio-pct "${hi_ratio_pct}" \
        --allow-code-execution
done

run_artifact "${output_root}/overhead_seed42.json" \
    --config dynaexq/configs/qwen30b_joint_a6000.yaml \
    --policy joint \
    --hash-model-files --seed 42 \
    --initial-map "${calibration_map}" \
    overhead --allow-code-execution

# The primary latency grid is last because each point requires a fresh,
# isolated process and whole-process NVML high-water monitoring.
for seed in 42 43 44 45 46; do
    for batch_size in 1 2 4 8 16 32; do
        run_artifact "${output_root}/joint_perf_seed${seed}_bs${batch_size}.json" \
            --config dynaexq/configs/qwen30b_joint_a6000.yaml \
            --policy joint \
            --hash-model-files \
            --seed "${seed}" \
            --initial-map "${calibration_map}" \
            perf --batch-size "${batch_size}" --input-length 2048 \
            --output-length 256 --n-warmup 5 --n-repeats 100 --paper-protocol
    done
done

# Derive paper-facing rows and figures even when an individual point failed;
# the final audit remains strict and returns nonzero until every required raw
# artifact is present and valid.
python scripts/build_qwen30b_evaluation_data.py --allow-partial
python scripts/plot_qwen30b_evaluation.py --allow-partial
python scripts/audit_qwen30b_gpu0_evaluation.py
