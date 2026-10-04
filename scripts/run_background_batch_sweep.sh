#!/usr/bin/env bash
# Re-measure the Background transfer-pressure curve at nine batch sizes.
# Expert demand uses five disjoint WikiText blocks per batch. The overlap
# window uses static INT4 TTFT/TPOT, because the dynamic runtime already
# fills one 48GB GPU at batch 32.

set -euo pipefail

root="/home/kec23008/DynaQuant"
model="/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507-W4A16-AutoRound"
out="${root}/results/paper/background"
prompts="${out}/wikitext103_train_1280x2048.jsonl"
manifest="${out}/wikitext103_train_1280x2048.manifest.json"
activation="${out}/qwen30b_activation_density.json"
perf_dir="${out}/performance"
log_dir="${out}/logs"
batches=(1 2 4 8 16 32 64 128 256)
batch_csv="1,2,4,8,16,32,64,128,256"

mkdir -p "${perf_dir}" "${log_dir}"
cd "${root}"

wait_for_gpu_idle() {
    while true; do
        sample=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
        used=${sample%%,*}
        util=${sample##*,}
        used=${used//[[:space:]]/}
        util=${util//[[:space:]]/}
        if [[ "${used}" -le 1024 && "${util}" -le 5 ]]; then
            return
        fi
        printf '%s WAIT_GPU used_mib=%s utilization_pct=%s\n' "$(date --iso-8601=seconds)" "${used}" "${util}"
        sleep 30
    done
}

run_when_gpu_free() {
    local log=$1
    shift
    local attempt=1
    local max_attempts=2
    while true; do
        wait_for_gpu_idle
        printf '%s start attempt=%s %s\n' "$(date --iso-8601=seconds)" "${attempt}" "$*"
        if "$@" 2>&1 | tee "${log}"; then
            return
        fi
        printf '%s attempt=%s failed; retry after GPU is free\n' "$(date --iso-8601=seconds)" "${attempt}"
        if [[ "${attempt}" -ge "${max_attempts}" ]]; then
            printf '%s FAIL attempts=%s command=%s\n' "$(date --iso-8601=seconds)" "${attempt}" "$*"
            return 1
        fi
        attempt=$((attempt + 1))
        sleep 15
    done
}

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${prompts}" ]]; then
    python scripts/build_independent_calibration.py \
        --output "${prompts}" \
        --manifest "${manifest}" \
        --count 1280
fi

run_when_gpu_free "${log_dir}/activation_density.log" \
    python scripts/collect_activation_density.py \
    --paper-model qwen30b \
    --model-path "${model}" \
    --prompts "${prompts}" \
    --output "${activation}" \
    --batch-sizes "${batch_csv}" \
    --repeats 5 \
    --max-input-tokens 2048 \
    --device cuda:0 \
    --hash-model-files

for batch in "${batches[@]}"; do
    artifact="${perf_dir}/qwen30b_static_int4_bs${batch}.json"
    if [[ -f "${artifact}" ]]; then
        echo "skip existing ${artifact}"
        continue
    fi
    run_when_gpu_free "${log_dir}/perf_bs${batch}.log" \
        python -m dynaexq.experiments.eval_perf \
        --model "${model}" \
        --paper-model qwen30b \
        --method quantized_checkpoint \
        --quantization int4 \
        --batch-size "${batch}" \
        --input-length 2048 \
        --output-length 256 \
        --n-warmup 5 \
        --n-repeats 100 \
        --device-map cuda:0 \
        --seed 42 \
        --hash-model-files \
        --output "${artifact}"
done

python scripts/build_background_motivation_data.py \
    --activation-density "${activation}" \
    --performance-dir "${perf_dir}" \
    --batches "${batch_csv}" \
    --only transfer
python scripts/plot_background_motivation.py
