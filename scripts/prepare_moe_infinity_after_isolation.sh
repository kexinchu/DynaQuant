#!/usr/bin/env bash
# Prepare the pinned MoE-Infinity runtime only after every latency-sensitive
# native and isolation run has finished. Until then this supervisor sleeps and
# consumes no GPU memory.

set -euo pipefail

artifact_root="/dev/shm/dynaexq-formal-ee5283b"
final_status="${artifact_root}/native_qwen80_bs4_isolation_backfill_gpu0.status"
status_file="${artifact_root}/moe_infinity_environment.status"
ready_file="${artifact_root}/moe_infinity_environment.ready"
official_repo="/home/kec23008/third_party/MoE-Infinity-ba56518"
build_tree="/home/kec23008/third_party/MoE-Infinity-store-build"
cutlass_dir="/home/kec23008/third_party/cutlass-v3.9.2"
venv_dir="/home/kec23008/.venvs/moe-infinity-ba56518"
nvme_device="/dev/nvme0n1p4"
nvme_mount="/mnt/oldroot"
cache_root="${nvme_mount}/data/dynaexq-moe-infinity-cache"
expected_commit="ba5651897a80d9c9b7a1500cef2c68adaa63db0f"
minimum_free_bytes=$((120 * 1024 * 1024 * 1024))

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${status_file}"
}

fail_status() {
    exit_code=$?
    record_status "FAILED exit=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap fail_status ERR

wait_for_isolation() {
    while true; do
        if [[ -s "${final_status}" ]] &&
           tail -n 1 "${final_status}" |
               grep -q 'QWEN80_BS4_ISOLATION_BACKFILL_COMPLETE$'; then
            return
        fi
        sleep 60
    done
}

gpus_idle() {
    while IFS=',' read -r used util; do
        used=${used//[[:space:]]/}
        util=${util//[[:space:]]/}
        if [[ "${used}" -gt 1024 || "${util}" -ne 0 ]]; then
            return 1
        fi
    done < <(
        nvidia-smi \
            --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )
}

wait_for_isolation_idle() {
    consecutive=0
    while ((consecutive < 3)); do
        idle_pct=$(vmstat 1 2 | tail -n 1 | awk '{print $15}')
        if gpus_idle && [[ "${idle_pct}" -ge 80 ]]; then
            consecutive=$((consecutive + 1))
            record_status \
                "IDLE_SAMPLE pass=${consecutive}/3 cpu_idle_pct=${idle_pct}"
        else
            consecutive=0
            record_status "WAIT_IDLE cpu_idle_pct=${idle_pct}"
        fi
        if ((consecutive < 3)); then
            sleep 30
        fi
    done
}

verify_sources() {
    [[ -z "$(git -C "${official_repo}" status --porcelain)" ]]
    [[ "$(git -C "${official_repo}" rev-parse HEAD)" == "${expected_commit}" ]]
    [[ "$(git -C "${build_tree}" rev-parse HEAD)" == "${expected_commit}" ]]
    [[ "$(git -C "${cutlass_dir}" describe --tags --exact-match)" == "v3.9.2" ]]
    [[ -s "${cutlass_dir}/include/cutlass/cutlass.h" ]]
}

prepare_venv() {
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
        python -m venv --system-site-packages "${venv_dir}"
    fi
    "${venv_dir}/bin/python" -m pip install \
        --disable-pip-version-check \
        --no-deps \
        'setuptools>=78.1.1,<82' \
        wheel \
        chardet \
        hjson \
        nvtx
}

build_store_extension() {
    if compgen -G "${official_repo}/moe_infinity/_store*.so" >/dev/null; then
        record_status "SKIP_BUILD existing_store_extension=true"
        return
    fi
    (
        cd "${build_tree}"
        CUDA_HOME=/usr/local/cuda-12.4 \
        CUTLASS_DIR="${cutlass_dir}" \
        MAX_JOBS=8 \
        MOE_ENABLE_SM90=0 \
        MOE_ENABLE_SM120=0 \
        MOE_STORE_ONLY=1 \
            "${venv_dir}/bin/python" setup.py build_ext --inplace
    )
    mapfile -t extensions < <(
        find "${build_tree}/moe_infinity" -maxdepth 1 \
            -type f -name '_store*.so' -print
    )
    [[ "${#extensions[@]}" -eq 1 ]]
    cp "${extensions[0]}" "${official_repo}/moe_infinity/"
}

verify_runtime_import() {
    [[ -z "$(git -C "${official_repo}" status --porcelain)" ]]
    PYTHONPATH="${official_repo}" "${venv_dir}/bin/python" - <<'PY'
import moe_infinity
import moe_infinity._store
from moe_infinity import MoE

assert moe_infinity.__file__ is not None
assert MoE.__name__ == "MoE"
PY
}

prepare_nvme_cache() {
    mounted_target=$(findmnt -n -o TARGET -S "${nvme_device}" || true)
    if [[ -n "${mounted_target}" && "${mounted_target}" != "${nvme_mount}" ]]; then
        record_status "NVME_MOUNT_CONFLICT target=${mounted_target}"
        return 1
    fi
    if [[ -z "${mounted_target}" ]]; then
        sudo mount "${nvme_device}" "${nvme_mount}"
    fi
    [[ "$(findmnt -n -o SOURCE --target "${nvme_mount}")" == "${nvme_device}" ]]
    available=$(df --output=avail -B1 "${nvme_mount}" | tail -n 1)
    available=${available//[[:space:]]/}
    [[ "${available}" -ge "${minimum_free_bytes}" ]]
    sudo mkdir -p "${cache_root}/gpu0" "${cache_root}/gpu1"
    sudo chown -R "$(id -u):$(id -g)" "${cache_root}"
    record_status \
        "NVME_READY mount=${nvme_mount} available_bytes=${available} cache=${cache_root}"
}

record_status "MOE_INFINITY_ENV_SUPERVISOR_STARTED"
wait_for_isolation
record_status "NATIVE_AND_ISOLATION_COMPLETE"
wait_for_isolation_idle
verify_sources
prepare_venv
record_status "VENV_READY path=${venv_dir}"
build_store_extension
verify_runtime_import
record_status "STORE_EXTENSION_READY"
prepare_nvme_cache
touch "${ready_file}"
record_status "MOE_INFINITY_ENVIRONMENT_READY"
