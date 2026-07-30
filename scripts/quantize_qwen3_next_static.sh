#!/usr/bin/env bash
# Register Intel's official mixed-W4 checkpoint and derive mixed-W2 from it.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

mode="${1:-all}"
repository="Intel/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound"
requested_revision="master"
model_root="${DYNAEXQ_MODEL_ROOT:-/home/kec23008/Models}"
int4_output="${model_root}/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound"
int2_output="${model_root}/Qwen3-Next-80B-A3B-Instruct-int2-from-int4-formal"
int4_manifest="results/model_manifests/qwen3_next_80b_int4_mixed_autoround_official.json"
int4_catalog="results/model_manifests/qwen3_next_80b_int4_mixed_autoround_official_modelscope_catalog.json"
int2_manifest="results/model_manifests/qwen3_next_80b_int2_from_int4.json"

fetch_int4() {
    mkdir -p "${model_root}"
    modelscope download "${repository}" \
        --revision "${requested_revision}" \
        --local_dir "${int4_output}" \
        --max-workers 8
}

register_int4() {
    python scripts/register_modelscope_snapshot.py \
        --model-dir "${int4_output}" \
        --repository "${repository}" \
        --requested-revision "${requested_revision}" \
        --output "${int4_manifest}" \
        --catalog-output "${int4_catalog}"
}

derive_int2() {
    python scripts/derive_autoround_int2_from_int4.py \
        --parent "${int4_output}" \
        --parent-manifest "${int4_manifest}" \
        --output "${int2_output}" \
        --target-group-size 64
    python scripts/build_quantized_model_manifest.py \
        --model-dir "${int2_output}" \
        --output "${int2_manifest}"
}

case "${mode}" in
    fetch-int4)
        fetch_int4
        ;;
    register-int4)
        register_int4
        ;;
    int4)
        fetch_int4
        register_int4
        ;;
    int2)
        derive_int2
        ;;
    all)
        fetch_int4
        register_int4
        derive_int2
        ;;
    *)
        echo "Usage: $0 [fetch-int4|register-int4|int4|int2|all]" >&2
        exit 2
        ;;
esac
