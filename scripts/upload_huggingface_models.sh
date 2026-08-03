#!/usr/bin/env bash
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI ('hf') is required." >&2
  exit 2
fi

if [[ -z "${DYNAEXQ_MODEL_ROOT:-}" ]]; then
  echo "Set DYNAEXQ_MODEL_ROOT to the directory containing the model folders." >&2
  exit 2
fi

namespace="${HF_NAMESPACE:-Kris2017}"
workers="${HF_UPLOAD_WORKERS:-4}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

local_names=(
  "Phi-3.5-MoE-instruct-W4A16-AutoRound-formal"
  "Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
  "Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound"
  "Qwen3-Next-80B-A3B-Instruct-int2-from-int4-formal"
)

hub_names=(
  "Phi-3.5-MoE-instruct-W4A16-AutoRound"
  "Qwen3-30B-A3B-Instruct-2507-W4A16-AutoRound"
  "Qwen3-Next-80B-A3B-Instruct-W4A16-AutoRound"
  "Qwen3-Next-80B-A3B-Instruct-W2A16-AutoRound-derived"
)

hf auth whoami >/dev/null

for index in "${!local_names[@]}"; do
  local_dir="${DYNAEXQ_MODEL_ROOT}/${local_names[$index]}"
  hub_name="${hub_names[$index]}"
  repo_id="${namespace}/${hub_name}"
  card="${repository_root}/release/huggingface/${hub_name}/README.md"

  if [[ ! -f "${local_dir}/model.safetensors.index.json" ]]; then
    echo "Missing model index: ${local_dir}" >&2
    exit 2
  fi
  if [[ ! -f "${card}" ]]; then
    echo "Missing release model card: ${card}" >&2
    exit 2
  fi

  hf repos create "${repo_id}" --type model --no-private --exist-ok
  hf upload-large-folder \
    "${repo_id}" \
    "${local_dir}" \
    --type model \
    --exclude README.md \
    --num-workers "${workers}"
  hf upload \
    "${repo_id}" \
    "${card}" \
    README.md \
    --type model \
    --commit-message "Add DynaExQ model card"
done
