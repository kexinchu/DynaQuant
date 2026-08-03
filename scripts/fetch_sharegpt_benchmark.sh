#!/usr/bin/env bash
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI ('hf') is required." >&2
  exit 2
fi

repository="anon8231489123/ShareGPT_Vicuna_unfiltered"
revision="192ab2185289094fc556ec8ce5ce1e8e587154ca"
filename="ShareGPT_V3_unfiltered_cleaned_split.json"
expected_sha256="35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${SHAREGPT_PATH:-${repository_root}/${filename}}"

if [[ -e "${output}" ]]; then
  actual_sha256="$(sha256sum "${output}" | awk '{print $1}')"
  if [[ "${actual_sha256}" == "${expected_sha256}" ]]; then
    echo "ShareGPT workload is already present and verified: ${output}"
    exit 0
  fi
  echo "Existing output has the wrong SHA-256: ${output}" >&2
  exit 1
fi

temporary_directory="$(mktemp -d)"
trap 'rm -rf "${temporary_directory}"' EXIT

hf download \
  "${repository}" \
  "${filename}" \
  --repo-type dataset \
  --revision "${revision}" \
  --local-dir "${temporary_directory}"

downloaded="${temporary_directory}/${filename}"
actual_sha256="$(sha256sum "${downloaded}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  echo "Downloaded ShareGPT workload failed SHA-256 verification." >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
mv "${downloaded}" "${output}"
echo "Downloaded and verified ShareGPT workload: ${output}"
