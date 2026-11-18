#!/usr/bin/env bash
# M8 step 3 - publish a LoRA adapter as an OCI artifact (ORAS) in GHCR; prints the DIGEST reference to pin in Git.
# Usage: mlops/push_adapter.sh <adapter-dir> <tag>     e.g. mlops/push_adapter.sh results/adapters/plover-style-v1 style-v1
# Needs: oras CLI, `oras login ghcr.io -u <you>` with a PAT that has write:packages. Make the package PUBLIC (synthetic
# data, no PHI) or give the NIM init containers a pull secret - the chart's puller runs `oras pull` without auth.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
DIR="${1:?usage: mlops/push_adapter.sh <adapter-dir> <tag>}"; TAG="${2:?tag}"
REPO="ghcr.io/$(echo "${GITHUB_OWNER}" | tr '[:upper:]' '[:lower:]')/plover-adapters"
[ -f "${DIR}/adapter_config.json" ] && [ -f "${DIR}/adapter_model.safetensors" ] || { echo "not a PEFT adapter dir: ${DIR}" >&2; exit 1; }
RUN_ID="$(cat "${DIR}/mlflow_run_id" 2>/dev/null || echo none)"
( cd "${DIR}" && oras push "${REPO}:${TAG}" --artifact-type application/vnd.plover.lora.v1 \
    --annotation "org.opencontainers.image.source=${GIT_REPO_URL}" --annotation "dev.plover.mlflow.run_id=${RUN_ID}" \
    adapter_config.json adapter_model.safetensors )
DIGEST="$(oras resolve "${REPO}:${TAG}")"
echo
echo "Pushed. Pin THIS in clusters/<cluster>/values.yaml (a digest cannot be re-pointed; a tag can):"
echo "  ${REPO}@${DIGEST}"
