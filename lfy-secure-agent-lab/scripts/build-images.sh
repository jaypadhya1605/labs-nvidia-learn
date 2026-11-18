#!/usr/bin/env bash
# Build the two Lab 03 images in Azure Container Registry (no local Docker needed). Context = repo root (.dockerignore applies).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
: "${ACR_NAME:?ACR_NAME missing - infra/deploy-landing-zone.sh up writes it to env/outputs.env}"
az acr build -r "${ACR_NAME}" -t "lfy/guarded-gateway:${APP_TAG}" -f workloads/gateway/Dockerfile .
az acr build -r "${ACR_NAME}" -t "lfy/soc-agent:${APP_TAG}" -f workloads/agent/Dockerfile .
az acr repository show-tags -n "${ACR_NAME}" --repository lfy/soc-agent -o table
