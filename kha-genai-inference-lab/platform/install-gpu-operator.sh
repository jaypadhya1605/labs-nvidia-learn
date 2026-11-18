#!/usr/bin/env bash
# M2: NVIDIA GPU Operator (driver, toolkit, device plugin, GFD, DCGM exporter, MIG Manager, validator).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null
helm repo update >/dev/null
kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace gpu-operator pod-security.kubernetes.io/enforce=privileged --overwrite
helm upgrade --install gpu-operator nvidia/gpu-operator --version "${GPU_OPERATOR_VERSION}" \
  -n gpu-operator -f platform/gpu-operator-values.yaml --wait --timeout 20m
echo "Operator installed. Driver install + MIG geometry (with a possible node reboot) takes 10-20 minutes."
platform/set-gpu-mode.sh wait
kubectl -n gpu-operator wait --for=condition=Ready pod -l app=nvidia-operator-validator --timeout=30m || true
platform/set-gpu-mode.sh status
