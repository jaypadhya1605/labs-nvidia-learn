#!/usr/bin/env bash
# M4: NVIDIA NIM Operator (NIMCache, NIMService, NIMPipeline, NeMo CRDs).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null
helm repo update >/dev/null
kubectl create namespace nim-operator --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install nim-operator nvidia/k8s-nim-operator -n nim-operator --version "${NIM_OPERATOR_VERSION}" --wait --timeout 10m
kubectl get pods -n nim-operator
kubectl get crd | grep apps.nvidia.com
