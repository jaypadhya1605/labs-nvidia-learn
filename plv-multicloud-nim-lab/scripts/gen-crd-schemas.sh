#!/usr/bin/env bash
# Generate kubeconform JSON schemas from the CRDs of the PINNED charts (NIM Operator, ESO, Argo CD, kube-prometheus-stack)
# so CI validates NIMService/NIMCache/ExternalSecret/Application/ServiceMonitor manifests offline-exactly.
# Usage: scripts/gen-crd-schemas.sh [out-dir=.schemas]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/pins.env
OUT="${1:-.schemas}"; mkdir -p "${OUT}"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo add external-secrets https://charts.external-secrets.io >/dev/null 2>&1 || true
helm repo add argo https://argoproj.github.io/argo-helm >/dev/null 2>&1 || true
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null
{
  helm template x nvidia/k8s-nim-operator --version "${PIN_NIM_OPERATOR}" --include-crds
  helm template x external-secrets/external-secrets --version "${PIN_ESO}" --include-crds --set installCRDs=true
  helm template x argo/argo-cd --version "${PIN_ARGOCD_CHART}" --include-crds
  helm template x prometheus-community/kube-prometheus-stack --version "${PIN_KPS}" --include-crds
} | python3 scripts/crd2schema.py "${OUT}"
