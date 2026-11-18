#!/usr/bin/env bash
# Install Argo CD into one cluster and hand it the root Application (app-of-apps) for that cluster.
# Usage: platform/install-argocd.sh aks|gke|eks|oke
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
C="${1:?usage: platform/install-argocd.sh <cluster>}"
helm repo add argo https://argoproj.github.io/argo-helm >/dev/null 2>&1 || true
helm repo update argo >/dev/null
h "$C" upgrade --install argocd argo/argo-cd --version "${PIN_ARGOCD_CHART}" -n argocd --create-namespace \
  -f platform/argocd/values.yaml --wait --timeout 10m
grep -q "YOUR_GITHUB_USER" gitops/bootstrap/root-"$C".yaml && die "run gitops/set-repo-url.sh first, then commit and push"
k "$C" apply -f "gitops/bootstrap/root-${C}.yaml"
echo "Argo CD UI: kubectl --context $(ctx "$C") -n argocd port-forward svc/argocd-server 8080:80"
echo "Password:   kubectl --context $(ctx "$C") -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d"
