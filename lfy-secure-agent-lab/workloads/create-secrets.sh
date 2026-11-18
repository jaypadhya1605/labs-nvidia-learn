#!/usr/bin/env bash
# Key Vault is the source of truth for the NGC API key and HF token. This script:
#   1) pushes them from env/lab.env into Key Vault (only if set locally), then
#   2) projects them into each namespace as Kubernetes secrets:
#      ngc-secret (image pull), ngc-api-secret (NGC_API_KEY for NIMCache/NIMService), hf-token-secret (HF_TOKEN)
# Production upgrade path: Secrets Store CSI driver + Workload Identity instead of copying secrets.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
: "${KV_NAME:?KV_NAME missing - run infra/deploy-landing-zone.sh up}"

kv_set() {
  for _ in 1 2 3 4 5 6; do
    if az keyvault secret set --vault-name "${KV_NAME}" -n "$1" --value "$2" -o none 2>/dev/null; then return 0; fi
    echo "  waiting for Key Vault RBAC to propagate..."; sleep 20
  done
  echo "Could not write $1 to Key Vault"; exit 1
}
if [ -n "${NGC_API_KEY}" ]; then kv_set ngc-api-key "${NGC_API_KEY}"; fi
if [ -n "${HF_TOKEN}" ]; then kv_set hf-token "${HF_TOKEN}"; fi

NGC="$(az keyvault secret show --vault-name "${KV_NAME}" -n ngc-api-key --query value -o tsv)"
HF="$(az keyvault secret show --vault-name "${KV_NAME}" -n hf-token --query value -o tsv 2>/dev/null || echo "")"

for ns in lfy-agent lfy-safety lfy-data lfy-bench; do
  kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  # shellcheck disable=SC2016
  kubectl -n "${ns}" create secret docker-registry ngc-secret --docker-server=nvcr.io \
    --docker-username='$oauthtoken' --docker-password="${NGC}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl -n "${ns}" create secret generic ngc-api-secret --from-literal=NGC_API_KEY="${NGC}" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl -n "${ns}" create secret generic hf-token-secret --from-literal=HF_TOKEN="${HF}" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  echo "secrets ready in ${ns}"
done
