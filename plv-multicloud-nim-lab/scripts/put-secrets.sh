#!/usr/bin/env bash
# Upload NGC_API_KEY (and HF_TOKEN) from env/lab.env into the CLOUD secret manager of one cluster.
# External Secrets Operator then syncs them into Kubernetes - no secret values in Git, Terraform state, or Helm values.
# Usage: scripts/put-secrets.sh aks|gke|eks|oke
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
C="${1:?usage: scripts/put-secrets.sh aks|gke|eks|oke}"
: "${NGC_API_KEY:?NGC_API_KEY is empty}"
put() {  # put <name> <value>
  case "$C" in
    aks) for _ in 1 2 3 4 5 6; do az keyvault secret set --vault-name "${AZ_KV_NAME:?}" -n "$1" --value "$2" -o none 2>/dev/null && return 0; echo "  waiting for Key Vault RBAC..."; sleep 20; done; return 1 ;;
    gke) printf '%s' "$2" | gcloud secrets versions add "$1" --project "${GCP_PROJECT}" --data-file=- >/dev/null ;;
    eks) aws secretsmanager put-secret-value --region "${AWS_REGION}" --secret-id "$1" --secret-string "$2" >/dev/null ;;
    oke) oci vault secret update-base64 --secret-id "$(eval echo "\${OCI_SECRET_OCID_$(echo "$1" | tr '-' '_' | tr '[:lower:]' '[:upper:]')}")" \
           --secret-content-content "$(printf '%s' "$2" | base64)" >/dev/null ;;
    *) echo "unknown cluster $C"; exit 1 ;;
  esac
}
put plv-ngc-api-key "${NGC_API_KEY}" && echo "NGC key stored in ${C}'s secret manager"
if [ -n "${HF_TOKEN:-}" ]; then put plv-hf-token "${HF_TOKEN}" && echo "HF token stored in ${C}'s secret manager"; fi
