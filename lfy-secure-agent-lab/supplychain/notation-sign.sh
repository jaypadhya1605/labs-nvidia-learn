#!/usr/bin/env bash
# M11 step 4: sign both images in ACR with a Key Vault certificate (Notation + azure-kv plugin), then verify.
# Prereqs: brew install notation ; install the azure-kv plugin per Microsoft's "Sign container images with Notation and
# Azure Key Vault" guide. Production enforcement at admission: Ratify + Gatekeeper (see ADR-09).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
CERT=lfy-image-signing
SUBJECT="CN=lanternfly.lab,O=Lanternfly Security,L=Detroit,ST=MI,C=US"
az keyvault certificate show -n "${CERT}" --vault-name "${KV_NAME}" >/dev/null 2>&1 || \
  az keyvault certificate create -n "${CERT}" --vault-name "${KV_NAME}" -p @supplychain/notation-cert-policy.json -o none
KEY_ID="$(az keyvault certificate show -n "${CERT}" --vault-name "${KV_NAME}" --query kid -o tsv)"
notation key list | grep -q lfy-signing || notation key add --plugin azure-kv --id "${KEY_ID}" --plugin-config self_signed=true lfy-signing
TOKEN="$(az acr login -n "${ACR_NAME}" --expose-token --query accessToken -o tsv)"
notation login -u 00000000-0000-0000-0000-000000000000 -p "${TOKEN}" "${ACR_LOGIN_SERVER}"
az keyvault certificate download -n "${CERT}" --vault-name "${KV_NAME}" -f supplychain/out/lfy-signing.pem
notation cert add --type ca --store lfy supplychain/out/lfy-signing.pem 2>/dev/null || true
cat > supplychain/out/trustpolicy.json << JSON
{"version": "1.0", "trustPolicies": [{"name": "lfy-images",
  "registryScopes": ["${ACR_LOGIN_SERVER}/lfy/soc-agent", "${ACR_LOGIN_SERVER}/lfy/guarded-gateway"],
  "signatureVerification": {"level": "strict"}, "trustStores": ["ca:lfy"], "trustedIdentities": ["x509.subject: ${SUBJECT}"]}]}
JSON
notation policy import --force supplychain/out/trustpolicy.json
for app in soc-agent guarded-gateway; do
  DIGEST="$(az acr repository show -n "${ACR_NAME}" --image "lfy/${app}:${APP_TAG}" --query digest -o tsv)"
  notation sign --key lfy-signing "${ACR_LOGIN_SERVER}/lfy/${app}@${DIGEST}"
  notation verify "${ACR_LOGIN_SERVER}/lfy/${app}@${DIGEST}"
done
