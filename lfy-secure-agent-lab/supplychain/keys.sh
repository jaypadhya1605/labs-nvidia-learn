#!/usr/bin/env bash
# M11 step 1: an EC P-256 signing key for policy bundles (guardrails config, playbooks). Private key -> Key Vault;
# public key -> ConfigMap the init containers read. Signature enforcement starts OFF (require_signature=false).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
mkdir -p supplychain/keys
if [ ! -f supplychain/keys/signing-key.pem ]; then
  openssl ecparam -name prime256v1 -genkey -noout -out supplychain/keys/signing-key.pem
  openssl ec -in supplychain/keys/signing-key.pem -pubout -out supplychain/keys/signing-pub.pem
fi
az keyvault secret set --vault-name "${KV_NAME}" -n lfy-policy-signing-key --file supplychain/keys/signing-key.pem -o none
for ns in lfy-safety lfy-agent; do
  kubectl -n "${ns}" create configmap lfy-signing --from-file=signing-pub.pem=supplychain/keys/signing-pub.pem \
    --from-literal=require_signature="${REQUIRE_SIGNATURE:-false}" --dry-run=client -o yaml | kubectl apply -f -
done
echo "Signing key ready (private key also in Key Vault ${KV_NAME} as lfy-policy-signing-key)."
