#!/usr/bin/env bash
# M11 step 2: sign the safety policy and the playbooks with OpenSSF model-signing, publish the signatures, and turn
# enforcement ON. From now on a changed policy must be re-signed (this script) or the pods will not start.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
command -v model_signing >/dev/null || { echo "pip install 'model-signing>=1.0'"; exit 1; }
model_signing sign key workloads/gateway/config --private_key supplychain/keys/signing-key.pem --signature supplychain/signatures/guardrails-config.sig
model_signing sign key workloads/agent/playbooks --private_key supplychain/keys/signing-key.pem --signature supplychain/signatures/playbooks.sig
for ns in lfy-safety lfy-agent; do
  kubectl -n "${ns}" create configmap lfy-signatures --from-file=supplychain/signatures/ --dry-run=client -o yaml | kubectl apply -f -
done
REQUIRE_SIGNATURE=true supplychain/keys.sh
kubectl -n lfy-safety create configmap guardrails-config --from-file=workloads/gateway/config/ --dry-run=client -o yaml | kubectl apply -f -
kubectl -n lfy-agent create configmap lfy-playbooks --from-file=workloads/agent/playbooks/ --dry-run=client -o yaml | kubectl apply -f -
kubectl -n lfy-safety rollout restart deploy/guarded-gateway
kubectl -n lfy-agent rollout restart deploy/soc-agent deploy/soc-tools
echo "Check: kubectl -n lfy-safety logs deploy/guarded-gateway -c verify-policy   (expect: bundle signature OK)"
