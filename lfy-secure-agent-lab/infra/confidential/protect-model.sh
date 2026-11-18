#!/usr/bin/env bash
# M12 (laptop): encrypt a model directory with a fresh AES-256 key (DEK), wrap the DEK with the attestation-gated
# Key Vault key, and copy both to the CVM. Only an attested CVM can ever unwrap the DEK.
# Usage: infra/confidential/protect-model.sh <model-dir>   e.g. Lab 02's triage ONNX model or any small HF model dir
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=/dev/null
. env/cvm.env
SRC="${1:?model directory}"
mkdir -p results/cvm
openssl rand 32 > results/cvm/dek.bin
tar -C "${SRC}" -czf - . | openssl enc -aes-256-cbc -pbkdf2 -salt -pass file:results/cvm/dek.bin -out results/cvm/model.tar.gz.enc
az keyvault key encrypt --vault-name "${CVM_KV}" --name lfy-model-key --algorithm RSA-OAEP-256 \
  --data-type base64 --value "$(base64 < results/cvm/dek.bin | tr -d '\n')" --query result -o tsv > results/cvm/dek.wrapped.b64
shred -u results/cvm/dek.bin 2>/dev/null || rm -f results/cvm/dek.bin
scp results/cvm/model.tar.gz.enc results/cvm/dek.wrapped.b64 infra/confidential/on-cvm/*.sh "azureuser@${CVM_IP}:~/"
echo "Encrypted model + wrapped key copied. The plaintext key no longer exists outside Key Vault."
