#!/usr/bin/env bash
# Run ON the CVM. CPU attestation (SEV-SNP report -> Microsoft Azure Attestation) releases the Key Vault key; the
# released key unwraps the DEK; the DEK decrypts the model in memory-backed storage; Triton serves it.
# Secure-key-release client: build AzureAttestSKR from github.com/Azure/confidential-computing-cvm-guest-attestation
# (cvm-securekey-release-app) and check its README for the exact flags of your version.
set -euo pipefail
: "${CVM_MAA:?export CVM_MAA and CVM_KEY_KID (values from env/cvm.env on your laptop)}"
: "${CVM_KEY_KID:?}"
sudo mkdir -p /dev/shm/model && sudo chown "$(id -u)" /dev/shm/model
sudo ./AzureAttestSKR -a "${CVM_MAA}" -k "${CVM_KEY_KID}" -c imds -u "$(cat ~/dek.wrapped.b64)" > /dev/shm/dek.b64
base64 -d /dev/shm/dek.b64 > /dev/shm/dek.bin
openssl enc -d -aes-256-cbc -pbkdf2 -pass file:/dev/shm/dek.bin -in ~/model.tar.gz.enc | tar -C /dev/shm/model -xzf -
shred -u /dev/shm/dek.bin /dev/shm/dek.b64
ls -la /dev/shm/model
echo "Serve it, e.g. Triton: docker run --rm --gpus all -p 8000:8000 -v /dev/shm/model:/models nvcr.io/nvidia/tritonserver:26.08-py3 tritonserver --model-repository=/models"
