#!/usr/bin/env bash
# M12: deploy the confidential GPU VM + attestation-gated key. Writes env/cvm.env. Delete with: az group delete -n rg-lfy-cvm
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=/dev/null
. env/lab.env
RGC="rg-lfy-cvm"
[ -f "${HOME}/.ssh/id_rsa.pub" ] || ssh-keygen -t rsa -b 4096 -N "" -f "${HOME}/.ssh/id_rsa"
MYIP="$(curl -s https://api.ipify.org)"
az group create -n "${RGC}" -l "${CVM_LOCATION}" -o none
az deployment group create -g "${RGC}" -n lfy-cvm -f infra/confidential/cvm.bicep \
  -p location="${CVM_LOCATION}" sshPublicKey="$(cat "${HOME}/.ssh/id_rsa.pub")" allowedSshCidr="${MYIP}/32" \
     userPrincipalId="$(az ad signed-in-user show --query id -o tsv)" -o none
OUT="$(az deployment group show -g "${RGC}" -n lfy-cvm --query properties.outputs -o json)"
{
  echo "export CVM_IP=\"$(echo "${OUT}" | jq -r .publicIp.value)\""
  echo "export CVM_KV=\"$(echo "${OUT}" | jq -r .kvName.value)\""
  echo "export CVM_KEY_KID=\"$(echo "${OUT}" | jq -r .keyKid.value)\""
  echo "export CVM_MAA=\"$(echo "${OUT}" | jq -r .maaEndpoint.value)\""
} > env/cvm.env
cat env/cvm.env
echo "ssh azureuser@\$CVM_IP   (source env/cvm.env first)"
