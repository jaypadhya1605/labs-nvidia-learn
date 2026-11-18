#!/usr/bin/env bash
# Delete everything the lab created (resource group + node pools + disks + Grafana + Key Vault).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
printf 'Delete resource group %s and ALL lab resources? [y/N] ' "${RG}"
read -r answer
[ "${answer}" = "y" ] || { echo "Aborted."; exit 0; }
az group delete -n "${RG}" --yes --no-wait
echo "Deletion started. Key Vault ${KV_NAME:-<kv>} stays soft-deleted for 7 days; purge with: az keyvault purge -n ${KV_NAME:-<kv>}"
rm -f env/outputs.env
