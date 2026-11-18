#!/usr/bin/env bash
# Landing zone lifecycle. Run from anywhere; operates from the repo root.
#   up                  resource group + Bicep (AKS, Key Vault, Azure Monitor workspace, Managed Grafana) + MIG GPU pool
#   add-dynamo-pool     2-GPU pool for KAI (M8) and Dynamo (M10)
#   remove-dynamo-pool
#   autoscale-mig on|off   cluster autoscaler 1..2 on the MIG pool (M9 cold-start experiment)
#   pause | resume      az aks stop/start: stops ALL node billing between study sessions
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
CMD="${1:-help}"

spot_args() {
  if [ "${GPU_PRIORITY}" = "Spot" ]; then echo "--priority Spot --eviction-policy Delete --spot-max-price -1"; fi
}

# gpu_pool_add <pool-name> <vm-size> <kha.io/pool value> <initial nvidia.com/mig.config>
gpu_pool_add() {
  echo "Adding GPU pool $1 ($2, ${GPU_PRIORITY}) - GPU Operator will own the driver (--gpu-driver none)"
  # shellcheck disable=SC2046
  az aks nodepool add -g "${RG}" --cluster-name "${AKS_NAME}" -n "$1" \
    --node-vm-size "$2" --node-count 1 --os-sku Ubuntu \
    --gpu-driver none \
    --node-taints "nvidia.com/gpu=present:NoSchedule" \
    --labels "kha.io/pool=$3" "nvidia.com/mig.config=$4" \
    --node-osdisk-size 256 --max-pods 60 \
    $(spot_args) -o none \
  || { echo "Pool creation failed. If Spot capacity is the cause, set GPU_PRIORITY=Regular in env/lab.env and retry."; exit 1; }
}

case "${CMD}" in
  up)
    az group create -n "${RG}" -l "${LOCATION}" --tags project=kha-genai-inference-lab -o none
    USER_PRINCIPAL_ID="$(az ad signed-in-user show --query id -o tsv)"
    export USER_PRINCIPAL_ID PREFIX LOCATION K8S_VERSION
    echo "Deploying Bicep landing zone (8-12 min)..."
    az deployment group create -g "${RG}" -n kha-landing-zone --parameters infra/main.bicepparam -o none
    OUT="$(az deployment group show -g "${RG}" -n kha-landing-zone --query properties.outputs -o json)"
    {
      echo "# Written by infra/deploy-landing-zone.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "export KV_NAME=\"$(echo "${OUT}" | jq -r .kvName.value)\""
      echo "export AMW_ID=\"$(echo "${OUT}" | jq -r .amwId.value)\""
      echo "export GRAFANA_NAME=\"$(echo "${OUT}" | jq -r .grafanaName.value)\""
      echo "export GRAFANA_ENDPOINT=\"$(echo "${OUT}" | jq -r .grafanaEndpoint.value)\""
      echo "export KUBELET_CLIENT_ID=\"$(echo "${OUT}" | jq -r .kubeletClientId.value)\""
      echo "export KUBELET_OBJECT_ID=\"$(echo "${OUT}" | jq -r .kubeletObjectId.value)\""
      echo "export ACR_NAME=\"$(echo "${OUT}" | jq -r .acrName.value)\""
    } > env/outputs.env
    echo "Wrote env/outputs.env"
    az aks get-credentials -g "${RG}" -n "${AKS_NAME}" --overwrite-existing
    gpu_pool_add "${MIG_POOL}" "${GPU_VM_SKU}" mig all-balanced
    kubectl get nodes -L kha.io/pool,node.kubernetes.io/instance-type,kubernetes.azure.com/scalesetpriority
    az aks nodepool show -g "${RG}" --cluster-name "${AKS_NAME}" -n "${MIG_POOL}" --query gpuProfile -o json
    ;;
  add-dynamo-pool)    gpu_pool_add "${DYN_POOL}" "${DYN_VM_SKU}" dynamo all-disabled ;;
  remove-dynamo-pool) az aks nodepool delete -g "${RG}" --cluster-name "${AKS_NAME}" -n "${DYN_POOL}" -o none ;;
  autoscale-mig)
    if [ "${2:-}" = "on" ]; then
      az aks nodepool update -g "${RG}" --cluster-name "${AKS_NAME}" -n "${MIG_POOL}" --enable-cluster-autoscaler --min-count 1 --max-count 2 -o none
    else
      az aks nodepool update -g "${RG}" --cluster-name "${AKS_NAME}" -n "${MIG_POOL}" --disable-cluster-autoscaler -o none
      az aks nodepool scale -g "${RG}" --cluster-name "${AKS_NAME}" -n "${MIG_POOL}" --node-count 1 -o none
    fi ;;
  pause)  az aks stop  -g "${RG}" -n "${AKS_NAME}" ;;
  resume) az aks start -g "${RG}" -n "${AKS_NAME}" && az aks get-credentials -g "${RG}" -n "${AKS_NAME}" --overwrite-existing ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
