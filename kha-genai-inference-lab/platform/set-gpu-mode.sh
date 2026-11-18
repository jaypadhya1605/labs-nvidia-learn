#!/usr/bin/env bash
# Switch a GPU pool between sharing modes, or inspect it.
#   mig    -> MIG all-balanced (3g + 2g + 1g + 1g), device plugin mixed strategy   (default tenancy)
#   full   -> MIG disabled, one exclusive nvidia.com/gpu
#   ts     -> MIG disabled, time-slicing: nvidia.com/gpu x2 (no memory isolation)
#   mps    -> MIG disabled, MPS: nvidia.com/gpu x2 (memory limit + compute share per client)
#   status | wait
# Usage: platform/set-gpu-mode.sh <mode> [pool: mig|dynamo]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
MODE="${1:-status}"
POOL="${2:-mig}"
NODE="$(kubectl get nodes -l "kha.io/pool=${POOL}" -o jsonpath='{.items[0].metadata.name}')"
[ -n "${NODE}" ] || { echo "No node with label kha.io/pool=${POOL}"; exit 1; }

gpu_pods() {  # user pods on this node holding NVIDIA resources
  kubectl get pods -A --field-selector "spec.nodeName=${NODE}" -o json | jq -r '
    .items[]
    | select(.metadata.namespace != "gpu-operator")
    | select(.status.phase == "Running" or .status.phase == "Pending")
    | select([.spec.containers[].resources.limits // {} | keys[] | select(startswith("nvidia.com/"))] | length > 0)
    | "\(.metadata.namespace)/\(.metadata.name)"'
}

drain_gpu_pods() {
  kubectl cordon "${NODE}"
  for p in $(gpu_pods); do
    echo "  evicting ${p}"
    kubectl -n "${p%%/*}" delete pod "${p#*/}" --wait=false
  done
  for _ in $(seq 1 60); do
    if [ -z "$(gpu_pods)" ]; then return 0; fi
    sleep 5
  done
  echo "GPU pods still present; MIG Manager needs the GPU idle. Delete their owners (NIMService/Deployment) and retry."
  exit 1
}

wait_mig() {
  echo "Waiting for nvidia.com/mig.config.state=success on ${NODE} (a reboot can take 5-15 min)..."
  for _ in $(seq 1 180); do
    s="$(kubectl get node "${NODE}" -o jsonpath='{.metadata.labels.nvidia\.com/mig\.config\.state}' 2>/dev/null || true)"
    case "${s}" in
      success) echo "  state=success"; return 0 ;;
      failed)  echo "  state=failed -> kubectl logs -n gpu-operator -l app=nvidia-mig-manager -c nvidia-mig-manager --tail=50"; exit 1 ;;
    esac
    sleep 10
  done
  echo "Timed out waiting for MIG Manager"; exit 1
}

set_labels() {  # <mig.config> <device-plugin.config or "-">
  if [ "$2" = "-" ]; then
    kubectl label node "${NODE}" nvidia.com/device-plugin.config- >/dev/null 2>&1 || true
  else
    kubectl label node "${NODE}" "nvidia.com/device-plugin.config=$2" --overwrite
  fi
  kubectl label node "${NODE}" "nvidia.com/mig.config=$1" --overwrite
}

show() {
  kubectl get node "${NODE}" -o json | jq '{node: .metadata.name,
    mig_config: .metadata.labels["nvidia.com/mig.config"],
    mig_state: .metadata.labels["nvidia.com/mig.config.state"],
    device_plugin_config: (.metadata.labels["nvidia.com/device-plugin.config"] // "default (mig-mixed)"),
    gpu_product: .metadata.labels["nvidia.com/gpu.product"],
    allocatable: (.status.allocatable | with_entries(select(.key | startswith("nvidia.com/"))))}'
}

case "${MODE}" in
  mig)  drain_gpu_pods; set_labels all-balanced -; wait_mig ;;
  full) drain_gpu_pods; set_labels all-disabled -; wait_mig ;;
  ts)   drain_gpu_pods; set_labels all-disabled ts-2; wait_mig ;;
  mps)  drain_gpu_pods; set_labels all-disabled mps-2; wait_mig ;;
  wait) wait_mig ;;
  status) show; exit 0 ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
kubectl uncordon "${NODE}"
echo "Letting the device plugin re-register resources..."; sleep 30
show
