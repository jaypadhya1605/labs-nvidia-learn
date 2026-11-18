#!/usr/bin/env bash
# M9 step 4 (laptop, through the tunnel, AFTER infra/tf.sh dc airgap) - install the GPU stack and Plover with ZERO
# Internet: charts come from your laptop/the VM, images from the mirror, the model from the host cache.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
CHART_DIR="results/airgap-charts"; mkdir -p "${CHART_DIR}"
if [ ! -f "${CHART_DIR}/gpu-operator-${PIN_GPU_OPERATOR}.tgz" ]; then   # the laptop side of the gap has Internet
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
  helm repo update nvidia >/dev/null
  helm pull nvidia/gpu-operator --version "${PIN_GPU_OPERATOR}" -d "${CHART_DIR}"
fi
echo "== 1/3 GPU Operator (host driver, RKE2 containerd socket)"
h rke2 upgrade --install gpu-operator "${CHART_DIR}/gpu-operator-${PIN_GPU_OPERATOR}.tgz" -n gpu-operator --create-namespace \
  -f platform/gpu-operator/rke2.yaml --wait --timeout 20m
i=0   # the operator creates the validator DaemonSet asynchronously after helm returns
until [ -n "$(k rke2 -n gpu-operator get pod -l app=nvidia-operator-validator -o name 2>/dev/null)" ]; do
  i=$((i + 1)); [ "$i" -le 90 ] || die "validator pods never appeared - kubectl -n gpu-operator get pods,clusterpolicy"
  sleep 10
done
k rke2 -n gpu-operator wait --for=condition=Ready pod -l app=nvidia-operator-validator --timeout=15m
k rke2 get node -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,PRODUCT:.metadata.labels.nvidia\.com/gpu\.product'
echo "== 2/3 Plover (LLM as a plain Deployment, hostPath model cache, no NGC key)"
API_IMG="ghcr.io/$(echo "${GITHUB_OWNER}" | tr '[:upper:]' '[:lower:]')/plover-api"
h rke2 upgrade --install plover charts/plover -n plv-serving --create-namespace -f clusters/rke2/values.yaml \
  --set "api.image.repository=${API_IMG}" --wait --timeout 30m
echo "== 3/3 helm test (summarize through the API, inside the air gap)"
h rke2 test plover -n plv-serving --logs
echo "Prove it is air-gapped: datacenter/verify-airgap.sh ; run the suites: kubectl --context ${KCTX_RKE2} -n plv-serving port-forward svc/plover-llm 8000"
