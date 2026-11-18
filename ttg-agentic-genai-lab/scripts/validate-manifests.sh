#!/usr/bin/env bash
# Render the chart for every values file and every engine kind x variant the chart supports, then validate it all
# with kubeconform against Kubernetes 1.34 schemas. CRDs (ExternalSecret, ServiceMonitor, KEDA ScaledObject) are
# skipped BY NAME - a missing core schema is still an error, never a silent skip. 🔁 Lab 04 validate-manifests.
# Usage: scripts/validate-manifests.sh            (needs helm + kubeconform; CI installs both)
set -euo pipefail
cd "$(dirname "$0")/.."
CHART=deploy/helm/ttg
KUBE_VERSION="${KUBE_VERSION:-1.34.0}"
OUT="$(mktemp -d)"
trap 'rm -rf "${OUT}"' EXIT

render() {   # render <name> <helm args...>
  _name="$1"; shift
  helm template ttg "${CHART}" -n ttg "$@" > "${OUT}/${_name}.yaml"
}

for v in kind aks gke; do
  render "values-${v}" -f "deploy/values/${v}.yaml"
done

# every engine kind x every variant it supports (unsupported pairs must FAIL to render - checked below)
variants_for() {
  case "$1" in
    vllm|sglang) echo "base nocache fp8 fp8kv spec eager lora tp2 pp2 dp2" ;;
    trtllm) echo "base nocache fp8 fp8kv tp2 pp2" ;;
    nim) echo "base nocache" ;;
  esac
}
for kind in vllm sglang trtllm nim; do
  for variant in $(variants_for "${kind}"); do
    render "${kind}-${variant}" -f deploy/values/aks.yaml --set "engine.kind=${kind}" --set "engine.variant=${variant}" \
      --set engine.lora.pvcName=ttg-adapters --set engine.fp8Mode=checkpoint
  done
done
# online FP8 (aks.yaml default: A100 cannot run the ModelOpt FP8 checkpoint)
for kind in vllm sglang; do
  render "${kind}-fp8-online" -f deploy/values/aks.yaml --set "engine.kind=${kind}" --set engine.variant=fp8
done

# autoscaling paths + the hosted-endpoint path (no engine, agent egress on 443 only)
render engine-hpa -f deploy/values/aks.yaml --set engine.autoscaling.enabled=true
render engine-keda -f deploy/values/aks.yaml --set engine.kind=sglang --set engine.autoscaling.enabled=true \
  --set engine.autoscaling.mode=keda
render hosted -f deploy/values/kind.yaml --set engine.enabled=false --set agent.baseUrl=https://integrate.api.nvidia.com/v1 \
  --set agent.apiKeySecret.name=nvidia-api-key
# M8: the engine-only VLM release, and the main release pointing its guest MCP server at it
render vlm-addon -f deploy/values/gke.yaml -f deploy/values/vlm-addon.yaml
render vlm-client -f deploy/values/gke.yaml --set mcp.guest.vlm.baseUrl=http://ttg-vlm:8000/v1 --set mcp.guest.vlm.model=ttg-vlm

# guard rails must refuse these at render time
for bad in "engine.kind=trtllm engine.variant=spec" "engine.kind=nim engine.variant=fp8" "engine.kind=vllm engine.variant=lora" \
           "engine.kind=mock engine.variant=fp8" "image.tag=latest" "engine.images.vllm.tag=latest" \
           "engine.kind=trtllm engine.variant=fp8 engine.fp8Mode=online" "engine.kind=trtllm engine.autoscaling.enabled=true"; do
  set -- -f deploy/values/aks.yaml
  for kv in ${bad}; do set -- "$@" --set "${kv}"; done
  if helm template ttg "${CHART}" -n ttg "$@" >/dev/null 2>&1; then
    echo "guard rail missing: rendering with '${bad}' should fail" >&2
    exit 1
  fi
done

# shellcheck disable=SC2016  # literal ${ on purpose
if grep -n '\${[A-Za-z_]*}' "${OUT}"/*.yaml | grep -v 'SUB\|NAME_ARGS'; then
  echo "unrendered \${VAR} above" >&2
  exit 1
fi
echo "rendered $(find "${OUT}" -name '*.yaml' | wc -l | tr -d ' ') manifests sets"
kubeconform -strict -summary -kubernetes-version "${KUBE_VERSION}" \
  -skip ExternalSecret,ServiceMonitor,ScaledObject "${OUT}"/*.yaml
