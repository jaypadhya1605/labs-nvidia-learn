#!/usr/bin/env bash
# M11 break-fix drills.  Usage: drills/inject.sh <1-10>    |    drills/inject.sh reset <1-10>
# Read drills/README.md first. Diagnose from symptoms before reading the root cause.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
NODE="$(kubectl get nodes -l kha.io/pool=mig -o jsonpath='{.items[0].metadata.name}')"
TOLERATIONS='  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
    - {key: kubernetes.azure.com/scalesetpriority, operator: Equal, value: spot, effect: NoSchedule}'
drill_ns() { kubectl create namespace kha-drills --dry-run=client -o yaml | kubectl apply -f - >/dev/null; }
reapply_shared() {
  if kubectl -n kha-shared get hpa >/dev/null 2>&1 && [ -n "$(kubectl -n kha-shared get hpa -o name 2>/dev/null)" ]; then
    scripts/render.sh workloads/nim/nimservice-shared-hpa.yaml | kubectl apply -f -
  else
    scripts/render.sh workloads/nim/nimservice-shared.yaml | kubectl apply -f -
  fi
}

inject() {
  case "$1" in
    1) drill_ns; kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill1-gpu-request, namespace: kha-drills}
spec:
  nodeSelector: {kha.io/pool: mig}
${TOLERATIONS}
  containers:
    - name: c
      image: ${CUDA_BASE_IMAGE}
      command: ["sleep", "infinity"]
      resources: {limits: {nvidia.com/gpu: 1}}
YAML
    ;;
    2) kubectl label node "${NODE}" nvidia.com/mig.config=all-9g.90gb --overwrite ;;
    3) kubectl label node "${NODE}" nvidia.com/mig.config=all-disabled --overwrite ;;
    4) kubectl -n kha-shared patch nimservice shared-llm --type merge -p '{"spec":{"groupID":2000}}' ;;
    5) drill_ns; kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill5-no-toleration, namespace: kha-drills}
spec:
  nodeSelector: {kha.io/pool: mig}
  containers:
    - name: c
      image: ${CUDA_BASE_IMAGE}
      command: ["sleep", "infinity"]
      resources: {limits: {${MIG_SMALL}: 1}}
YAML
    ;;
    6) drill_ns
       kubectl -n kha-drills create secret docker-registry ngc-secret --docker-server=nvcr.io \
         --docker-username='$oauthtoken' --docker-password=not-a-real-key --dry-run=client -o yaml | kubectl apply -f -
       kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill6-pull, namespace: kha-drills}
spec:
  imagePullSecrets: [{name: ngc-secret}]
  containers:
    - name: c
      image: ${NIM_EMBED_IMAGE}:${NIM_EMBED_TAG}
      command: ["sleep", "infinity"]
YAML
    ;;
    7) kubectl -n kha-shared patch nimservice shared-llm --type json \
         -p '[{"op":"replace","path":"/spec/env/0/value","value":"--max-model-len 131072"}]' ;;
    8) kubectl -n kha-shared patch nimservice shared-llm --type json \
         -p '[{"op":"replace","path":"/spec/scale/hpa/metrics/0/pods/metric/name","value":"vllm_kv_cache_usage"}]' ;;
    9) echo "Analysis drill (no injection). In Prometheus (localhost:9090) compare:"
       echo "  sum(DCGM_FI_DEV_POWER_USAGE)"
       echo "  sum(max by (UUID) (DCGM_FI_DEV_POWER_USAGE))"
       echo "Explain the difference and which one belongs in an ESG report." ;;
    10) kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill10-quota, namespace: kha-um}
spec:
  nodeSelector: {kha.io/pool: mig}
${TOLERATIONS}
  containers:
    - name: c
      image: ${CUDA_BASE_IMAGE}
      command: ["sleep", "infinity"]
      resources: {limits: {${MIG_LARGE}: 1}}
YAML
    ;;
    *) echo "unknown drill $1"; exit 1 ;;
  esac
  echo "Drill $1 injected. Symptoms first: kubectl get pods -A | grep -v Running ; kubectl describe ... ; logs ..."
}

reset() {
  case "$1" in
    1|5|6) kubectl delete namespace kha-drills --ignore-not-found ;;
    2|3) kubectl label node "${NODE}" nvidia.com/mig.config=all-balanced --overwrite
         platform/set-gpu-mode.sh wait ;;
    4|7|8) reapply_shared ;;
    9) echo "nothing to reset" ;;
    10) kubectl -n kha-um delete pod drill10-quota --ignore-not-found ;;
    *) echo "unknown drill $1"; exit 1 ;;
  esac
}

if [ "${1:-}" = "reset" ]; then reset "${2:?drill number}"; else inject "${1:?drill number 1-10}"; fi
