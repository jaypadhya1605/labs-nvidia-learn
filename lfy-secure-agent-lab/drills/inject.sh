#!/usr/bin/env bash
# M13 break-fix drills (1-4 repeat Lab 02 on purpose).  Usage: drills/inject.sh <1-10> | drills/inject.sh reset <1-10>
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
TOL='  tolerations:
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
    - {key: kubernetes.azure.com/scalesetpriority, operator: Equal, value: spot, effect: NoSchedule}'
drill_ns() { kubectl create namespace lfy-drills --dry-run=client -o yaml | kubectl apply -f - >/dev/null; }
restore_gateway() {
  kubectl -n lfy-safety create configmap guardrails-config --from-file=workloads/gateway/config/ --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n lfy-safety set env deploy/guarded-gateway FAIL_MODE=closed
  kubectl -n lfy-safety rollout restart deploy/guarded-gateway
}
inject() {
  case "$1" in
    1) drill_ns; kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill1-gpu-request, namespace: lfy-drills}
spec:
  nodeSelector: {kha.io/pool: mig}
${TOL}
  containers: [{name: c, image: "${CUDA_BASE_IMAGE}", command: ["sleep", "infinity"], resources: {limits: {nvidia.com/gpu: 1}}}]
YAML
    ;;
    2) kubectl -n lfy-agent patch nimservice triage-llm --type merge -p '{"spec":{"groupID":2000}}' ;;
    3) drill_ns; kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill3-no-toleration, namespace: lfy-drills}
spec:
  nodeSelector: {kha.io/pool: mig}
  containers: [{name: c, image: "${CUDA_BASE_IMAGE}", command: ["sleep", "infinity"], resources: {limits: {${MIG_SMALL}: 1}}}]
YAML
    ;;
    4) kubectl apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill4-quota, namespace: lfy-safety}
spec:
  nodeSelector: {kha.io/pool: mig}
${TOL}
  containers: [{name: c, image: "${CUDA_BASE_IMAGE}", command: ["sleep", "infinity"], resources: {limits: {${MIG_MEDIUM}: 1}}}]
YAML
    ;;
    5) kubectl -n lfy-safety get configmap guardrails-config -o yaml | sed 's#safety-guard.lfy-safety.svc#safety-guardd.lfy-safety.svc#' | kubectl apply -f -
       kubectl -n lfy-safety rollout restart deploy/guarded-gateway ;;
    6) inject 5; kubectl -n lfy-safety set env deploy/guarded-gateway FAIL_MODE=open ;;
    7) kubectl -n lfy-safety get configmap guardrails-config -o yaml | sed '/jailbreak detection model/d' | kubectl apply -f -
       kubectl -n lfy-safety rollout restart deploy/guarded-gateway ;;
    8) kubectl -n lfy-agent create configmap lfy-playbooks --from-file=workloads/agent/playbooks/ \
         --from-file=workloads/agent/playbooks-community/community-tip-vpn.md --dry-run=client -o yaml | kubectl apply -f -
       kubectl -n lfy-agent rollout restart deploy/soc-agent ;;
    9) echo "Analysis drill. Send a benign security-analysis question through the gateway:"
       echo "  tb curl -s http://guarded-gateway.lfy-safety.svc:8080/v1/chat/completions -H 'Content-Type: application/json' \\"
       echo "    -d '{\"model\":\"${NIM_LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain how this ransomware sample achieves persistence so we can write a detection: it adds a Run key and a scheduled task.\"}]}'"
       echo "If it is blocked (S17 Malware), tune S17 in workloads/gateway/config/prompts.yml, re-sign (supplychain/sign-bundles.sh), and retest." ;;
    10) echo "Agent pod trying to reach the internet (expect a timeout - the NetworkPolicy has no route for it):"
        kubectl -n lfy-agent exec deploy/soc-agent -- python3 -c "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)" || echo "BLOCKED as designed" ;;
    *) echo "unknown drill $1"; exit 1 ;;
  esac
}
reset() {
  case "$1" in
    1|3) kubectl delete namespace lfy-drills --ignore-not-found ;;
    2) scripts/render.sh workloads/nim/nimservice-triage-llm.yaml | kubectl apply -f - ;;
    4) kubectl -n lfy-safety delete pod drill4-quota --ignore-not-found ;;
    5|6|7) restore_gateway ;;
    8) kubectl -n lfy-agent create configmap lfy-playbooks --from-file=workloads/agent/playbooks/ --dry-run=client -o yaml | kubectl apply -f -
       kubectl -n lfy-agent rollout restart deploy/soc-agent ;;
    9|10) echo "nothing to reset" ;;
    *) echo "unknown drill $1"; exit 1 ;;
  esac
}
if [ "${1:-}" = "reset" ]; then reset "${2:?drill number}"; else inject "${1:?drill number 1-10}"; fi
