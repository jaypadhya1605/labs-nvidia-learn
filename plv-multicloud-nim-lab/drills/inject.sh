#!/usr/bin/env bash
# M12 break-fix drills - multi-cloud edition (1 repeats Lab 02 on purpose). Diagnose from SYMPTOMS only, fix, then
# say the one-liner to the customer. Drills that hot-patch live objects pause Argo CD auto-sync first - exactly what
# happens in real outages ("someone turned off sync and patched prod"); reset re-enables sync and Git heals the rest.
# Usage: drills/inject.sh <cluster> <1-9>        drills/inject.sh <cluster> reset <1-9>
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
CL="${1:?usage: drills/inject.sh <cluster> [reset] <n>}"; shift
valid_cluster "${CL}" || die "unknown cluster ${CL}"
NS=plv-serving
LLM_SEL="app.kubernetes.io/name=plover-llm"
# App-of-apps: plv-root owns the 'plover' Application object, so pausing only 'plover' would be self-healed back
# by the root within minutes. Pause root first, resume it last.
AUTO='{"spec":{"syncPolicy":{"automated":{"prune":true,"selfHeal":true}}}}'
OFF='{"spec":{"syncPolicy":{"automated":null}}}'
pause_sync()  { for a in plv-root plover; do k "${CL}" -n argocd patch application "$a" --type merge -p "${OFF}" >/dev/null; done; }
resume_sync() { for a in plover plv-root; do k "${CL}" -n argocd patch application "$a" --type merge -p "${AUTO}" >/dev/null; done
                echo "auto-sync restored on ${CL}: Argo CD re-applies Git"; }
gpu_nodes()   { k "${CL}" get nodes -l plv.io/pool=gpu -o jsonpath='{.items[*].metadata.name}'; }

inject() {
  case "$1" in
    1) k "${CL}" create namespace plv-drills --dry-run=client -o yaml | k "${CL}" apply -f - >/dev/null
       k "${CL}" apply -f - << YAML
apiVersion: v1
kind: Pod
metadata: {name: drill1-no-toleration, namespace: plv-drills}
spec:
  nodeSelector: {plv.io/pool: gpu}
  containers: [{name: c, image: "busybox:1.37", command: ["sleep", "3600"], resources: {limits: {nvidia.com/gpu: 1}}}]
YAML
       ;;
    2) pause_sync
       k "${CL}" -n "${NS}" patch nimservice plover-llm --type json \
         -p '[{"op":"add","path":"/spec/env/-","value":{"name":"NIM_MODEL_PROFILE","value":"vllm-bf16-tp2-pp1"}}]' ;;
    3) pause_sync
       k "${CL}" -n "${NS}" patch externalsecret ngc-api-secret --type json \
         -p '[{"op":"replace","path":"/spec/data/0/remoteRef/key","value":"plv-ngc-api-key-renamed"}]'
       k "${CL}" -n "${NS}" delete secret ngc-api-secret --ignore-not-found
       k "${CL}" -n "${NS}" delete pod -l "${LLM_SEL}" --wait=false ;;
    4) pause_sync
       k "${CL}" -n "${NS}" set env deploy/plover-api CANARY_MODEL=plover-style-v9 CANARY_WEIGHT=50 ;;
    5) for n in $(gpu_nodes); do k "${CL}" cordon "$n"; done
       k "${CL}" -n "${NS}" delete pod -l "${LLM_SEL}" --wait=false ;;
    6) k "${CL}" -n "${NS}" patch nimservice plover-llm --type merge -p '{"spec":{"replicas":0}}'
       echo "Now watch who wins: kubectl --context $(ctx "${CL}") -n argocd get application plover -w" ;;
    7) pause_sync
       k "${CL}" -n "${NS}" patch nimservice plover-llm --type merge -p '{"spec":{"replicas":2}}' ;;
    8) [ "${CL}" = "rke2" ] || die "drill 8 is for the air-gapped rke2 cluster"
       h rke2 upgrade plover charts/plover -n "${NS}" --reuse-values --set llm.image.tag=2.0.14 ;;
    9) [ "${CL}" = "aks" ] || die "drill 9 needs networkPolicy.allowNamespaces (AKS values)"
       pause_sync
       k "${CL}" -n "${NS}" patch networkpolicy plover-ingress --type json -p '[{"op":"remove","path":"/spec/ingress/0/from/2"}]'
       echo "Probe: kubectl --context $(ctx "${CL}") -n plv-mlops run np-probe --rm -i --restart=Never --image=curlimages/curl:8.17.0 -- curl -m 5 -sS http://plover-llm.plv-serving.svc:8000/v1/models" ;;
    *) die "unknown drill $1 (1-9)" ;;
  esac
  echo "Drill $1 injected on ${CL}. Start from the symptom (kubectl get/describe/logs/events, Grafana, Argo CD) - not from this script."
}

reset() {
  case "$1" in
    1) k "${CL}" delete namespace plv-drills --ignore-not-found ;;
    2|3|4|7|9) resume_sync
       if [ "$1" = "3" ]; then k "${CL}" -n "${NS}" annotate externalsecret ngc-api-secret force-sync="$(date +%s)" --overwrite >/dev/null || true; fi ;;
    5) for n in $(gpu_nodes); do k "${CL}" uncordon "$n"; done ;;
    6) echo "Nothing to reset - Argo CD self-heal already did it (that was the point)." ;;
    8) h rke2 upgrade plover charts/plover -n "${NS}" --reuse-values --set "llm.image.tag=${PIN_NIM_LLM_TAG}" ;;
    *) die "unknown drill $1" ;;
  esac
}

if [ "${1:-}" = "reset" ]; then reset "${2:?drill number}"; else inject "${1:?drill number}"; fi
