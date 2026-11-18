#!/usr/bin/env bash
# M6 engine bake-off on ONE cluster's GPU, inside a GitOps pause (Argo CD would otherwise undo the NIM scale-down).
# Usage: engines/bakeoff.sh <cluster> <pause|trtllm|trtllm-down|triton|triton-perf|triton-down|resume|status>
#   pause        disable Argo CD auto-sync for 'plover' and scale the NIMService to 0 (frees the GPU)
#   trtllm       deploy trtllm-serve with TRTLLM_MODEL (default per cloud), wait, print the AIPerf command
#   triton       deploy the Triton triage pipeline (ENGINE_MODE=build|build-compat|import)
#   resume       delete bake-off workloads and re-enable auto-sync -> Argo CD restores the NIM from Git
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
CL="${1:?usage: engines/bakeoff.sh <cluster> <step>}"; STEP="${2:?step}"
valid_cluster "${CL}" || die "unknown cluster ${CL}"
NS=plv-serving
case "${CL}" in
  gke) DEFAULT_MODEL="nvidia/Llama-3.1-8B-Instruct-NVFP4" ;;   # Blackwell: NVFP4 tensor cores
  eks) DEFAULT_MODEL="nvidia/Llama-3.1-8B-Instruct-FP8" ;;     # L40S (Ada): FP8
  *)   DEFAULT_MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;     # Ampere: BF16
esac
export TRTLLM_MODEL="${TRTLLM_MODEL:-${DEFAULT_MODEL}}"
export ENGINE_MODE="${ENGINE_MODE:-build}"
case "${STEP}" in
  pause)
    # app-of-apps: pause the root FIRST, otherwise it self-heals the child Application's sync policy back on
    for a in plv-root plover; do k "${CL}" -n argocd patch application "$a" --type merge -p '{"spec":{"syncPolicy":{"automated":null}}}'; done
    k "${CL}" -n "${NS}" patch nimservice plover-llm --type merge -p '{"spec":{"replicas":0}}' 2>/dev/null \
      || k "${CL}" -n "${NS}" scale deploy/plover-llm --replicas=0
    echo "GitOps paused and NIM scaled to 0 on ${CL}. The GPU is free for the bake-off. ALWAYS finish with: $0 ${CL} resume" ;;
  trtllm)
    scripts/render.sh engines/trtllm-serve.yaml | k "${CL}" apply -f -
    echo "Waiting for trtllm-serve (${TRTLLM_MODEL}) - first start downloads the checkpoint..."
    k "${CL}" -n "${NS}" rollout status deploy/trtllm-serve --timeout=40m
    echo "Benchmark it with the SAME preset as the NIM:"
    echo "  PLV_URL=http://trtllm-serve.${NS}.svc:8000 PLV_MODEL=${TRTLLM_MODEL} bench/run_sweep.sh ${CL} synth-600-200 \"1 8 32 64 128\"" ;;
  trtllm-down) scripts/render.sh engines/trtllm-serve.yaml | k "${CL}" delete -f - --ignore-not-found ;;
  triton)
    k "${CL}" -n "${NS}" create configmap triage-code --from-file=engines/triton/ --dry-run=client -o yaml | k "${CL}" apply -f -
    scripts/render.sh engines/triton/triage.yaml | k "${CL}" apply -f -
    echo "Init containers: gen-data -> train-export -> build-engine (${ENGINE_MODE}). Follow with:"
    echo "  kubectl --context $(ctx "${CL}") -n ${NS} logs -f deploy/triage-triton -c build-engine" ;;
  triton-perf)
    scripts/render.sh engines/triton/perf-job.yaml | k "${CL}" apply -f -
    k "${CL}" -n "${NS}" wait --for=condition=complete job/triage-perf --timeout=20m
    k "${CL}" -n "${NS}" logs job/triage-perf | tee "results/${CL}-triton-perf-$(date -u +%Y%m%dT%H%M%SZ).txt" ;;
  triton-down)
    scripts/render.sh engines/triton/triage.yaml | k "${CL}" delete -f - --ignore-not-found
    scripts/render.sh engines/triton/perf-job.yaml | k "${CL}" delete -f - --ignore-not-found ;;
  resume)
    "$0" "${CL}" trtllm-down || true
    "$0" "${CL}" triton-down || true
    for a in plover plv-root; do
      k "${CL}" -n argocd patch application "$a" --type merge -p '{"spec":{"syncPolicy":{"automated":{"prune":true,"selfHeal":true}}}}'
    done
    echo "Auto-sync restored: Argo CD re-applies the NIMService from Git (replicas back to the values file)." ;;
  status)
    k "${CL}" -n argocd get application plover -o jsonpath='{.spec.syncPolicy}{"\n"}'
    k "${CL}" -n "${NS}" get nimservice,deploy,pods -o wide ;;
  *) die "unknown step ${STEP}" ;;
esac
