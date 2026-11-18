# shellcheck shell=bash
# shellcheck disable=SC2034  # preset variables are consumed by the scripts that source this file
# Shared helpers for bench scripts - the SAME library as Lab 02, new presets. bash 3.2 compatible.
TOOLBOX_NS="lfy-bench"

toolbox_pod() {
  kubectl -n "${TOOLBOX_NS}" get pod -l app=bench-toolbox --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

# Presets: traffic shape, SLO for AIPerf goodput, GPU share for chargeback (3g=3/7, 2g=2/7, 1g=1/7).
# The guarded path uses the 3g + 2g + 1g slices -> share 6/7: safety is not free, and chargeback should say so.
set_preset() {
  PER_LEVEL=6; MIN_REQ=16; GOODPUT=""; POOL="mig"; NODE_HOURLY="${GPU_HOURLY_USD}"; STREAM=0
  KIND="file"; DATA=""; ISL=""; OSL=""; MODEL="${NIM_LLM_MODEL}"; TOK="${TOKENIZER}"
  case "$1" in
    agent-llm)          URL="http://triage-llm.lfy-agent.svc:8000"; NS="lfy-agent"; DATA="/results/agent-steps.jsonl"; SHARE=0.4286; GOODPUT="request_latency:4000" ;;
    agent-llm-guarded)  URL="http://guarded-gateway.lfy-safety.svc:8080"; NS="lfy-agent"; DATA="/results/agent-steps.jsonl"; SHARE=0.8571; GOODPUT="request_latency:4000" ;;
    agent-llm-stream)   URL="http://triage-llm.lfy-agent.svc:8000"; NS="lfy-agent"; DATA="/results/agent-steps.jsonl"; SHARE=0.4286; STREAM=1; GOODPUT="time_to_first_token:1000 inter_token_latency:60" ;;
    safety-guard)       URL="http://safety-guard.lfy-safety.svc:8000"; NS="lfy-safety"; MODEL="${NIM_SAFETY_MODEL}"; DATA="/results/safety-checks.jsonl"; SHARE=0.2857; GOODPUT="request_latency:500"; PER_LEVEL=10 ;;
    safety-flood)       URL="http://safety-guard.lfy-safety.svc:8000"; NS="lfy-safety"; MODEL="${NIM_SAFETY_MODEL}"; KIND="synthetic"; ISL=900; OSL=12; SHARE=0.2857 ;;
    *) echo "unknown preset: $1 (agent-llm|agent-llm-guarded|agent-llm-stream|safety-guard|safety-flood)" >&2; return 1 ;;
  esac
}

# aiperf_run <concurrency> <request-count> <remote-artifact-dir> [extra aiperf args...]
aiperf_run() {
  _c="$1"; _n="$2"; _out="$3"; shift 3
  set -- --model "${MODEL}" --url "${URL}" --endpoint-type chat --tokenizer "${TOK}" \
         --concurrency "${_c}" --request-count "${_n}" --warmup-request-count "${_c}" \
         --artifact-dir "${_out}" --ui-type simple ${1+"$@"}
  if [ "${STREAM}" = "1" ]; then set -- "$@" --streaming; fi
  if [ "${KIND}" = "synthetic" ]; then
    set -- "$@" --isl "${ISL}" --isl-stddev 0 --osl "${OSL}" --osl-stddev 0 --extra-inputs "min_tokens:${OSL}" "ignore_eos:true"
  else
    set -- "$@" --input-file "${DATA}" --custom-dataset-type single_turn
  fi
  if [ -n "${GOODPUT}" ]; then set -- "$@" --goodput "${GOODPUT}"; fi
  kubectl -n "${TOOLBOX_NS}" exec "$(toolbox_pod)" -- aiperf profile "$@"
}

fetch_results() {
  mkdir -p "$(dirname "$2")"
  kubectl -n "${TOOLBOX_NS}" cp "$(toolbox_pod):$1" "$2" >/dev/null
}
