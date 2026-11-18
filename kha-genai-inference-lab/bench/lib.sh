# shellcheck shell=bash
# shellcheck disable=SC2034  # preset variables are consumed by the scripts that source this file
# Shared helpers for bench scripts (bash 3.2 compatible - macOS default). Source after env/lab.env.
TOOLBOX_NS="kha-bench"

toolbox_pod() {
  kubectl -n "${TOOLBOX_NS}" get pod -l app=bench-toolbox --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

# Tenant presets: traffic shape (ISL/OSL), SLOs for AIPerf goodput, slice share for chargeback.
set_preset() {
  PER_LEVEL=6; MIN_REQ=16; GOODPUT=""; POOL="mig"; NODE_HOURLY="${GPU_HOURLY_USD}"
  KIND="synthetic"; DATA=""; ISL=""; OSL=""; MODEL="${NIM_LLM_MODEL}"; TOK="${TOKENIZER}"
  UM_SLO="time_to_first_token:1500 inter_token_latency:60"
  MEMBER_SLO="time_to_first_token:800 inter_token_latency:50"
  case "$1" in
    priorauth)        URL="http://priorauth-llm.kha-um.svc:8000"; NS="kha-um"; ISL=6000; OSL=350; SHARE=0.4286; GOODPUT="${UM_SLO}" ;;
    priorauth-fg)     URL="http://priorauth-llm-fg.kha-um.svc:8000"; NS="kha-um"; ISL=6000; OSL=350; SHARE=0.5; GOODPUT="${UM_SLO}" ;;
    member)           URL="http://shared-llm.kha-shared.svc:8000"; NS="kha-shared"; ISL=600; OSL=200; SHARE=0.2857; GOODPUT="${MEMBER_SLO}" ;;
    member-fg)        URL="http://shared-llm-fg.kha-shared.svc:8000"; NS="kha-shared"; ISL=600; OSL=200; SHARE=0.5; GOODPUT="${MEMBER_SLO}" ;;
    batch)            URL="http://shared-llm.kha-shared.svc:8000"; NS="kha-shared"; ISL=1200; OSL=150; SHARE=0.2857; PER_LEVEL=8 ;;
    batch-fg)         URL="http://shared-llm-fg.kha-shared.svc:8000"; NS="kha-shared"; ISL=1200; OSL=150; SHARE=0.5; PER_LEVEL=8 ;;
    prefix-priorauth) URL="http://priorauth-llm.kha-um.svc:8000"; NS="kha-um"; KIND="file"; DATA="/results/prefix-llama.jsonl"; SHARE=0.4286; GOODPUT="${UM_SLO}" ;;
    dynamo:*)         URL="http://${1#dynamo:}-frontend.kha-dynamo.svc:8000"; NS="kha-dynamo"; KIND="file"; DATA="/results/prefix-qwen3.jsonl"
                      MODEL="${DYNAMO_MODEL}"; TOK="${DYNAMO_MODEL}"; POOL="dynamo"; NODE_HOURLY="${DYN_HOURLY_USD}"; SHARE=1 ;;
    *) echo "unknown preset: $1 (priorauth|priorauth-fg|member|member-fg|batch|batch-fg|prefix-priorauth|dynamo:<dgd>)" >&2; return 1 ;;
  esac
}

# aiperf_run <concurrency> <request-count> <remote-artifact-dir> [extra aiperf args...]
aiperf_run() {
  _c="$1"; _n="$2"; _out="$3"; shift 3
  set -- --model "${MODEL}" --url "${URL}" --endpoint-type chat --streaming --tokenizer "${TOK}" \
         --concurrency "${_c}" --request-count "${_n}" --warmup-request-count "${_c}" \
         --artifact-dir "${_out}" --ui-type simple ${1+"$@"}
  if [ "${KIND}" = "synthetic" ]; then
    set -- "$@" --isl "${ISL}" --isl-stddev 0 --osl "${OSL}" --osl-stddev 0 --extra-inputs "min_tokens:${OSL}" "ignore_eos:true"
  else
    set -- "$@" --input-file "${DATA}" --custom-dataset-type single_turn
  fi
  if [ -n "${GOODPUT}" ]; then set -- "$@" --goodput "${GOODPUT}"; fi
  kubectl -n "${TOOLBOX_NS}" exec "$(toolbox_pod)" -- aiperf profile "$@"
}

# fetch_results <remote-dir> <local-dir>
fetch_results() {
  mkdir -p "$(dirname "$2")"
  kubectl -n "${TOOLBOX_NS}" cp "$(toolbox_pod):$1" "$2" >/dev/null
}
