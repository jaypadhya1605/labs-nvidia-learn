# shellcheck shell=bash
# shellcheck disable=SC2034  # preset variables are consumed by the scripts that source this file
# 🔁 Lab 02/03's bench library, now cluster-aware: every function takes the cluster first (aks|gke|eks|oke|rke2).
# Source after env/lab.env and scripts/lib.sh. bash 3.2 compatible.
TOOLBOX_NS="plv-serving"     # the chart deploys the AIPerf toolbox next to the product (bench.toolbox.enabled)

toolbox_pod() {
  k "$1" -n "${TOOLBOX_NS}" get pod -l app=bench-toolbox --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

# Hourly GPU node price for cost columns: GPU_HOURLY_<CLUSTER> from env/lab.env (defaults = selector/instances.yaml).
gpu_hourly() {
  _u="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  eval "echo \"\${GPU_HOURLY_${_u}:-0}\""
}

# Presets: traffic shape + AIPerf goodput SLO. PLV_MODEL / PLV_TOKENIZER switch the model (add-on: Nemotron 3.5).
set_preset() {
  PER_LEVEL=6; MIN_REQ=16; GOODPUT=""; STREAM=0; KIND="file"; DATA=""; ISL=""; OSL=""
  MODEL="${PLV_MODEL:-meta/llama-3.1-8b-instruct}"; TOK="${PLV_TOKENIZER:-meta-llama/Llama-3.1-8B-Instruct}"
  URL="${PLV_URL:-http://plover-llm.plv-serving.svc:8000}"   # M6: point at trtllm-serve with PLV_URL
  case "$1" in
    llm-long)     DATA="/results/priorauth-long.jsonl"; GOODPUT="request_latency:8000" ;;
    llm-adapter)  DATA="/results/priorauth-adapter.jsonl"; MODEL="${PLV_ADAPTER:-plover-style-v1}"; GOODPUT="request_latency:8000" ;;
    llm-stream)   DATA="/results/priorauth-long.jsonl"; STREAM=1; GOODPUT="time_to_first_token:1000 inter_token_latency:40" ;;
    api)          URL="http://plover-api.plv-serving.svc:8080"; DATA="/results/priorauth-long.jsonl"; GOODPUT="request_latency:8000" ;;
    synth-600-200)  KIND="synthetic"; ISL=600; OSL=200; STREAM=1; GOODPUT="time_to_first_token:1000 inter_token_latency:40" ;;
    synth-3000-400) KIND="synthetic"; ISL=3000; OSL=400; STREAM=1; GOODPUT="time_to_first_token:2000 inter_token_latency:30" ;;
    *) echo "unknown preset: $1 (llm-long|llm-adapter|llm-stream|api|synth-600-200|synth-3000-400)" >&2; return 1 ;;
  esac
}

# aiperf_run <cluster> <concurrency> <request-count> <remote-artifact-dir> [extra aiperf args...]
aiperf_run() {
  _k="$1"; _c="$2"; _n="$3"; _out="$4"; shift 4
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
  k "${_k}" -n "${TOOLBOX_NS}" exec "$(toolbox_pod "${_k}")" -- aiperf profile "$@"
}

fetch_results() {   # fetch_results <cluster> <remote-dir> <local-dir>
  mkdir -p "$(dirname "$3")"
  k "$1" -n "${TOOLBOX_NS}" cp "$(toolbox_pod "$1"):$2" "$3" >/dev/null
}

push_datasets() {   # push_datasets <cluster>: generate the prior-auth datasets locally, copy them into the toolbox
  _tmp="$(mktemp -d)"
  python3 bench/make_datasets.py --out-dir "${_tmp}" >/dev/null
  for f in "${_tmp}"/*.jsonl; do k "$1" -n "${TOOLBOX_NS}" cp "$f" "$(toolbox_pod "$1"):/results/$(basename "$f")"; done
  rm -rf "${_tmp}"
}

# Where does the LLM actually run? -> instance type + GPU product, recorded with every run (compare_clouds.py).
llm_node_facts() {
  _ns="${PLV_LLM_NS:-plv-serving}"   # add-on: PLV_LLM_NS=plv-addon
  _node="$(k "$1" -n "${_ns}" get pod -l "app.kubernetes.io/name=plover-llm" -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null || true)"
  if [ -z "${_node}" ]; then _node="$(k "$1" -n "${_ns}" get pod -o wide 2>/dev/null | awk '/plover-llm/ {print $7; exit}')"; fi
  if [ -z "${_node}" ]; then echo '{}'; return; fi
  k "$1" get node "${_node}" -o json | python3 -c '
import json, sys
n = json.load(sys.stdin)["metadata"]["labels"]
print(json.dumps({"node": n.get("kubernetes.io/hostname"), "instance_type": n.get("node.kubernetes.io/instance-type"),
                  "gpu_product": n.get("nvidia.com/gpu.product"), "gpu_count": n.get("nvidia.com/gpu.count"),
                  "driver": n.get("nvidia.com/cuda.driver-version.full")}))'
}
