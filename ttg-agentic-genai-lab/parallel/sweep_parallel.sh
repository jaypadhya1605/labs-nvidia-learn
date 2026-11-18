#!/usr/bin/env bash
# M7 - tensor vs pipeline vs data parallelism for SERVING, measured on one 8x H100 SXM node with vLLM 0.31.
# Every configuration serves the same Tern agent (served model name ttg-llm) and gets the same two benchmarks:
#   1. the agent itself: python -m ttg.bench.agent_bench (task latency, tokens, $ per 1,000 disruptions)
#   2. AIPerf with an agent-shaped synthetic load (3,000 tokens in, 100 out) - 🔁 Labs 02-04 sweeps
#
# Usage: parallel/sweep_parallel.sh [plan] [--dry-run]
#   plan all  (default) both plans below, 8 configurations (~60-90 min with first-time downloads)
#   plan 70b  Llama 3.1 70B FP8 (nvidia/Llama-3.1-70B-Instruct-FP8, ModelOpt checkpoint, ~71 GB):
#             TP 2, TP 4, TP 8, and PP 2 x TP 4                                -> latency vs GPUs, TP vs PP
#   plan 8b   Llama 3.1 8B BF16: TP 1, then TP 2, PP 2, DP 2 (2 GPUs each)     -> same 2 GPUs, three ways
# Env: HF_TOKEN (Llama license accepted), GPU_HOURLY or GPU_HOURLY_USD (USD per GPU-hour; default 2.99), GPU_NAME,
#      LEVELS (agent_bench concurrency levels, default 1,8,32), N (tasks per level, default 48),
#      PIN_VLLM_IMAGE / PIN_VLLM_TAG (env/pins.env; default vllm/vllm-openai:v0.31.0).
# Same conventions as engines/launch.sh: container ttg-engine (engines/stop.sh removes it), port 8000, served model
# name ttg-llm, usage with cached-token details on every response. One engine owns the GPUs at a time.
# Predictions to beat: python3 -m parallel.parallelism_calc --infer --model llama-3.1-70b --precision fp8 --tp 2,4,8
set -euo pipefail
cd "$(dirname "$0")/.."
for f in env/pins.env env/lab.env; do      # pins + your keys (HF_TOKEN) and GPU_HOURLY_USD (env/lab.env.example)
  if [ -f "${f}" ]; then
    # shellcheck source=/dev/null
    . "${f}"
  fi
done

PLAN="all"
DRY=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY=1 ;;
    70b|8b|all) PLAN="${arg}" ;;
    *) echo "unknown argument: ${arg} (plans: 70b, 8b, all)" >&2; exit 2 ;;
  esac
done

IMAGE="${PIN_VLLM_IMAGE:-vllm/vllm-openai}:${PIN_VLLM_TAG:-v0.31.0}"
GPU_HOURLY="${GPU_HOURLY:-${GPU_HOURLY_USD:-2.99}}"
GPU_NAME="${GPU_NAME:-H100-SXM}"
LEVELS="${LEVELS:-1,8,32}"
N="${N:-48}"
NAME="ttg-engine"
OUT="results/parallel/$(date -u +%Y%m%dT%H%M%SZ)"

# label|model|tp|pp|dp|extra vLLM flags. The FP8 checkpoint is pre-quantized (vLLM detects ModelOpt from its config), so
# no --quantization flag and no BF16 load spike (online --quantization fp8 would first load 141 GB of BF16 weights).
configs_70b() {
  cat <<'EOF'
vllm-70b-tp2|nvidia/Llama-3.1-70B-Instruct-FP8|2|1|1|
vllm-70b-tp4|nvidia/Llama-3.1-70B-Instruct-FP8|4|1|1|
vllm-70b-tp8|nvidia/Llama-3.1-70B-Instruct-FP8|8|1|1|
vllm-70b-pp2tp4|nvidia/Llama-3.1-70B-Instruct-FP8|4|2|1|
EOF
}
configs_8b() {
  cat <<'EOF'
vllm-8b-tp1|meta-llama/Llama-3.1-8B-Instruct|1|1|1|
vllm-8b-tp2|meta-llama/Llama-3.1-8B-Instruct|2|1|1|
vllm-8b-pp2|meta-llama/Llama-3.1-8B-Instruct|1|2|1|
vllm-8b-dp2|meta-llama/Llama-3.1-8B-Instruct|1|1|2|
EOF
}

device_list() {   # first N GPUs: 0,1,...
  _n="$1"; _out=""; _i=0
  while [ "${_i}" -lt "${_n}" ]; do _out="${_out}${_out:+,}${_i}"; _i=$((_i + 1)); done
  echo "${_out}"
}

run_one() {
  _label="$1"; _model="$2"; _tp="$3"; _pp="$4"; _dp="$5"; _extra="$6"
  _gpus=$((_tp * _pp * _dp))
  # shellcheck disable=SC2086  # _extra holds zero or more vLLM flags
  set -- docker run -d --name "${NAME}" --gpus "\"device=$(device_list "${_gpus}")\"" --ipc=host -p 127.0.0.1:8000:8000 \
    -e HF_TOKEN -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" --entrypoint vllm "${IMAGE}" \
    serve "${_model}" --served-model-name ttg-llm --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "${_tp}" --pipeline-parallel-size "${_pp}" --data-parallel-size "${_dp}" \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --enable-prefix-caching \
    --enable-prompt-tokens-details --enable-force-include-usage ${_extra}
  if [ "${DRY}" = "1" ]; then
    printf '%q ' "$@"; echo
    echo "TTG_BASE_URL=http://localhost:8000/v1 TTG_MODEL=ttg-llm python3 -m ttg.bench.agent_bench --label ${_label} --engine vllm --levels ${LEVELS} --n ${N} --gpus ${_gpus} --gpu-hourly ${GPU_HOURLY} --gpu-name ${GPU_NAME} --metrics-url http://localhost:8000/metrics"
    echo "aiperf profile --model ttg-llm --url http://localhost:8000 --endpoint-type chat --streaming --tokenizer ${_model} --concurrency 32 --request-count 256 --warmup-request-count 32 --isl 3000 --isl-stddev 0 --osl 100 --osl-stddev 0 --artifact-dir ${OUT}/${_label}/aiperf-c32 --ui-type simple"
    return 0
  fi
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  mkdir -p "${OUT}/${_label}"
  "$@" >/dev/null
  echo "== ${_label}: TP ${_tp} x PP ${_pp} x DP ${_dp} on ${_gpus} GPU(s) - waiting for /health (first run downloads weights)"
  _t=0
  until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
    sleep 10; _t=$((_t + 10))
    if ! docker ps -q -f name="${NAME}" | grep -q .; then
      docker logs "${NAME}" > "${OUT}/${_label}/engine.log" 2>&1 || true
      echo "   engine exited - see ${OUT}/${_label}/engine.log"; return 1
    fi
    if [ "${_t}" -ge 2400 ]; then echo "   timeout after 40 min"; return 1; fi
  done
  echo "   ready after ${_t}s"
  TTG_BASE_URL=http://localhost:8000/v1 TTG_MODEL=ttg-llm python3 -m ttg.bench.agent_bench --label "${_label}" \
    --engine vllm --levels "${LEVELS}" --n "${N}" --gpus "${_gpus}" --gpu-hourly "${GPU_HOURLY}" \
    --gpu-name "${GPU_NAME}" --metrics-url http://localhost:8000/metrics | tee "${OUT}/${_label}/agent_bench.txt"
  if command -v aiperf >/dev/null 2>&1; then
    aiperf profile --model ttg-llm --url http://localhost:8000 --endpoint-type chat --streaming --tokenizer "${_model}" \
      --concurrency 32 --request-count 256 --warmup-request-count 32 --isl 3000 --isl-stddev 0 --osl 100 \
      --osl-stddev 0 --artifact-dir "${OUT}/${_label}/aiperf-c32" --ui-type simple | tail -25 \
      | tee "${OUT}/${_label}/aiperf.txt"
  else
    echo "   (aiperf not installed: pip install aiperf==${PIN_AIPERF:-0.13.0} to add the engine-level view)"
  fi
  docker logs "${NAME}" > "${OUT}/${_label}/engine.log" 2>&1 || true
  docker rm -f "${NAME}" >/dev/null
}

run_plan() {
  "configs_$1" | while IFS='|' read -r label model tp pp dp extra; do
    run_one "${label}" "${model}" "${tp}" "${pp}" "${dp}" "${extra}" || echo "   ${label} failed - continuing"
  done
}

if [ "${DRY}" = "0" ]; then
  mkdir -p "${OUT}"
  nvidia-smi topo -m > "${OUT}/topo.txt" 2>&1 || true
  echo "Results -> ${OUT}   (image ${IMAGE}, ${GPU_HOURLY} USD per GPU-hour)"
fi
case "${PLAN}" in
  70b) run_plan 70b ;;
  8b) run_plan 8b ;;
  all) run_plan 70b; run_plan 8b ;;
esac
if [ "${DRY}" = "0" ]; then
  echo
  echo "Compare: python3 -m ttg.bench.ledger results/bench/vllm-70b-*.json results/bench/vllm-8b-*.json --slo-p95-s 12"
  echo "Predicted: python3 -m parallel.parallelism_calc --infer --model llama-3.1-70b --precision fp8 --tp 2,4,8 --batch 1,32"
fi
