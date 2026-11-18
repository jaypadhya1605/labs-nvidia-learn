#!/usr/bin/env bash
# engines/launch.sh - start ONE inference engine for Lab 05 (M4-M8) as container "ttg-engine" on port 8000.
#
# Usage: engines/launch.sh <engine> <variant>
#   engine : vllm | sglang | trtllm | nim | hf
#   variant: base | nocache | fp8 | fp8kv | spec | eager | lora | tp2 | pp2 | dp2 | vlm
# Every engine serves the model as "ttg-llm" at http://localhost:8000/v1 - the agent, the bench and the prompts never
# change between runs, only the engine does. That is what makes the M4 bake-off fair.
#
# Variants: base = BF16, prefix caching ON, CUDA graphs ON | nocache = prefix caching OFF | fp8 = FP8 weights
#   (FP8_MODE=checkpoint: the ModelOpt checkpoint, same weights on every engine; FP8_MODE=online: vllm/sglang quantize
#   the BF16 weights at load) | fp8kv = FP8 KV cache | spec = n-gram speculative decoding (vllm, sglang) |
#   eager = no CUDA graphs | lora = + adapter ./adapters/ttg-short (vllm, sglang) | tp2/pp2/dp2 = tensor / pipeline /
#   data parallel over every GPU in GPUS (default 0,1; GPUS=0,1,2,3 gives 4-way) | vlm = vLLM + Qwen2.5-VL (M8)
#
# Env (optional): GPUS=0  MODEL=<hf id or path>  FP8_MODE=checkpoint|online  MAX_MODEL_LEN=16384  PORT=8000
#   BIND=127.0.0.1 (host address the port is published on: the engine has no auth and vLLM runs with dev endpoints
#   for cache resets, so reach it from your laptop through an SSH tunnel, not an open port)
#   GPU_MEM_UTIL=0.90 (vllm)  MEM_FRACTION=0.85 (sglang)  HF_HOME_HOST=~/.cache/huggingface  NIM_CACHE_HOST=~/.cache/nim
#   VLLM_CACHE_HOST=~/.cache/vllm (torch.compile cache: the second start of a variant skips compilation)
#   ADAPTER_DIR=./adapters  NIM_MODEL_PROFILE=<id>  EXTRA_ARGS='<engine flags>'  EXTRA_DOCKER='<docker flags>'
#   ENTRYPOINT_WRAPPER='<nsys profile ...>' (vllm only, see profiling/nsys_capture.sh)  WAIT_S=1800
#   DRY_RUN=1  print the docker command + an ENGINE_FLAGS line and exit (CI tests and engines/preflight.sh use this)
# Keys come from the environment by NAME only (HF_TOKEN, NGC_API_KEY) - they are never printed.
#
# Examples:
#   engines/launch.sh vllm base                    # then: python -m ttg.bench.agent_bench --label vllm-base ...
#   engines/launch.sh sglang nocache               # prefix caching off (RadixAttention disabled)
#   FP8_MODE=online engines/launch.sh vllm fp8     # on-the-fly FP8 from BF16 weights
#   GPUS=0,1,2,3 engines/launch.sh vllm tp2        # 4-way tensor parallel (M7)
#   DRY_RUN=1 engines/launch.sh trtllm fp8         # show, do not run
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
if [ -f env/pins.env ]; then . env/pins.env; fi

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "launch.sh: $*" >&2; exit 2; }

ENGINE="${1:-}"
VARIANT="${2:-}"
if [ -z "${ENGINE}" ] || [ -z "${VARIANT}" ] || [ "${ENGINE}" = "-h" ] || [ "${ENGINE}" = "--help" ]; then
  usage
  exit 2
fi
case "${ENGINE}" in vllm|sglang|trtllm|nim|hf) ;; *) die "unknown engine '${ENGINE}' (vllm|sglang|trtllm|nim|hf)" ;; esac
case "${VARIANT}" in
  base|nocache|fp8|fp8kv|spec|eager|lora|tp2|pp2|dp2|vlm) ;;
  *) die "unknown variant '${VARIANT}' (base|nocache|fp8|fp8kv|spec|eager|lora|tp2|pp2|dp2|vlm)" ;;
esac
unsupported() { die "${ENGINE} + ${VARIANT} is not supported: $*"; }

NAME="ttg-engine"
PORT="${PORT:-8000}"
BIND="${BIND:-127.0.0.1}"
case "${BIND}" in 127.0.0.1|0.0.0.0|localhost|"") CLIENT_HOST="localhost" ;; *) CLIENT_HOST="${BIND}" ;; esac
SERVED="${PIN_SERVED_MODEL_NAME:-ttg-llm}"
LORA_NAME="${PIN_LORA_NAME:-ttg-short}"
BASE_MODEL="${PIN_AGENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
FP8_MODEL="${PIN_AGENT_MODEL_FP8:-nvidia/Llama-3.1-8B-Instruct-FP8}"
VLM_MODEL="${PIN_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FP8_MODE="${FP8_MODE:-checkpoint}"
HF_HOME_HOST="${HF_HOME_HOST:-${HOME}/.cache/huggingface}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${HOME}/.cache/vllm}"
NIM_CACHE_HOST="${NIM_CACHE_HOST:-${HOME}/.cache/nim}"
ADAPTER_DIR="${ADAPTER_DIR:-${PWD}/adapters}"
WAIT_S="${WAIT_S:-1800}"
DRY_RUN="${DRY_RUN:-0}"
case "${FP8_MODE}" in checkpoint|online) ;; *) die "FP8_MODE must be checkpoint or online" ;; esac

case "${VARIANT}" in
  tp2|pp2|dp2) GPUS="${GPUS:-0,1}" ;;
  *) GPUS="${GPUS:-0}" ;;
esac
N_GPUS="$(echo "${GPUS}" | tr ',' '\n' | grep -c '[0-9]' || true)"
case "${VARIANT}" in
  tp2|pp2|dp2) if [ "${N_GPUS}" -lt 2 ]; then die "${VARIANT} needs at least 2 GPUs in GPUS (got '${GPUS}')"; fi ;;
esac
case "${VARIANT}" in
  vlm) MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" ;;
  *) MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" ;;
esac

MODEL_ID="${MODEL:-${BASE_MODEL}}"
if [ "${VARIANT}" = "fp8" ] && [ -z "${MODEL:-}" ] && [ "${FP8_MODE}" = "checkpoint" ]; then MODEL_ID="${FP8_MODEL}"; fi
if [ "${VARIANT}" = "vlm" ] && [ -z "${MODEL:-}" ]; then MODEL_ID="${VLM_MODEL}"; fi

# Token names only: docker copies the values from this shell's environment, so nothing secret is ever printed.
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
DOCKER=(docker run -d --name "${NAME}" --gpus "\"device=${GPUS}\"" -p "${BIND}:${PORT}:8000" -e HF_TOKEN
        -e HUGGING_FACE_HUB_TOKEN)
ENTRY=""
ARGS=()
HEALTH="/health"
METRICS="/metrics"
RESET=""          # endpoint the bench POSTs before each concurrency level (--reset-url) so runs start with a cold cache
AGENT_MODEL="${SERVED}"

need_adapter() {
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${ADAPTER_DIR}/${LORA_NAME}/adapter_config.json" ]; then
    die "no adapter at ${ADAPTER_DIR}/${LORA_NAME} - train it first (finetune/train_lora.py, M6)"
  fi
}

case "${ENGINE}" in
  vllm)
    IMAGE="${PIN_VLLM_IMAGE:-vllm/vllm-openai}:${PIN_VLLM_TAG:-v0.31.0}"
    ENTRY="vllm"
    # --enable-prompt-tokens-details + --enable-force-include-usage: usage.prompt_tokens_details.cached_tokens on
    # every response, so the bench can show prefix-cache hits per task (not only from /metrics).
    ARGS=(serve "${MODEL_ID}" --served-model-name "${SERVED}" --host 0.0.0.0 --port 8000
          --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
          --enable-prompt-tokens-details --enable-force-include-usage)
    case "${VARIANT}" in
      base) ARGS+=(--enable-prefix-caching) ;;
      nocache) ARGS+=(--no-enable-prefix-caching) ;;
      fp8)
        ARGS+=(--enable-prefix-caching)
        if [ "${FP8_MODE}" = "online" ]; then ARGS+=(--quantization fp8); fi ;;
      fp8kv) ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8) ;;
      spec) ARGS+=(--enable-prefix-caching
                   --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}') ;;
      eager) ARGS+=(--enable-prefix-caching --enforce-eager) ;;
      lora)
        need_adapter
        ARGS+=(--enable-prefix-caching --enable-lora --lora-modules "${LORA_NAME}=/adapters/${LORA_NAME}" --max-lora-rank 16)
        DOCKER+=(-v "${ADAPTER_DIR}:/adapters:ro")
        AGENT_MODEL="${LORA_NAME}" ;;
      tp2) ARGS+=(--enable-prefix-caching --tensor-parallel-size "${N_GPUS}") ;;
      pp2) ARGS+=(--enable-prefix-caching --pipeline-parallel-size "${N_GPUS}") ;;
      dp2) ARGS+=(--enable-prefix-caching --data-parallel-size "${N_GPUS}") ;;
      vlm) ARGS+=(--enable-prefix-caching --limit-mm-per-prompt '{"image":2}') ;;
    esac
    # VLLM_SERVER_DEV_MODE=1 adds POST /reset_prefix_cache (the bench's --reset-url). It also adds /sleep,
    # /wake_up and /collective_rpc - lab use only, never in a production deployment (M9 Helm does not set it).
    # ~/.cache/vllm holds the torch.compile cache: mounted, the second start of the same variant skips compilation.
    DOCKER+=(--ipc=host -v "${HF_HOME_HOST}:/root/.cache/huggingface" -v "${VLLM_CACHE_HOST}:/root/.cache/vllm"
             -e VLLM_SERVER_DEV_MODE=1)
    RESET="/reset_prefix_cache"
    ;;
  sglang)
    IMAGE="${PIN_SGLANG_IMAGE:-lmsysorg/sglang}:${PIN_SGLANG_TAG:-v0.5.22}"
    ENTRY="python3"
    # --enable-metrics: Prometheus /metrics (sglang:cache_hit_rate ...); --enable-cache-report: cached_tokens in usage.
    ARGS=(-m sglang.launch_server --model-path "${MODEL_ID}" --served-model-name "${SERVED}" --host 0.0.0.0 --port 8000
          --context-length "${MAX_MODEL_LEN}" --mem-fraction-static "${MEM_FRACTION:-0.85}" --enable-metrics
          --enable-cache-report)
    case "${VARIANT}" in
      base) ;;
      nocache) ARGS+=(--disable-radix-cache) ;;
      fp8) if [ "${FP8_MODE}" = "online" ]; then ARGS+=(--quantization fp8); fi ;;
      fp8kv) ARGS+=(--kv-cache-dtype fp8_e4m3) ;;
      spec) ARGS+=(--speculative-algorithm NGRAM --speculative-num-draft-tokens 4) ;;
      eager) ARGS+=(--disable-cuda-graph) ;;
      lora)
        need_adapter
        ARGS+=(--enable-lora --lora-paths "${LORA_NAME}=/adapters/${LORA_NAME}" --max-lora-rank 16)
        DOCKER+=(-v "${ADAPTER_DIR}:/adapters:ro")
        AGENT_MODEL="${SERVED}:${LORA_NAME}" ;;          # SGLang selects adapters with 'base-model:adapter-name'
      tp2) ARGS+=(--tp-size "${N_GPUS}") ;;
      pp2) ARGS+=(--pp-size "${N_GPUS}") ;;
      dp2) ARGS+=(--dp-size "${N_GPUS}") ;;
      vlm) unsupported "M8 uses 'engines/launch.sh vllm vlm' (Qwen2.5-VL on vLLM)" ;;
    esac
    DOCKER+=(--ipc=host -v "${HF_HOME_HOST}:/root/.cache/huggingface")
    RESET="/flush_cache"                                  # SGLang: empties the radix cache (GET or POST)
    ;;
  trtllm)
    IMAGE="${PIN_TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release}:${PIN_TRTLLM_TAG:-1.3.0rc29}"
    YAML="base.yaml"
    FLAGS=(--host 0.0.0.0 --port 8000 --served_model_name "${SERVED}" --max_batch_size "${MAX_BATCH:-256}"
           --max_num_tokens "${MAX_NUM_TOKENS:-8192}" --max_seq_len "${MAX_MODEL_LEN}")
    case "${VARIANT}" in
      base) ;;
      nocache) YAML="nocache.yaml" ;;
      fp8)
        if [ "${FP8_MODE}" = "online" ]; then
          unsupported "TensorRT-LLM needs a pre-quantized ModelOpt checkpoint - use FP8_MODE=checkpoint (${FP8_MODEL})"
        fi ;;
      fp8kv) YAML="fp8kv.yaml" ;;                       # kv_cache_config.dtype: fp8 (no CLI flag)
      eager) YAML="eager.yaml" ;;
      tp2) FLAGS+=(--tp_size "${N_GPUS}") ;;
      pp2) FLAGS+=(--pp_size "${N_GPUS}") ;;
      spec) unsupported "stretch: verify the speculative_config keys (NGram / Eagle3) for your TensorRT-LLM tag, then add them to a YAML in engines/trtllm/" ;;
      lora) unsupported "stretch: the M6 adapter is served on vllm or sglang (or NIM, Lab 04)" ;;
      dp2) unsupported "run two single-GPU containers behind a load balancer, or use vllm/sglang dp2 (--enable_attention_dp is a different, MoE-oriented feature)" ;;
      vlm) unsupported "M8 uses 'engines/launch.sh vllm vlm'" ;;
    esac
    FLAGS+=(--extra_llm_api_options "/engines/trtllm/${YAML}")
    # trtllm-serve became a command group ('trtllm-serve serve MODEL'); older images take 'trtllm-serve MODEL'.
    DETECT='if trtllm-serve --help 2>&1 | grep -qE "^[[:space:]]+serve([[:space:]]|$)"; then exec trtllm-serve serve "$@"; else exec trtllm-serve "$@"; fi'
    ENTRY="bash"
    ARGS=(-c "${DETECT}" trtllm "${MODEL_ID}" "${FLAGS[@]}")
    DOCKER+=(--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "${HF_HOME_HOST}:/root/.cache/huggingface"
             -v "${PWD}/engines/trtllm:/engines/trtllm:ro")
    ;;
  nim)
    IMAGE="${PIN_NIM_LLM_IMAGE:-nvcr.io/nim/meta/llama-3.1-8b-instruct}:${PIN_NIM_LLM_TAG:-2.0.13}"
    if [ "${DRY_RUN}" != "1" ] && [ -z "${NGC_API_KEY:-}" ]; then die "NGC_API_KEY is not set (NIM downloads its model from NGC)"; fi
    # NIM_SERVED_MODEL_NAME / NIM_ENABLE_KV_CACHE_REUSE / NIM_MODEL_PROFILE: NIM LLM env vars - check the NIM docs
    # for YOUR tag (engines/preflight.sh nim lists them). NIM picks the engine profile per GPU and GPU count.
    DOCKER+=(--shm-size=16g -u "$(id -u)" -e NGC_API_KEY -e "NIM_SERVED_MODEL_NAME=${SERVED}"
             -v "${NIM_CACHE_HOST}:/opt/nim/.cache")
    case "${VARIANT}" in
      base) DOCKER+=(-e NIM_ENABLE_KV_CACHE_REUSE=1) ;;
      nocache) DOCKER+=(-e NIM_ENABLE_KV_CACHE_REUSE=0) ;;
      tp2) DOCKER+=(-e NIM_ENABLE_KV_CACHE_REUSE=1) ;;        # 2 GPUs visible -> NIM selects a TP2 profile
      fp8)
        if [ -z "${NIM_MODEL_PROFILE:-}" ]; then
          unsupported "pick an FP8 profile id with 'list-model-profiles' (🔁 Lab 04 M4) and export NIM_MODEL_PROFILE=<id>"
        fi
        DOCKER+=(-e NIM_ENABLE_KV_CACHE_REUSE=1 -e NIM_MODEL_PROFILE) ;;
      lora) unsupported "NIM LoRA works through NIM_PEFT_SOURCE - see Lab 04 M8; this lab serves the adapter on vllm/sglang" ;;
      *) unsupported "NIM chooses precision, CUDA graphs and parallelism through its profile - try base, nocache, tp2 or fp8 (with NIM_MODEL_PROFILE)" ;;
    esac
    HEALTH="/v1/health/ready"
    METRICS="/v1/metrics"
    ;;
  hf)
    # The naive baseline runs inside the vLLM image (it already has torch, transformers, fastapi and uvicorn).
    IMAGE="${PIN_VLLM_IMAGE:-vllm/vllm-openai}:${PIN_VLLM_TAG:-v0.31.0}"
    if [ "${VARIANT}" != "base" ]; then unsupported "hf is the naive 'before' baseline: base only"; fi
    ENTRY="python3"
    ARGS=(/engines/hf_naive_server.py --model "${MODEL_ID}" --served-model-name "${SERVED}" --port 8000
          --dtype bfloat16)
    DOCKER+=(--ipc=host -v "${HF_HOME_HOST}:/root/.cache/huggingface" -v "${PWD}/engines:/engines:ro")
    METRICS=""
    RESET="/reset_prefix_cache"                           # a no-op: nothing is ever cached in the naive server
    ;;
esac

if [ -n "${EXTRA_DOCKER:-}" ]; then
  read -r -a _extra_docker <<< "${EXTRA_DOCKER}"
  DOCKER+=("${_extra_docker[@]}")
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
  read -r -a _extra_args <<< "${EXTRA_ARGS}"
  ARGS+=("${_extra_args[@]}")        # trtllm: lands in the wrapper's "$@", i.e. after the model, like every other flag
fi
ENGINE_ARGS=("${ARGS[@]+"${ARGS[@]}"}")     # before any wrapper: these are the flags preflight checks
if [ -n "${ENTRYPOINT_WRAPPER:-}" ]; then
  if [ "${ENGINE}" != "vllm" ]; then die "ENTRYPOINT_WRAPPER is supported for vllm only"; fi
  read -r -a _wrap <<< "${ENTRYPOINT_WRAPPER}"
  ARGS=("${_wrap[@]:1}" "${ENTRY}" "${ARGS[@]}")
  ENTRY="${_wrap[0]}"
fi

CMD=("${DOCKER[@]}")
if [ -n "${ENTRY}" ]; then CMD+=(--entrypoint "${ENTRY}"); fi
CMD+=("${IMAGE}")
if [ "${#ARGS[@]}" -gt 0 ]; then CMD+=("${ARGS[@]}"); fi

q() {   # shell-quote one word for display (single quotes, readable JSON)
  case "$1" in
    *[!A-Za-z0-9_./:=,@%+-]*|'') printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}
show() { local w out=""; for w in "$@"; do out="${out}$(q "${w}") "; done; printf '%s\n' "${out% }"; }
engine_flags() {
  local w
  { for w in "${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}"; do case "${w}" in --*) printf '%s\n' "${w}" ;; esac; done
    if [ "${ENGINE}" = "trtllm" ]; then for w in "${FLAGS[@]}"; do case "${w}" in --*) printf '%s\n' "${w}" ;; esac; done; fi
  } | grep -v -- '--entrypoint' | awk '!seen[$0]++' | tr '\n' ' ' | sed 's/ $//'
}

METRICS_URL=""
RESET_URL=""
PROMPT_EXPORT=""
if [ "${VARIANT}" = "lora" ]; then PROMPT_EXPORT=" TTG_PROMPT_VARIANT=short"; fi   # prompt baked into the adapter (M6)
if [ -n "${METRICS}" ]; then METRICS_URL="http://${CLIENT_HOST}:${PORT}${METRICS}"; fi
if [ -n "${RESET}" ]; then RESET_URL="http://${CLIENT_HOST}:${PORT}${RESET}"; fi
bench_cmd() {   # the M4 bench line for THIS engine; ${GPU_HOURLY_USD:-0} is left for your shell to expand
  local levels="1,4,16,32" n="48"
  if [ "${ENGINE}" = "hf" ]; then levels="1,4"; n="8"; fi    # one request at a time: bigger runs only time out
  local c="python -m ttg.bench.agent_bench --label ${ENGINE}-${VARIANT} --engine ${ENGINE} --levels ${levels} --n ${n}"
  if [ "${VARIANT}" = "lora" ]; then c="TTG_PROMPT_VARIANT=short TTG_MODEL=${AGENT_MODEL} ${c}"; fi   # the adapter needs the short prompt
  if [ -n "${1:-}" ]; then c="${c} --gpu-name $(q "$1")"; fi
  c="${c} --gpus ${N_GPUS} --gpu-hourly \${GPU_HOURLY_USD:-0}"
  if [ -n "${METRICS_URL}" ]; then c="${c} --metrics-url ${METRICS_URL}"; fi
  if [ -n "${RESET_URL}" ]; then c="${c} --reset-url ${RESET_URL}"; fi
  printf '%s\n' "${c}"
}

if [ "${DRY_RUN}" = "1" ]; then
  show "${CMD[@]}"
  echo "ENGINE_FLAGS: $(engine_flags)"
  echo "AGENT: export TTG_BASE_URL=http://${CLIENT_HOST}:${PORT}/v1 TTG_MODEL=${AGENT_MODEL}${PROMPT_EXPORT}  # label ${ENGINE}-${VARIANT}"
  echo "BENCH: $(bench_cmd)"
  exit 0
fi

command -v docker >/dev/null || die "docker not found (scripts/gpu-vm-bootstrap.sh installs it on a GPU VM)"
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Removing the previous ${NAME} container (one engine at a time owns the GPU)..."
  docker rm -f "${NAME}" >/dev/null
fi
mkdir -p "${HF_HOME_HOST}" results
if [ "${ENGINE}" = "nim" ]; then mkdir -p "${NIM_CACHE_HOST}"; fi
if [ "${ENGINE}" = "vllm" ]; then mkdir -p "${VLLM_CACHE_HOST}"; fi
echo "+ $(show "${CMD[@]}")"
"${CMD[@]}" >/dev/null

echo "Waiting for ${ENGINE} (${VARIANT}) on :${PORT}${HEALTH} - the first start downloads ${MODEL_ID} ..."
deadline=$((SECONDS + WAIT_S))
until curl -sf "http://${CLIENT_HOST}:${PORT}${HEALTH}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
    docker logs --tail 80 "${NAME}" >&2 || true
    die "${NAME} exited before it became healthy (logs above)"
  fi
  if [ "${SECONDS}" -ge "${deadline}" ]; then
    docker logs --tail 80 "${NAME}" >&2 || true
    die "not healthy after ${WAIT_S}s"
  fi
  sleep 5
done
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS%%,*}" 2>/dev/null | head -1 || true)"
fi
python3 - "${ENGINE}" "${VARIANT}" "${MODEL_ID}" "${IMAGE}" "${GPUS}" "${AGENT_MODEL}" "${METRICS_URL}" \
  "${RESET_URL}" "${GPU_NAME}" << 'PY'
import json, sys, time
e, v, m, img, gpus, agent_model, metrics_url, reset_url, gpu_name = sys.argv[1:10]
json.dump({"engine": e, "variant": v, "label": f"{e}-{v}", "model": m, "image": img, "gpus": gpus,
           "gpu_name": gpu_name, "agent_model": agent_model, "metrics_url": metrics_url, "reset_url": reset_url,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open("results/engine-current.json", "w"), indent=2)
PY
echo
echo "Ready: ${ENGINE} ${VARIANT} serving ${MODEL_ID} as '${AGENT_MODEL}' (published on ${BIND}:${PORT})"
echo "  export TTG_BASE_URL=http://${CLIENT_HOST}:${PORT}/v1 TTG_MODEL=${AGENT_MODEL}${PROMPT_EXPORT}"
if [ -n "${METRICS_URL}" ]; then echo "  metrics: ${METRICS_URL}   (python -m ttg.bench.scrape_metrics --url ${METRICS_URL})"; fi
if [ -n "${RESET_URL}" ]; then
  echo "  reset:   curl -X POST ${RESET_URL}   (empties the prefix cache; the bench calls it with --reset-url)"
else
  echo "  reset:   none on ${ENGINE} - relaunch the engine when you need a cold prefix cache"
fi
echo "  bench:   $(bench_cmd "${GPU_NAME}")"
echo "  stop:    engines/stop.sh"
