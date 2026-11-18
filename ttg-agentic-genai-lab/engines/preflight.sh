#!/usr/bin/env bash
# engines/preflight.sh - prove every flag engines/launch.sh uses exists in the PINNED image, before you pay for GPU time.
#
# Usage: engines/preflight.sh <engine> [variant ...]   # docker pulls the image and greps its --help (no GPU needed)
#        engines/preflight.sh --list <engine>          # just print the flags launch.sh would use (no docker)
#   engine: vllm | sglang | trtllm | nim | hf      variants default to every variant the engine supports.
# Why: engine CLIs move fast (vLLM renamed and regrouped flags many times; trtllm-serve became a command group).
# A MISSING line here costs nothing; the same mistake found after a 20-minute model download costs a session.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
if [ -f env/pins.env ]; then . env/pins.env; fi

LIST=0
if [ "${1:-}" = "--list" ]; then LIST=1; shift; fi
ENGINE="${1:?usage: engines/preflight.sh [--list] <engine> [variant ...]}"
shift || true
VARIANTS="$*"
if [ -z "${VARIANTS}" ]; then VARIANTS="base nocache fp8 fp8kv spec eager lora tp2 pp2 dp2 vlm"; fi

flags=""
for v in ${VARIANTS}; do
  modes="checkpoint"
  if [ "${v}" = "fp8" ]; then modes="checkpoint online"; fi      # FP8_MODE=online adds --quantization fp8
  for m in ${modes}; do
    if out="$(DRY_RUN=1 GPUS=0,1 FP8_MODE="${m}" NIM_MODEL_PROFILE=x engines/launch.sh "${ENGINE}" "${v}" 2>/dev/null)"; then
      flags="${flags} $(printf '%s\n' "${out}" | sed -n 's/^ENGINE_FLAGS: //p')"
    else
      echo "  (skip ${ENGINE} ${v} FP8_MODE=${m}: not supported by launch.sh)" >&2
    fi
  done
done
# shellcheck disable=SC2086  # word-splitting the collected flags is intended
FLAGS="$(printf '%s\n' ${flags} | awk 'NF && !seen[$0]++')"

if [ "${LIST}" = "1" ]; then
  printf '%s\n' "${FLAGS}"
  exit 0
fi

case "${ENGINE}" in
  vllm)
    IMAGE="${PIN_VLLM_IMAGE:-vllm/vllm-openai}:${PIN_VLLM_TAG:-v0.31.0}"
    HELP="$(docker run --rm --entrypoint vllm "${IMAGE}" serve --help=all 2>&1 || true)" ;;
  sglang)
    IMAGE="${PIN_SGLANG_IMAGE:-lmsysorg/sglang}:${PIN_SGLANG_TAG:-v0.5.22}"
    HELP="$(docker run --rm --entrypoint python3 "${IMAGE}" -m sglang.launch_server --help 2>&1 || true)" ;;
  trtllm)
    IMAGE="${PIN_TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release}:${PIN_TRTLLM_TAG:-1.3.0rc29}"
    HELP="$(docker run --rm "${IMAGE}" bash -c 'trtllm-serve serve --help 2>/dev/null || trtllm-serve --help' 2>&1 || true)" ;;
  hf)
    IMAGE="(local script)"
    HELP="$(python3 engines/hf_naive_server.py --help 2>&1 || true)" ;;
  nim)
    IMAGE="${PIN_NIM_LLM_IMAGE:-nvcr.io/nim/meta/llama-3.1-8b-instruct}:${PIN_NIM_LLM_TAG:-2.0.13}"
    echo "NIM is configured by environment variables, not flags. launch.sh sets:"
    echo "  NIM_SERVED_MODEL_NAME, NIM_ENABLE_KV_CACHE_REUSE, NIM_MODEL_PROFILE (fp8), NGC_API_KEY"
    echo "Check them against the NIM for LLMs configuration docs for ${IMAGE}, and list profiles on the GPU VM with:"
    echo "  docker run --rm --gpus all -e NGC_API_KEY ${IMAGE} list-model-profiles      # 🔁 Lab 04 M4"
    exit 0 ;;
  *) echo "unknown engine ${ENGINE}" >&2; exit 2 ;;
esac

if [ -z "${HELP}" ] || ! printf '%s' "${HELP}" | grep -q -- '--'; then
  echo "Could not read --help from ${IMAGE} (pull/login problem?):" >&2
  printf '%s\n' "${HELP}" | tail -20 >&2
  exit 1
fi

missing=0
echo "Flags used by engines/launch.sh ${ENGINE} vs ${IMAGE}:"
for f in ${FLAGS}; do
  if printf '%s' "${HELP}" | grep -qE -e "${f}([^a-z0-9_-]|\$)"; then
    printf '  OK       %s\n' "${f}"
  else
    printf '  MISSING  %s\n' "${f}"
    missing=$((missing + 1))
  fi
done
if [ "${missing}" -gt 0 ]; then
  echo "${missing} flag(s) missing: fix engines/launch.sh or the pin in env/pins.env before renting the GPU." >&2
  exit 1
fi
echo "All flags present."
