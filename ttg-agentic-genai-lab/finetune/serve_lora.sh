#!/usr/bin/env bash
# finetune/serve_lora.sh - serve the M6 adapter next to the base model and print the three-way evaluation.
#
# Usage: finetune/serve_lora.sh [vllm|sglang] [adapter_dir]       # defaults: vllm, adapters/ttg-short
#   env: DRY_RUN=1 (print the engine command only)   any engines/launch.sh variable (GPUS, PORT, ...)
# One engine serves BOTH the base model ("ttg-llm") and the adapter ("ttg-short" on vLLM, "ttg-llm:ttg-short" on
# SGLang), so the before/after comparison runs on the same GPU, same weights, same kernels.
# The adapter must have adapter_config.json + adapter_model.safetensors (finetune/train_lora.py writes both).
# Rank: launch.sh passes --max-lora-rank 16; a bigger r gets the next size vLLM accepts (32, 64, ...).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
if [ -f env/pins.env ]; then . env/pins.env; fi

ENGINE="${1:-vllm}"
ADAPTER="${2:-adapters/${PIN_LORA_NAME:-ttg-short}}"
NAME="${PIN_LORA_NAME:-ttg-short}"
case "${ENGINE}" in vllm|sglang) ;; *) echo "engine must be vllm or sglang (NIM: NIM_PEFT_SOURCE, Lab 04 M8)" >&2; exit 2 ;; esac

R=16
if [ -f "${ADAPTER}/adapter_config.json" ]; then
  R="$(python3 -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["r"]))' "${ADAPTER}/adapter_config.json")"
  if [ ! -f "${ADAPTER}/adapter_model.safetensors" ]; then
    echo "warning: ${ADAPTER}/adapter_model.safetensors is missing (an FSDP run saved shards?)" >&2
  fi
elif [ "${DRY_RUN:-0}" != "1" ]; then
  echo "no adapter at ${ADAPTER}: python3 finetune/make_sft_data.py && python3 finetune/train_lora.py" >&2
  exit 1
fi
MAXR=16
for c in 8 16 32 64 128 256; do if [ "${R}" -le "${c}" ]; then MAXR="${c}"; break; fi; done

# launch.sh mounts ADAPTER_DIR at /adapters and loads /adapters/<name>: point it at the adapter's parent folder.
PARENT="$(cd "$(dirname "${ADAPTER}")" 2>/dev/null && pwd || echo "${PWD}/$(dirname "${ADAPTER}")")"
if [ "$(basename "${ADAPTER}")" != "${NAME}" ]; then
  echo "the adapter folder must be named ${NAME} (got $(basename "${ADAPTER}")): rename it or set PIN_LORA_NAME" >&2
  exit 2
fi
EXTRA=""
if [ "${MAXR}" != "16" ]; then EXTRA="--max-lora-rank ${MAXR}"; fi     # the last --max-lora-rank wins
ADAPTER_DIR="${PARENT}" EXTRA_ARGS="${EXTRA}" engines/launch.sh "${ENGINE}" lora

LORA_MODEL="${NAME}"
if [ "${ENGINE}" = "sglang" ]; then LORA_MODEL="${PIN_SERVED_MODEL_NAME:-ttg-llm}:${NAME}"; fi
cat << EOF

Evaluate (held-out seed 13, step by step, teacher-forced):
  python3 finetune/eval_toolcalls.py --model ${PIN_SERVED_MODEL_NAME:-ttg-llm} --system-variant full     # before: base + full prompt
  python3 finetune/eval_toolcalls.py --model ${PIN_SERVED_MODEL_NAME:-ttg-llm} --system-variant short    # control: base + short prompt
  python3 finetune/eval_toolcalls.py --model ${LORA_MODEL} --system-variant short    # after: adapter + short prompt
Task level (real tool calls, golden set):
  TTG_PROMPT_VARIANT=short TTG_MODEL=${LORA_MODEL} python -m ttg.bench.agent_bench --label ${ENGINE}-lora --engine ${ENGINE} --levels 1,4,16,32 --n 48
EOF
