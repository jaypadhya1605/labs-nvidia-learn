#!/usr/bin/env bash
# M7 - data-parallel scaling of the M6 LoRA job: 1 GPU vs DDP vs FSDP2 on 2 / 4 / 8 GPUs of one node.
# Same script, same data, same steps - only the launcher changes. Logs + nvidia-smi memory samples per run, then
# python3 -m parallel.scaling_report prints samples/s, speedup, scaling efficiency and peak memory per GPU.
#
# Usage: TRAIN_ARGS="<the flags you used for finetune/train_lora.py in M6>" parallel/train_scaling.sh [gpus...] [--dry-run]
#   parallel/train_scaling.sh                                   # 1 GPU vs DDP x8 vs FSDP2 x8, train_lora.py defaults
#   TRAIN_ARGS="<M6 flags + its max-steps flag set to ~60>" parallel/train_scaling.sh 1 2 4 8   # the full curve
#   Keep the per-device batch the same in every run (weak scaling): the global batch grows with GPUs.
# Launchers (the M6 script and its FSDP2 config; DDP needs no config file):
#   1 GPU : python3 finetune/train_lora.py $TRAIN_ARGS
#   DDP   : accelerate launch --multi_gpu --num_processes N finetune/train_lora.py $TRAIN_ARGS
#   FSDP2 : accelerate launch --config_file finetune/accelerate_fsdp2.yaml --num_processes N finetune/train_lora.py $TRAIN_ARGS
#           (--num_processes on the command line overrides the num_processes value in the YAML)
# Predictions first: python3 -m parallel.parallelism_calc --examples   (E1: LoRA memory; E2: what FSDP shards)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f env/lab.env ]; then                 # HF_TOKEN for the gated base model (env/lab.env.example)
  # shellcheck source=/dev/null
  . env/lab.env
fi

DRY=0
GPUS_LIST=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY=1 ;;
    ''|*[!0-9]*) echo "unknown argument: ${arg}" >&2; exit 2 ;;
    *) GPUS_LIST="${GPUS_LIST} ${arg}" ;;
  esac
done
GPUS_LIST="${GPUS_LIST:- 1 8}"
if [ -z "${TRAIN_ARGS:-}" ]; then
  TRAIN_ARGS=""
  echo "TRAIN_ARGS is empty: every run uses finetune/train_lora.py defaults. For short, comparable runs pass your M6"
  echo "flags with a step cap (see: python3 finetune/train_lora.py --help)."
fi
OUT="results/scaling/$(date -u +%Y%m%dT%H%M%SZ)"
if [ "${DRY}" = "0" ] && ! python3 -c "import torch, trl, peft, accelerate" 2>/dev/null; then
  echo "torch / TRL / PEFT / accelerate are not importable here. Run this script INSIDE the NGC PyTorch container" >&2
  echo "with the pinned Hugging Face stack, exactly as M6 trains (finetune/README.md, step 3)." >&2
  exit 2
fi

launch() {   # launch <name> <gpus> <command...>
  _name="$1"; _n="$2"; shift 2
  if [ "${DRY}" = "1" ]; then
    echo "[${_name}] CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_n - 1))) $*"
    return 0
  fi
  mkdir -p "${OUT}"
  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "${OUT}/${_name}.mem.csv" &
  _mon=$!
  if CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))" "$@" > "${OUT}/${_name}.log" 2>&1; then
    echo "  ${_name}: done"
  else
    echo "  ${_name}: FAILED - tail ${OUT}/${_name}.log"
  fi
  kill "${_mon}" 2>/dev/null || true
}

# Every run writes its adapter and metrics under ${OUT}: the M6 adapter (adapters/ttg-short) and its results file are
# never overwritten. These flags come AFTER ${TRAIN_ARGS}, so they win over any --out / --run-name in it.
own() { printf -- '--out %s --run-name %s' "${OUT}/$1-adapter" "scaling-$1"; }
for n in ${GPUS_LIST}; do
  # shellcheck disable=SC2086,SC2046  # TRAIN_ARGS and own() are flag lists on purpose
  if [ "${n}" = "1" ]; then
    launch "single-1gpu" 1 python3 finetune/train_lora.py ${TRAIN_ARGS} $(own single-1gpu)
  else
    launch "ddp-${n}gpu" "${n}" accelerate launch --multi_gpu --num_processes "${n}" finetune/train_lora.py \
      ${TRAIN_ARGS} $(own "ddp-${n}gpu")
    launch "fsdp2-${n}gpu" "${n}" accelerate launch --config_file finetune/accelerate_fsdp2.yaml --num_processes "${n}" \
      finetune/train_lora.py ${TRAIN_ARGS} $(own "fsdp2-${n}gpu")
  fi
done
if [ "${DRY}" = "0" ]; then
  python3 -m parallel.scaling_report "${OUT}"
fi
