#!/usr/bin/env bash
# profiling/nsys_capture.sh - Nsight Systems timeline of vLLM serving the agent: kernels, NVTX ranges, CUDA graphs (M5).
#
# Usage: profiling/nsys_capture.sh [variant] [window_s]        # variant: base (default) | eager | fp8 | tp2 ...
#   env: NSYS=/path/to/nsys (default: nsys on the host PATH)   NSYS_IN_IMAGE=1 (the image has its own nsys)
#        PROFILES=results/profiles/nsys/<variant>  CONCURRENCY=8  N_TASKS=24  DELAY_ITERS=20  MAX_ITERS=60
#        NSYS_WAIT_S=300  PORT=8000  DRY_RUN=1 (print the launch command, run nothing)
# How: the vLLM image has no nsys, so the host's Nsight Systems install is mounted read-only at /nsys and wraps
# `vllm serve` (engines/launch.sh ENTRYPOINT_WRAPPER). vLLM runs with --profiler-config '{"profiler":"cuda"}', so
# POST /start_profile and /stop_profile call cudaProfilerStart/Stop and nsys records only that window
# (--capture-range=cudaProfilerApi). Flags follow the vLLM profiling guide: --trace-fork-before-exec=true
# (the engine core is a child process) and --cuda-graph-trace=node (see kernels inside CUDA graphs).
# Output: PROFILES/<variant>.nsys-rep (open it in the Nsight Systems GUI) + kern_sum.csv + the analyzer summary.
# Install nsys on an Ubuntu GPU VM (NVIDIA devtools repo):  sudo apt-get install -y nsight-systems-cli
# (repo setup: https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html). The NGC PyTorch image
# (nvcr.io/nvidia/pytorch:${PIN_NGC_RELEASE}-py3) also ships nsys.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
if [ -f env/pins.env ]; then . env/pins.env; fi

VARIANT="${1:-base}"
WINDOW_S="${2:-20}"
PORT="${PORT:-8000}"
PROFILES="${PROFILES:-${PWD}/results/profiles/nsys/${VARIANT}}"
CONCURRENCY="${CONCURRENCY:-8}"
N_TASKS="${N_TASKS:-24}"
BASE="http://localhost:${PORT}"
REP="${PROFILES}/${VARIANT}.nsys-rep"
case "${PROFILES}" in *" "*) echo "PROFILES must not contain spaces (${PROFILES})" >&2; exit 2 ;; esac

realpath_py() { python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"; }

MOUNT=""
if [ "${NSYS_IN_IMAGE:-0}" = "1" ]; then
  NSYS_IN_CONTAINER="nsys"
else
  NSYS_BIN="${NSYS:-$(command -v nsys || true)}"
  if [ -z "${NSYS_BIN}" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
      NSYS_BIN="/opt/nvidia/nsight-systems-cli/VERSION/target-linux-x64/nsys"     # placeholder for the dry run
    else
      echo "nsys not found on this host. Install it (sudo apt-get install -y nsight-systems-cli after adding the" >&2
      echo "NVIDIA devtools repo), or set NSYS=/path/to/nsys, or NSYS_IN_IMAGE=1 for an image that ships nsys." >&2
      exit 1
    fi
  fi
  REAL="$(realpath_py "${NSYS_BIN}")"
  NSYS_ROOT="$(dirname "$(dirname "${REAL}")")"        # .../<version>  (holds target-linux-x64/, host-linux-x64/)
  NSYS_IN_CONTAINER="/nsys/${REAL#"${NSYS_ROOT}"/}"
  MOUNT="-v ${NSYS_ROOT}:/nsys:ro"
fi

CFG="{\"profiler\":\"cuda\",\"delay_iterations\":${DELAY_ITERS:-20},\"max_iterations\":${MAX_ITERS:-60}}"
WRAP="${NSYS_IN_CONTAINER} profile -t cuda,nvtx,osrt -s none --cpuctxsw=none --trace-fork-before-exec=true"
WRAP="${WRAP} --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop"
WRAP="${WRAP} -o /profiles/${VARIANT} --force-overwrite=true"
# --init: a real PID 1 forwards SIGTERM to nsys, so `docker stop` lets nsys finish writing the report.
# VLLM_WORKER_MULTIPROC_METHOD=spawn: recommended by the vLLM guide when profiling with nsys.
DOCKER_EXTRA="--init ${MOUNT} -v ${PROFILES}:/profiles -e VLLM_WORKER_MULTIPROC_METHOD=spawn"

if [ "${DRY_RUN:-0}" = "1" ]; then
  DRY_RUN=1 ENTRYPOINT_WRAPPER="${WRAP}" EXTRA_DOCKER="${DOCKER_EXTRA}" EXTRA_ARGS="--profiler-config ${CFG}" \
    engines/launch.sh vllm "${VARIANT}"
  echo "PLAN: warm-up -> agent bench (c=${CONCURRENCY}) -> /start_profile -> ${WINDOW_S}s -> /stop_profile -> ${REP}"
  exit 0
fi

mkdir -p "${PROFILES}"
ENTRYPOINT_WRAPPER="${WRAP}" EXTRA_DOCKER="${DOCKER_EXTRA}" EXTRA_ARGS="--profiler-config ${CFG}" \
  engines/launch.sh vllm "${VARIANT}"
MODEL="ttg-llm"
PROMPT="full"
if [ "${VARIANT}" = "lora" ]; then MODEL="${PIN_LORA_NAME:-ttg-short}"; PROMPT="short"; fi   # adapter = short prompt (M6)
export TTG_BASE_URL="${BASE}/v1" TTG_MODEL="${MODEL}" TTG_PROMPT_VARIANT="${PROMPT}"

echo "== warm-up (thrown away)"
python3 -m ttg.bench.agent_bench --label "nsys-warmup-${VARIANT}" --engine vllm --levels 4 --n 8 \
  --out "${PROFILES}/warmup.json" > /dev/null
echo "== agent traffic: concurrency ${CONCURRENCY}, ${N_TASKS} tasks (background)"
python3 -m ttg.bench.agent_bench --label "nsys-${VARIANT}" --engine vllm --levels "${CONCURRENCY}" --n "${N_TASKS}" \
  --out "${PROFILES}/bench.json" > "${PROFILES}/bench.log" 2>&1 &
BENCH_PID=$!
trap 'kill "${BENCH_PID}" 2>/dev/null || true' EXIT
sleep 5
curl -sf -X POST "${BASE}/start_profile" > /dev/null
sleep "${WINDOW_S}"
curl -sf -m 900 -X POST "${BASE}/stop_profile" > /dev/null || true
wait "${BENCH_PID}" || true
trap - EXIT

# --capture-range-end=stop: collection ends at cudaProfilerStop and vLLM keeps serving. If the report does not
# appear, stopping the container ends the nsys session, which writes it.
deadline=$((SECONDS + ${NSYS_WAIT_S:-300}))
while [ ! -s "${REP}" ] && [ "${SECONDS}" -lt "${deadline}" ]; do sleep 5; done
if [ ! -s "${REP}" ]; then
  echo "No report yet - stopping the engine so nsys finalizes it ..."
  docker stop -t 180 ttg-engine > /dev/null || true
  for _ in $(seq 1 36); do if [ -s "${REP}" ]; then break; fi; sleep 5; done
fi
if [ ! -s "${REP}" ]; then
  echo "No ${REP}. Check: docker logs ttg-engine | grep -i -e nsys -e 'capture range'" >&2
  exit 1
fi
echo "== report: ${REP}"

HOST_NSYS="${NSYS:-$(command -v nsys || true)}"
if [ -n "${HOST_NSYS}" ]; then
  "${HOST_NSYS}" stats --report cuda_gpu_kern_sum --format csv "${REP}" > "${PROFILES}/kern_sum.csv" 2> "${PROFILES}/stats.log" || true
  python3 profiling/analyze_trace.py --nsys-csv "${PROFILES}/kern_sum.csv" --json "${PROFILES}/summary.json"
else
  echo "Summarize later where nsys is installed:"
  echo "  nsys stats --report cuda_gpu_kern_sum --format csv ${REP} > kern_sum.csv"
  echo "  python3 profiling/analyze_trace.py --nsys-csv kern_sum.csv"
fi
echo "Open ${REP} in the Nsight Systems GUI: look for gaps between CUDA-graph launches (CPU time) and NCCL on tp2."
