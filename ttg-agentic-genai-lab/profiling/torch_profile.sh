#!/usr/bin/env bash
# profiling/torch_profile.sh - PyTorch-profiler trace of vLLM serving the REAL agent traffic, then the analysis (M5).
#
# Usage: profiling/torch_profile.sh [variant] [window_s]      # variant: base (default) | eager | fp8 | nocache | spec ...
#   env: PROFILES=results/profiles/torch/<variant>  CONCURRENCY=8  N_TASKS=24  DELAY_ITERS=20  MAX_ITERS=60  PORT=8000
#        SKIP_LAUNCH=1  (reuse a running engine that was started with the same --profiler-config)
#        DRY_RUN=1      (print the launch command and the plan, run nothing)
# Steps: 1) engines/launch.sh vllm <variant> with --profiler-config (torch profiler; traces -> PROFILES via /profiles)
#        2) warm-up bench (CUDA graphs, torch.compile, prefix cache)   3) the agent bench in the background
#        4) POST /start_profile: vLLM skips DELAY_ITERS engine steps, records MAX_ITERS steps, then stops by itself
#        5) POST /stop_profile (a no-op if it already stopped; it blocks while traces are written)
#        6) profiling/analyze_trace.py on the newest trace -> PROFILES/summary.json
# Compare: run it for base and eager - same traffic, CUDA graphs on vs off - and diff the "GPU busy" line.
# Traces are small on purpose (60 steps). Open the .json.gz in https://ui.perfetto.dev to see the timeline.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
if [ -f env/pins.env ]; then . env/pins.env; fi

VARIANT="${1:-base}"
WINDOW_S="${2:-20}"
PORT="${PORT:-8000}"
PROFILES="${PROFILES:-${PWD}/results/profiles/torch/${VARIANT}}"
CONCURRENCY="${CONCURRENCY:-8}"
N_TASKS="${N_TASKS:-24}"
BASE="http://localhost:${PORT}"
# No spaces inside the JSON: launch.sh splits EXTRA_ARGS on whitespace.
CFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/profiles\",\"torch_profiler_with_stack\":false,\"delay_iterations\":${DELAY_ITERS:-20},\"max_iterations\":${MAX_ITERS:-60},\"ignore_frontend\":true}"

case "${PROFILES}" in *" "*) echo "PROFILES must not contain spaces (${PROFILES})" >&2; exit 2 ;; esac
if [ "${DRY_RUN:-0}" = "1" ]; then
  DRY_RUN=1 EXTRA_DOCKER="-v ${PROFILES}:/profiles" EXTRA_ARGS="--profiler-config ${CFG}" engines/launch.sh vllm "${VARIANT}"
  echo "PLAN: warm-up bench -> agent bench (c=${CONCURRENCY}, n=${N_TASKS}) -> POST ${BASE}/start_profile -> ${WINDOW_S}s -> POST ${BASE}/stop_profile -> analyze ${PROFILES}"
  exit 0
fi

mkdir -p "${PROFILES}"
if [ "${SKIP_LAUNCH:-0}" != "1" ]; then
  EXTRA_DOCKER="-v ${PROFILES}:/profiles" EXTRA_ARGS="--profiler-config ${CFG}" engines/launch.sh vllm "${VARIANT}"
fi
MODEL="ttg-llm"
PROMPT="full"
if [ "${VARIANT}" = "lora" ]; then MODEL="${PIN_LORA_NAME:-ttg-short}"; PROMPT="short"; fi   # adapter = short prompt (M6)
export TTG_BASE_URL="${BASE}/v1" TTG_MODEL="${MODEL}" TTG_PROMPT_VARIANT="${PROMPT}"

echo "== warm-up (thrown away)"
python3 -m ttg.bench.agent_bench --label "prof-warmup-${VARIANT}" --engine vllm --levels 4 --n 8 \
  --out "${PROFILES}/warmup.json" > /dev/null

echo "== agent traffic: concurrency ${CONCURRENCY}, ${N_TASKS} tasks (background)"
python3 -m ttg.bench.agent_bench --label "prof-${VARIANT}" --engine vllm --levels "${CONCURRENCY}" --n "${N_TASKS}" \
  --out "${PROFILES}/bench.json" > "${PROFILES}/bench.log" 2>&1 &
BENCH_PID=$!
trap 'kill "${BENCH_PID}" 2>/dev/null || true' EXIT
sleep 5                                         # let the batch fill up: profile the steady state, not the ramp

MARK="${PROFILES}/.started"
touch "${MARK}"
echo "== POST /start_profile (records ${MAX_ITERS:-60} engine steps after ${DELAY_ITERS:-20})"
curl -sf -X POST "${BASE}/start_profile" > /dev/null
sleep "${WINDOW_S}"
echo "== POST /stop_profile (blocks while the trace is written)"
curl -sf -m 900 -X POST "${BASE}/stop_profile" > /dev/null || echo "   (stop_profile returned an error - the profiler may have stopped itself already)"
wait "${BENCH_PID}" || true
trap - EXIT

TRACE=""
for _ in $(seq 1 60); do
  TRACE="$(find "${PROFILES}" -name '*.pt.trace.json*' -newer "${MARK}" ! -name '*async_llm*' 2>/dev/null | head -1 || true)"
  if [ -n "${TRACE}" ]; then break; fi
  sleep 5
done
if [ -z "${TRACE}" ]; then
  echo "No trace in ${PROFILES} yet. Check: engines/logs.sh (look for 'Profiler started' / 'Max profiling iterations')." >&2
  exit 1
fi
echo "== trace: ${TRACE}"
python3 profiling/analyze_trace.py "${PROFILES}" --json "${PROFILES}/summary.json"
echo
echo "Timeline: open ${TRACE} in https://ui.perfetto.dev   Bench of the profiled run: ${PROFILES}/bench.json"
