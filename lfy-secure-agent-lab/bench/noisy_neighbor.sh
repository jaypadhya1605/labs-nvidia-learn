#!/usr/bin/env bash
# Lab 02's noisy-neighbor test, generalized: does flooding one MIG slice hurt the tenant on another?
# Usage: bench/noisy_neighbor.sh <mode-label> [victim-preset] [aggressor-preset]
#   default: victim = agent-llm (3g slice), aggressor = safety-flood (2g slice, concurrency 64)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=bench/lib.sh
. bench/lib.sh
MODE="${1:?usage: bench/noisy_neighbor.sh <mode-label> [victim] [aggressor]}"
VICTIM="${2:-agent-llm}"; AGGR="${3:-safety-flood}"
RUN_ID="nn-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"; L="results/${RUN_ID}"; mkdir -p "${L}"
POD="$(toolbox_pod)"
echo "1/4 victim (${VICTIM}) alone"; set_preset "${VICTIM}"
aiperf_run 8 48 "/results/${RUN_ID}/victim-alone"; fetch_results "/results/${RUN_ID}/victim-alone" "${L}/victim-alone"
echo "2/4 aggressor flood (${AGGR}, concurrency 64) in the background"; set_preset "${AGGR}"
kubectl -n "${TOOLBOX_NS}" exec "${POD}" -- bash -c "nohup aiperf profile --model ${MODEL} --url ${URL} \
  --endpoint-type chat --tokenizer ${TOK} --isl ${ISL:-900} --isl-stddev 0 --osl ${OSL:-12} --osl-stddev 0 \
  --concurrency 64 --request-count 100000 --benchmark-duration 480 --ui-type none \
  --artifact-dir /results/${RUN_ID}/aggressor > /results/${RUN_ID}-aggressor.log 2>&1 &"
sleep 60
echo "3/4 victim again while the neighbor floods"; set_preset "${VICTIM}"
aiperf_run 8 48 "/results/${RUN_ID}/victim-flooded"; fetch_results "/results/${RUN_ID}/victim-flooded" "${L}/victim-flooded"
echo "4/4 stopping the flood"; kubectl -n "${TOOLBOX_NS}" exec "${POD}" -- pkill -f "/results/${RUN_ID}/aggressor" || true
python3 bench/aiperf_summary.py --compare "${L}/victim-alone" "${L}/victim-flooded" | tee "${L}/comparison.txt"
