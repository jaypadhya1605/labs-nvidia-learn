#!/usr/bin/env bash
# M8: does a claims-batch flood on the neighbor hurt the prior-auth SLO?  Run once per sharing mode.
# Usage: bench/noisy_neighbor.sh <mode-label>      e.g. mig | timeslice | mps
# Uses the MIG tenants (priorauth/shared) or, if priorauth-llm-fg exists, the full-GPU tenants.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=bench/lib.sh
. bench/lib.sh
MODE="${1:?usage: bench/noisy_neighbor.sh <mode-label>}"
if kubectl -n kha-um get svc priorauth-llm-fg >/dev/null 2>&1; then VICTIM="priorauth-fg"; AGGR="batch-fg"; else VICTIM="priorauth"; AGGR="batch"; fi
RUN_ID="nn-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
L="results/${RUN_ID}"
mkdir -p "${L}"
POD="$(toolbox_pod)"

echo "1/4 victim (${VICTIM}) alone: concurrency 8"
set_preset "${VICTIM}"
aiperf_run 8 48 "/results/${RUN_ID}/victim-alone"
fetch_results "/results/${RUN_ID}/victim-alone" "${L}/victim-alone"

echo "2/4 starting aggressor flood (${AGGR}, concurrency 64) in the background"
set_preset "${AGGR}"
kubectl -n "${TOOLBOX_NS}" exec "${POD}" -- bash -c "nohup aiperf profile --model ${MODEL} --url ${URL} \
  --endpoint-type chat --streaming --tokenizer ${TOK} --isl ${ISL} --isl-stddev 0 --osl ${OSL} --osl-stddev 0 \
  --concurrency 64 --request-count 100000 --benchmark-duration 480 --ui-type none \
  --artifact-dir /results/${RUN_ID}/aggressor > /results/${RUN_ID}-aggressor.log 2>&1 &"
echo "    ramping 60 s..."
sleep 60

echo "3/4 victim again while the neighbor floods"
set_preset "${VICTIM}"
aiperf_run 8 48 "/results/${RUN_ID}/victim-flooded"
fetch_results "/results/${RUN_ID}/victim-flooded" "${L}/victim-flooded"

echo "4/4 stopping the flood"
kubectl -n "${TOOLBOX_NS}" exec "${POD}" -- pkill -f "/results/${RUN_ID}/aggressor" || true
echo
echo "Mode: ${MODE}   (prior-auth SLO: TTFT p95 <= 1500 ms, ITL p95 <= 60 ms; isolation target: < 10% change)"
python3 bench/aiperf_summary.py --compare "${L}/victim-alone" "${L}/victim-flooded" | tee "${L}/comparison.txt"
