#!/usr/bin/env bash
# 🔁 Lab 02's noisy-neighbor test, M8 edition: does LoRA TRAINING on one MIG slice hurt SERVING on the other?
# Usage: bench/noisy_neighbor.sh <cluster> [aggressor-manifest]   (default aggressor: mlops/k8s/train-lora-job.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
# shellcheck source=bench/lib.sh
. bench/lib.sh
CL="${1:?usage: bench/noisy_neighbor.sh <cluster> [aggressor-manifest]}"
AGGR="${2:-mlops/k8s/train-lora-job.yaml}"
export ADAPTER_NAME="${ADAPTER_NAME:-nn-aggressor}"   # a throwaway training run: only the GPU load matters here
RUN_ID="nn-${CL}-$(date -u +%Y%m%dT%H%M%SZ)"; L="results/${RUN_ID}"; mkdir -p "${L}"
set_preset synth-600-200
echo "1/4 victim (serving) alone"
aiperf_run "${CL}" 16 96 "/results/${RUN_ID}/victim-alone"; fetch_results "${CL}" "/results/${RUN_ID}/victim-alone" "${L}/victim-alone"
echo "2/4 start the aggressor (${AGGR}) and wait until it is Running"
scripts/render.sh "${AGGR}" | k "${CL}" apply -f -
k "${CL}" -n plv-mlops wait --for=condition=Ready pod -l app=plover-train --timeout=15m
sleep 90   # let training reach steady state (model loaded, steps running)
echo "3/4 victim again while training runs on the neighbouring slice"
aiperf_run "${CL}" 16 96 "/results/${RUN_ID}/victim-neighbour"; fetch_results "${CL}" "/results/${RUN_ID}/victim-neighbour" "${L}/victim-neighbour"
echo "4/4 stop the aggressor"
scripts/render.sh "${AGGR}" | k "${CL}" delete -f - --ignore-not-found
python3 bench/aiperf_summary.py --compare "${L}/victim-alone" "${L}/victim-neighbour" | tee "${L}/comparison.txt"
