#!/usr/bin/env bash
# Latency-throughput sweep for one tenant, with energy and cost per concurrency level.
# Usage: bench/run_sweep.sh <preset> "<levels>" [extra aiperf args...]
#   presets: agent-llm | agent-llm-guarded | agent-llm-stream | safety-guard | safety-flood
# Needs the bench toolbox (M4) and, for energy/cost:  kubectl -n monitoring port-forward svc/kps-prometheus 9090:9090
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=bench/lib.sh
. bench/lib.sh
PRESET="${1:?usage: bench/run_sweep.sh <preset> \"<levels>\" [extra aiperf args]}"
LEVELS="${2:-1 2 4 8 16}"
shift
if [ $# -gt 0 ]; then shift; fi
set_preset "${PRESET}"
RUN_ID="$(echo "${PRESET}" | tr ':' '-')-$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_DIR="results/${RUN_ID}"
mkdir -p "${LOCAL_DIR}"
echo "Run ${RUN_ID} -> ${URL} (model ${MODEL}); levels: ${LEVELS}"
for C in ${LEVELS}; do
  N=$((C * PER_LEVEL))
  if [ "${N}" -lt "${MIN_REQ}" ]; then N="${MIN_REQ}"; fi
  echo "== concurrency ${C}: ${N} requests"
  aiperf_run "${C}" "${N}" "/results/${RUN_ID}/c${C}" ${1+"$@"}
  fetch_results "/results/${RUN_ID}/c${C}" "${LOCAL_DIR}/c${C}"
  python3 bench/energy_probe.py --aiperf "${LOCAL_DIR}/c${C}" --prom "${PROM_URL:-http://localhost:9090}" \
    --pool "${POOL}" --namespace "${NS}" --node-hourly "${NODE_HOURLY}" --slice-share "${SHARE}" \
    --label "${PRESET}" --concurrency "${C}" --csv "${LOCAL_DIR}/summary.csv" >/dev/null \
    || echo "  (energy probe skipped: is Prometheus port-forwarded to localhost:9090?)"
done
echo
if [ -f "${LOCAL_DIR}/summary.csv" ]; then
  cut -d, -f1-10,13,17,18,20 "${LOCAL_DIR}/summary.csv" | column -s, -t
else
  python3 bench/aiperf_summary.py "${LOCAL_DIR}"/c*
fi
echo "Full results: ${LOCAL_DIR}/summary.csv"
