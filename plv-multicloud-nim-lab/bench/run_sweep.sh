#!/usr/bin/env bash
# 🔁 Lab 02's latency-throughput sweep, now per cluster: same preset, same levels, any cloud -> comparable CSVs.
# Usage: bench/run_sweep.sh <cluster> <preset> "<levels>" [extra aiperf args...]
#   bench/run_sweep.sh aks synth-600-200 "1 8 32 64 128"
#   bench/run_sweep.sh gke synth-600-200 "1 8 32 64 128"
# The served precision is recorded from clusters/<c>/values.yaml (override: PLV_PRECISION=nvfp4 for M6/add-on runs).
# Energy/cost columns need Prometheus for THAT cluster on PROM_URL (default http://localhost:9090):
#   kubectl --context plv-aks -n monitoring port-forward svc/kps-prometheus 9090:9090
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
# shellcheck source=bench/lib.sh
. bench/lib.sh
CL="${1:?usage: bench/run_sweep.sh <cluster> <preset> \"<levels>\" [extra aiperf args]}"
PRESET="${2:?preset}"
LEVELS="${3:-1 4 16 32 64}"
shift 2
if [ $# -gt 0 ]; then shift; fi
valid_cluster "${CL}" || die "unknown cluster ${CL}"
set_preset "${PRESET}"
RUN_ID="${CL}-${PRESET}-$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_DIR="results/${RUN_ID}"
mkdir -p "${LOCAL_DIR}"
if [ "${KIND}" = "file" ]; then push_datasets "${CL}"; fi
HOURLY="$(gpu_hourly "${CL}")"
PRECISION="${PLV_PRECISION:-$(python3 -c "import yaml,sys; v=yaml.safe_load(open('clusters/${CL}/values.yaml')) or {}; print((v.get('llm') or {}).get('precision', 'bf16'))")}"
python3 - "${LOCAL_DIR}/run.json" "${CL}" "${PRESET}" "${MODEL}" "${ISL:-}" "${OSL:-}" "${HOURLY}" "$(llm_node_facts "${CL}")" "${PRECISION}" << 'PY'
import json, sys
out, cl, preset, model, isl, osl, hourly, facts, precision = sys.argv[1:10]
json.dump({"cluster": cl, "preset": preset, "model": model, "precision": precision, "isl": int(isl) if isl else None,
           "osl": int(osl) if osl else None, "gpu_hourly_usd": float(hourly or 0), **json.loads(facts or "{}")},
          open(out, "w"), indent=2)
PY
echo "Run ${RUN_ID} -> ${URL} (model ${MODEL}) on $(python3 -c "import json;d=json.load(open('${LOCAL_DIR}/run.json'));print(d.get('instance_type'), d.get('gpu_product'))"); levels: ${LEVELS}"
for C in ${LEVELS}; do
  N=$((C * PER_LEVEL))
  if [ "${N}" -lt "${MIN_REQ}" ]; then N="${MIN_REQ}"; fi
  echo "== concurrency ${C}: ${N} requests"
  aiperf_run "${CL}" "${C}" "${N}" "/results/${RUN_ID}/c${C}" ${1+"$@"}
  fetch_results "${CL}" "/results/${RUN_ID}/c${C}" "${LOCAL_DIR}/c${C}"
  python3 bench/energy_probe.py --aiperf "${LOCAL_DIR}/c${C}" --prom "${PROM_URL:-http://localhost:9090}" \
    --context "$(ctx "${CL}")" --namespace plv-serving --node-hourly "${HOURLY}" \
    --label "${CL}:${PRESET}" --concurrency "${C}" --csv "${LOCAL_DIR}/summary.csv" >/dev/null \
    || echo "  (energy probe skipped: is Prometheus for ${CL} port-forwarded to ${PROM_URL:-http://localhost:9090}?)"
done
echo
if [ -f "${LOCAL_DIR}/summary.csv" ]; then
  cut -d, -f1-10,13,17,18,20 "${LOCAL_DIR}/summary.csv" | column -s, -t
else
  python3 bench/aiperf_summary.py "${LOCAL_DIR}"/c*
fi
echo "Results: ${LOCAL_DIR}  ->  python3 bench/compare_clouds.py results/*-${PRESET}-*"
