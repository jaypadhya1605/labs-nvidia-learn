#!/usr/bin/env bash
# profiling/dcgm_watch.sh - sample GPU counters once a second while a bench runs: what "GPU utilization" hides (M5).
#
# Usage: profiling/dcgm_watch.sh [seconds] [out_dir]      # default 120 s -> results/profiles/gpu-<UTC timestamp>/
#   env: GPUS=0 (comma list)   INTERVAL_MS=1000 (dcgmi)   SUMMARY_ONLY=<out_dir> (re-print the summary, sample nothing)
# Run it next to the bench:  profiling/dcgm_watch.sh 180 & python -m ttg.bench.agent_bench --label vllm-base ...
# Writes:
#   nvidia-smi-dmon.txt  power, SM / memory "utilization", clocks, PCIe MB/s (nvidia-smi dmon -s pucvmet)
#   dcgmi-dmon.txt       DCGM profiling counters by field ID (needs dcgmi + a running nv-hostengine):
#                        1002 SM active | 1003 SM occupancy | 1004 tensor pipe active | 1005 DRAM active |
#                        1009 PCIe TX bytes/s | 1010 PCIe RX bytes/s  (IDs are stable across DCGM versions; names
#                        changed, e.g. 1002 DCGM_FI_PROF_SM_ACTIVE -> DCGM_FI_PROF_SM_UTIL_RATIO)
#   summary.txt          mean and max per column; *.csv = the same samples as CSV (pandas / notebook)
# Read it: nvidia-smi "sm %" = share of time ANY kernel ran. Decode at batch 1 shows ~100% while tensor-pipe active
# stays low and DRAM active is high: memory-bound (🔁 Lab 01). DRAM active x peak bandwidth ~ achieved bandwidth -
# compare it with profiling/roofline_check.py. On Kubernetes the same counters come from dcgm-exporter (🔁 Lab 04).
set -euo pipefail
cd "$(dirname "$0")/.."

summarize() {   # $1 = directory with the dmon files
  python3 - "$1" << 'PY'
import csv
import sys
from pathlib import Path

def table(path):
    cols, rows = None, []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            if cols is None:
                cols = line.lstrip("#").split()
            continue
        parts = line.split()
        if parts[0] == "GPU" and len(parts) > 1:      # dcgmi rows: "GPU 0 ..." against a "#Entity" header
            parts = [parts[0] + parts[1]] + parts[2:]
        if len(parts) > 1:                            # skips dcgmi's bare "ID" units line
            rows.append(parts)
    return cols or [], rows

out = []
for name in ("nvidia-smi-dmon.txt", "dcgmi-dmon.txt"):
    p = Path(sys.argv[1]) / name
    if not p.exists() or p.stat().st_size == 0:
        continue
    cols, rows = table(p)
    with open(p.with_suffix(".csv"), "w", newline="") as fh:           # the same samples as CSV for a notebook
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    out.append(f"{name}: {len(rows)} samples")
    for i, c in enumerate(cols):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[i]))
            except (IndexError, ValueError):
                pass
        if vals and c.lower() not in ("date", "time", "gpu", "idx", "entity", "id"):
            out.append(f"  {c:<10} mean {sum(vals) / len(vals):>14,.3f}   max {max(vals):>14,.3f}")
text = "\n".join(out) or "no samples"
print(text)
(Path(sys.argv[1]) / "summary.txt").write_text(text + "\n")
PY
}

if [ -n "${SUMMARY_ONLY:-}" ]; then summarize "${SUMMARY_ONLY}"; exit 0; fi

DURATION="${1:-120}"
OUT="${2:-results/profiles/gpu-$(date -u +%Y%m%dT%H%M%SZ)}"
GPUS="${GPUS:-0}"
INTERVAL_MS="${INTERVAL_MS:-1000}"
command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: run this on the GPU VM" >&2; exit 1; }
mkdir -p "${OUT}"

echo "Sampling GPU ${GPUS} for ${DURATION}s -> ${OUT}"
nvidia-smi dmon -i "${GPUS}" -s pucvmet -o DT -d 1 -c "${DURATION}" > "${OUT}/nvidia-smi-dmon.txt" 2>&1 &
SMI_PID=$!
DCGM_PID=""
if command -v dcgmi > /dev/null && dcgmi discovery -l > /dev/null 2>&1; then
  COUNT=$(( DURATION * 1000 / INTERVAL_MS ))
  dcgmi dmon -i "${GPUS}" -e 1002,1003,1004,1005,1009,1010 -d "${INTERVAL_MS}" -c "${COUNT}" \
    > "${OUT}/dcgmi-dmon.txt" 2>&1 &
  DCGM_PID=$!
else
  echo "  (dcgmi or nv-hostengine not available: nvidia-smi only. Install DCGM - datacenter-gpu-manager - for SM /"
  echo "   tensor / DRAM activity, or read the same counters from dcgm-exporter on Kubernetes.)"
fi
wait "${SMI_PID}" || true
if [ -n "${DCGM_PID}" ]; then wait "${DCGM_PID}" || true; fi
summarize "${OUT}"
