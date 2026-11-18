#!/usr/bin/env bash
# engines/logs.sh - follow the engine logs (startup, weight loading, KV cache size, CUDA graph capture, errors).
# Usage: engines/logs.sh [--save results/logs/<label>.log]
#   --save  write the full log, with docker's RFC 3339 timestamp on every line, to a file instead of following it.
#           profiling/coldstart.py reads these files: the timestamps give the cold-start wall time on EVERY engine,
#           even the ones that do not print their own load timings.
set -euo pipefail
if [ "${1:-}" = "--save" ]; then
  out="${2:?usage: engines/logs.sh --save <file>}"
  mkdir -p "$(dirname "${out}")"
  docker logs -t ttg-engine > "${out}" 2>&1
  echo "saved $(wc -l < "${out}") lines -> ${out}   (next: python3 profiling/coldstart.py ${out})"
else
  docker logs -f --tail "${TAIL:-200}" ttg-engine
fi
