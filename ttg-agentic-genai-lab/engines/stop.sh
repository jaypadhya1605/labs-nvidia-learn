#!/usr/bin/env bash
# engines/stop.sh - stop and remove the Lab 05 engine container (ttg-engine). Frees the GPU for the next variant.
# Usage: engines/stop.sh
set -euo pipefail
if docker ps -a --format '{{.Names}}' | grep -qx ttg-engine; then
  docker rm -f ttg-engine >/dev/null
  echo "ttg-engine removed"
else
  echo "no ttg-engine container"
fi
