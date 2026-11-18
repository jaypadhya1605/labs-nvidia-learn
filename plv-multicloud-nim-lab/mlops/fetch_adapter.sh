#!/usr/bin/env bash
# Copy the trained adapter off the cluster's MLOps volume (no registry credentials ever live in the cluster).
# Usage: mlops/fetch_adapter.sh <cluster> <adapter-name>   -> results/adapters/<adapter-name>/
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
CL="${1:?usage: mlops/fetch_adapter.sh <cluster> <adapter-name>}"; NAME="${2:?adapter name}"
k "${CL}" -n plv-mlops run adapter-fetch --image=busybox:1.37 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"adapter-fetch","image":"busybox:1.37","command":["sleep","600"],"volumeMounts":[{"name":"d","mountPath":"/data"}]}],"volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"plv-mlops-data"}}]}}'
k "${CL}" -n plv-mlops wait --for=condition=Ready pod/adapter-fetch --timeout=5m
mkdir -p results/adapters
k "${CL}" -n plv-mlops cp "adapter-fetch:/data/adapters/${NAME}" "results/adapters/${NAME}"
k "${CL}" -n plv-mlops cp adapter-fetch:/data/eval "results/adapters/${NAME}-eval" 2>/dev/null || true
k "${CL}" -n plv-mlops delete pod adapter-fetch --wait=false
ls -la "results/adapters/${NAME}"
