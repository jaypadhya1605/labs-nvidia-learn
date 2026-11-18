#!/usr/bin/env bash
# M8: split the AKS A100 into two 3g.40gb MIG slices (serve on one, train on the other) and back (🔁 Lab 02 MIG).
# Usage: mlops/mig.sh <cluster> <all-3g.40gb|all-disabled>
# Then commit llm.gpuResource=nvidia.com/mig-3g.40gb (or back to nvidia.com/gpu) in clusters/<cluster>/values.yaml.
# The label is reset if the node is re-created (Terraform owns the pool labels) - fine for a lab, not for prod.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
CL="${1:?usage: mlops/mig.sh <cluster> <all-3g.40gb|all-disabled>}"; CFG="${2:?mig config}"
for n in $(k "${CL}" get nodes -l plv.io/pool=gpu -o jsonpath='{.items[*].metadata.name}'); do
  k "${CL}" label node "$n" "nvidia.com/mig.config=${CFG}" --overwrite
done
echo "MIG Manager is reconfiguring (GPU pods are evicted; the VM may reboot). Watch:"
echo "  kubectl --context $(ctx "${CL}") get nodes -l plv.io/pool=gpu -L nvidia.com/mig.config.state -w"
echo "Valid configs on this node: kubectl --context $(ctx "${CL}") -n gpu-operator get cm -o name | grep mig-config"
