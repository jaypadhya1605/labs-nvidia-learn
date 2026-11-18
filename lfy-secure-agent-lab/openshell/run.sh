#!/usr/bin/env bash
# M10 on your Mac (Apple Silicon) or Linux box with Docker. OpenShell 0.1.x.
#   1) curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
#   2) kubectl -n lfy-safety port-forward svc/guarded-gateway 8080:8080   (separate terminal)
#   3) register http://localhost:8080/v1 as the sandbox's inference provider (docs: Providers -> Inference)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-soc-analyst}"
openshell sandbox create --name "${NAME}" --from ./openshell/sandbox -- python3 /sandbox/investigator.py --probe
openshell logs "${NAME}" --tail --source sandbox || true
echo "Next: openshell sandbox connect ${NAME}  ->  python3 /sandbox/investigator.py --analyze"
echo "Clean up: openshell sandbox delete ${NAME}"
