#!/usr/bin/env bash
# M11 step 3: SBOM (syft) -> vulnerability scan (grype) -> VEX statement (vexctl) -> re-scan honoring VEX.
# Tools: brew install syft grype ; go install github.com/openvex/vexctl@latest
# Usage: supplychain/sbom-scan-vex.sh [soc-agent|guarded-gateway] [CVE-ID-you-assessed-as-not-exploitable]
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
APP="${1:-soc-agent}"; CVE="${2:-}"
IMG="${ACR_LOGIN_SERVER}/lfy/${APP}:${APP_TAG}"
TOKEN="$(az acr login -n "${ACR_NAME}" --expose-token --query accessToken -o tsv)"
export SYFT_REGISTRY_AUTH_AUTHORITY="${ACR_LOGIN_SERVER}" SYFT_REGISTRY_AUTH_USERNAME=00000000-0000-0000-0000-000000000000 SYFT_REGISTRY_AUTH_PASSWORD="${TOKEN}"
mkdir -p supplychain/out
syft "registry:${IMG}" -o "spdx-json=supplychain/out/${APP}.spdx.json"
grype "sbom:supplychain/out/${APP}.spdx.json" -o table | head -40 || true
if [ -n "${CVE}" ]; then
  vexctl create --product="pkg:oci/${APP}" --vuln="${CVE}" --status=not_affected \
    --justification=vulnerable_code_not_in_execute_path > "supplychain/out/${APP}.vex.json"
  echo "Re-scan honoring VEX:"; grype "sbom:supplychain/out/${APP}.spdx.json" --vex "supplychain/out/${APP}.vex.json" -o table | head -40 || true
fi
