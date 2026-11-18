#!/usr/bin/env bash
# Verify every pinned nvcr.io image:tag exists and is pullable with YOUR NGC key (registry v2 HEAD on the manifest).
# Runs on your machine; nothing is downloaded. Usage: scripts/check-images.sh [extra-image:tag ...]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
. env/pins.env
: "${NGC_API_KEY:?NGC_API_KEY is empty}"
IMAGES="${PIN_NIM_LLM_IMAGE}:${PIN_NIM_LLM_TAG} ${PIN_TRTLLM_IMAGE}:${PIN_TRTLLM_TAG} nvcr.io/nvidia/pytorch:${PIN_NGC_RELEASE}-py3 nvcr.io/nvidia/tensorrt:${PIN_NGC_RELEASE}-py3 nvcr.io/nvidia/tritonserver:${PIN_NGC_RELEASE}-py3 ${PIN_CUDA_DEVEL} $*"
RC=0
for ref in ${IMAGES}; do
  repo="${ref#nvcr.io/}"; repo="${repo%:*}"; tag="${ref##*:}"
  token="$(curl -s -u "\$oauthtoken:${NGC_API_KEY}" "https://nvcr.io/proxy_auth?scope=repository:${repo}:pull" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("token",""))' 2>/dev/null)"
  code="$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer ${token}" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://nvcr.io/v2/${repo}/manifests/${tag}")"
  if [ "${code}" = "200" ]; then printf '  [ok]   %s\n' "${ref}"; else printf '  [FAIL] %s (HTTP %s) - check the tag on catalog.ngc.nvidia.com and update env/pins.env\n' "${ref}" "${code}"; RC=1; fi
done
exit "${RC}"
