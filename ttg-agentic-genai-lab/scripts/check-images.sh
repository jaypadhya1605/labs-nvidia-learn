#!/usr/bin/env bash
# Prove every pinned container image:tag exists BEFORE you rent a GPU (registry v2 HEAD on the manifest; nothing is
# downloaded). Works for nvcr.io (with your NGC key, anonymous otherwise), Docker Hub and gcr.io via the standard
# registry token challenge. 🔁 Lab 04's check-images, now multi-registry.
# Usage: scripts/check-images.sh [extra-image:tag ...]
#   NGC_API_KEY=nvapi-... scripts/check-images.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/pins.env
if [ -f env/lab.env ]; then
  # shellcheck source=/dev/null
  . env/lab.env
fi

IMAGES="${PIN_VLLM_IMAGE}:${PIN_VLLM_TAG} ${PIN_SGLANG_IMAGE}:${PIN_SGLANG_TAG} ${PIN_TRTLLM_IMAGE}:${PIN_TRTLLM_TAG} \
${PIN_NIM_LLM_IMAGE}:${PIN_NIM_LLM_TAG} ${PIN_NIM_LIGHTNING_IMAGE}:${PIN_NIM_LIGHTNING_TAG} \
${PIN_PYTORCH_IMAGE}:${PIN_NGC_RELEASE}-py3 ${PIN_CUDA_DEVEL} ${PIN_PYTHON_IMAGE} ${PIN_KANIKO_IMAGE} \
${PIN_HELM_IMAGE} ${PIN_SHELLCHECK_IMAGE} ${PIN_KIND_NODE_IMAGE} $*"
ACCEPT='application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'

# split "registry/repo:tag" -> registry repo tag (Docker Hub defaults: registry-1.docker.io, library/<name>)
parse_ref() {
  _ref="$1"; _tag="${_ref##*:}"; _name="${_ref%:*}"
  case "${_name}" in
    */*) _first="${_name%%/*}" ;;
    *) _first="" ;;
  esac
  case "${_first}" in
    *.*|*:*|localhost) _reg="${_first}"; _repo="${_name#*/}" ;;
    *) _reg="registry-1.docker.io"; _repo="${_name}"; case "${_repo}" in */*) ;; *) _repo="library/${_repo}" ;; esac ;;
  esac
  echo "${_reg} ${_repo} ${_tag}"
}

check_one() {
  read -r reg repo tag <<EOF2
$(parse_ref "$1")
EOF2
  url="https://${reg}/v2/${repo}/manifests/${tag}"
  # 1) unauthenticated probe -> 401 with a Bearer challenge (realm, service)
  challenge="$(curl -s -o /dev/null -D - -I -H "Accept: ${ACCEPT}" "${url}" | tr -d '\r' | grep -i '^www-authenticate:' || true)"
  token=""
  if [ -n "${challenge}" ]; then
    realm="$(printf '%s' "${challenge}" | sed -n 's/.*realm="\([^"]*\)".*/\1/p')"
    service="$(printf '%s' "${challenge}" | sed -n 's/.*service="\([^"]*\)".*/\1/p')"
    turl="${realm}?scope=repository:${repo}:pull"
    if [ -n "${service}" ]; then turl="${turl}&service=${service}"; fi
    if [ "${reg}" = "nvcr.io" ] && [ -n "${NGC_API_KEY:-}" ]; then
      tok_json="$(curl -s -u "\$oauthtoken:${NGC_API_KEY}" "${turl}")"
    else
      tok_json="$(curl -s "${turl}")"
    fi
    token="$(printf '%s' "${tok_json}" | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin); print(d.get("token") or d.get("access_token") or "")
except Exception:
    print("")')"
  fi
  if [ -n "${token}" ]; then
    code="$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer ${token}" -H "Accept: ${ACCEPT}" "${url}")"
  else
    code="$(curl -s -o /dev/null -w '%{http_code}' -I -H "Accept: ${ACCEPT}" "${url}")"
  fi
  if [ "${code}" = "200" ]; then
    printf '  [ok]   %s\n' "$1"
    return 0
  fi
  hint="check the tag in the registry UI and update env/pins.env"
  if [ "${reg}" = "nvcr.io" ] && [ -z "${NGC_API_KEY:-}" ]; then hint="set NGC_API_KEY (NIM images need it), then retry"; fi
  printf '  [FAIL] %s (HTTP %s) - %s\n' "$1" "${code}" "${hint}"
  return 1
}

RC=0
for ref in ${IMAGES}; do
  check_one "${ref}" || RC=1
done
exit "${RC}"
