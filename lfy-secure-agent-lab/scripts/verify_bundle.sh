#!/usr/bin/env bash
# Init-container helper (baked into both images): copy a ConfigMap bundle to a plain directory - ConfigMaps mount
# through symlinks - and, once M11 turns it on, verify its OpenSSF model-signing signature before the app may start.
set -euo pipefail
SRC="$1"; DST="$2"; SIG="$3"
mkdir -p "${DST}"
for f in "${SRC}"/*; do cp -L "${f}" "${DST}/"; done
if [ "${REQUIRE_SIGNATURE:-false}" != "true" ]; then
  echo "signature check disabled (REQUIRE_SIGNATURE=${REQUIRE_SIGNATURE:-false}); M11 turns it on"; exit 0
fi
[ -f "${SIG}" ] || { echo "REQUIRE_SIGNATURE=true but no signature at ${SIG}"; exit 1; }
model_signing verify key "${DST}" --signature "${SIG}" --public_key /keys/signing-pub.pem
echo "bundle signature OK: ${DST}"
