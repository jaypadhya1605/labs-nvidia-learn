#!/usr/bin/env bash
# M9 step 5 (laptop) - evidence for the customer's security team: no egress, every image from the mirror, no key.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
echo "1) Egress from a pod (expect FAIL/timeout):"
k rke2 run egress-probe --rm -i --restart=Never --image=curlimages/curl:8.17.0 --command -- \
  curl -sS -m 8 -o /dev/null -w "%{http_code}\n" https://nvcr.io/v2/ 2>&1 | tail -1
echo "2) Images actually running (all must be in the mirror list /opt/airgap/images.txt on the VM):"
k rke2 get pods -A -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.image}{"\n"}{end}{end}' | sort -u
echo "3) No NGC key anywhere in the serving namespace (expect nothing):"
k rke2 -n plv-serving get deploy -o yaml | grep -n "NGC_API_KEY" || echo "   none"
echo "4) NIM served from the local cache (look for the profile and no download progress):"
k rke2 -n plv-serving logs deploy/plover-llm --tail=400 2>/dev/null | grep -iE "profile|cache|download" | tail -8
