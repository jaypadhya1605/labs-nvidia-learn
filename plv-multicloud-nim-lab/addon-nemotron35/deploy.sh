#!/usr/bin/env bash
# Add-on A2: deploy a Nemotron 3.5 NIM NEXT TO Plover with the same chart - a separate release in plv-addon, on the
# cluster's GPU freed by the M6 GitOps pause. Not in Argo CD on purpose: day-0 evaluation, not a release.
# Usage: addon-nemotron35/deploy.sh <aks|gke> <lightning|super-vl> <up|down|profiles>
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
CL="${1:?usage: addon-nemotron35/deploy.sh <aks|gke> <lightning|super-vl> <up|down|profiles>}"; M="${2:?model}"; A="${3:?action}"
VALUES="addon-nemotron35/values/${CL}-${M}.yaml"
[ -f "${VALUES}" ] || die "no overlay ${VALUES} (provided: aks-lightning, gke-super-vl - copy one for other pairs)"
case "${A}" in
  up)
    engines/bakeoff.sh "${CL}" pause
    h "${CL}" upgrade --install plover-addon charts/plover -n plv-addon --create-namespace \
      -f "clusters/${CL}/values.yaml" -f "clusters/${CL}/generated/plover-generated-values.yaml" -f "${VALUES}"
    echo "Watch the first start (model download): kubectl --context $(ctx "${CL}") -n plv-addon logs -f -l app.kubernetes.io/name=plover-llm"
    echo "Then: kubectl --context $(ctx "${CL}") -n plv-addon port-forward svc/plover-llm 8010:8000   and  addon-nemotron35/run-eval.sh" ;;
  profiles)
    POD="$(k "${CL}" -n plv-addon get pod -l app.kubernetes.io/name=plover-llm -o jsonpath='{.items[0].metadata.name}')"
    k "${CL}" -n plv-addon exec "${POD}" -- list-model-profiles | tee "results/${CL}-${M}-profiles.txt" ;;
  down)
    h "${CL}" uninstall plover-addon -n plv-addon || true
    k "${CL}" delete namespace plv-addon --ignore-not-found
    engines/bakeoff.sh "${CL}" resume ;;
  *) die "unknown action ${A}" ;;
esac
