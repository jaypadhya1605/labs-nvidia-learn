#!/usr/bin/env bash
# Render per-cluster platform config from Terraform outputs into clusters/<c>/generated/ (commit these: no secrets inside).
# Argo CD cannot substitute variables, so we render once and let Git be the source of truth.
# Usage: scripts/render-cluster-config.sh aks|gke|eks|oke|rke2
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
C="${1:?usage: scripts/render-cluster-config.sh <cluster>}"
OUT="clusters/${C}/generated"
mkdir -p "${OUT}"
for t in platform/templates/"${C}"/*.tmpl.yaml; do
  [ -f "$t" ] || continue
  scripts/render.sh "$t" | sed '$d' | sed '$d' > "${OUT}/$(basename "${t%.tmpl.yaml}").yaml"
  echo "rendered ${OUT}/$(basename "${t%.tmpl.yaml}").yaml"
done
scripts/render.sh platform/templates/plover-generated-values.tmpl.yaml | sed '$d' | sed '$d' > "${OUT}/plover-generated-values.yaml"
echo "rendered ${OUT}/plover-generated-values.yaml"
echo "Next: git add clusters/${C}/generated && git commit -m 'render ${C} config' && git push"
