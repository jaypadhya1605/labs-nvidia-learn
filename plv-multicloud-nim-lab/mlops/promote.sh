#!/usr/bin/env bash
# M8 step 5 - release = a Git commit. Edits clusters/<cluster>/values.yaml with yq, commits, and lets Argo CD roll it out.
# Usage:
#   mlops/promote.sh <cluster> canary   <adapter-name> <oci-ref@digest> <weight%>   load the adapter + route weight% to it
#   mlops/promote.sh <cluster> promote  <adapter-name>                            adapter becomes STABLE, canary cleared
#   mlops/promote.sh <cluster> rollback                                            canary weight -> 0 (adapter stays loaded)
set -euo pipefail
cd "$(dirname "$0")/.."
CL="${1:?usage: mlops/promote.sh <cluster> <canary|promote|rollback> ...}"; ACTION="${2:?action}"
F="clusters/${CL}/values.yaml"
[ -f "$F" ] || { echo "no $F" >&2; exit 1; }
command -v yq >/dev/null || { echo "yq (mikefarah v4) required: brew install yq" >&2; exit 1; }
case "${ACTION}" in
  canary)
    export N="${3:?adapter name}" R="${4:?oci ref}" W="${5:?weight 0-100}"
    yq -i '.llm.lora.adapters = ((.llm.lora.adapters // []) | map(select(.name != strenv(N))) + [{"name": strenv(N), "ref": strenv(R)}])' "$F"
    yq -i '.api.routing.canaryModel = strenv(N) | .api.routing.canaryWeight = (strenv(W) | tonumber)' "$F"
    MSG="canary: ${N} at ${W}% on ${CL} (${R})" ;;
  promote)
    export N="${3:?adapter name}"
    yq -i '.api.routing.stableModel = strenv(N) | .api.routing.canaryModel = "" | .api.routing.canaryWeight = 0' "$F"
    MSG="promote: ${N} is stable on ${CL}" ;;
  rollback)
    yq -i '.api.routing.canaryWeight = 0' "$F"
    MSG="rollback: canary weight 0 on ${CL}" ;;
  *) echo "unknown action ${ACTION}" >&2; exit 1 ;;
esac
helm template plover charts/plover -f "$F" --set api.image.repository=ghcr.io/x/plover-api >/dev/null   # schema check before commit
git add "$F" && git commit -m "${MSG}" && echo "Committed: ${MSG}.  git push  -> Argo CD syncs in ~3 min (or: argocd app sync plover)."
