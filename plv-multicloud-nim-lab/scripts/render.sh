#!/usr/bin/env bash
# 🔁 Lab 02's safe envsubst: substitute ONLY the listed variables (other $ signs survive).
# Usage: scripts/render.sh template.yaml [...] > out.yaml
set -euo pipefail
VARS="GIT_REPO_URL GIT_REVISION GITHUB_OWNER GITHUB_OWNER_LC AZ_KV_URI AZ_ESO_CLIENT_ID AZ_TENANT_ID AZ_ACR_LOGIN_SERVER GCP_PROJECT GCP_REGION GCP_AR_REPO AWS_REGION OCI_REGION OCI_VAULT_OCID PIN_AIPERF PIN_CUDA_DEVEL PIN_NGC_RELEASE PIN_TRTLLM_IMAGE PIN_TRTLLM_TAG PIN_NIM_LLM_IMAGE PIN_NIM_LLM_TAG PIN_MLFLOW PIN_PEFT PIN_TRANSFORMERS CUDA_ARCH TRTLLM_MODEL ENGINE_MODE PIN_ORAS ADAPTER_NAME PIN_REGISTRY_IMAGE"
subst() {  # envsubst restricted to $VARS; a python3 fallback keeps this working where gettext is not installed
  if command -v envsubst >/dev/null 2>&1; then
    envsubst "$FMT" < "$1"
  else
    # shellcheck disable=SC2016  # the ${...} below is Python text, not shell
    VARS="$VARS" python3 -c 'import os, sys
t = open(sys.argv[1]).read()
for v in os.environ["VARS"].split():
    t = t.replace("${" + v + "}", os.environ.get(v, ""))
sys.stdout.write(t)' "$1"
  fi
}
FMT=""
for v in $VARS; do FMT="${FMT} \${${v}}"; done
for f in "$@"; do
  for v in $VARS; do
    eval "_val=\"\${${v}:-}\""
    if grep -q "\${${v}}" "$f" && [ -z "${_val}" ]; then
      echo "render.sh: \${${v}} is used in $f but is empty - source env/lab.env and the cluster outputs first" >&2; exit 1
    fi
  done
  subst "$f"
  printf '\n---\n'
done
