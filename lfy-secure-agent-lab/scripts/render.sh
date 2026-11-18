#!/usr/bin/env bash
# Render manifest templates with ONLY the lab variables substituted (other $ signs are left alone).
# Usage: scripts/render.sh file.yaml [file2.yaml ...] | kubectl apply -f -
set -euo pipefail
VARS="MIG_LARGE MIG_MEDIUM MIG_SMALL MIG_POOL NIM_LLM_IMAGE NIM_LLM_TAG NIM_LLM_MODEL NIM_SAFETY_IMAGE NIM_SAFETY_TAG NIM_SAFETY_MODEL NIM_JAILBREAK_IMAGE NIM_JAILBREAK_TAG NIM_EMBED_IMAGE NIM_EMBED_TAG NIM_EMBED_MODEL LLM_PRECISION AGENT_MAX_MODEL_LEN AGENT_EXTRA_ARGS AGENT_CACHE_FLAG SAFETY_MAX_MODEL_LEN NGC_RELEASE CUDA_DEVEL_IMAGE CUDA_BASE_IMAGE CUDA_ARCH AIPERF_VERSION GARAK_VERSION NAT_VERSION TOKENIZER KUSTAINER_IMAGE ACR_LOGIN_SERVER APP_TAG FAIL_MODE GPU_HOURLY_USD"
command -v envsubst >/dev/null 2>&1 || { echo "envsubst missing (macOS: brew install gettext)" >&2; exit 1; }
SHELL_FORMAT=""
for v in $VARS; do SHELL_FORMAT="${SHELL_FORMAT} \${${v}}"; done
for f in "$@"; do
  for v in $VARS; do
    case "$v" in *_EXTRA_ARGS) continue ;; esac
    eval "_val=\"\${${v}:-}\""
    if grep -q "\${${v}}" "$f" && [ -z "${_val}" ]; then
      echo "render.sh: \${${v}} is used in $f but is empty - did you 'source env/lab.env' from the repo root?" >&2; exit 1
    fi
  done
  envsubst "$SHELL_FORMAT" < "$f"
  printf '\n---\n'
done
