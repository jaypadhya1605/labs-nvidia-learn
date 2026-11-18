#!/usr/bin/env bash
# Preflight: tools, Azure login, GPU quota and SKU availability, NGC + Hugging Face access.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
. env/lab.env
FAILED=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAILED=1; }

echo "Tools"
for t in az kubectl helm jq envsubst python3 openssl curl column; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else bad "$t missing (macOS: brew install azure-cli kubernetes-cli helm jq gettext python openssl)"; fi
done

echo "Azure"
if az account show >/dev/null 2>&1; then
  if [ -n "${SUBSCRIPTION_ID}" ]; then az account set --subscription "${SUBSCRIPTION_ID}"; fi
  ok "subscription: $(az account show --query name -o tsv)"
else
  bad "not logged in: run 'az login'"
fi

echo "GPU quota in ${LOCATION} (MIG pool needs a ${GPU_VM_SKU}; Dynamo pool needs a ${DYN_VM_SKU})"
az vm list-usage -l "${LOCATION}" -o json 2>/dev/null \
  | jq -r --arg g "${GPU_QUOTA_GREP}" '.[] | select((.localName|test($g;"i")) or (.name.value|test("lowPriority";"i"))) | "    \(.localName): \(.currentValue)/\(.limit)"' \
  || warn "could not read quota"
echo "    (Spot pools consume 'Total Regional Low-priority vCPUs'; Regular pools consume the GPU family quota.)"

for sku in "${GPU_VM_SKU}" "${DYN_VM_SKU}"; do
  r="$(az vm list-skus -l "${LOCATION}" --size "${sku}" --query "[0].restrictions" -o json 2>/dev/null)"
  if [ "$r" = "[]" ]; then ok "${sku} offered in ${LOCATION}"; else warn "${sku} restricted or unknown in ${LOCATION}: ${r:-no data}"; fi
done

echo "NVIDIA NGC"
if [ -z "${NGC_API_KEY}" ]; then bad "NGC_API_KEY is empty"; else
  code="$(curl -s -o /dev/null -w '%{http_code}' -u "\$oauthtoken:${NGC_API_KEY}" "https://nvcr.io/proxy_auth?scope=repository:${NIM_LLM_IMAGE#nvcr.io/}:pull")"
  if [ "$code" = "200" ]; then ok "NGC key can pull ${NIM_LLM_IMAGE}"; else bad "NGC auth returned HTTP ${code} for ${NIM_LLM_IMAGE}"; fi
fi

echo "Hugging Face"
if [ -z "${HF_TOKEN}" ]; then warn "HF_TOKEN empty: gated tokenizer ${TOKENIZER} will fail"; else
  code="$(curl -s -o /dev/null -w '%{http_code}' -L -H "Authorization: Bearer ${HF_TOKEN}" "https://huggingface.co/${TOKENIZER}/resolve/main/tokenizer_config.json")"
  if [ "$code" = "200" ]; then ok "HF token can read ${TOKENIZER}"; else warn "HF returned ${code} for ${TOKENIZER}: accept the model license on huggingface.co"; fi
fi

echo
if [ "$FAILED" = "0" ]; then echo "Preflight passed."; else echo "Preflight found blocking issues."; exit 1; fi
