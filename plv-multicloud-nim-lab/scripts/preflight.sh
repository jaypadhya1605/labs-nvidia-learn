#!/usr/bin/env bash
# shellcheck disable=SC2015  # ok/warn/bad always succeed, so A && B || C is safe here
# Preflight for Lab 04: tools, cloud logins, GPU quota per cloud, NGC + Hugging Face access.
# Usage: scripts/preflight.sh            (checks the clouds listed in PLV_CLUSTERS)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
. env/lab.env
FAILED=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAILED=1; }
has()  { command -v "$1" >/dev/null 2>&1; }

echo "Common tools"
for t in kubectl helm jq envsubst python3 terraform curl git; do
  if has "$t"; then ok "$t"; else bad "$t missing (macOS: brew install kubernetes-cli helm jq gettext python terraform)"; fi
done
python3 -c "import yaml" 2>/dev/null && ok "python yaml" || bad "pip3 install pyyaml"
has oras && ok "oras (M8)" || warn "oras missing - needed in M8 (brew install oras)"

for c in ${PLV_CLUSTERS}; do
  case "$c" in
    aks)
      echo "Azure"
      if has az && az account show >/dev/null 2>&1; then
        [ -n "${AZ_SUBSCRIPTION_ID}" ] && az account set --subscription "${AZ_SUBSCRIPTION_ID}"
        ok "subscription: $(az account show --query name -o tsv)"
        az vm list-usage -l "${AZ_LOCATION}" -o json 2>/dev/null | jq -r '.[] | select((.localName|test("A100|Low-priority";"i"))) | "    \(.localName): \(.currentValue)/\(.limit)"'
      else bad "az CLI missing or not logged in (az login)"; fi ;;
    gke)
      echo "Google Cloud"
      if has gcloud && gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
        ok "gcloud account: $(gcloud config get-value account 2>/dev/null)"
        [ -n "${GCP_PROJECT}" ] || bad "GCP_PROJECT is empty"
        gcloud compute regions describe "${GCP_REGION}" --project "${GCP_PROJECT}" --format=json 2>/dev/null \
          | jq -r '.quotas[] | select(.metric|test("GPU";"i")) | "    \(.metric): \(.usage)/\(.limit)"'
        gcloud compute accelerator-types list --project "${GCP_PROJECT}" --filter="zone:${GCP_ZONE}" --format='value(name)' 2>/dev/null | sed 's/^/    available in zone: /'
        has gke-gcloud-auth-plugin && ok "gke-gcloud-auth-plugin" || bad "gcloud components install gke-gcloud-auth-plugin"
      else bad "gcloud missing or not logged in (gcloud auth login && gcloud auth application-default login)"; fi ;;
    eks)
      echo "AWS"
      if has aws && aws sts get-caller-identity >/dev/null 2>&1; then ok "aws: $(aws sts get-caller-identity --query Arn --output text)"
        aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region "${AWS_REGION}" --query 'Quota.Value' --output text 2>/dev/null | sed 's/^/    G and VT on-demand vCPU quota: /'
      else bad "aws CLI missing or not logged in"; fi ;;
    oke)
      echo "Oracle Cloud"
      if has oci && oci iam region list >/dev/null 2>&1; then ok "oci CLI configured"; else bad "oci CLI missing or not configured (oci setup config)"; fi ;;
  esac
done

echo "NVIDIA NGC"
if [ -z "${NGC_API_KEY}" ]; then bad "NGC_API_KEY is empty"; else
  scripts/check-images.sh || bad "one or more pinned images are not pullable with your NGC key"
fi
echo "Hugging Face"
if [ -z "${HF_TOKEN}" ]; then warn "HF_TOKEN empty: AIPerf tokenizer and M8 LoRA training need it"; else
  code="$(curl -s -o /dev/null -w '%{http_code}' -L -H "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json)"
  [ "$code" = "200" ] && ok "HF token can read meta-llama/Llama-3.1-8B-Instruct" || warn "HF returned ${code}: accept the Llama 3.1 license on huggingface.co"
fi
echo
if [ "$FAILED" = "0" ]; then echo "Preflight passed."; else echo "Preflight found blocking issues."; exit 1; fi
