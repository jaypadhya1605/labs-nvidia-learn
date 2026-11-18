#!/usr/bin/env bash
# Render EVERY manifest the lab applies (chart x 5 clusters + add-on overlay, GitOps apps, observability, engines,
# MLOps, data center, CUDA probe, secret stores) and validate them with kubeconform: Kubernetes schemas + CRD schemas.
# Usage: scripts/validate-manifests.sh   (run scripts/gen-crd-schemas.sh first; CI does both)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/pins.env
SCHEMAS="${SCHEMAS:-.schemas}"
OUT="$(mktemp -d)"; trap 'rm -rf "${OUT}"' EXIT
# placeholder values for templates that are normally filled from env/lab.env + Terraform outputs
export GIT_REPO_URL="https://github.com/example/plv-multicloud-nim-lab.git" GIT_REVISION=main GITHUB_OWNER=Example GITHUB_OWNER_LC=example \
  AZ_KV_URI="https://kv-example.vault.azure.net/" AZ_ESO_CLIENT_ID="00000000-0000-0000-0000-000000000000" \
  AZ_TENANT_ID="00000000-0000-0000-0000-000000000000" AZ_ACR_LOGIN_SERVER=example.azurecr.io GCP_PROJECT=example-project \
  GCP_REGION=us-central1 GCP_AR_REPO=us-central1-docker.pkg.dev/example-project/plv-images AWS_REGION=us-east-1 \
  OCI_REGION=us-ashburn-1 OCI_VAULT_OCID=ocid1.vault.oc1..example CUDA_ARCH=80 \
  TRTLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct ENGINE_MODE=build ADAPTER_NAME=plover-style-v1
API="--set api.image.repository=ghcr.io/example/plover-api"
for c in aks gke eks oke rke2; do
  # shellcheck disable=SC2086
  helm template plover charts/plover -n plv-serving -f "clusters/${c}/values.yaml" ${API} > "${OUT}/chart-${c}.yaml"
done
# shellcheck disable=SC2086
helm template plover charts/plover -n plv-serving -f clusters/aks/values.yaml -f addon-nemotron35/values/aks-lightning.yaml ${API} \
  --set llm.lora.adapters[0].name=plover-style-v1 --set llm.lora.adapters[0].ref=ghcr.io/example/plover-adapters:style-v1 > "${OUT}/chart-addon.yaml"
for f in gitops/bootstrap/*.yaml; do printf -- '---\n'; cat "$f"; printf '\n'; done > "${OUT}/gitops-bootstrap.yaml"
for f in gitops/clusters/*/*.yaml; do printf -- '---\n'; cat "$f"; printf '\n'; done > "${OUT}/gitops-apps.yaml"
cp observability/recording-rules.yaml observability/grafana-dashboard.yaml platform/gke/gpu-operator-quota.yaml "${OUT}/"
scripts/render.sh engines/trtllm-serve.yaml engines/triton/triage.yaml engines/triton/perf-job.yaml \
  mlops/k8s/mlflow.yaml mlops/k8s/train-lora-job.yaml datacenter/k8s/registry.yaml bench/physics/bandwidth-job.yaml \
  platform/templates/aks/clustersecretstore.tmpl.yaml platform/templates/gke/clustersecretstore.tmpl.yaml \
  platform/templates/eks/clustersecretstore.tmpl.yaml platform/templates/oke/clustersecretstore.tmpl.yaml > "${OUT}/rendered.yaml"
cp mlops/k8s/dataset-job.yaml mlops/k8s/hf-token-externalsecret.yaml "${OUT}/"
scripts/render.sh platform/templates/plover-generated-values.tmpl.yaml > "${OUT}/values-check.txt"
# a variable missing from render.sh VARS would silently ship as a literal "${...}" (schema-valid, but broken in the cluster)
# shellcheck disable=SC2016  # the pattern is a literal ${, on purpose
if grep -n '\${[A-Z_]*}' "${OUT}"/*.yaml "${OUT}/values-check.txt"; then echo "unrendered \${VAR} above: add it to scripts/render.sh VARS" >&2; exit 1; fi
rm -f "${OUT}/values-check.txt"
kubeconform -strict -summary -kubernetes-version 1.34.0 \
  -schema-location default -schema-location "${SCHEMAS}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json" \
  "${OUT}"/*.yaml
