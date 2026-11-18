#!/usr/bin/env bash
# M6: send in-cluster Prometheus data to the Azure Monitor workspace (managed identity auth) and
# import the KHA dashboard into Azure Managed Grafana.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
: "${AMW_ID:?AMW_ID missing - run infra/deploy-landing-zone.sh up first}"
DCR_ID="$(az resource show --ids "${AMW_ID}" --query properties.defaultIngestionSettings.dataCollectionRuleResourceId -o tsv)"
DCE_ID="$(az resource show --ids "${AMW_ID}" --query properties.defaultIngestionSettings.dataCollectionEndpointResourceId -o tsv)"
DCE_URL="$(az resource show --ids "${DCE_ID}" --query properties.metricsIngestion.endpoint -o tsv)"
DCR_IMMUTABLE_ID="$(az resource show --ids "${DCR_ID}" --query properties.immutableId -o tsv)"
INGEST_URL="${DCE_URL}/dataCollectionRules/${DCR_IMMUTABLE_ID}/streams/Microsoft-PrometheusMetrics/api/v1/write?api-version=2023-04-24"
echo "Ingestion endpoint: ${INGEST_URL}"

# The kubelet (user-assigned) identity publishes metrics; Prometheus gets its token from IMDS.
az role assignment create --assignee-object-id "${KUBELET_OBJECT_ID}" --assignee-principal-type ServicePrincipal \
  --role "Monitoring Metrics Publisher" --scope "${DCR_ID}" -o none 2>/dev/null || echo "(role assignment may already exist)"

cat > platform/.remote-write.values.yaml << YAML
prometheus:
  prometheusSpec:
    remoteWrite:
      - url: "${INGEST_URL}"
        azureAd:
          cloud: AzurePublic
          managedIdentity:
            clientId: "${KUBELET_CLIENT_ID}"
        writeRelabelConfigs:
          # Ship only what the executive view needs (keeps ingestion cost low)
          - sourceLabels: [__name__]
            regex: "lfy:.*|guardrail_.*|agent_.*|DCGM_FI_DEV_POWER_USAGE|DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION|DCGM_FI_PROF_.*|vllm:.*"
            action: keep
YAML
helm upgrade kps prometheus-community/kube-prometheus-stack --version "${KPS_VERSION}" -n monitoring \
  --reuse-values -f platform/.remote-write.values.yaml --wait --timeout 10m
echo "Remote write configured. Data appears in Azure Monitor within ~5 minutes."

if [ -n "${GRAFANA_NAME:-}" ]; then
  az extension add --name amg --upgrade -o none 2>/dev/null || true
  az grafana dashboard import -g "${RG}" -n "${GRAFANA_NAME}" --definition observability/grafana-dashboard-lfy.json -o none \
    && echo "Dashboard imported: ${GRAFANA_ENDPOINT}"
fi
