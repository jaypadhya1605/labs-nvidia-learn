#!/usr/bin/env bash
# M2 (first): kube-prometheus-stack + prometheus-adapter + KHA rules and dashboard.
# Installed BEFORE the GPU Operator so the ServiceMonitor CRD exists when the operator creates DCGM's ServiceMonitor.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo update >/dev/null
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

GRAFANA_PW="$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-20)"
helm upgrade --install kps prometheus-community/kube-prometheus-stack --version "${KPS_VERSION}" \
  -n monitoring -f platform/kube-prometheus-values.yaml \
  --set grafana.adminPassword="${GRAFANA_PW}" --wait --timeout 15m
if [ -n "${KV_NAME:-}" ]; then
  az keyvault secret set --vault-name "${KV_NAME}" -n grafana-admin-password --value "${GRAFANA_PW}" -o none \
    && echo "Grafana admin password stored in Key Vault ${KV_NAME} (secret grafana-admin-password)"
else
  echo "KV_NAME not set; Grafana admin password: ${GRAFANA_PW}"
fi

helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter --version "${PROM_ADAPTER_VERSION}" \
  -n monitoring -f platform/prometheus-adapter-values.yaml --wait --timeout 10m

kubectl apply -f observability/recording-rules.yaml
kubectl -n monitoring create configmap kha-genai-dashboard \
  --from-file=kha-genai-dashboard.json=observability/grafana-dashboard-kha.json --dry-run=client -o yaml \
  | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
echo "Grafana: kubectl -n monitoring port-forward svc/kps-grafana 3000:80   (user: admin)"
