#!/usr/bin/env bash
# M10: NVIDIA Dynamo platform (operator + CRDs). Since Dynamo 1.x no NATS/etcd is needed by default
# (Kubernetes-native discovery, TCP request plane, ZMQ event plane).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
helm upgrade --install dynamo-platform \
  "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${DYNAMO_VERSION}.tgz" \
  --namespace dynamo-system --create-namespace --wait --timeout 10m
kubectl get pods -n dynamo-system
kubectl get crd | grep -i dynamo
kubectl create namespace kha-dynamo --dry-run=client -o yaml | kubectl apply -f -
echo "Next: workloads/create-secrets.sh (creates hf-token-secret in kha-dynamo), then dynamo/model-cache.yaml"
