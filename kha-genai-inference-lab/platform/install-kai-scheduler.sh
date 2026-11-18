#!/usr/bin/env bash
# M8 part F: KAI Scheduler (CNCF sandbox, originally from NVIDIA Run:ai) + KHA queues.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
helm upgrade -i kai-scheduler oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace --version "${KAI_VERSION}" --wait --timeout 10m
kubectl apply -f platform/kai-queues.yaml
kubectl get queues.scheduling.run.ai
kubectl get priorityclass | grep -E 'train|build|inference' || true
