#!/usr/bin/env bash
# M9 step 3 (connected) - put the model in the host cache with NIM's own download-to-cache, for ONE profile.
# The NGC key exists only for the length of this Job; the air-gapped pod never gets one (NIM air-gap rule).
# Run ON THE VM:  sudo -E bash datacenter/prefetch-model.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
export KUBECONFIG=/etc/rancher/rke2/rke2.yaml
PROFILE="$(python3 -c "import yaml;print(yaml.safe_load(open('clusters/rke2/values.yaml'))['llm']['profile'])")"
IMAGE="$(python3 -c "import yaml;v=yaml.safe_load(open('charts/plover/values.yaml'))['llm']['image'];print(v['repository']+':'+v['tag'])")"
echo "Profile ${PROFILE} from ${IMAGE} -> /opt/nim-cache"
kubectl create namespace plv-prefetch --dry-run=client -o yaml | kubectl apply -f -
kubectl -n plv-prefetch create secret generic ngc-api --from-literal=NGC_API_KEY="${NGC_API_KEY}" --dry-run=client -o yaml | kubectl apply -f -
cat << YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata: {name: nim-prefetch, namespace: plv-prefetch}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      securityContext: {runAsUser: 1000, runAsGroup: 0, fsGroup: 0}
      containers:
        - name: prefetch
          image: ${IMAGE}
          command: ["download-to-cache", "--profile", "${PROFILE}"]
          env:
            - {name: NIM_CACHE_PATH, value: /opt/nim/.cache}
            - name: NGC_API_KEY
              valueFrom: {secretKeyRef: {name: ngc-api, key: NGC_API_KEY}}
          volumeMounts: [{name: cache, mountPath: /opt/nim/.cache}]
      volumes: [{name: cache, hostPath: {path: /opt/nim-cache, type: Directory}}]
YAML
kubectl -n plv-prefetch wait --for=condition=complete job/nim-prefetch --timeout=60m || {
  kubectl -n plv-prefetch logs job/nim-prefetch | tail -30
  echo "If it complains about GPUs: install the GPU Operator first and add a nvidia.com/gpu limit to the Job." >&2; exit 1; }
kubectl -n plv-prefetch logs job/nim-prefetch | tail -5
kubectl delete namespace plv-prefetch           # the key leaves the cluster with the Job
du -sh /opt/nim-cache
echo "Model cached. From your laptop: infra/tf.sh dc airgap   (cuts Internet egress), then datacenter/deploy-airgapped.sh"
