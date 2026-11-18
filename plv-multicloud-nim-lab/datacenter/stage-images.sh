#!/usr/bin/env bash
# M9 step 2 (connected) - fill the local registry with EXACTLY the images the charts need, then point RKE2 at it.
# Run ON THE VM:  sudo -E bash datacenter/stage-images.sh      (needs env/lab.env with NGC_API_KEY + GITHUB_OWNER)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
export KUBECONFIG=/etc/rancher/rke2/rke2.yaml
[ -n "${NGC_API_KEY}" ] || { echo "NGC_API_KEY empty" >&2; exit 1; }
echo "== 1/4 local registry"
scripts/render.sh datacenter/k8s/registry.yaml | kubectl apply -f -
kubectl -n plv-registry rollout status deploy/registry --timeout=5m
echo "== 2/4 charts -> /opt/airgap/charts, image list from the RENDERED charts"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update nvidia >/dev/null
helm pull nvidia/gpu-operator --version "${PIN_GPU_OPERATOR}" -d /opt/airgap/charts
API_IMG="ghcr.io/$(echo "${GITHUB_OWNER}" | tr '[:upper:]' '[:lower:]')/plover-api"
{ helm template gpu-operator "/opt/airgap/charts/gpu-operator-${PIN_GPU_OPERATOR}.tgz" -n gpu-operator -f platform/gpu-operator/rke2.yaml
  helm template plover charts/plover -n plv-serving -f clusters/rke2/values.yaml --set "api.image.repository=${API_IMG}"
} | python3 datacenter/list-images.py | tee /opt/airgap/images.txt
echo "== 3/4 skopeo copy -> 127.0.0.1:5000/<registry>/<path>  (linux/amd64 only: multi-arch NIM images would double the copy)"
while read -r ref; do
  [ -n "${ref}" ] || continue
  creds=""
  case "${ref}" in nvcr.io/nim/*) creds="--src-creds=\$oauthtoken:${NGC_API_KEY}" ;; esac
  echo "-> ${ref}"
  # shellcheck disable=SC2086
  skopeo copy --retry-times 3 --dest-tls-verify=false --override-os linux --override-arch amd64 ${creds} "docker://${ref}" "docker://127.0.0.1:5000/${ref}"
done < /opt/airgap/images.txt
curl -s http://127.0.0.1:5000/v2/_catalog | jq -r '.repositories[]' | sed 's/^/   mirrored: /'
echo "== 4/4 registries.yaml + restart RKE2 (containerd now resolves every registry through the mirror)"
cp datacenter/registries.yaml.tmpl /etc/rancher/rke2/registries.yaml
systemctl restart rke2-server
until kubectl get nodes >/dev/null 2>&1; do sleep 5; done
echo "Staged $(wc -l < /opt/airgap/images.txt) images. Next: sudo -E bash datacenter/prefetch-model.sh"
