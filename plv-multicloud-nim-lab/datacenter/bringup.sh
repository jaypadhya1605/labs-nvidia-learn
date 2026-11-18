#!/usr/bin/env bash
# M9 step 1 - turn the DC-sim VM into a single-node "customer data center" Kubernetes: host driver + RKE2 + tools.
# Run ON THE VM (while it still has Internet):  sudo -E bash datacenter/bringup.sh
# Why host driver: in most customer DCs the platform team owns the driver (LTSB, change-controlled), so the GPU
# Operator runs with driver.enabled=false (platform/gpu-operator/rke2.yaml). Lab 02/03 (AKS) let the operator own it.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/pins.env
[ "$(id -u)" = "0" ] || { echo "run with sudo -E" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
echo "== 1/5 NVIDIA driver (${PIN_DC_DRIVER_BRANCH})"
if ! nvidia-smi >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq ubuntu-drivers-common skopeo jq curl >/dev/null
  ubuntu-drivers install --gpgpu "nvidia:${PIN_DC_DRIVER_BRANCH}"
  apt-get install -y -qq "nvidia-utils-${PIN_DC_DRIVER_BRANCH}" >/dev/null || true
  modprobe nvidia 2>/dev/null || { echo ">>> Driver installed. REBOOT the VM (sudo reboot), then re-run this script."; exit 0; }
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
apt-get install -y -qq skopeo jq curl gettext-base python3-yaml >/dev/null
echo "== 2/5 RKE2 ${PIN_RKE2}"
if ! systemctl is-active --quiet rke2-server; then
  curl -sfL https://get.rke2.io | INSTALL_RKE2_VERSION="${PIN_RKE2}" sh -
  systemctl enable --now rke2-server
fi
ln -sf /var/lib/rancher/rke2/bin/kubectl /usr/local/bin/kubectl
export KUBECONFIG=/etc/rancher/rke2/rke2.yaml
until kubectl get nodes >/dev/null 2>&1; do sleep 5; done
kubectl wait --for=condition=Ready node --all --timeout=10m
echo "== 3/5 helm ${PIN_HELM}"
if ! helm version --short 2>/dev/null | grep -q "${PIN_HELM}"; then
  curl -sfL "https://get.helm.sh/helm-${PIN_HELM}-linux-amd64.tar.gz" | tar -xz -C /tmp
  install -m 0755 /tmp/linux-amd64/helm /usr/local/bin/helm
fi
echo "== 4/5 host paths: model cache (uid 1000 / gid 0, the NIM user) and registry storage"
mkdir -p /opt/nim-cache /opt/registry /opt/airgap/charts
chown 1000:0 /opt/nim-cache && chmod 0775 /opt/nim-cache
echo "== 5/5 label the GPU node like every cloud pool (plv.io/pool=gpu)"
kubectl label node --all plv.io/pool=gpu --overwrite
kubectl get nodes -o wide
echo "Next (still connected): sudo -E bash datacenter/stage-images.sh && sudo -E bash datacenter/prefetch-model.sh"
