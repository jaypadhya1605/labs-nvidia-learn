#!/usr/bin/env bash
# M9 (laptop) - reach the DC cluster through an SSH tunnel: you are the jump host, like a customer's bastion.
# Usage: datacenter/kubeconfig.sh   then keep the printed ssh -L command running in another terminal.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
[ -n "${DC_IP:-}" ] || { echo "DC_IP unknown: infra/tf.sh dc output" >&2; exit 1; }
tmp="$(mktemp)"
ssh -o StrictHostKeyChecking=accept-new "${DC_USER:-plvadmin}@${DC_IP}" "sudo cat /etc/rancher/rke2/rke2.yaml" > "$tmp"
python3 - "$tmp" "${KCTX_RKE2}" << 'PY'
import sys, yaml
p, name = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open(p))
c["clusters"][0]["cluster"]["server"] = "https://127.0.0.1:16443"
c["clusters"][0]["name"] = c["users"][0]["name"] = c["contexts"][0]["name"] = name
c["contexts"][0]["context"] = {"cluster": name, "user": name}
c["current-context"] = name
yaml.safe_dump(c, open(p, "w"))
PY
KUBECONFIG="${HOME}/.kube/config:${tmp}" kubectl config view --flatten > "${tmp}.merged"
mv "${tmp}.merged" "${HOME}/.kube/config"; chmod 600 "${HOME}/.kube/config"; rm -f "$tmp"
echo "Context ${KCTX_RKE2} added. Open the tunnel (separate terminal, keep it running):"
echo "  ssh -N -L 16443:127.0.0.1:6443 ${DC_USER:-plvadmin}@${DC_IP}"
