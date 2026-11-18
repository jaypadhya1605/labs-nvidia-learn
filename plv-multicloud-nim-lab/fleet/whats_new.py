#!/usr/bin/env python3
"""M11 "what's new": every pinned component vs its newest upstream release - the input to the weekly sync's
"new features" section and to planned upgrades (pins.env is the single source of truth; CI enforces it).
  python3 fleet/whats_new.py                  # public sources only (git tags, Helm indexes, PyPI)
  python3 fleet/whats_new.py --json results/whats-new.json
NIM image tags need YOUR NGC key (scripts/check-images.sh verifies the pinned ones); this script never lists them.
"""
import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [  # pin, kind, where, release notes
    ("PIN_GPU_OPERATOR", "git", "https://github.com/NVIDIA/gpu-operator", "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/release-notes.html"),
    ("PIN_NIM_OPERATOR", "git", "https://github.com/NVIDIA/k8s-nim-operator", "https://github.com/NVIDIA/k8s-nim-operator/releases"),
    ("PIN_ESO", "helm", "https://charts.external-secrets.io/index.yaml#external-secrets", "https://github.com/external-secrets/external-secrets/releases"),
    ("PIN_ARGOCD_CHART", "helm", "https://argoproj.github.io/argo-helm/index.yaml#argo-cd", "https://github.com/argoproj/argo-cd/releases"),
    ("PIN_KPS", "helm", "https://prometheus-community.github.io/helm-charts/index.yaml#kube-prometheus-stack", "https://github.com/prometheus-community/helm-charts/releases"),
    ("PIN_PROM_ADAPTER", "helm", "https://prometheus-community.github.io/helm-charts/index.yaml#prometheus-adapter", "https://github.com/prometheus-community/helm-charts/releases"),
    ("PIN_ORAS", "git", "https://github.com/oras-project/oras", "https://github.com/oras-project/oras/releases"),
    ("PIN_RKE2", "git-rke2", "https://github.com/rancher/rke2", "https://github.com/rancher/rke2/releases"),
    ("PIN_HELM", "git", "https://github.com/helm/helm", "https://github.com/helm/helm/releases"),
    ("PIN_AIPERF", "pypi", "aiperf", "https://github.com/ai-dynamo/aiperf/releases"),
    ("PIN_MLFLOW", "pypi", "mlflow", "https://github.com/mlflow/mlflow/releases"),
    ("PIN_PEFT", "pypi", "peft", "https://github.com/huggingface/peft/releases"),
    ("PIN_TRANSFORMERS", "pypi", "transformers", "https://github.com/huggingface/transformers/releases"),
]
SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:\+rke2r(\d+))?$")


def parse(v):
    m = SEMVER.match(v or "")
    return tuple(int(x or 0) for x in m.groups()) if m else None


def pins():
    out = {}
    for line in (ROOT / "env" / "pins.env").read_text().splitlines():
        m = re.match(r'export (PIN_[A-Z0-9_]+)="([^"]*)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def git_versions(url):
    r = subprocess.run(["git", "ls-remote", "--tags", url], capture_output=True, text=True, timeout=60)
    return [t.split("refs/tags/")[1] for t in r.stdout.splitlines() if "refs/tags/" in t and not t.endswith("^{}")]


def http_get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def helm_versions(spec):
    url, chart = spec.split("#")
    idx = yaml.safe_load(http_get(url))
    return [e["version"] for e in idx["entries"].get(chart, [])]


def pypi_versions(pkg):
    return list(json.loads(http_get(f"https://pypi.org/pypi/{pkg}/json"))["releases"].keys())


def newest(versions, same_major_as=None):
    stable = [v for v in versions if parse(v)]
    if same_major_as and parse(same_major_as):
        stable = [v for v in stable if parse(v)[0] == parse(same_major_as)[0]] or stable
    return max(stable, key=parse) if stable else None


def status(pinned, latest):
    p, l = parse(pinned), parse(latest)
    if not p or not l:
        return "unknown (pre-release pin or unparseable)"
    if l <= p:
        return "current"
    if l[0] > p[0]:
        return "MAJOR behind"
    if l[1] > p[1]:
        return "minor behind"
    return "patch behind"


def check(pin_name, pinned, kind, where):
    try:
        if kind.startswith("git"):
            vs = git_versions(where)
            if kind == "git-rke2":
                vs = [v for v in vs if "+rke2r" in v and "-rc" not in v]
        elif kind == "helm":
            vs = helm_versions(where)
        else:
            vs = pypi_versions(where)
        latest = newest(vs)
        st = status(pinned, latest)
        if st == "MAJOR behind":   # also report the newest release on the pinned major (the safe upgrade)
            same = newest(vs, pinned)
            st += f" (newest on v{parse(pinned)[0]}: {same}, {status(pinned, same)})"
        return latest, st
    except Exception as exc:  # noqa: BLE001 - one unreachable source must not hide the others
        return None, f"unreachable ({type(exc).__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pin, rows = pins(), []
    print(f"{'component':<20}{'pinned':<18}{'latest':<18}status")
    for name, kind, where, notes in SOURCES:
        pinned = pin.get(name, "?")
        latest, st = check(name, pinned, kind, where)
        rows.append({"pin": name, "pinned": pinned, "latest": latest, "status": st, "notes": notes})
        print(f"{name.replace('PIN_', '').lower():<20}{pinned:<18}{(latest or '-'):<18}{st}")
    print("\nNIM image: scripts/check-images.sh proves the pinned tag is pullable; read the NIM release notes for newer tags.")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
