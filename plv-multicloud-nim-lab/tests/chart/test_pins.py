"""Pin-drift test: every version in GitOps manifests must equal env/pins.env (single source of truth)."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def pins():
    out = {}
    for line in (ROOT / "env" / "pins.env").read_text().splitlines():
        m = re.match(r'export (PIN_[A-Z0-9_]+)="([^"]*)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


CHART_TO_PIN = {"gpu-operator": "PIN_GPU_OPERATOR", "k8s-nim-operator": "PIN_NIM_OPERATOR",
                "kube-prometheus-stack": "PIN_KPS", "prometheus-adapter": "PIN_PROM_ADAPTER",
                "external-secrets": "PIN_ESO"}


def test_gitops_chart_versions_match_pins():
    p = pins()
    checked = 0
    for f in (ROOT / "gitops" / "clusters").glob("*/*.yaml"):
        app = yaml.safe_load(f.read_text())
        sources = app["spec"].get("sources") or [app["spec"]["source"]]
        for s in sources:
            if s.get("chart") in CHART_TO_PIN:
                assert s["targetRevision"] == p[CHART_TO_PIN[s["chart"]]], f"{f}: {s['chart']} drifted from env/pins.env"
                checked += 1
            if "container-engine-accelerators" in s.get("repoURL", ""):
                assert s["targetRevision"] == p["PIN_GKE_DRIVER_INSTALLER_REF"]
    assert checked >= 20


def test_chart_default_nim_tag_matches_pin():
    v = yaml.safe_load((ROOT / "charts" / "plover" / "values.yaml").read_text())
    p = pins()
    assert v["llm"]["image"]["repository"] == p["PIN_NIM_LLM_IMAGE"]
    assert v["llm"]["image"]["tag"] == p["PIN_NIM_LLM_TAG"]


def test_addon_and_lora_puller_match_pins():
    p = pins()
    lt = yaml.safe_load((ROOT / "addon-nemotron35" / "values" / "aks-lightning.yaml").read_text())["llm"]["image"]
    assert (lt["repository"], lt["tag"]) == (p["PIN_NIM_LIGHTNING_IMAGE"], p["PIN_NIM_LIGHTNING_TAG"])
    v = yaml.safe_load((ROOT / "charts" / "plover" / "values.yaml").read_text())
    assert v["llm"]["lora"]["puller"]["tag"] == p["PIN_ORAS"]
    assert v["bench"]["toolbox"]["aiperfVersion"] == p["PIN_AIPERF"]
