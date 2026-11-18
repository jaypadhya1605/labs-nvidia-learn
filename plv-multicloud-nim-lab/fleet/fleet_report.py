#!/usr/bin/env python3
"""M11 fleet report: one Markdown page for every cluster the customer runs (AKS, GKE, EKS, OKE, RKE2) -
GPU nodes, driver versions, GitOps health, NIM version/state, LoRA routing, $/hour, and what needs attention.
It is the first page of the weekly technical sync (fleet/weekly_sync.py).
  python3 fleet/fleet_report.py                       # clusters from PLV_CLUSTERS, contexts from KCTX_*
  python3 fleet/fleet_report.py --save-dir results/fleet-raw     # also keep raw JSON (replay with --from-dir)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = {  # name -> kubectl args (JSON output)
    "nodes": ["get", "nodes"],
    "apps": ["-n", "argocd", "get", "applications.argoproj.io"],
    "nims": ["-n", "plv-serving", "get", "nimservices.apps.nvidia.com"],
    "deploys": ["-n", "plv-serving", "get", "deployments"],
    "pods": ["-n", "plv-serving", "get", "pods"],
}


def pins():
    out = {}
    for line in (ROOT / "env" / "pins.env").read_text().splitlines():
        m = re.match(r'export (PIN_[A-Z0-9_]+)="([^"]*)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def kubectl_json(ctx, args):
    r = subprocess.run(["kubectl", "--context", ctx, *args, "-o", "json", "--request-timeout=20s"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"items": [], "error": (r.stderr.strip().splitlines() or ["unreachable"])[-1][:160]}
    return json.loads(r.stdout)


def collect(ctx):
    return {k: kubectl_json(ctx, v) for k, v in QUERIES.items()}


def env_of(deploy, container_index=0):
    c = deploy["spec"]["template"]["spec"]["containers"][container_index]
    return {e["name"]: e.get("value") for e in c.get("env", [])}


def analyze(cluster, raw, hourly, pin):
    s = {"cluster": cluster, "attention": [], "errors": [v["error"] for v in raw.values() if v.get("error")]}
    gpu_nodes = [n for n in raw["nodes"]["items"] if int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", "0") or 0) > 0
                 or n["metadata"]["labels"].get("plv.io/pool") == "gpu"]
    lab = [n["metadata"]["labels"] for n in gpu_nodes]
    s["gpu_nodes"] = len(gpu_nodes)
    s["gpus"] = sum(int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", "0") or 0) for n in gpu_nodes) + \
        sum(int(v) for n in gpu_nodes for k, v in n["status"].get("allocatable", {}).items() if k.startswith("nvidia.com/mig-"))
    s["instance_types"] = sorted({x.get("node.kubernetes.io/instance-type", "?") for x in lab})
    s["gpu_products"] = sorted({x.get("nvidia.com/gpu.product", "?") for x in lab})
    s["drivers"] = sorted({x.get("nvidia.com/cuda.driver-version.full", "?") for x in lab})
    s["mig"] = sorted({x.get("nvidia.com/mig.config", "-") for x in lab})
    s["usd_per_hour"] = round(hourly * len(gpu_nodes), 3)
    apps = raw["apps"]["items"]
    s["apps_total"] = len(apps)
    bad = [(a["metadata"]["name"], a.get("status", {}).get("sync", {}).get("status"), a.get("status", {}).get("health", {}).get("status"))
           for a in apps]
    bad = [b for b in bad if b[1] != "Synced" or b[2] != "Healthy"]
    s["apps_unhealthy"] = [f"{n} ({sy}/{he})" for n, sy, he in bad]
    nims = raw["nims"]["items"]
    s["nim"] = [{"name": n["metadata"]["name"], "image": f"{n['spec']['image']['repository']}:{n['spec']['image']['tag']}",
                 "state": n.get("status", {}).get("state", "?")} for n in nims]
    deps = {d["metadata"]["name"]: d for d in raw["deploys"]["items"]}
    if not nims and "plover-llm" in deps:   # air-gapped RKE2 runs the NIM as a plain Deployment
        d = deps["plover-llm"]
        ready = d.get("status", {}).get("readyReplicas", 0) or 0
        s["nim"] = [{"name": "plover-llm", "image": d["spec"]["template"]["spec"]["containers"][0]["image"],
                     "state": "Ready" if ready else "NotReady"}]
    api = deps.get("plover-api")
    s["routing"] = None
    if api:
        e = env_of(api)
        s["routing"] = {"stable": e.get("STABLE_MODEL") or e.get("BASE_MODEL"), "canary": e.get("CANARY_MODEL") or None,
                        "weight": int(e.get("CANARY_WEIGHT") or 0)}
    restarts = [(p["metadata"]["name"], sum(c.get("restartCount", 0) for c in p.get("status", {}).get("containerStatuses", []) or []))
                for p in raw["pods"]["items"]]
    s["restarting_pods"] = [f"{n} ({r})" for n, r in restarts if r > 3]
    want = f"{pin.get('PIN_NIM_LLM_IMAGE')}:{pin.get('PIN_NIM_LLM_TAG')}"
    for n in s["nim"]:
        if n["state"] not in ("Ready", "ready"):
            s["attention"].append(f"NIM {n['name']} is {n['state']}")
        if n["image"].startswith(pin.get("PIN_NIM_LLM_IMAGE", "~")) and n["image"] != want:
            s["attention"].append(f"NIM image {n['image']} differs from the pin {want}")
    s["attention"] += [f"Argo CD app {x}" for x in s["apps_unhealthy"]]
    s["attention"] += [f"pod restarting: {x}" for x in s["restarting_pods"]]
    if s["routing"] and s["routing"]["canary"]:
        s["attention"].append(f"canary {s['routing']['canary']} at {s['routing']['weight']}% - decide: promote or roll back")
    if s["errors"]:
        s["attention"].append("could not read: " + "; ".join(sorted(set(s["errors"])))[:200])
    return s


def render(summaries, when):
    lines = [f"# Plover fleet report - {when}", "",
             "| Cluster | GPU nodes | GPU | Instance | Driver | NIM | Routing | GitOps | $/h |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        nim = ", ".join(f"{n['image'].rsplit('/', 1)[-1]} ({n['state']})" for n in s["nim"]) or "-"
        r = s["routing"]
        route = "-" if not r else (f"{r['stable']}" + (f" + {r['canary']} {r['weight']}%" if r["canary"] else ""))
        gitops = f"{s['apps_total'] - len(s['apps_unhealthy'])}/{s['apps_total']} healthy" if s["apps_total"] else "n/a"
        lines.append(f"| {s['cluster']} | {s['gpu_nodes']} ({s['gpus']} GPU/MIG) | {', '.join(s['gpu_products'])} | "
                     f"{', '.join(s['instance_types'])} | {', '.join(s['drivers'])} | {nim} | {route} | {gitops} | {s['usd_per_hour']:.2f} |")
    total = sum(s["usd_per_hour"] for s in summaries)
    lines += ["", f"**Fleet GPU spend right now: ${total:.2f}/hour (${total * 730:,.0f}/month if left running).**", ""]
    drivers = {d for s in summaries for d in s["drivers"] if d != "?"}
    if len(drivers) > 1:
        lines.append(f"> Driver versions differ across the fleet ({', '.join(sorted(drivers))}) - expected when clouds own "
                     "the driver; record it in the support matrix and test NIM upgrades on the oldest one first.\n")
    lines.append("## Needs attention")
    items = [f"- **{s['cluster']}**: {a}" for s in summaries for a in s["attention"]]
    lines += items or ["- Nothing. Every cluster is synced, healthy and on the pinned NIM."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters", default=os.environ.get("PLV_CLUSTERS", "aks gke"))
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--from-dir", default=None, help="render from saved raw JSON instead of calling kubectl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pin, summaries = pins(), []
    for c in a.clusters.split():
        if a.from_dir:
            raw = json.loads((Path(a.from_dir) / f"{c}.json").read_text())
        else:
            raw = collect(os.environ.get(f"KCTX_{c.upper()}", f"plv-{c}"))
            if a.save_dir:
                Path(a.save_dir).mkdir(parents=True, exist_ok=True)
                (Path(a.save_dir) / f"{c}.json").write_text(json.dumps(raw))
        summaries.append(analyze(c, raw, float(os.environ.get(f"GPU_HOURLY_{c.upper()}", "0") or 0), pin))
    md = render(summaries, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        Path(a.out).with_suffix(".json").write_text(json.dumps(summaries, indent=2))
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
