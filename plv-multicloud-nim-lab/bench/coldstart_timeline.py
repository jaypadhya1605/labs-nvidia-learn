#!/usr/bin/env python3
"""Reconstruct a pod's cold-start timeline: scale-up -> node ready -> scheduled -> image pulled -> model loaded -> ready.

Usage: python3 bench/coldstart_timeline.py --context plv-aks -n plv-serving --newest -l app.kubernetes.io/name=plover-llm
       python3 bench/coldstart_timeline.py --context plv-gke -n plv-serving <pod-name>
M7 runs it on every cloud: the SAME chart cold-starts very differently (node provisioning, image pull, model cache).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime


CONTEXT = []


def kc(*args):
    out = subprocess.run(["kubectl", *CONTEXT, *args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(out.stderr)
    return json.loads(out.stdout)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pod", nargs="?")
    ap.add_argument("-n", "--namespace", required=True)
    ap.add_argument("-l", "--selector")
    ap.add_argument("--newest", action="store_true")
    ap.add_argument("--context", default=None)
    a = ap.parse_args()
    if a.context:
        CONTEXT.extend(["--context", a.context])
    if a.newest:
        items = kc("get", "pods", "-n", a.namespace, "-l", a.selector or "", "-o", "json")["items"]
        pod = sorted(items, key=lambda p: p["metadata"]["creationTimestamp"])[-1]
    else:
        pod = kc("get", "pod", a.pod, "-n", a.namespace, "-o", "json")
    name = pod["metadata"]["name"]
    t0 = ts(pod["metadata"]["creationTimestamp"])
    marks = [("pod created", t0)]
    for c in pod["status"].get("conditions", []):
        marks.append((f"condition {c['type']}", ts(c.get("lastTransitionTime"))))
    node = pod["spec"].get("nodeName")
    if node:
        n = kc("get", "node", node, "-o", "json")
        marks.append((f"node {node} created", ts(n["metadata"]["creationTimestamp"])))
        for c in n["status"].get("conditions", []):
            if c["type"] == "Ready":
                marks.append(("node Ready", ts(c.get("lastTransitionTime"))))
    ev = kc("get", "events", "-n", a.namespace, "--field-selector", f"involvedObject.name={name}", "-o", "json")["items"]
    for e in ev:
        t = ts(e.get("eventTime") or e.get("firstTimestamp") or e.get("lastTimestamp"))
        marks.append((f"event {e.get('reason')}: {(e.get('message') or '')[:70]}", t))
    print(f"Cold-start timeline for {a.namespace}/{name} (seconds from pod creation)")
    for label, t in sorted((m for m in marks if m[1]), key=lambda m: m[1]):
        print(f"{(t - t0).total_seconds():>8.0f}s  {label}")


if __name__ == "__main__":
    main()
