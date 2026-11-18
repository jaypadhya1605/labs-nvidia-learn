#!/usr/bin/env python3
"""Every image the air-gapped install will pull - computed from the charts, never typed by hand.
  helm template ... | python3 datacenter/list-images.py            (one or more rendered manifests on stdin)
Reads container/initContainer images from workloads AND the GPU Operator ClusterPolicy components (which are
repository/image/version fields, not 'image:' lines - the classic air-gap miss). Sandbox/vGPU/confidential
components are skipped unless sandboxWorkloads.enabled. Output: one fully-qualified reference per line.
"""
import sys

import yaml

SANDBOX_ONLY = {"vfioManager", "vgpuManager", "vgpuDeviceManager", "ccManager", "sandboxDevicePlugin", "kataSandboxDevicePlugin"}


def normalize(ref):
    """busybox -> docker.io/library/busybox:latest ; foo/bar:1 -> docker.io/foo/bar:1"""
    name, _, digest = ref.partition("@")
    first = name.split("/")[0]
    if "." not in first and ":" not in first and first != "localhost":
        name = ("docker.io/library/" if "/" not in name else "docker.io/") + name
    if not digest and ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name + (f"@{digest}" if digest else "")


def pod_images(spec):
    for key in ("initContainers", "containers"):
        for c in spec.get(key) or []:
            if c.get("image"):
                yield c["image"]


def clusterpolicy_images(spec):
    sandbox = (spec.get("sandboxWorkloads") or {}).get("enabled", False)
    for name, comp in spec.items():
        if not isinstance(comp, dict) or "image" not in comp or not comp.get("repository"):
            continue
        if comp.get("enabled") is False or (name in SANDBOX_ONLY and not sandbox):
            continue
        yield f"{comp['repository']}/{comp['image']}:{comp['version']}"


def images(docs):
    out = set()
    for d in docs:
        if not isinstance(d, dict):
            continue
        kind, spec = d.get("kind"), d.get("spec") or {}
        if kind in ("Deployment", "DaemonSet", "StatefulSet", "ReplicaSet", "Job"):
            out.update(pod_images(spec["template"]["spec"]))
        elif kind == "CronJob":
            out.update(pod_images(spec["jobTemplate"]["spec"]["template"]["spec"]))
        elif kind == "Pod":
            out.update(pod_images(spec))
        elif kind == "ClusterPolicy":
            out.update(clusterpolicy_images(spec))
    return sorted(normalize(i) for i in out)


def main():
    for ref in images(yaml.safe_load_all(sys.stdin)):
        print(ref)


if __name__ == "__main__":
    main()
