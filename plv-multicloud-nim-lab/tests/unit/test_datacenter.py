"""Air-gap image list: computed from rendered charts, including the GPU Operator's ClusterPolicy components."""
import importlib.util
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("list_images", ROOT / "datacenter" / "list-images.py")
li = importlib.util.module_from_spec(spec)
spec.loader.exec_module(li)


def test_normalize_short_names():
    assert li.normalize("busybox") == "docker.io/library/busybox:latest"
    assert li.normalize("curlimages/curl:8.17.0") == "docker.io/curlimages/curl:8.17.0"
    assert li.normalize("nvcr.io/nvidia/gpu-operator:v26.7.2") == "nvcr.io/nvidia/gpu-operator:v26.7.2"
    assert li.normalize("localhost/x:1") == "localhost/x:1"


def test_clusterpolicy_components_are_included_and_sandbox_skipped():
    cp = {"kind": "ClusterPolicy", "spec": {
        "toolkit": {"enabled": True, "repository": "nvcr.io/nvidia/k8s", "image": "container-toolkit", "version": "v1"},
        "driver": {"enabled": False, "repository": "nvcr.io/nvidia", "image": "driver", "version": "595"},
        "validator": {"repository": "nvcr.io/nvidia", "image": "gpu-operator", "version": "v26"},
        "vfioManager": {"enabled": True, "repository": "nvcr.io/nvidia/cloud-native", "image": "k8s-driver-manager", "version": "v0"}}}
    out = li.images([cp])
    assert "nvcr.io/nvidia/k8s/container-toolkit:v1" in out and "nvcr.io/nvidia/gpu-operator:v26" in out
    assert not any("driver" in i for i in out), "disabled driver and sandbox-only components are not mirrored"


def test_rke2_chart_images():
    out = subprocess.run(["helm", "template", "plover", str(ROOT / "charts" / "plover"), "-f", str(ROOT / "clusters" / "rke2" / "values.yaml"),
                          "--set", "api.image.repository=ghcr.io/test/plover-api"], capture_output=True, text=True, check=True).stdout
    imgs = li.images(yaml.safe_load_all(out))
    assert "nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.13" in imgs
    assert not any("oras" in i for i in imgs), "LoRA puller is off in the air gap"
