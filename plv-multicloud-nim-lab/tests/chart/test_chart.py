"""Policy tests for the Helm chart: render it with EVERY cluster's values and assert the portability contract.
Runs in CI with only `helm` installed - no cluster needed."""
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "charts" / "plover"
CLUSTERS = ["aks", "gke", "eks", "oke", "rke2"]


def render(cluster, *extra):
    cmd = ["helm", "template", "plover", str(CHART), "-n", "plv-serving",
           "-f", str(ROOT / "clusters" / cluster / "values.yaml"),
           "--set", "api.image.repository=ghcr.io/test/plover-api", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def by_kind(docs, kind):
    return [d for d in docs if d["kind"] == kind]


@pytest.mark.parametrize("cluster", CLUSTERS)
def test_renders_and_requests_one_gpu(cluster):
    docs = render(cluster)
    gpu_specs = [d["spec"] for d in by_kind(docs, "NIMService")] + \
                [d["spec"]["template"]["spec"]["containers"][0] for d in by_kind(docs, "Deployment") if d["metadata"]["name"] == "plover-llm"]
    assert len(gpu_specs) == 1, "exactly one LLM workload per cluster"
    limits = gpu_specs[0]["resources"]["limits"]
    assert any(k.startswith("nvidia.com/") for k in limits), "LLM must request an NVIDIA GPU resource"


@pytest.mark.parametrize("cluster", ["aks", "gke", "eks", "oke"])
def test_gpu_toleration_everywhere(cluster):
    ns = by_kind(render(cluster), "NIMService")[0]["spec"]
    assert any(t["key"] == "nvidia.com/gpu" for t in ns["tolerations"])


def test_aks_spot_toleration_and_rwx_cache():
    docs = render("aks")
    ns = by_kind(docs, "NIMService")[0]["spec"]
    assert any(t["key"] == "kubernetes.azure.com/scalesetpriority" for t in ns["tolerations"])
    cache = by_kind(docs, "NIMCache")[0]["spec"]["storage"]["pvc"]
    assert cache["volumeAccessMode"] == "ReadWriteMany" and cache["storageClass"] == "azurefile-csi-premium"


@pytest.mark.parametrize("cluster", ["gke", "eks", "oke"])
def test_no_azure_storage_leaks_into_other_clouds(cluster):
    docs = render(cluster)
    assert not by_kind(docs, "NIMCache")
    assert "emptyDir" in by_kind(docs, "NIMService")[0]["spec"]["storage"]


def test_rke2_is_airgap_safe():
    docs = render("rke2")
    assert not by_kind(docs, "ExternalSecret") and not by_kind(docs, "NIMService")
    llm = [d for d in by_kind(docs, "Deployment") if d["metadata"]["name"] == "plover-llm"][0]
    env = {e["name"]: e.get("value") for e in llm["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert "NGC_API_KEY" not in env, "air-gapped NIM must not carry an NGC key"
    assert env["NIM_MODEL_PROFILE"]


def test_lora_adapters_become_init_containers():
    docs = render("gke", "--set", "llm.lora.adapters[0].name=plover-style-v1",
                  "--set", "llm.lora.adapters[0].ref=ghcr.io/test/plover-adapters:style-v1")
    spec = by_kind(docs, "NIMService")[0]["spec"]
    ic = spec["initContainers"][0]
    assert ic["args"] == ["pull", "ghcr.io/test/plover-adapters:style-v1", "-o", "/scratch/loras/plover-style-v1"]
    env = {e["name"]: e.get("value") for e in spec["env"]}
    assert env["NIM_PEFT_SOURCE"] == "/scratch/loras"


def test_autoscaling_omits_replicas():
    spec = by_kind(render("aks", "--set", "llm.autoscaling.enabled=true"), "NIMService")[0]["spec"]
    assert "replicas" not in spec and spec["scale"]["hpa"]["maxReplicas"] == 2


def test_schema_rejects_latest_tag_and_bad_weight():
    for bad in (["--set", "llm.image.tag=latest"], ["--set", "api.routing.canaryWeight=150"]):
        r = subprocess.run(["helm", "template", "plover", str(CHART), "-f", str(ROOT / "clusters/aks/values.yaml"), *bad],
                           capture_output=True, text=True)
        assert r.returncode != 0


def test_no_secret_values_in_any_values_file():
    for f in (ROOT / "clusters").glob("*/values.yaml"):
        text = f.read_text().lower()
        assert "nvapi-" not in text and "hf_" not in text, f"secret-looking value in {f}"


def test_extra_env_and_extra_body_render():
    docs = render("aks", "--set", "llm.extraEnv[0].name=NIM_MODEL_NAME", "--set", "llm.extraEnv[0].value=nvidia/x",
                  "--set", "api.llmExtraBody.chat_template_kwargs.enable_thinking=false")
    env = {e["name"]: e.get("value") for e in by_kind(docs, "NIMService")[0]["spec"]["env"]}
    assert env["NIM_MODEL_NAME"] == "nvidia/x"
    api = [d for d in by_kind(docs, "Deployment") if d["metadata"]["name"] == "plover-api"][0]
    aenv = {e["name"]: e.get("value") for e in api["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert '"enable_thinking":false' in aenv["LLM_EXTRA_BODY"]


def test_schema_blocks_ngc_key_in_extra_env():
    with pytest.raises(subprocess.CalledProcessError):
        render("gke", "--set", "llm.extraEnv[0].name=NGC_API_KEY", "--set", "llm.extraEnv[0].value=nvapi-x")


def test_day0_verify_placeholders_block_rendering():
    with pytest.raises(subprocess.CalledProcessError):
        render("gke", "--set", "llm.image.tag=VERIFY-in-NGC")


def test_empty_profile_lets_nim_autoselect():
    spec = by_kind(render("gke", "--set", "llm.profile="), "NIMService")[0]["spec"]
    assert "NIM_MODEL_PROFILE" not in {e["name"] for e in spec["env"]}
