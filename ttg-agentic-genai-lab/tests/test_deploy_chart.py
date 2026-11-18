"""Helm chart deploy/helm/ttg (M9): every values file and every engine kind x variant renders, the guard rails refuse
what would crash on a GPU, and the invariants that keep the lab honest hold (Recreate + GPU limit + /dev/shm +
startupProbe on the engine, least-privilege NetworkPolicy, no secret literals, the agent wired to the engine and
both MCP servers with the env names ttg/agent/config.py actually reads, engine flags identical to engines/launch.sh).

Needs the helm binary (CI installs it); kubeconform is used when present. No cluster, no network, no GPU.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CHART = REPO / "deploy" / "helm" / "ttg"
HELM = shutil.which("helm")
pytestmark = pytest.mark.skipif(not HELM, reason="helm is not installed (CI installs it; brew install helm)")

ALL_VARIANTS = ["base", "nocache", "fp8", "fp8kv", "spec", "eager", "lora", "tp2", "pp2", "dp2"]
MATRIX = {                                    # = the support matrix in _helpers.tpl and engines/launch.sh
    "mock": ["base"],
    "vllm": ALL_VARIANTS,
    "sglang": ALL_VARIANTS,
    "trtllm": ["base", "nocache", "fp8", "fp8kv", "tp2", "pp2"],
    "nim": ["base", "nocache"],
}
GPU_KINDS = ["vllm", "sglang", "trtllm", "nim"]
SECRET_ENV = {"NGC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NVIDIA_API_KEY", "TTG_API_KEY", "TTG_VLM_API_KEY"}


def pins() -> dict:
    out = {}
    for line in (REPO / "env" / "pins.env").read_text().splitlines():
        m = re.match(r'^export (PIN_[A-Z0-9_]+)="([^"]*)"', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# ------------------------------------------------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def _helm(args: tuple, ns: str = "ttg", release: str = "ttg") -> subprocess.CompletedProcess:
    return subprocess.run([HELM, "template", release, str(CHART), "-n", ns, *args],
                          cwd=REPO, capture_output=True, text=True, timeout=120)


def render(*args, ns="ttg", release="ttg") -> list[dict]:
    r = _helm(tuple(args), ns, release)
    assert r.returncode == 0, f"helm template {' '.join(args)} failed:\n{r.stderr}"
    return [d for d in yaml.safe_load_all(r.stdout) if d]


def render_fails(*args) -> str:
    r = _helm(tuple(args))
    assert r.returncode != 0, f"rendering with {args} should have failed"
    return r.stderr


def engine_args(kind, variant, *extra):
    a = ["--set", f"engine.kind={kind}", "--set", f"engine.variant={variant}",
         "--set", "engine.lora.pvcName=ttg-adapters", "--set", "secrets.mode=existing"]
    return [*a, *extra]


def get(docs, kind, name=None):
    hits = [d for d in docs if d["kind"] == kind and (name is None or d["metadata"]["name"] == name)]
    assert hits, f"no {kind} {name or ''} rendered (have: {sorted((d['kind'], d['metadata']['name']) for d in docs)})"
    return hits[0]


def names(docs, kind):
    return sorted(d["metadata"]["name"] for d in docs if d["kind"] == kind)


def container(deploy):
    return deploy["spec"]["template"]["spec"]["containers"][0]


def env_of(c) -> dict:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in c.get("env", [])}


def cmdline(c) -> list[str]:
    return list(c.get("command", [])) + list(c.get("args", []))


def flags(c) -> set[str]:
    text = " ".join(cmdline(c))
    return set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9_-]*)", text)) - {"--help"}   # --help = trtllm-serve probing


# ----------------------------------------------------------------------------------------------- values files
@pytest.mark.parametrize("values", ["kind", "aks", "gke"])
def test_values_file_lints_and_renders_the_whole_stack(values):
    lint = subprocess.run([HELM, "lint", str(CHART), "-f", f"deploy/values/{values}.yaml"], cwd=REPO,
                          capture_output=True, text=True)
    assert lint.returncode == 0, lint.stdout + lint.stderr
    docs = render("-f", f"deploy/values/{values}.yaml")
    assert names(docs, "Deployment") == ["ttg-agent", "ttg-engine", "ttg-mcp-guest", "ttg-mcp-ops"]
    assert names(docs, "Service") == ["ttg-agent", "ttg-engine", "ttg-mcp-guest", "ttg-mcp-ops"]
    assert names(docs, "NetworkPolicy") == ["ttg-agent", "ttg-engine-ingress", "ttg-mcp-guest", "ttg-mcp-ops"]
    assert get(docs, "Pod", "ttg-test-disruption")["metadata"]["annotations"]["helm.sh/hook"] == "test"


def test_kind_values_run_on_cpu_for_zero_dollars():
    docs = render("-f", "deploy/values/kind.yaml")
    eng = container(get(docs, "Deployment", "ttg-engine"))
    assert cmdline(eng)[:2] == ["uvicorn", "ttg.mock.fake_llm:app"]
    assert "nvidia.com/gpu" not in json.dumps(eng.get("resources", {}))
    assert not [d for d in docs if d["kind"] in ("ExternalSecret", "PersistentVolumeClaim", "HorizontalPodAutoscaler")]
    agent = get(docs, "Deployment", "ttg-agent")
    assert agent["spec"]["replicas"] == 1
    assert container(agent)["image"] == "ttg-agent:dev" and container(agent)["imagePullPolicy"] == "IfNotPresent"


def test_aks_values_match_lab04_cluster():
    docs = render("-f", "deploy/values/aks.yaml")
    eng = get(docs, "Deployment", "ttg-engine")["spec"]["template"]["spec"]
    assert eng["nodeSelector"] == {"plv.io/pool": "gpu"}
    tol = {(t["key"], t.get("value")) for t in eng["tolerations"]}
    assert ("nvidia.com/gpu", None) in tol and ("kubernetes.azure.com/scalesetpriority", "spot") in tol
    pvc = get(docs, "PersistentVolumeClaim", "ttg-engine-cache")
    assert pvc["spec"]["storageClassName"] == "managed-csi-premium"
    assert pvc["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"     # weights survive helm uninstall
    assert get(docs, "Deployment", "ttg-agent")["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name": "gitlab-registry"}]
    assert names(docs, "ServiceMonitor") == ["ttg-agent", "ttg-engine"]
    assert names(docs, "HorizontalPodAutoscaler") == ["ttg-agent"]
    assert "replicas" not in get(docs, "Deployment", "ttg-agent")["spec"]                # the HPA owns the count


def test_gke_values_use_local_ssd_emptydir_cache():
    docs = render("-f", "deploy/values/gke.yaml")
    vols = {v["name"]: v for v in get(docs, "Deployment", "ttg-engine")["spec"]["template"]["spec"]["volumes"]}
    assert vols["cache"]["emptyDir"]["sizeLimit"] == "80Gi"
    assert not names(docs, "PersistentVolumeClaim")


def test_external_secrets_come_from_the_lab04_store():
    docs = render("-f", "deploy/values/aks.yaml")
    es = {d["metadata"]["name"]: d for d in docs if d["kind"] == "ExternalSecret"}
    assert sorted(es) == ["hf-token-secret", "ngc-api-secret", "ngc-secret"]
    for d in es.values():
        assert d["spec"]["secretStoreRef"] == {"kind": "ClusterSecretStore", "name": "plv-cloud-secrets"}
    assert es["ngc-secret"]["spec"]["target"]["template"]["type"] == "kubernetes.io/dockerconfigjson"


# ------------------------------------------------------------------------------------------ support matrix + rails
@pytest.mark.parametrize("kind", list(MATRIX))
def test_support_matrix(kind):
    for variant in ALL_VARIANTS:
        r = _helm(tuple(engine_args(kind, variant)))
        if variant in MATRIX[kind]:
            assert r.returncode == 0, f"{kind}/{variant} should render:\n{r.stderr}"
        else:
            assert r.returncode != 0 and "is not supported" in r.stderr, f"{kind}/{variant} should be refused"


@pytest.mark.parametrize("bad, message", [
    (["--set", "engine.variant=lora"], "engine.lora.pvcName"),
    (["--set", "engine.kind=nim", "--set", "secrets.mode=none"], "NGC_API_KEY"),
    (["--set", "engine.kind=trtllm", "--set", "engine.variant=fp8", "--set", "engine.fp8Mode=online"], "pre-quantized"),
    (["--set", "engine.kind=trtllm", "--set", "secrets.mode=existing", "--set", "engine.autoscaling.enabled=true"],
     "JSON iteration stats"),
    (["--set", "engine.kind=mock", "--set", "engine.autoscaling.enabled=true"], "the mock has no queue"),
    (["--set", "engine.enabled=false"], "agent.baseUrl"),
])
def test_guard_rails_fail_at_render_time(bad, message):
    assert message in render_fails(*bad)


@pytest.mark.parametrize("bad", [
    ["--set", "image.tag=latest"], ["--set", "engine.images.vllm.tag=latest"], ["--set", "engine.images.nim.tag=VERIFY"],
    ["--set", "engine.gpuResource=amd.com/gpu"], ["--set", "engine.maxModelLen=512"], ["--set", "engine.variant=turbo"],
    ["--set", "engine.fp8Mode=sometimes"], ["--set", "cloud=azure"], ["--set", "agent.baseUrl=ftp://example.com"],
    ["--set", "engine.extraEnv[0].name=NGC_API_KEY", "--set", "engine.extraEnv[0].value=nvapi-123"],
    ["--set", "agent.extraEnv[0].name=NVIDIA_API_KEY", "--set", "agent.extraEnv[0].value=nvapi-123"],
    ["--set", "engine.gpuMemoryUtilization=0.99"], ["--set", "agent.maxSteps=2"],
    ["--set", "engine.extraEnv[0].name=VLLM_SERVER_DEV_MODE", "--set-string", "engine.extraEnv[0].value=1"],
])
def test_values_schema_rejects(bad):
    render_fails(*bad)


# ---------------------------------------------------------------------------------------------- engine invariants
SUPPORTED_GPU = [(k, v) for k in GPU_KINDS for v in MATRIX[k]]


@pytest.mark.parametrize("kind, variant", SUPPORTED_GPU, ids=[f"{k}-{v}" for k, v in SUPPORTED_GPU])
def test_gpu_engine_deployment_invariants(kind, variant):
    p = pins()
    docs = render(*engine_args(kind, variant))
    dep = get(docs, "Deployment", "ttg-engine")
    spec = dep["spec"]["template"]["spec"]
    c = container(dep)
    # a GPU engine is never rolled: two pods would each want the GPU (and the RWO cache volume)
    assert dep["spec"]["strategy"] == {"type": "Recreate"}
    want_gpus = 2 if variant in ("tp2", "pp2", "dp2") else 1
    assert c["resources"]["limits"]["nvidia.com/gpu"] == want_gpus
    vols = {v["name"]: v for v in spec["volumes"]}
    assert vols["dshm"]["emptyDir"]["medium"] == "Memory"                       # NCCL / PyTorch shared memory
    assert {"name": "dshm", "mountPath": "/dev/shm"} in c["volumeMounts"]
    sp = c["startupProbe"]                                                      # first start downloads ~16 GB
    assert sp["failureThreshold"] * sp.get("periodSeconds", 10) >= 1800
    assert c["readinessProbe"]["httpGet"]["path"] == ("/v1/health/ready" if kind == "nim" else "/health")
    assert vols["cache"]["persistentVolumeClaim"]["claimName"] == "ttg-engine-cache"
    image = {"vllm": f"{p['PIN_VLLM_IMAGE']}:{p['PIN_VLLM_TAG']}", "sglang": f"{p['PIN_SGLANG_IMAGE']}:{p['PIN_SGLANG_TAG']}",
             "trtllm": f"{p['PIN_TRTLLM_IMAGE']}:{p['PIN_TRTLLM_TAG']}",
             "nim": f"{p['PIN_NIM_LLM_IMAGE']}:{p['PIN_NIM_LLM_TAG']}"}[kind]
    assert c["image"] == image
    served = p["PIN_SERVED_MODEL_NAME"]
    text = " ".join(cmdline(c))
    if kind == "nim":
        assert env_of(c)["NIM_SERVED_MODEL_NAME"] == served
        assert spec["securityContext"]["runAsUser"] == 1000                     # NIM images run as uid 1000
    elif kind == "trtllm":
        assert f"--served_model_name {served}" in text
    else:
        assert f"--served-model-name {served}" in text
    if kind in ("nim", "trtllm"):
        assert spec["imagePullSecrets"] == [{"name": "ngc-secret"}]
    svc = get(docs, "Service", "ttg-engine")
    assert svc["spec"]["ports"][0]["port"] == 8000


@pytest.mark.parametrize("kind", GPU_KINDS)
def test_nocache_really_turns_prefix_caching_off(kind):
    on, off = render(*engine_args(kind, "base")), render(*engine_args(kind, "nocache"))
    c_on, c_off = container(get(on, "Deployment", "ttg-engine")), container(get(off, "Deployment", "ttg-engine"))
    if kind == "vllm":
        assert "--enable-prefix-caching" in flags(c_on) and "--no-enable-prefix-caching" in flags(c_off)
    elif kind == "sglang":
        assert "--disable-radix-cache" not in flags(c_on) and "--disable-radix-cache" in flags(c_off)
    elif kind == "trtllm":
        opts = lambda docs: yaml.safe_load(get(docs, "ConfigMap", "ttg-engine-trtllm-options")["data"]["extra-llm-api-options.yaml"])  # noqa: E731
        assert opts(on)["kv_cache_config"]["enable_block_reuse"] is True
        assert opts(off)["kv_cache_config"]["enable_block_reuse"] is False
    else:
        assert env_of(c_on)["NIM_ENABLE_KV_CACHE_REUSE"] == "1" and env_of(c_off)["NIM_ENABLE_KV_CACHE_REUSE"] == "0"


def test_trtllm_options_file_and_rollout_checksum():
    docs = {v: render(*engine_args("trtllm", v)) for v in ("base", "fp8kv", "nocache")}
    opts = {v: yaml.safe_load(get(d, "ConfigMap", "ttg-engine-trtllm-options")["data"]["extra-llm-api-options.yaml"])
            for v, d in docs.items()}
    assert opts["fp8kv"]["kv_cache_config"]["dtype"] == "fp8"                  # KvCacheConfig.dtype, not a CLI flag
    assert "dtype" not in opts["base"]["kv_cache_config"]
    assert opts["base"]["enable_iter_perf_stats"] is True                       # GET /metrics needs it
    assert opts["base"]["kv_cache_config"]["free_gpu_memory_fraction"] == 0.85
    sums = {v: get(d, "Deployment", "ttg-engine")["spec"]["template"]["metadata"]["annotations"]["checksum/trtllm-options"]
            for v, d in docs.items()}
    assert len(set(sums.values())) == 3                                         # a new options file restarts the pod
    script = container(get(docs["base"], "Deployment", "ttg-engine"))["args"][0]
    assert "--kv_cache_dtype" not in script and "--extra_llm_api_options /config/extra-llm-api-options.yaml" in script


def test_fp8_checkpoint_vs_online():
    p = pins()
    for kind in ("vllm", "sglang", "trtllm"):
        c = container(get(render(*engine_args(kind, "fp8")), "Deployment", "ttg-engine"))
        assert p["PIN_AGENT_MODEL_FP8"] in " ".join(cmdline(c)) and "--quantization" not in flags(c)
    for kind in ("vllm", "sglang"):
        c = container(get(render(*engine_args(kind, "fp8", "--set", "engine.fp8Mode=online")), "Deployment", "ttg-engine"))
        text = " ".join(cmdline(c))
        assert p["PIN_AGENT_MODEL"] in text and "--quantization fp8" in text
    aks = container(get(render("-f", "deploy/values/aks.yaml", "--set", "engine.variant=fp8"), "Deployment", "ttg-engine"))
    assert "--quantization" in flags(aks)                                       # A100: no ModelOpt FP8 checkpoint


@pytest.mark.parametrize("kind, model", [("vllm", "ttg-short"), ("sglang", "ttg-llm:ttg-short")])
def test_lora_adapter_volume_and_agent_model(kind, model):
    docs = render(*engine_args(kind, "lora"))
    dep = get(docs, "Deployment", "ttg-engine")
    vols = {v["name"]: v for v in dep["spec"]["template"]["spec"]["volumes"]}
    assert vols["adapters"]["persistentVolumeClaim"]["claimName"] == "ttg-adapters"
    assert "ttg-short=/adapters/ttg-short" in cmdline(container(dep))
    agent = env_of(container(get(docs, "Deployment", "ttg-agent")))
    assert agent["TTG_MODEL"] == model and agent["TTG_PROMPT_VARIANT"] == "short"   # M6: the adapter knows the policy


def test_mock_engine_is_least_privilege():
    c = container(get(render("-f", "deploy/values/kind.yaml"), "Deployment", "ttg-engine"))
    assert c["securityContext"]["runAsNonRoot"] is True and c["securityContext"]["readOnlyRootFilesystem"] is True
    assert {"name": "tmp", "mountPath": "/tmp"} in c["volumeMounts"]


# ------------------------------------------------------------------- agent + MCP wiring (contract with ttg/ code)
def _settings_env_names() -> set[str]:
    return set(re.findall(r'_env\("(TTG_[A-Z0-9_]+)"', (REPO / "ttg" / "agent" / "config.py").read_text()))


def test_agent_env_is_exactly_what_ttg_settings_reads(monkeypatch):
    docs = render("-f", "deploy/values/aks.yaml")
    env = env_of(container(get(docs, "Deployment", "ttg-agent")))
    unknown = {k for k in env if k.startswith("TTG_")} - _settings_env_names()
    assert not unknown, f"chart sets TTG_* variables ttg/agent/config.py never reads: {sorted(unknown)}"
    assert env["TTG_BASE_URL"] == "http://ttg-engine:8000/v1"
    assert env["TTG_MCP_MODE"] == "http"
    assert env["TTG_MCP_OPS_URL"] == "http://ttg-mcp-ops:8101/mcp"
    assert env["TTG_MCP_GUEST_URL"] == "http://ttg-mcp-guest:8102/mcp"
    # the rendered values must pass the agent's own validation (what would otherwise crash at pod start)
    sys.path.insert(0, str(REPO))
    from ttg.agent.config import Settings
    for k, v in env.items():
        if k.startswith("TTG_") and isinstance(v, str):
            monkeypatch.setenv(k, v)
    s = Settings().validate()
    assert s.mcp_mode == "http" and s.model == "ttg-llm"


def test_agent_and_mcp_commands_match_the_image_entry_points():
    docs = render("-f", "deploy/values/kind.yaml")
    agent = container(get(docs, "Deployment", "ttg-agent"))
    assert cmdline(agent) == ["uvicorn", "ttg.agent.api:app", "--host", "0.0.0.0", "--port", "8080"]
    assert agent["readinessProbe"]["httpGet"]["path"] == "/healthz"
    for role, port in (("ops", 8101), ("guest", 8102)):
        c = container(get(docs, "Deployment", f"ttg-mcp-{role}"))
        assert cmdline(c) == ["python", "-m", f"ttg.mcp.{role}_server", "--transport", "streamable-http",
                              "--host", "0.0.0.0", "--port", str(port)]
        assert (REPO / "ttg" / "mcp" / f"{role}_server.py").is_file()
    assert (REPO / "ttg" / "agent" / "api.py").is_file() and (REPO / "ttg" / "mock" / "fake_llm.py").is_file()


@pytest.mark.parametrize("ns", ["ttg", "ttg-staging"])
def test_mcp_servers_accept_the_in_cluster_host_names(ns):
    docs = render("-f", "deploy/values/kind.yaml", ns=ns)
    for name in ("ttg-mcp-ops", "ttg-mcp-guest"):
        hosts = env_of(container(get(docs, "Deployment", name)))["TTG_MCP_ALLOWED_HOSTS"].split(",")
        assert hosts == [f"{name}:*", f"{name}.{ns}:*", f"{name}.{ns}.svc:*", f"{name}.{ns}.svc.cluster.local:*"]


def test_guest_vlm_path_env_and_key_from_secret():
    docs = render("-f", "deploy/values/aks.yaml", "--set", "mcp.guest.vlm.baseUrl=https://integrate.api.nvidia.com/v1",
                  "--set", "mcp.guest.vlm.model=meta/llama-3.2-11b-vision-instruct",
                  "--set", "mcp.guest.vlm.apiKeySecret.name=nvidia-api-key")
    env = env_of(container(get(docs, "Deployment", "ttg-mcp-guest")))
    assert env["TTG_VLM_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert env["TTG_VLM_API_KEY"] == {"secretKeyRef": {"name": "nvidia-api-key", "key": "NVIDIA_API_KEY"}}
    assert "TTG_VLM_BASE_URL" not in env_of(container(get(docs, "Deployment", "ttg-mcp-ops")))


def test_hosted_endpoint_path_has_no_engine():
    docs = render("-f", "deploy/values/kind.yaml", "--set", "engine.enabled=false",
                  "--set", "agent.baseUrl=https://integrate.api.nvidia.com/v1", "--set", "agent.apiKeySecret.name=nvidia-api-key",
                  "--set", "agent.model=meta/llama-3.1-8b-instruct")
    assert "ttg-engine" not in names(docs, "Deployment")
    env = env_of(container(get(docs, "Deployment", "ttg-agent")))
    assert env["TTG_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert env["TTG_API_KEY"] == {"secretKeyRef": {"name": "nvidia-api-key", "key": "NVIDIA_API_KEY"}}
    egress = get(docs, "NetworkPolicy", "ttg-agent")["spec"]["egress"]
    assert {"ports": [{"port": 443, "protocol": "TCP"}]} in egress


def test_securitycontext_on_every_ttg_container():
    docs = render("-f", "deploy/values/kind.yaml")
    for name in ("ttg-agent", "ttg-mcp-ops", "ttg-mcp-guest", "ttg-engine"):
        c = container(get(docs, "Deployment", name))
        sc = c["securityContext"]
        assert sc["runAsNonRoot"] and sc["readOnlyRootFilesystem"] and not sc["allowPrivilegeEscalation"]
        assert sc["capabilities"]["drop"] == ["ALL"]
        assert any(m["mountPath"] == "/tmp" for m in c["volumeMounts"]), f"{name}: read-only root needs a /tmp volume"


def test_no_secret_is_ever_a_literal():
    renders = [render("-f", f"deploy/values/{v}.yaml") for v in ("kind", "aks", "gke")]
    renders += [render(*engine_args(k, "base")) for k in GPU_KINDS]
    for docs in renders:
        for d in docs:
            for c in d.get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
                for e in c.get("env", []):
                    if e["name"] in SECRET_ENV and "value" in e:
                        assert e["name"] == "TTG_API_KEY" and e["value"] == "not-used", f"{d['metadata']['name']}: {e}"


def test_chart_never_enables_vllm_dev_mode():
    """engines/launch.sh sets VLLM_SERVER_DEV_MODE=1 on the lab VM (the bench's --reset-url). It also exposes /sleep,
    /wake_up and /collective_rpc - never in a cluster. No render of this chart may contain it."""
    sets = [("-f", f"deploy/values/{v}.yaml") for v in ("kind", "aks", "gke")]
    sets += [tuple(engine_args(k, v)) for k in MATRIX for v in MATRIX[k]]
    sets += [("-f", "deploy/values/gke.yaml", "-f", "deploy/values/vlm-addon.yaml")]
    for args in sets:
        r = _helm(tuple(args))
        assert r.returncode == 0, r.stderr
        assert "VLLM_SERVER_DEV_MODE" not in r.stdout, f"helm template {' '.join(args)} enables vLLM dev mode"
        assert "reset_prefix_cache" not in r.stdout


def test_helm_test_hook_posts_a_golden_disruption():
    pod = get(render("-f", "deploy/values/kind.yaml", "--set", "helmTest.scenarioId=S0042"), "Pod", "ttg-test-disruption")
    text = json.dumps(pod)
    assert "http://ttg-agent:8080/v1/disruptions" in text and "S0042" in text
    assert pod["spec"]["restartPolicy"] == "Never"


# ---------------------------------------------------------------------------------------- NetworkPolicy (🔁 Lab 03)
def test_networkpolicy_least_privilege():
    docs = render("-f", "deploy/values/aks.yaml")
    eng = get(docs, "NetworkPolicy", "ttg-engine-ingress")["spec"]
    assert eng["policyTypes"] == ["Ingress"]
    peers = eng["ingress"][0]["from"]
    assert {"podSelector": {"matchLabels": {"app": "ttg-agent"}}} in peers
    assert {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "monitoring"}}} in peers
    assert eng["ingress"][0]["ports"] == [{"port": 8000, "protocol": "TCP"}]
    for name, port in (("ttg-mcp-ops", 8101), ("ttg-mcp-guest", 8102)):
        mcp = get(docs, "NetworkPolicy", name)["spec"]
        assert mcp["ingress"] == [{"from": [{"podSelector": {"matchLabels": {"app": "ttg-agent"}}}],
                                   "ports": [{"port": port, "protocol": "TCP"}]}]
        assert mcp["egress"] == []                                              # deterministic tools: no egress
    agent = get(docs, "NetworkPolicy", "ttg-agent")["spec"]
    assert agent["policyTypes"] == ["Ingress", "Egress"]
    allowed = []
    for rule in agent["egress"]:
        for port in rule.get("ports", []):
            allowed.append((json.dumps(rule.get("to", []), sort_keys=True), port["port"]))
    assert {p for _, p in allowed} == {8000, 8101, 8102, 53}                    # no 443: a fooled model cannot exfiltrate
    assert all(to != "[]" for to, p in allowed if p != 53)                      # every non-DNS rule names its pods


def test_networkpolicy_opens_guest_egress_only_for_the_vlm():
    docs = render("-f", "deploy/values/aks.yaml", "--set", "mcp.guest.vlm.baseUrl=http://ttg-vlm:8000/v1")
    assert get(docs, "NetworkPolicy", "ttg-mcp-guest")["spec"]["egress"] == [{}]
    assert get(docs, "NetworkPolicy", "ttg-mcp-ops")["spec"]["egress"] == []


def test_networkpolicy_can_be_disabled_and_extra_namespaces_allowed():
    assert not names(render("--set", "networkPolicy.enabled=false"), "NetworkPolicy")
    docs = render("--set", "networkPolicy.allowNamespaces[0]=loadtest")
    peer = {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "loadtest"}}}
    assert peer in get(docs, "NetworkPolicy", "ttg-agent")["spec"]["ingress"][0]["from"]
    assert peer in get(docs, "NetworkPolicy", "ttg-engine-ingress")["spec"]["ingress"][0]["from"]


def test_vlm_addon_release_is_engine_only():
    docs = render("-f", "deploy/values/gke.yaml", "-f", "deploy/values/vlm-addon.yaml", release="ttg-vlm")
    assert sorted((d["kind"], d["metadata"]["name"]) for d in docs) == [
        ("Deployment", "ttg-vlm"), ("NetworkPolicy", "ttg-vlm-ingress"), ("Service", "ttg-vlm"), ("ServiceMonitor", "ttg-vlm")]
    c = container(get(docs, "Deployment", "ttg-vlm"))
    assert pins()["PIN_VLM_MODEL"] in cmdline(c) and "--limit-mm-per-prompt" in flags(c)
    peers = get(docs, "NetworkPolicy", "ttg-vlm-ingress")["spec"]["ingress"][0]["from"]
    assert {"podSelector": {"matchLabels": {"app": "ttg-mcp-guest"}}} in peers


# ------------------------------------------------------------------------------------- autoscaling + monitoring
@pytest.mark.parametrize("kind, metric", [("vllm", "vllm_num_requests_waiting"), ("sglang", "sglang_num_queue_reqs"),
                                          ("nim", "vllm_num_requests_waiting")])
def test_engine_hpa_scales_on_queue_depth(kind, metric):
    docs = render(*engine_args(kind, "base", "--set", "engine.autoscaling.enabled=true"))
    hpa = get(docs, "HorizontalPodAutoscaler", "ttg-engine")["spec"]
    m = hpa["metrics"][0]
    assert m["type"] == "Pods" and m["pods"]["metric"]["name"] == metric
    assert m["pods"]["target"] == {"type": "AverageValue", "averageValue": "4"}
    assert hpa["behavior"]["scaleDown"]["stabilizationWindowSeconds"] >= 600   # GPUs are slow to come back
    assert (hpa["minReplicas"], hpa["maxReplicas"]) == (1, 4)


def test_engine_keda_option():
    docs = render(*engine_args("sglang", "base", "--set", "engine.autoscaling.enabled=true",
                               "--set", "engine.autoscaling.mode=keda"))
    so = get(docs, "ScaledObject", "ttg-engine")["spec"]
    trig = so["triggers"][0]
    assert trig["type"] == "prometheus" and trig["metadata"]["query"] == 'sum(sglang:num_queue_reqs{namespace="ttg"})'
    assert trig["metadata"]["serverAddress"].startswith("http://")
    assert not [d for d in docs if d["kind"] == "HorizontalPodAutoscaler" and d["metadata"]["name"] == "ttg-engine"]


@pytest.mark.parametrize("kind, expected", [("vllm", ["ttg-agent", "ttg-engine"]), ("trtllm", ["ttg-agent"]),
                                            ("nim", ["ttg-agent", "ttg-engine"])])
def test_servicemonitors(kind, expected):
    docs = render(*engine_args(kind, "base", "--set", "serviceMonitor.enabled=true"))
    assert names(docs, "ServiceMonitor") == expected
    if "ttg-engine" in expected:
        path = get(docs, "ServiceMonitor", "ttg-engine")["spec"]["endpoints"][0]["path"]
        assert path == ("/v1/metrics" if kind == "nim" else "/metrics")
    assert get(docs, "ServiceMonitor", "ttg-agent")["spec"]["endpoints"][0]["path"] == "/metrics"


# --------------------------------------------------------- same engine flags in Kubernetes as on the GPU VM (M4 -> M9)
LAUNCH = REPO / "engines" / "launch.sh"
# known, documented differences (chart side first): none. (TensorRT-LLM's FP8 KV cache is kv_cache_config.dtype in the
# options YAML on both sides - engines/trtllm/fp8kv.yaml and the chart's ConfigMap - not a CLI flag.)
ALLOWED_LAUNCH_ONLY: dict = {}


def _launch_flags(kind, variant):
    env = {**os.environ, "DRY_RUN": "1", "GPUS": "0,1"}
    r = subprocess.run(["bash", str(LAUNCH), kind, variant], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    m = re.search(r"^ENGINE_FLAGS:(.*)$", r.stdout, re.M)
    return None if (r.returncode or not m) else set(m.group(1).split())


@pytest.mark.skipif(not LAUNCH.is_file(), reason="engines/launch.sh not in this checkout")
@pytest.mark.parametrize("kind, variant", SUPPORTED_GPU, ids=[f"{k}-{v}" for k, v in SUPPORTED_GPU])
def test_engine_flags_match_engines_launch_sh(kind, variant):
    want = _launch_flags(kind, variant)
    if want is None:
        pytest.skip("engines/launch.sh DRY_RUN=1 printed no ENGINE_FLAGS line for this pair")
    have = flags(container(get(render(*engine_args(kind, variant)), "Deployment", "ttg-engine")))
    assert not (have - want), f"chart-only flags {sorted(have - want)}: add them to engines/launch.sh or drop them"
    missing = want - have - ALLOWED_LAUNCH_ONLY.get((kind, variant), set())
    assert not missing, f"engines/launch.sh passes {sorted(missing)} that the chart does not"


# ---------------------------------------------------------------------------------------------- kubeconform -strict
@pytest.mark.skipif(not shutil.which("kubeconform"), reason="kubeconform is not installed (CI installs it)")
def test_validate_manifests_script():
    r = subprocess.run(["bash", "scripts/validate-manifests.sh"], cwd=REPO, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    if r.returncode and re.search(r"could not find schema|no such host|dial tcp|i/o timeout|TLS handshake", out):
        pytest.skip("Kubernetes JSON schemas unreachable (offline)")
    assert r.returncode == 0, out[-4000:]
    assert "Invalid: 0, Errors: 0" in out
