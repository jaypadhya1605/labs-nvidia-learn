"""env/pins.env is the ONE source of truth for versions (🔁 Lab 04): the chart, the image, both CI systems and the
requirements files must agree with it, and no file may float on "latest". Pure file parsing - instant, offline."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PIN_LINE = re.compile(r'^export (PIN_[A-Z0-9_]+)="([^"]+)"(\s+#.*)?$')


def pins() -> dict:
    out = {}
    for line in (REPO / "env" / "pins.env").read_text().splitlines():
        m = PIN_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def requirements(path: str) -> dict:
    reqs = {}
    for raw in (REPO / path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-r "):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)==([0-9][A-Za-z0-9.\-+]*)$", line)
        assert m, f"{path}: '{line}' is not pinned with == (no ranges in a lab others must reproduce)"
        reqs[m.group(1).lower()] = m.group(2)
    return reqs


def pin_for(package: str) -> str:
    return "PIN_" + re.sub(r"[^A-Z0-9]", "_", package.upper())


# --------------------------------------------------------------------------------------------------- pins.env itself
def test_pins_env_is_well_formed_and_sourceable():
    seen = set()
    for n, line in enumerate((REPO / "env" / "pins.env").read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = PIN_LINE.match(line)
        assert m, f"env/pins.env:{n} is not 'export PIN_X=\"value\"  # comment': {line!r}"
        assert m.group(1) not in seen, f"{m.group(1)} pinned twice"
        seen.add(m.group(1))
        assert m.group(2).lower() not in ("latest", "verify", "tbd"), f"{m.group(1)} is not a real version"
    p = pins()
    shell = subprocess.run(["bash", "-c", '. env/pins.env && printf "%s|%s" "$PIN_VLLM_TAG" "$PIN_SERVED_MODEL_NAME"'],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert shell == f"{p['PIN_VLLM_TAG']}|{p['PIN_SERVED_MODEL_NAME']}"


def test_every_pinned_image_is_existence_checked():
    script = (REPO / "scripts" / "check-images.sh").read_text()
    image_pins = [k for k in pins() if k.endswith("_IMAGE") or k == "PIN_CUDA_DEVEL"]
    missing = [k for k in image_pins if "${" + k + "}" not in script]
    assert not missing, f"scripts/check-images.sh never checks {missing}"


# ---------------------------------------------------------------------------------------------------- Helm chart
def test_chart_engine_images_and_model_names_match_pins():
    p = pins()
    v = yaml.safe_load((REPO / "deploy" / "helm" / "ttg" / "values.yaml").read_text())["engine"]
    want = {"vllm": ("PIN_VLLM_IMAGE", "PIN_VLLM_TAG"), "sglang": ("PIN_SGLANG_IMAGE", "PIN_SGLANG_TAG"),
            "trtllm": ("PIN_TRTLLM_IMAGE", "PIN_TRTLLM_TAG"), "nim": ("PIN_NIM_LLM_IMAGE", "PIN_NIM_LLM_TAG")}
    for kind, (img, tag) in want.items():
        assert v["images"][kind]["repository"] == p[img], kind
        assert str(v["images"][kind]["tag"]) == p[tag], kind
    assert v["model"] == p["PIN_AGENT_MODEL"]
    assert v["fp8Model"] == p["PIN_AGENT_MODEL_FP8"]
    assert v["servedModelName"] == p["PIN_SERVED_MODEL_NAME"]
    assert v["lora"]["name"] == p["PIN_LORA_NAME"]
    addon = yaml.safe_load((REPO / "deploy" / "values" / "vlm-addon.yaml").read_text())
    assert addon["engine"]["model"] == p["PIN_VLM_MODEL"]


def test_kind_node_kubeconform_target_and_chart_kube_version_agree():
    node = pins()["PIN_KIND_NODE_IMAGE"].split(":v")[1]                   # kindest/node:v1.34.0 -> 1.34.0
    assert f'KUBE_VERSION="${{KUBE_VERSION:-{node}}}"' in (REPO / "scripts" / "validate-manifests.sh").read_text()
    chart = yaml.safe_load((REPO / "deploy" / "helm" / "ttg" / "Chart.yaml").read_text())
    floor = re.match(r">=\s*(\d+)\.(\d+)", chart["kubeVersion"]).groups()
    assert tuple(map(int, node.split(".")[:2])) >= tuple(map(int, floor))


# ---------------------------------------------------------------------------------------------- image + requirements
def test_dockerfile_base_is_pinned_and_runs_non_root():
    text = (REPO / "deploy" / "Dockerfile").read_text()
    froms = re.findall(r"^FROM\s+(\S+)", text, re.M)
    assert froms == [pins()["PIN_PYTHON_IMAGE"]]
    assert re.search(r"^USER 10001$", text, re.M), "numeric non-root USER (runAsNonRoot needs a numeric uid)"
    assert "requirements-runtime.txt" in text and not re.search(r"^ADD\s+https?://", text, re.M)


def test_runtime_requirements_match_pins():
    reqs = requirements("deploy/requirements-runtime.txt")
    p = pins()
    must = ["langgraph", "langchain-core", "langchain-mcp-adapters", "mcp", "openai", "fastapi", "uvicorn", "httpx",
            "pydantic"]
    for pkg in must:
        assert pkg in reqs, f"deploy/requirements-runtime.txt lacks {pkg} (imported by ttg/)"
        assert reqs[pkg] == p[pin_for(pkg)], f"{pkg}: requirements {reqs[pkg]} != {pin_for(pkg)} {p[pin_for(pkg)]}"
    gpu_side = {"torch", "vllm", "sglang", "tensorrt-llm", "trl", "peft", "transformers"}
    assert not gpu_side & set(reqs), "the agent image is CPU-only: GPU packages live in the engine containers"


def test_dev_requirements_extend_runtime_and_pin_ruff():
    text = (REPO / "requirements-dev.txt").read_text()
    assert "-r deploy/requirements-runtime.txt" in text.splitlines()
    reqs = requirements("requirements-dev.txt")
    assert reqs["ruff"] == pins()["PIN_RUFF"]
    assert "pytest" in reqs and "pyyaml" in reqs
    assert not {"torch", "vllm", "sglang", "tensorrt-llm"} & set(reqs)


# ------------------------------------------------------------------------------------------------------------- CI
def test_gitlab_ci_versions_match_pins():
    v = yaml.safe_load((REPO / ".gitlab-ci.yml").read_text())["variables"]
    p = pins()
    assert v["PYTHON_IMAGE"] == p["PIN_PYTHON_IMAGE"]
    assert v["HELM_IMAGE"] == p["PIN_HELM_IMAGE"]
    assert v["KANIKO_IMAGE"] == p["PIN_KANIKO_IMAGE"]
    assert v["SHELLCHECK_IMAGE"] == p["PIN_SHELLCHECK_IMAGE"]
    assert v["HELM_VERSION"] == p["PIN_HELM"]
    assert v["KUBECONFORM_VERSION"] == p["PIN_KUBECONFORM"]
    assert v["RUFF_VERSION"] == p["PIN_RUFF"]


def test_github_ci_versions_match_pins():
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    p = pins()
    assert wf["env"]["HELM_VERSION"] == p["PIN_HELM"]
    assert wf["env"]["KUBECONFORM_VERSION"] == p["PIN_KUBECONFORM"]
    assert wf["env"]["RUFF_VERSION"] == p["PIN_RUFF"]
    python = re.search(r"python:(\d+\.\d+)", p["PIN_PYTHON_IMAGE"]).group(1)
    versions = {s["with"]["python-version"] for job in wf["jobs"].values() for s in job["steps"]
                if str(s.get("uses", "")).startswith("actions/setup-python")}
    assert versions == {python}


def test_gpu_vm_bootstrap_installs_the_pinned_uv():
    text = (REPO / "scripts" / "gpu-vm-bootstrap.sh").read_text()
    assert 'https://astral.sh/uv/${PIN_UV}/install.sh' in text and pins()["PIN_UV"]
    assert "--python 3.12" in text and pins()["PIN_PYTHON_IMAGE"].startswith("python:3.12")


def test_helm_image_and_helm_binary_are_the_same_release():
    p = pins()
    assert p["PIN_HELM_IMAGE"].split(":")[1] == p["PIN_HELM"].lstrip("v")


@pytest.mark.parametrize("path", ["deploy", ".gitlab-ci.yml", ".github", "scripts", "env"])
def test_nothing_floats_on_latest(path):
    root = REPO / path
    files = [root] if root.is_file() else [f for f in root.rglob("*") if f.is_file() and f.suffix in
                                            (".yaml", ".yml", ".sh", ".env", ".txt", ".tpl", ".json", "") and
                                            "__pycache__" not in f.parts]
    for f in files:
        for n, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            code = line.split("#", 1)[0]
            assert not re.search(r":latest\b", code), f"{f.relative_to(REPO)}:{n} uses a :latest tag"


# ------------------------------------------------------------------------------------------------- keys stay local
def test_lab_env_example_ships_no_keys():
    text = (REPO / "env" / "lab.env.example").read_text()
    for key in ("HF_TOKEN", "NGC_API_KEY", "NVIDIA_API_KEY"):
        m = re.search(rf"^(?:export\s+)?{key}=(.*)$", text, re.M)
        assert m, f"env/lab.env.example should list {key} (empty)"
        assert m.group(1).split("#", 1)[0].strip().strip('"\'') == "", f"{key} must be empty in the example"
    assert not re.search(r"nvapi-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{20,}", text)


def test_gpus_is_a_device_list_and_the_count_is_gpus_per_replica():
    """engines/launch.sh reads GPUS as docker's device list ("0", "0,1"); a count of "1" would pick GPU #1."""
    text = (REPO / "env" / "lab.env.example").read_text()
    assert re.search(r'^export GPUS_PER_REPLICA="1"', text, re.M)
    exported = re.search(r'^export GPUS="([^"]*)"', text, re.M)
    assert exported is None or exported.group(1) == "0"
    ci = yaml.safe_load((REPO / ".gitlab-ci.yml").read_text())["perf-gate"]
    assert re.fullmatch(r"\d+(,\d+)*", ci["variables"]["GPUS"]) and ci["variables"]["GPUS"].startswith("0")
    assert ci["variables"]["GPUS_PER_REPLICA"] == "1"
    assert '--gpus "$GPUS_PER_REPLICA"' in "\n".join(ci["script"])


def test_gitignore_keeps_keys_and_results_out_of_git():
    lines = {line.strip() for line in (REPO / ".gitignore").read_text().splitlines()}
    for entry in ("env/lab.env", "results/*", "!results/.gitkeep", "adapters/", "data/receipts/", "data/sft/",
                  "__pycache__/"):
        assert entry in lines, f".gitignore lacks {entry}"
