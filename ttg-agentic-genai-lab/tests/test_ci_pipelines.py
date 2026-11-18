"""CI structure (.gitlab-ci.yml is primary - JR2019035 names GitLab; .github/workflows/ci.yml mirrors it).

Checks the release order (lint -> test -> eval-gate -> perf-gate -> build -> deploy), that the GPU perf gate and the
deploys never run by accident, that every file a job calls exists, and that every CLI flag a job passes is one the
target program actually accepts (`--help`) - the drift that otherwise only shows up on a GPU runner.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
GL = yaml.safe_load((REPO / ".gitlab-ci.yml").read_text())
GH = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
STAGES = ["lint", "test", "eval-gate", "perf-gate", "build", "deploy"]
RESERVED = {"stages", "variables", "workflow", "default", "include", "image", "services", "cache", "before_script",
            "after_script"}
JOBS = {k: v for k, v in GL.items() if k not in RESERVED and not k.startswith(".") and isinstance(v, dict)}


def resolved(job: dict) -> dict:
    """GitLab `extends` (single level is all this file uses), merged the way GitLab does for top-level keys."""
    parent = GL.get(job.get("extends")) if isinstance(job.get("extends"), str) else None
    return {**(parent or {}), **job}


def lines(job: dict, key: str = "script") -> list[str]:
    out = []
    for item in resolved(job).get(key) or []:
        if isinstance(item, list):
            out += [str(x) for x in item]
        else:
            out.append(str(item))
    return out


def all_gitlab_commands() -> str:
    return "\n".join(line for job in JOBS.values() for key in ("before_script", "script", "after_script")
                     for line in lines(job, key))


def all_github_commands() -> str:
    return "\n".join(str(s.get("run", "")) for job in GH["jobs"].values() for s in job["steps"])


# ------------------------------------------------------------------------------------------------- GitLab: shape
def test_stages_are_the_release_order():
    assert GL["stages"] == STAGES


def test_expected_jobs_and_their_stages():
    want = {"shellcheck": "lint", "ruff": "lint", "helm-lint": "lint", "pytest": "test", "manifests": "test",
            "eval-gate": "eval-gate", "perf-gate": "perf-gate", "build-image": "build",
            "deploy-staging": "deploy", "deploy-prod": "deploy"}
    assert {name: resolved(job)["stage"] for name, job in JOBS.items()} == want


def test_eval_gate_runs_every_push_on_the_kubernetes_code_path():
    job = resolved(JOBS["eval-gate"])
    assert "rules" not in job and "when" not in job                       # every pipeline, no opt-out
    script = "\n".join(lines(JOBS["eval-gate"]))
    assert "uvicorn ttg.mock.fake_llm:app" in script
    assert "python -m ttg.mcp.ops_server --transport streamable-http --port 8101" in script
    assert "python -m ttg.mcp.guest_server --transport streamable-http --port 8102" in script
    assert job["variables"]["TTG_MCP_MODE"] == "http"                    # MCP over HTTP, like the agent pod
    assert re.search(r"python -m ttg\.bench\.agent_bench --label ci-mock .*--engine mock", script)
    assert re.search(r"scripts/gate\.py eval results/bench/ci-mock\.json --min-success 0\.95 --junit", script)
    assert job["artifacts"]["reports"]["junit"] == "results/eval-gate.xml"
    assert job["artifacts"]["when"] == "always"


def test_perf_gate_only_on_a_gpu_runner_manual_or_scheduled():
    job = resolved(JOBS["perf-gate"])
    assert job["tags"] == ["gpu"]
    rules = job["rules"]
    assert {"if": '$CI_PIPELINE_SOURCE == "schedule"'} in rules
    manual = [r for r in rules if r.get("when") == "manual"]
    assert manual and manual[0]["if"] == "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH" and manual[0]["allow_failure"] is True
    assert all(r.get("when") != "always" for r in rules)
    script = "\n".join(lines(JOBS["perf-gate"]))
    assert "uv venv -q --python 3.12 .venv" in script and ". .venv/bin/activate" in script   # no --user pip (PEP 668)
    assert 'engines/launch.sh "$ENGINE" "$VARIANT"' in script
    assert '--gpu-name "$GPU_NAME"' in script and "nvidia-smi --query-gpu=name" in script
    assert re.search(r'gate\.py perf .* --baseline "\$BASELINE" --max-p95-regression 0\.10 --max-cost-regression 0\.10',
                     script)
    assert any("engines/stop.sh" in s for s in lines(JOBS["perf-gate"], "after_script"))   # never leave a GPU running
    assert job["variables"]["BASELINE"].startswith("bench/baselines/")


def test_build_uses_kaniko_and_the_repo_dockerfile():
    job = resolved(JOBS["build-image"])
    assert job["image"] == {"name": "$KANIKO_IMAGE", "entrypoint": [""]}
    script = "\n".join(lines(JOBS["build-image"]))
    assert '--dockerfile "${CI_PROJECT_DIR}/deploy/Dockerfile"' in script
    assert "${IMAGE}:${CI_COMMIT_SHORT_SHA}" in script
    assert {r["if"] for r in job["rules"]} == {"$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH", "$CI_COMMIT_TAG"}
    assert "CI_REGISTRY_PASSWORD" in script and "docker login" not in script   # no daemon, no privileged runner


def test_deploys_are_manual_staged_and_helm_tested():
    base = GL[".deploy"]
    assert all(r.get("when") == "manual" for r in base["rules"])
    script = "\n".join(base["script"])
    assert "helm upgrade --install ttg deploy/helm/ttg" in script and "--wait" in script
    assert '-f "deploy/values/$CLUSTER.yaml"' in script and '--kube-context "$KUBE_CONTEXT"' in script
    assert "helm test ttg" in script
    staging, prod = resolved(JOBS["deploy-staging"]), resolved(JOBS["deploy-prod"])
    assert staging["environment"]["name"] == "staging" and prod["environment"]["name"] == "production"
    assert prod["needs"] == ["deploy-staging"]
    for job in (staging, prod):
        assert (REPO / "deploy" / "values" / f"{job['variables']['CLUSTER']}.yaml").is_file()


def test_workflow_avoids_duplicate_pipelines():
    rules = GL["workflow"]["rules"]
    assert {"if": "$CI_COMMIT_BRANCH && $CI_OPEN_MERGE_REQUESTS", "when": "never"} in rules


# ------------------------------------------------------------------------------- files + flags the jobs depend on
def _referenced_paths(text: str) -> set[str]:
    text = text.replace("${CI_PROJECT_DIR}/", "")
    found = set(re.findall(r"(?<![\w$/.-])((?:scripts|engines|deploy)/[\w./-]+\.(?:py|sh|yaml|yml|txt)|deploy/Dockerfile"
                           r"|deploy/helm/ttg|requirements-dev\.txt)", text))
    return {p for p in found if "$" not in p}


@pytest.mark.parametrize("system", ["gitlab", "github"])
def test_every_referenced_file_exists(system):
    text = all_gitlab_commands() if system == "gitlab" else all_github_commands()
    paths = _referenced_paths(text)
    assert paths, "expected the jobs to call repo scripts"
    missing = sorted(p for p in paths if not (REPO / p).exists())
    assert not missing, f"{system} CI calls files that do not exist: {missing}"


def _help(*cmd: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, *cmd, "--help"], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{' '.join(cmd)} --help failed: {r.stderr[-1500:]}"
    return r.stdout


def _flags_after(text: str, program: str) -> set[str]:
    out = set()
    for line in re.split(r"\n|(?<!\\)\s&&\s", text.replace("\\\n", " ")):
        if program in line:
            out |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", line.split(program, 1)[1]))
    return out


@pytest.mark.parametrize("program, cmd", [
    ("ttg.bench.agent_bench", ["-m", "ttg.bench.agent_bench"]),
    ("ttg.mcp.ops_server", ["-m", "ttg.mcp.ops_server"]),
    ("ttg.mcp.guest_server", ["-m", "ttg.mcp.guest_server"]),
    ("scripts/wait_for.py", ["scripts/wait_for.py"]),
    ("scripts/gate.py eval", ["scripts/gate.py", "eval"]),
    ("scripts/gate.py perf", ["scripts/gate.py", "perf"]),
])
def test_every_flag_ci_passes_is_accepted(program, cmd):
    text = all_gitlab_commands() + "\n" + all_github_commands()
    used = _flags_after(text, program)
    if program == "ttg.bench.agent_bench":                     # perf-gate builds per-engine probe flags in $EXTRA
        for extra in re.findall(r'EXTRA="([^"]*)"', text):
            used |= set(re.findall(r"(--[a-z][a-z0-9-]*)", extra))
    assert used, f"no CI line calls {program}"
    accepted = set(re.findall(r"(--[a-z][a-z0-9-]*)", _help(*cmd)))
    assert not used - accepted, f"CI passes {sorted(used - accepted)} to {program}, which accepts {sorted(accepted)}"


def test_uvicorn_apps_the_ci_starts_import():
    for app in sorted(set(re.findall(r"uvicorn (ttg\.[\w.]+:app)", all_gitlab_commands() + all_github_commands()))):
        mod, attr = app.split(":")
        r = subprocess.run([sys.executable, "-c", f"import {mod} as m; assert hasattr(m, '{attr}')"], cwd=REPO,
                           env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{app}: {r.stderr[-1500:]}"


@pytest.mark.skipif(not (REPO / "engines" / "launch.sh").is_file(), reason="engines/launch.sh not in this checkout")
def test_perf_gate_default_engine_launch_is_valid():
    v = resolved(JOBS["perf-gate"])["variables"]
    r = subprocess.run(["bash", "engines/launch.sh", v["ENGINE"], v["VARIANT"]], cwd=REPO, capture_output=True,
                       text=True, env={**os.environ, "DRY_RUN": "1"}, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ttg-llm" in r.stdout                                          # = the perf-gate's TTG_MODEL
    assert v["TTG_MODEL"] == "ttg-llm"


# ------------------------------------------------------------------------------------------------ GitHub mirror
def test_github_mirror_has_the_same_gates():
    jobs = GH["jobs"]
    assert set(jobs) == {"lint", "test", "eval-gate"}
    assert jobs["eval-gate"]["needs"] == "test"
    assert GH["permissions"] == {"contents": "read"}
    gl_env = resolved(JOBS["eval-gate"])["variables"]
    assert {k: str(v) for k, v in jobs["eval-gate"]["env"].items()} == {k: str(v) for k, v in gl_env.items()}
    norm = lambda s: re.sub(r"\s+--junit\s+\S+", "", s).strip()           # noqa: E731 - JUnit is GitLab-only
    gl = {norm(s) for s in lines(JOBS["eval-gate"]) if "agent_bench" in s or "gate.py" in s}
    gh = {norm(str(s.get("run", ""))) for s in jobs["eval-gate"]["steps"]
          if "agent_bench" in str(s.get("run", "")) or "gate.py" in str(s.get("run", ""))}
    assert gl == gh, f"GitLab and GitHub eval gates drifted:\n{gl}\n{gh}"
    runs = all_github_commands()
    assert "scripts/validate-manifests.sh" in runs and "python -m pytest" in runs
    assert "--select E9,F63,F7,F82" in runs and "--select E9,F63,F7,F82" in all_gitlab_commands()


# ------------------------------------------------------------------------------------------------ the shell scripts
SCRIPTS = sorted((REPO / "scripts").glob("*.sh")) + sorted((REPO / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_scripts_are_executable_with_a_shebang(path):
    assert os.access(path, os.X_OK), f"chmod +x {path.relative_to(REPO)} (CI calls it directly)"
    assert path.read_text().startswith("#!/usr/bin/env "), f"{path.name}: portable shebang"


@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck is not installed (CI lint job runs it)")
def test_shell_scripts_pass_shellcheck():
    files = [str(p) for p in sorted((REPO / "scripts").glob("*.sh"))]
    r = subprocess.run(["shellcheck", "-x", *files], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:]


def test_bash_scripts_parse():
    for p in sorted((REPO / "scripts").glob("*.sh")):
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p.name}: {r.stderr}"


def test_manual_baseline_recipe_matches_the_perf_gate():
    """A baseline recorded by hand must be comparable with what the job measures (same levels, n, GPU name, probes)."""
    job = "\n".join(lines(JOBS["perf-gate"])).replace("\\\n", " ")
    readme = (REPO / "bench" / "baselines" / "README.md").read_text().replace("\\\n", " ")
    bench_line = lambda text: next(ln for ln in text.splitlines() if "ttg.bench.agent_bench" in ln)  # noqa: E731
    ci, doc = bench_line(job), bench_line(readme)
    vllm_extra = re.search(r'vllm\)\s+EXTRA="([^"]*)"', job).group(1)
    flags = lambda s: set(re.findall(r"(--[a-z][a-z0-9-]*)", re.sub(r"\$\([^)]*\)", "", s)))         # noqa: E731
    assert flags(ci) | flags(vllm_extra) == flags(doc)
    for opt in ("--levels", "--n"):
        val = lambda s: re.search(rf"{opt} (\S+)", s).group(1)                                          # noqa: E731
        assert val(ci) == val(doc), f"{opt} differs between the perf-gate job and bench/baselines/README.md"
