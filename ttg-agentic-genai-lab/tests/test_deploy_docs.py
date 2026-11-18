"""The M9 walkthrough must not lie: every command in deploy/README.md, bench/baselines/README.md and the chart's
NOTES.txt calls files that exist, with flags the programs accept (`--help`), and every relative link resolves."""
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
DOCS = ["deploy/README.md", "bench/baselines/README.md"]
PROGRAMS = {                                  # how the docs call it -> how to ask it for --help
    "python -m ttg.bench.agent_bench": ["-m", "ttg.bench.agent_bench"],
    "scripts/gate.py eval": ["scripts/gate.py", "eval"],
    "scripts/gate.py perf": ["scripts/gate.py", "perf"],
    "scripts/gate.py baseline": ["scripts/gate.py", "baseline"],
    "scripts/load_api.py": ["scripts/load_api.py"],
    "scripts/wait_for.py": ["scripts/wait_for.py"],
}
_HELP: dict[str, set[str]] = {}


def accepted(program: str) -> set[str]:
    if program not in _HELP:
        r = subprocess.run([sys.executable, *PROGRAMS[program], "--help"], cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=120)
        assert r.returncode == 0, r.stderr[-1500:]
        _HELP[program] = set(re.findall(r"(--[a-z][a-z0-9-]*)", r.stdout))
    return _HELP[program]


def code_lines(markdown: str) -> list[str]:
    blocks = re.findall(r"```(?:bash|sh)?\n(.*?)```", markdown, re.S)
    text = "\n".join(blocks).replace("\\\n", " ")
    return [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def rendered_notes(tmp_path: Path, values: str) -> str:
    """helm 3 renders NOTES.txt only on install; render it with `tpl` from a throwaway copy of the chart."""
    chart = tmp_path / "chart"
    shutil.copytree(REPO / "deploy" / "helm" / "ttg", chart)
    shutil.copy(chart / "templates" / "NOTES.txt", chart / "NOTES.src")
    (chart / "templates" / "zz-notes.yaml").write_text('notes: |\n{{- tpl (.Files.Get "NOTES.src") . | nindent 2 }}\n')
    out = subprocess.run(["helm", "template", "ttg", str(chart), "-n", "ttg", "-f", values,
                          "--show-only", "templates/zz-notes.yaml"], cwd=REPO, capture_output=True, text=True, check=True)
    return yaml.safe_load(out.stdout)["notes"]


def check_commands(lines: list[str], where: str) -> int:
    checked = 0
    for line in lines:
        for program in PROGRAMS:
            if program in line:
                rest = re.sub(r"\$\([^)]*\)", "", line.split(program, 1)[1].split("#", 1)[0])   # not $(nvidia-smi ...)
                used = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", rest))
                assert not used - accepted(program), f"{where}: `{line.strip()}` uses {sorted(used - accepted(program))}"
                checked += 1
        for path in re.findall(r"(?<![\w$/.-])((?:scripts|engines|deploy|bench|docs)/[\w./-]*[\w])", line):
            if path.startswith("deploy/ttg-"):                  # kubectl shorthand (deploy/ttg-engine), not a file
                continue
            if "." in path.rsplit("/", 1)[-1] and not path.startswith("bench/baselines/"):   # files only: not written yet
                assert (REPO / path).exists(), f"{where}: {path} does not exist"
    return checked


@pytest.mark.parametrize("doc", DOCS)
def test_doc_commands_use_real_files_and_flags(doc):
    assert check_commands(code_lines((REPO / doc).read_text()), doc) > 0


@pytest.mark.parametrize("doc", DOCS)
def test_doc_relative_links_resolve(doc):
    path = REPO / doc
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text()):
        if not re.match(r"[a-z]+://", target):
            assert (path.parent / target).exists(), f"{doc}: broken link {target}"


def test_readme_names_every_values_file_and_never_hardcodes_the_kind_node():
    text = (REPO / "deploy" / "README.md").read_text()
    for values in sorted((REPO / "deploy" / "values").glob("*.yaml")):
        assert f"deploy/values/{values.name}" in text, f"deploy/README.md does not explain {values.name}"
    assert "kindest/node:v" not in text, "use $PIN_KIND_NODE_IMAGE so the walkthrough follows env/pins.env"


@pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")
@pytest.mark.parametrize("values", ["deploy/values/kind.yaml", "deploy/values/aks.yaml"])
def test_chart_notes_commands_are_real(values, tmp_path):
    notes = rendered_notes(tmp_path, values)
    assert "helm test ttg -n ttg" in notes and "/v1/disruptions" in notes
    lines = [line for line in notes.splitlines() if line.strip()]
    assert check_commands(lines, f"NOTES.txt ({values})") >= 3
