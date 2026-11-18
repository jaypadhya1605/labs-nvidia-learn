"""FORK C area (docs 00-02, 09-10, workshop/, collateral/, sales/, drills/): every link, path and module resolves.

Offline, no GPU. Files that other builders write in parallel are allowed to be missing (PENDING), and generated
outputs (results/, adapters/, data/receipts/) are never required.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MINE = sorted({*REPO.glob("docs/0[0-2]-*.md"), *REPO.glob("docs/09-*.md"), *REPO.glob("docs/10-*.md"),
               *REPO.glob("workshop/*.md"), *REPO.glob("collateral/*.md"), *REPO.glob("sales/*.md"),
               REPO / "drills" / "README.md"})
EXPECTED = [
    "docs/00-recall.md", "docs/01-customer-brief.md", "docs/02-architecture-and-adrs.md",
    "docs/09-results-and-exec-readout.md", "docs/10-interview-drill.md",
    "workshop/README.md", "workshop/agenda.md", "workshop/slides-outline.md", "workshop/facilitator-guide.md",
    "workshop/exercises.md", "workshop/build_notebook.py", "workshop/ttg-workshop.ipynb",
    "collateral/README.md", "collateral/blog.md",
    "sales/README.md", "sales/discovery-questions.md", "sales/poc-charter.md", "sales/poc-evaluation-report.md",
    "sales/feature-request.md", "sales/technical-win-plan.md", "sales/mutual-action-plan.md",
    "drills/README.md", "drills/inject.sh", "drills/fixtures.py", "drills/diff_runs.py",
    "drills/files/broken-chat-template.jinja",
]
# Written by other builders in parallel (README by MAIN last, docs 03-08 by forks A and B, deploy notes by fork D).
PENDING = {"README.md", "docs/03-model-complexity.md", "docs/04-gpu-cluster-architecture.md",
           "docs/05-parallelism-math.md", "docs/06-multimodal.md", "docs/07-engine-cheatsheet.md",
           "docs/08-profiling-playbook.md", "deploy/README.md"}
PENDING_DIRS = ("vlm/", "finetune/", "profiling/")
GENERATED = ("results/", "adapters/", "data/receipts/")

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
TOP = (r"(?:ttg|drills|workshop|sales|collateral|docs|engines|scripts|deploy|parallel|archmath|env|data|vlm|finetune"
       r"|profiling|tests)")
CODE_PATH = re.compile(r"`(\.gitlab-ci\.yml|" + TOP + r"/[\w./-]*[\w/])`")
MODULE = re.compile(r"python3? -m ((?:ttg|vlm|parallel|archmath|finetune|profiling)(?:\.\w+)+)")


def _rel(p: Path) -> str:
    return p.resolve().relative_to(REPO).as_posix()


def _allowed_missing(rel: str) -> bool:
    return rel in PENDING or rel.startswith(PENDING_DIRS) or rel.startswith(GENERATED)


@pytest.mark.parametrize("rel", EXPECTED)
def test_deliverable_exists(rel):
    assert (REPO / rel).exists(), f"missing deliverable {rel}"


@pytest.mark.parametrize("md", MINE, ids=lambda p: _rel(p))
def test_relative_links_resolve(md):
    bad = []
    for target in LINK.findall(md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (md.parent / target.split("#", 1)[0]).resolve()
        rel = path.relative_to(REPO).as_posix() if str(path).startswith(str(REPO)) else str(path)
        if not path.exists() and not _allowed_missing(rel):
            bad.append(target)
    assert not bad, f"{_rel(md)}: broken links {bad}"


@pytest.mark.parametrize("md", MINE, ids=lambda p: _rel(p))
def test_code_span_paths_resolve(md):
    bad = [p for p in CODE_PATH.findall(md.read_text(encoding="utf-8"))
           if not (REPO / p.rstrip("/")).exists() and not _allowed_missing(p)]
    assert not bad, f"{_rel(md)}: paths that do not exist {bad}"


@pytest.mark.parametrize("md", MINE, ids=lambda p: _rel(p))
def test_python_modules_resolve(md):
    bad = []
    for mod in set(MODULE.findall(md.read_text(encoding="utf-8"))):
        base = REPO / Path(*mod.split("."))
        if not (base.with_suffix(".py").exists() or (base / "__main__.py").exists()):
            if not _allowed_missing(base.relative_to(REPO).as_posix() + "/"):
                bad.append(mod)
    assert not bad, f"{_rel(md)}: python -m targets that do not exist {bad}"


_HELP: dict = {}


def _help(cmd: list) -> str:
    key = " ".join(cmd)
    if key not in _HELP:
        r = subprocess.run([sys.executable, *cmd, "--help"], cwd=REPO, capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PYTHONPATH": str(REPO)})
        _HELP[key] = r.stdout + r.stderr
    return _HELP[key]


COMMAND = re.compile(r"python3? (-m ttg(?:\.\w+)+|drills/\w+\.py(?: [a-z][\w-]*)?)([^\n`|]*)")


@pytest.mark.parametrize("src", [*MINE, REPO / "drills" / "inject.sh"], ids=lambda p: _rel(p))
def test_python_flags_exist(src):
    """Every --flag the docs pass to a ttg module or a drills script appears in that command's --help."""
    text = re.sub(r"\\\n\s*", " ", src.read_text(encoding="utf-8"))      # join shell line continuations
    bad = []
    for target, rest in COMMAND.findall(text):
        cmd = target.split()
        if cmd[0] == "-m":
            cmd = ["-m", cmd[1]]
        help_text = _help(cmd)
        for flag in set(re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest.split(" | ")[0])):
            if flag not in help_text:
                bad.append(f"{' '.join(cmd)} {flag}")
    assert not bad, f"{_rel(src)}: flags missing from --help: {sorted(set(bad))}"
