"""The README is the lab: every link, path, module and flag it tells you to use must exist in this repo.

Commands are parsed from the README's bash blocks. For each `python -m <module>` / `python3 <script>.py` the test runs
`--help` and checks that every `--flag` the README uses appears in argparse's usage line; for bash scripts it checks
the script exists, passes `bash -n` and mentions the flag. Nothing here needs a GPU, Docker or network.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")
ENV = {**os.environ, "PYTHONPATH": str(REPO)}
PY_EXTERNAL = {"uvicorn", "venv", "pip", "pytest"}          # modules that come from pip, not from this repo


def _blocks() -> list[str]:
    return re.findall(r"```bash\n(.*?)```", README, flags=re.S)


def _commands() -> list[list[str]]:
    """Every simple command in the bash blocks, continuation lines joined, comments and env prefixes stripped."""
    out = []
    for block in _blocks():
        text = re.sub(r"\\\n\s*", " ", block)
        for line in text.splitlines():
            line = line.split(" # ")[0].split("  #")[0].strip()
            if not line or line.startswith("#"):
                continue
            for part in re.split(r"&&|\|\||;|\|", line):
                part = part.strip().rstrip("&").strip()
                if part.startswith(("for ", "do ", "done")):
                    part = re.sub(r"^(for .* in .*|do|done)\s*", "", part)
                try:
                    toks = shlex.split(part)
                except ValueError:
                    continue
                while toks and re.match(r"^[A-Z_][A-Z0-9_]*=", toks[0]):
                    toks = toks[1:]
                if toks:
                    out.append(toks)
    return out


COMMANDS = _commands()


def _flags(toks: list[str]) -> list[str]:
    return sorted({t.split("=")[0] for t in toks if t.startswith("--") and len(t) > 2})


def _usage(cmd: list[str]) -> str:
    p = subprocess.run(cmd + ["--help"], cwd=REPO, env=ENV, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, f"{' '.join(cmd)} --help failed: {p.stderr[-1500:]}"
    usage = p.stdout.split("\n\n")[0]                 # argparse prints every option in the usage paragraph
    return usage


def test_markdown_links_resolve():
    missing = []
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", README):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (REPO / target).exists():
            missing.append(target)
    assert not missing, missing


def test_python_modules_and_flags():
    seen, problems = set(), []
    for toks in COMMANDS:
        if toks[0] not in ("python", "python3") or len(toks) < 2 or toks[1] == "-c":
            continue
        if toks[1] == "-m":
            mod = toks[2]
            if mod.split(".")[0] in PY_EXTERNAL:
                continue
            path = REPO / (mod.replace(".", "/") + ".py")
            if not path.exists():
                problems.append(f"module {mod} not found")
                continue
            cmd, args = [sys.executable, "-m", mod], toks[3:]
        else:
            path = REPO / toks[1]
            if not path.exists():
                problems.append(f"script {toks[1]} not found")
                continue
            cmd, args = [sys.executable, str(path)], toks[2:]
        key = (tuple(cmd), tuple(_flags(args)))
        if key in seen:
            continue
        seen.add(key)
        usage = _usage(cmd)
        for f in _flags(args):
            if not re.search(re.escape(f) + r"(?![\w-])", usage):
                problems.append(f"{' '.join(toks[:3])}: flag {f} not in --help usage")
    assert not problems, "\n".join(problems)


def test_uvicorn_apps_exist():
    for toks in COMMANDS:
        if toks[0] == "uvicorn":
            mod, _, attr = toks[1].partition(":")
            assert (REPO / (mod.replace(".", "/") + ".py")).exists(), toks[1]
            src = (REPO / (mod.replace(".", "/") + ".py")).read_text()
            assert re.search(rf"^{attr}\s*=", src, flags=re.M), f"{toks[1]}: no {attr} ="


SCRIPT_RE = re.compile(r"^(engines|profiling|finetune|parallel|vlm|scripts|deploy|drills|workshop)/[\w./-]+\.sh$")


def test_bash_scripts_exist_parse_and_know_their_flags():
    problems = []
    for toks in COMMANDS:
        if not SCRIPT_RE.match(toks[0]):
            continue
        path = REPO / toks[0]
        if not path.exists():
            problems.append(f"{toks[0]} not found")
            continue
        if not os.access(path, os.X_OK):
            problems.append(f"{toks[0]} is not executable")
        p = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if p.returncode:
            problems.append(f"{toks[0]}: bash -n: {p.stderr.strip()}")
        src = path.read_text()
        for f in _flags(toks[1:]):
            if f not in src:
                problems.append(f"{toks[0]}: flag {f} not handled")
    assert not problems, "\n".join(problems)


def _launch_pairs() -> list[tuple[str, str]]:
    """Every `engines/launch.sh <engine> <variant>` the README mentions, in code blocks or in tables."""
    return sorted(set(re.findall(r"engines/launch\.sh ([a-z]+) ([a-z0-9]+)", README)))


@pytest.mark.parametrize("engine,variant", _launch_pairs())
def test_every_engine_variant_in_the_readme_launches_in_dry_run(engine, variant):
    env = {**ENV, "DRY_RUN": "1", "HF_TOKEN": "", "NGC_API_KEY": ""}
    p = subprocess.run(["bash", "engines/launch.sh", engine, variant], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-1500:]
    assert "docker run" in p.stdout or "docker" in p.stdout


def test_bench_labels_follow_the_engine_variant_convention():
    labels = []
    for toks in COMMANDS:
        if "ttg.bench.agent_bench" in toks and "--label" in toks:
            labels.append(toks[toks.index("--label") + 1])
    assert labels
    for lab in labels:
        assert re.match(r"^[a-z0-9]+-[a-z0-9]+$", lab), lab


def test_ledger_order_names_labels_the_readme_creates():
    created = {toks[toks.index("--label") + 1] for toks in COMMANDS if "--label" in toks}
    created |= {f"{e}-{v}" for e, v in _launch_pairs()}
    for toks in COMMANDS:
        if "ttg.bench.ledger" in toks:
            order = toks[toks.index("--order") + 1].split(",")
            missing = [o for o in order if o not in created]
            assert not missing, f"--order names labels the README never benchmarks: {missing}"


def test_modules_section_is_complete_and_ordered():
    heads = re.findall(r"^## (M\d+) - ", README, flags=re.M)
    assert heads == [f"M{i}" for i in range(14)], heads
    for i in range(14):
        body = README.split(f"## M{i} - ", 1)[1].split("\n## ", 1)[0]
        if i == 0:
            assert "**Checkpoint:**" in body
        elif i != 13:
            assert "**Say it:**" in body, f"M{i} has no Say it line"


def test_no_secrets_or_real_customer_names():
    assert not re.search(r"nvapi-[A-Za-z0-9_-]{20,}", README)
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", README)
    for name in ("Molina", "CommonSpirit", "UPMC", "Providence", "Bayer"):
        assert name not in README
