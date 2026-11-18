"""drills/: the injector is safe to eval, every drill reverts cleanly, and the fixtures and diff tool work offline."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
INJECT = REPO / "drills" / "inject.sh"
DRILLS = ["1", "2", "3", "4", "5", "6", "7", "8", "8b", "9", "10"]
SAFE_LINE = re.compile(r"^(export [A-Z_][A-Z0-9_]*='[^']*'|unset( [A-Z_][A-Z0-9_]*)+)$")
sys.path.insert(0, str(REPO / "drills"))


def sh(*args):
    return subprocess.run(["bash", str(INJECT), *args], capture_output=True, text=True, cwd=REPO, timeout=30)


def exported(out: str) -> dict:
    return dict(line[len("export "):].split("=", 1) for line in out.splitlines() if line.startswith("export "))


def test_syntax():
    assert subprocess.run(["bash", "-n", str(INJECT)]).returncode == 0


def test_shellcheck_clean():
    if not shutil.which("shellcheck"):
        pytest.skip("shellcheck not installed")
    r = subprocess.run(["shellcheck", str(INJECT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_help_and_list():
    for arg in ([], ["help"], ["--help"]):
        r = sh(*arg)
        assert r.returncode == 0 and "Usage" in r.stdout
    r = sh("list")
    assert r.returncode == 0
    assert [ln.split()[0] for ln in r.stdout.splitlines()] == DRILLS


@pytest.mark.parametrize("n", DRILLS)
def test_stdout_is_only_export_and_unset(n):
    inj, rev = sh(n), sh(n, "revert")
    assert inj.returncode == 0 and rev.returncode == 0, inj.stderr + rev.stderr
    for line in inj.stdout.splitlines() + rev.stdout.splitlines():
        assert SAFE_LINE.match(line), f"drill {n}: unsafe stdout line {line!r}"
    assert exported(inj.stdout).get("TTG_DRILL") == f"'{n}'"
    unset = {v for line in rev.stdout.splitlines() if line.startswith("unset ") for v in line.split()[1:]}
    assert set(exported(inj.stdout)) <= unset, f"drill {n}: revert leaves {set(exported(inj.stdout)) - unset} set"
    assert inj.stderr.strip(), f"drill {n}: no instructions on stderr"


def test_eval_round_trip():
    script = ('eval "$(bash drills/inject.sh 2 2>/dev/null)"; printf "%s|%s|%s\\n" "$MOCK_FAULTS" "$EXTRA_ARGS" '
              '"$TTG_DRILL"; eval "$(bash drills/inject.sh 2 revert 2>/dev/null)"; '
              'printf "%s|%s\\n" "${MOCK_FAULTS:-unset}" "${TTG_DRILL:-unset}"')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    assert out == ["invalid_json_once|--chat-template /drills/broken-chat-template.jinja|2", "unset|unset"]


@pytest.mark.parametrize("args", [["0"], ["11"], ["foo"], ["1", "sideways"]])
def test_bad_arguments_fail_with_empty_stdout(args):
    r = sh(*args)
    assert r.returncode != 0 and r.stdout == ""


def test_exported_variables_are_read_somewhere():
    """Every knob a drill turns must exist in the code it targets (catches interface drift between builders)."""
    code = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in
                     [*REPO.glob("ttg/**/*.py"), REPO / "engines" / "launch.sh"] if p.is_file())
    names = {k for n in DRILLS for k in exported(sh(n).stdout)} - {"TTG_DRILL"}
    missing = sorted(k for k in names if k not in code)
    assert not missing, f"drills export variables nothing reads: {missing}"


def test_readme_has_every_drill():
    text = (REPO / "drills" / "README.md").read_text(encoding="utf-8")
    for n in DRILLS:
        assert f"## Drill {n} - " in text, f"drills/README.md has no section for drill {n}"
    assert text.count("**Say it:**") >= len(DRILLS) + 1 and "**Checkpoint:**" in text


def test_broken_template_has_no_roles():
    jinja2 = pytest.importorskip("jinja2")
    src = (REPO / "drills" / "files" / "broken-chat-template.jinja").read_text(encoding="utf-8")
    out = jinja2.Template(src).render(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert out.strip() == "SU"


def test_fixture_lora_rank(tmp_path):
    import fixtures

    src = tmp_path / "ttg-short"
    src.mkdir()
    (src / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    (src / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    dst = fixtures.lora_rank(src, tmp_path / "out" / "ttg-short", 32)
    assert json.loads((dst / "adapter_config.json").read_text())["r"] == 32
    assert (dst / "adapter_model.safetensors").exists() and (dst / "DRILL-9-FAKE-RANK.txt").exists()
    assert json.loads((src / "adapter_config.json").read_text())["r"] == 16      # the original is untouched


def test_fixture_receipt_photos(tmp_path):
    image = pytest.importorskip("PIL.Image")
    import fixtures

    (tmp_path / "src").mkdir()
    image.new("RGB", (600, 900), "white").save(tmp_path / "src" / "R-S0001.png")
    out = fixtures.receipts_photo(tmp_path / "src", tmp_path / "dst")
    with image.open(out[0]) as im:
        assert im.size == (3024, 4032)                       # 12 MP: 108 x 144 = 15,552 Qwen2.5-VL tokens


def test_diff_runs(tmp_path):
    import diff_runs

    def run(bust, hit, tps):
        return {"engine": "mock", "settings": {"bust_cache": bust, "thoughts": "brief"},
                "levels": [{"concurrency": 8, "success_rate": 1.0, "tasks_per_s": tps, "cache_hit_ratio": hit}],
                "tasks": [{"error": None, "violations": []}]}

    d = diff_runs.diff(run(False, 0.95, 3.82), run(True, 0.0, 3.09))
    assert d["settings"] == {"bust_cache": (False, True)}
    assert d["levels"][8]["cache_hit_ratio"] == (0.95, 0.0)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run(False, 0.95, 3.82)))
    b.write_text(json.dumps(run(True, 0.0, 3.09)))
    assert diff_runs.main([str(a), str(b)]) == 0
