"""workshop/build_notebook.py builds a valid nbformat notebook with stable cell ids; the committed copy is current.

Executing it (~2-3 minutes on the mock) is opt-in: TTG_EXECUTE_NOTEBOOK=1 python -m pytest tests/test_docs_notebook.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "workshop" / "build_notebook.py"
COMMITTED = REPO / "workshop" / "ttg-workshop.ipynb"


def _builder():
    spec = importlib.util.spec_from_file_location("ttg_build_notebook", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builds_a_valid_notebook(tmp_path):
    out = tmp_path / "nb.ipynb"
    assert _builder().main(["--out", str(out)]) == 0
    nb = nbformat.read(out, as_version=4)
    nbformat.validate(nb)
    ids = [c["id"] for c in nb.cells]
    assert ids == [f"ttg-{i:02d}" for i in range(len(nb.cells))]
    for c in nb.cells:
        if c.cell_type == "code":
            compile(c.source, c["id"], "exec")
            assert not c.get("outputs")
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for n in range(1, 8):
        assert f"## {n}. " in text, f"section {n} missing"
    assert "synthetic" in text and "never a GPU result" in text


def test_committed_notebook_is_current():
    assert COMMITTED.exists(), "run: python workshop/build_notebook.py"
    committed = nbformat.read(COMMITTED, as_version=4)
    fresh = _builder().build()
    assert [(c["id"], c.source) for c in committed.cells] == [(c["id"], c.source) for c in fresh.cells], \
        "workshop/ttg-workshop.ipynb is stale: run python workshop/build_notebook.py"


@pytest.mark.skipif(os.environ.get("TTG_EXECUTE_NOTEBOOK") != "1", reason="opt-in: TTG_EXECUTE_NOTEBOOK=1")
def test_executes_on_the_mock(tmp_path, monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    monkeypatch.setenv("TTG_WORKSHOP_PORT", os.environ.get("TTG_WORKSHOP_PORT", "18030"))
    monkeypatch.delenv("TTG_BASE_URL", raising=False)
    nb = _builder().build()
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO / "workshop")}}).execute()
    outputs = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "mock-bustcache" in outputs and "cache hit 0%" in outputs and "mock stopped" in outputs
