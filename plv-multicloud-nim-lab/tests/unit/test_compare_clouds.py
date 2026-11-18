"""compare_clouds: non-streaming CSVs (empty ITL cells) must not crash; predictions use the precision actually served."""
import csv
import json
import sys

import compare_clouds

COLS = ["label", "concurrency", "out_tok_per_s", "itl_p95_ms", "slo_attainment", "energy_method"]


def write_run(d, rows, precision=None, product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"):
    d.mkdir()
    meta = {"cluster": "gke", "model": "meta/llama-3.1-8b-instruct", "isl": 600, "osl": 200, "gpu_hourly_usd": 1.63, "gpu_product": product}
    if precision:
        meta["precision"] = precision
    (d / "run.json").write_text(json.dumps(meta))
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(dict(zip(COLS, r)))


def test_empty_itl_cells_do_not_crash(tmp_path, capsys, monkeypatch):
    write_run(tmp_path / "gke-llm-long-x", [("gke:llm-long", 8, 1500, "", "", "avg-power")])
    monkeypatch.setattr(sys, "argv", ["compare_clouds.py", str(tmp_path / "gke-llm-long-x")])
    compare_clouds.main()
    assert "no level met ITL" in capsys.readouterr().out


def test_prediction_uses_served_precision(tmp_path, capsys, monkeypatch):
    write_run(tmp_path / "bf16", [("gke:s", 128, 8800, 33, 0.95, "avg-power")], precision="bf16")
    write_run(tmp_path / "nvfp4", [("gke:s", 128, 8800, 33, 0.95, "avg-power")], precision="nvfp4")
    monkeypatch.setattr(sys, "argv", ["compare_clouds.py", str(tmp_path / "bf16"), str(tmp_path / "nvfp4"), "--itl-slo-ms", "40"])
    compare_clouds.main()
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith(("bf16", "nvfp4"))]
    pred = {x.split()[0]: int(x.split()[-2]) for x in lines}
    assert pred["nvfp4"] > pred["bf16"], "BF16 and NVFP4 runs must be predicted with their own precision"
