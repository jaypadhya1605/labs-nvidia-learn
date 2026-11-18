#!/usr/bin/env python3
"""M8 step 1 - prompt distillation dataset: the TEACHER is today's product (base model + ~700-token style guide);
the STUDENT input is the one-line instruction. Only schema-valid, faithful teacher outputs are kept.
  python3 mlops/make_dataset.py --endpoint http://localhost:8000 --n 800 --out results/mlops      # laptop + port-forward
  (in cluster: mlops/k8s/dataset-job.yaml)       --endpoint mock  -> in-process mock (CI / dry run)
Case ids 0..n-1 train/val; the eval gate uses ids >= 9000 and the AIPerf datasets ids >= 5000 (no leakage).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tests")]
from plover_api import prompts, schema, synth  # noqa: E402


def client_for(endpoint):
    if endpoint == "mock":
        from fastapi.testclient import TestClient
        from mock_openai import app
        return TestClient(app)
    import httpx
    return httpx.Client(base_url=endpoint.rstrip("/"), timeout=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--out", default="results/mlops")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], 0
    with client_for(a.endpoint) as c:
        for case in synth.cases(a.n, 0):
            body = {"model": a.model, "messages": prompts.build_messages(case["text"], "long-prompt"),
                    "temperature": 0.0, "max_tokens": 500}
            r = c.post("/v1/chat/completions", json=body)
            if r.status_code != 200:
                dropped += 1
                continue
            obj = schema.extract_json(r.json()["choices"][0]["message"].get("content") or "")
            if schema.validate(obj) or schema.unfaithful_codes(obj, case["text"]):
                dropped += 1          # never teach the student a wrong or hallucinated answer
                continue
            student = prompts.build_messages(case["text"], "adapter")
            kept.append({"id": case["id"], "messages": student + [{"role": "assistant", "content": json.dumps(obj)}]})
    random.Random(7).shuffle(kept)
    n_val = max(1, int(len(kept) * a.val_fraction))
    for name, rows in (("val.jsonl", kept[:n_val]), ("train.jsonl", kept[n_val:])):
        with open(out / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    stats = {"teacher_model": a.model, "requested": a.n, "kept": len(kept), "dropped": dropped,
             "train": len(kept) - n_val, "val": n_val}
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
