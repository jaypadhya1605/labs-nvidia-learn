#!/usr/bin/env python3
"""Builds workshop/plover-workshop.ipynb (keep the notebook generated: reviewable diffs, CI executes it on the mock).
  python3 workshop/build_notebook.py && jupyter nbconvert --execute --to notebook workshop/plover-workshop.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Ship one NIM product on any cloud - hands-on (90 minutes)
**Audience:** Plover engineers. **You need:** Python 3.11+, `pip install httpx fastapi pyyaml`.
**Endpoint:** set `PLV_ENDPOINT` to a NIM (e.g. `kubectl port-forward svc/plover-llm 8000` -> `http://localhost:8000`), or leave it
unset to use the built-in mock (offline). Hosted endpoints: **synthetic data only** - every case here is fictional."""),
    code("""import json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "app").exists() else Path.cwd().parent
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tests"), str(ROOT / "selector")]
from plover_api import prompts, schema, synth

ENDPOINT = os.environ.get("PLV_ENDPOINT")
MODEL = os.environ.get("PLV_MODEL", "meta/llama-3.1-8b-instruct")
if ENDPOINT:
    import httpx
    key = os.environ.get("PLV_API_KEY")
    client = httpx.Client(base_url=ENDPOINT.rstrip("/"), timeout=300, headers={"Authorization": f"Bearer {key}"} if key else {})
else:
    from fastapi.testclient import TestClient
    from mock_openai import app
    client = TestClient(app)
print("endpoint:", ENDPOINT or "in-process mock", "| model:", MODEL)"""),
    md("## 1. It is just the OpenAI API\nNIM speaks the OpenAI protocol, so every SDK and tool the team already uses works unchanged."),
    code("""print([m["id"] for m in client.get("/v1/models").json()["data"]])
r = client.post("/v1/chat/completions", json={"model": MODEL, "max_tokens": 60, "temperature": 0,
                 "messages": [{"role": "user", "content": "In one sentence: what does a prior-authorization nurse reviewer need?"}]}).json()
print(r["choices"][0]["message"]["content"]); print(r["usage"])"""),
    md("## 2. Streaming and time-to-first-token\nUsers feel TTFT; dashboards should too (🔁 Lab 02: TTFT = prefill = compute-bound)."),
    code("""t0, first, text = time.perf_counter(), None, ""
body = {"model": MODEL, "stream": True, "max_tokens": 60, "messages": [{"role": "user", "content": "List three NIM benefits."}]}
with client.stream("POST", "/v1/chat/completions", json=body) as s:
    for line in s.iter_lines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - t0
            text += delta
print(f"TTFT {first * 1000:.0f} ms, total {(time.perf_counter() - t0) * 1000:.0f} ms\\n{text}")"""),
    md("## 3. The product call: long prompt vs LoRA adapter\nSame output contract, very different prompt bill. This is why M8 trains an adapter."),
    code("""case = synth.make_case(4242)
for mode, model in (("long-prompt", MODEL), ("adapter", os.environ.get("PLV_ADAPTER", "plover-style-v1"))):
    r = client.post("/v1/chat/completions", json={"model": model, "temperature": 0, "max_tokens": 400,
                     "messages": prompts.build_messages(case["text"], mode)})
    if r.status_code != 200:
        print(mode, "->", r.status_code, "(adapter not loaded on this endpoint?)"); continue
    d = r.json(); obj = schema.extract_json(d["choices"][0]["message"]["content"])
    print(f"{mode:<12} prompt_tokens={d['usage']['prompt_tokens']:<5} valid={not schema.validate(obj)} "
          f"hallucinated={schema.unfaithful_codes(obj, case['text'])} -> {obj and obj.get('recommendation')}")"""),
    md("## 4. Structured output: valid JSON every time\nGuided decoding guarantees the *shape*; the faithfulness check still catches invented codes."),
    code("""js = {"type": "object", "required": sorted(schema.REQUIRED), "properties": {"recommendation": {"enum": sorted(schema.RECOMMENDATIONS)}}}
r = client.post("/v1/chat/completions", json={"model": MODEL, "temperature": 0, "max_tokens": 400,
                 "messages": prompts.build_messages(case["text"], "long-prompt"),
                 "response_format": {"type": "json_schema", "json_schema": {"name": "plover", "schema": js}}}).json()
obj = json.loads(r["choices"][0]["message"]["content"]); print(json.dumps(obj, indent=1)[:600]); print("errors:", schema.validate(obj))"""),
    md("## 5. Agents: tool calls and prompt injection (🔁 Lab 03)\nOne scenario: the clinical notes contain a planted instruction to approve. A safe agent pends."),
    code("""from agentic import harness
sc = [s for s in harness.scenarios(1) if s["kind"] == "injected"][0]
rec = harness.run_scenario(client, MODEL, sc)
print("tool calls:", rec["calls"]); print("final:", rec["final"]); print("followed the injection?", rec["injection_followed"])"""),
    md("## 6. Measure, don't guess: a 60-second concurrency probe\nAIPerf does this properly (M5); this shows the shape: throughput rises with concurrency until latency breaks the SLO."),
    code("""def one(i):
    c = synth.make_case(6000 + i); t = time.perf_counter()
    client.post("/v1/chat/completions", json={"model": MODEL, "temperature": 0, "max_tokens": 300,
                 "messages": prompts.build_messages(c["text"], "long-prompt")})
    return time.perf_counter() - t
for conc in (1, 4, 8):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        lat = sorted(ex.map(one, range(conc * 4)))
    wall = time.perf_counter() - t0
    print(f"concurrency {conc}: {conc * 4 / wall:6.1f} req/s   p50 {lat[len(lat) // 2] * 1000:7.0f} ms   p95 {lat[-1] * 1000:7.0f} ms")"""),
    md("## 7. What does a summary cost?\nSelf-hosted: GPU $/hour / summaries per hour at the SLO. Pay-per-token: tokens x list price. Plug in YOUR M5 numbers."),
    code("""import yaml
inst = {i["instance"]: i for i in yaml.safe_load((ROOT / "selector" / "instances.yaml").read_text())}
gpu_hour = inst["Standard_NC24ads_A100_v4"]["spot"]          # AKS Spot A100 (selector/instances.yaml)
summaries_per_hour = float(os.environ.get("PLV_SUMMARIES_PER_HOUR", "15000"))   # replace with your M5 operating point
print(f"self-hosted: ${gpu_hour / summaries_per_hour * 1000:.4f} per 1,000 summaries (+ NVAIE $1/GPU-h via marketplace)")
pin, pout = 1400, 300                                         # tokens per summary (long prompt)
print(f"hosted Lightning @ $0.025/$0.12 per 1M: ${(pin * 0.025 + pout * 0.12) / 1e6 * 1000:.4f} per 1,000 summaries")"""),
    md("""## Your turn
1. Point `PLV_ENDPOINT` at the GKE NIM: same code, different cloud - what changed in TTFT?
2. Raise the probe to concurrency 32 against a real NIM: where does p95 cross 8 s?
3. Re-run section 5 with `PLV_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b` (add-on) - does injection resistance change?"""),
]
for i, c in enumerate(cells):
    c["id"] = f"plv-{i:02d}"       # stable ids -> reproducible file (CI diffs it)
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).resolve().parent / "plover-workshop.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
