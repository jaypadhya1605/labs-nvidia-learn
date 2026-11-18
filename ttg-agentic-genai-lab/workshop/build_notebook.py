#!/usr/bin/env python3
"""Builds workshop/ttg-workshop.ipynb, the attendee notebook for the 90-minute workshop (M12).

  python workshop/build_notebook.py                      # writes workshop/ttg-workshop.ipynb
  python workshop/build_notebook.py --out /tmp/x.ipynb   # anywhere else (tests do this)
  jupyter-execute --timeout=600 workshop/ttg-workshop.ipynb   # run it on the mock (~2-3 minutes; nbclient)

Keep the notebook generated, never hand-edited: cell ids are stable ("ttg-NN"), so diffs stay reviewable. It runs on a
laptop: it starts the mock engine itself unless TTG_BASE_URL points somewhere else. Synthetic data only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = [
    ("md", """# Storm day: an AI agent, measured - hands-on (90 minutes)
**Audience:** Tern Travel Group engineers (platform, ML, contact-center apps). **You need:** Python 3.11+ and this repo
(`pip install -r requirements-dev.txt`). **Engine:** the notebook starts the built-in mock engine on port 8000 unless
`TTG_BASE_URL` points at a real one. Tern Travel Group, its travelers and its policy are fictional; every scenario is
synthetic.

You will: run the Disruption Concierge on three travelers (§1), see where its tokens go (§2), measure and price it
(§3), break its prefix cache with one timestamp (§4), and diagnose a mystery fault with a partner (§5).
**Mock numbers show the shape of a result, never a GPU result.**"""),
    ("code", r'''import json, os, random, shlex, subprocess, sys, time, urllib.request
from pathlib import Path

from IPython.display import Image, Markdown, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "ttg" / "world.py").exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "workshop"
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("TTG_WORKSHOP_PORT", "8000"))
_direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost never goes through a proxy


def is_up(base_url):
    try:
        with _direct.open(base_url.rstrip("/") + "/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_mock(port, env=None):
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "ttg.mock.fake_llm:app", "--port", str(port)],
                            cwd=ROOT, env={**os.environ, **(env or {})},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if is_up(f"http://localhost:{port}/v1"):
            return proc
        time.sleep(0.25)
    proc.terminate()
    raise RuntimeError(f"the mock did not start on port {port} - is the port busy?")


BASE = os.environ.get("TTG_BASE_URL", "")
LOCAL = not BASE
MOCK = None
if LOCAL:
    BASE = f"http://localhost:{PORT}/v1"
    if not is_up(BASE):
        MOCK = start_mock(PORT)
ENV = {**os.environ, "TTG_BASE_URL": BASE, "TTG_MODEL": os.environ.get("TTG_MODEL", "ttg-llm"),
       "TTG_TRACE_DIR": str(OUT / "traces")}


def run(*args, env=None, quiet=False):
    """python -m <args> with the workshop settings; prints the tail of its output."""
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env={**ENV, **(env or {})},
                       capture_output=True, text=True)
    if not quiet or r.returncode:
        print((r.stdout or "")[-3000:] + (r.stderr or "")[-1500:] * (r.returncode != 0))
    return r


print("engine:", BASE, "| mock started by this notebook" if MOCK else "| already running")'''),
    ("md", """## 1. One disruption, step by step (Exercise 1)
Three travelers from the 200-scenario golden set. Read each message, then **write down how many tool calls the agent
will need** before you run the next cells. Hint: `flight_status`, `loyalty_profile` and `policy_search` always come
first; the policy is in `data/policies/tern-care-v3.md`."""),
    ("code", '''from ttg import world

GOLDEN = {s["scenario_id"]: s for s in world.make_scenarios(200, seed=7)}
PICKS = ["S0003", "S0005", "S0016"]
for sid in PICKS:
    v = world.public_view(GOLDEN[sid])
    print(f"{sid} | party of {v['party_size']} | {v['guest_message']}")'''),
    ("code", '''for sid in PICKS:
    run("ttg.agent.run", "--scenario", sid)'''),
    ("md", """Now compare with the policy oracle: the exact plan Tern Care Commitments v3 requires. Every LLM call is
one step of this plan, plus one for the final answer. Did the receipt in S0003 earn a reimbursement? Why not?"""),
    ("code", '''for sid in PICKS:
    plan = world.expected_plan(GOLDEN[sid])
    print(f"{sid}: {len(plan) - 1} tool calls, {len(plan)} LLM calls: "
          + " -> ".join(step.get("action", "final") for step in plan))'''),
    ("md", """## 2. Where the tokens go (Exercise 2, part 1)
An agent resends the whole conversation on every call. The system prompt (role, protocol, tools, policy digest, one
example) is identical for every call of every task. Predict: what share of S0016's prompt tokens is that one shared
prefix?"""),
    ("code", '''from ttg.agent import prompts, protocol

SYSTEM = prompts.build_system_prompt()
sys_tokens = len(SYSTEM) // 4                      # ~4 characters per token: an estimate, the engine counts exactly
msgs, per_call = [SYSTEM, prompts.render_scenario(GOLDEN["S0016"])], []
for step in world.expected_plan(GOLDEN["S0016"]):
    per_call.append(sum(len(m) for m in msgs) // 4)
    if "final" in step:
        break
    msgs.append(protocol.format_action(step["action"], step["args"], thought="Next step."))
    msgs.append(prompts.render_observation(step["action"], world.execute_tool(step["action"], step["args"])))
total = sum(per_call)
new_per_task = (per_call[0] - sys_tokens) + sum(b - a for a, b in zip(per_call, per_call[1:]))
print(f"system prompt: {len(SYSTEM):,} characters, ~{sys_tokens:,} tokens, byte-identical for every call")
print(f"S0016 prompt tokens per call: {per_call}")
print(f"total {total:,} tokens over {len(per_call)} calls; the shared system prompt is {sys_tokens * len(per_call):,} "
      f"of them ({sys_tokens * len(per_call) / total:.0%})")
print(f"with a warm prefix cache only ~{new_per_task:,} tokens are new: "
      f"an ideal hit ratio of {1 - new_per_task / total:.0%}")'''),
    ("md", """## 3. Measure it and price it (Exercise 2, part 2)
`ttg.bench.agent_bench` runs golden-set tasks at several concurrency levels and records success, latency, tokens,
the prefix-cache hit ratio and GPU $ per 1,000 tasks. Here the "GPU" is the mock priced at $2.49 per hour: the numbers
teach the shape, not the answer. Predict the cache hit ratio before you run it."""),
    ("code", '''def bench(label, env=None, levels="1,8", n=16):
    extra = ["--metrics-url", BASE[:-3] + "/metrics", "--reset-url", BASE[:-3] + "/reset"] if LOCAL else []
    run("ttg.bench.agent_bench", "--label", label, "--engine", "mock", "--levels", levels, "--n", str(n),
        "--warmup", "1", "--gpu-hourly", "2.49", "--out", str(OUT / f"{label}.json"), *extra, env=env, quiet=True)
    d = json.loads((OUT / f"{label}.json").read_text())
    for lv in d["levels"]:
        print(f"{label:<15} concurrency {lv['concurrency']:>2}: success {lv['success_rate']:.0%}, "
              f"{lv['tasks_per_s']:.2f} tasks/s, p95 {lv['p95_latency_s']:.2f} s, "
              f"cache hit {lv['cache_hit_ratio']:.0%}, ${lv['usd_per_1k_tasks']:.3f} per 1k tasks (mock)")
    return d


base = bench("mock-base")'''),
    ("md", """## 4. Break it with one timestamp (Exercise 2, part 3)
A teammate adds `Current time: ...` to the top of the system prompt "so the agent knows the date". Same agent, same
policy, same answers. Predict: success? cache hit ratio? $ per 1,000 tasks? Then run it (this is drill 1)."""),
    ("code", '''bust = bench("mock-bustcache", env={"TTG_BUST_CACHE": "1"})
run("ttg.bench.ledger", str(OUT / "mock-base.json"), str(OUT / "mock-bustcache.json"),
    "--order", "mock-base,mock-bustcache", "--out", str(OUT / "ledger"), quiet=True)
display(Markdown((OUT / "ledger" / "ledger.md").read_text()))
if (OUT / "ledger" / "waterfall.png").exists():
    display(Image(str(OUT / "ledger" / "waterfall.png")))'''),
    ("md", """On a GPU the same change costs more than on the mock: every call re-prefills ~1,840 system tokens plus the
history, so TTFT rises on every step, and at storm-day load the queue builds sooner. The fix: nothing dynamic above
the shared prefix (ADR-02 in `docs/02-architecture-and-adrs.md`)."""),
    ("md", """## 5. Mystery fault, in pairs (Exercise 3)
Person A runs `mystery()` and keeps the drill number to themselves. Person B reads only the symptoms, next to the
healthy run, and names the root cause. Then run `reveal()` and read that drill in `drills/README.md`. Swap and repeat.
The drills are the laptop ones from `drills/inject.sh` (1, 2, 8 and 10)."""),
    ("code", r'''def drill_env(n):
    out = subprocess.run(["bash", str(ROOT / "drills" / "inject.sh"), str(n)], capture_output=True, text=True,
                         check=True).stdout
    return {k: shlex.split(v)[0] for k, v in (line[7:].split("=", 1) for line in out.splitlines()
                                               if line.startswith("export "))}


def symptoms(env=None, tasks=12):
    r = run("ttg.agent.run", "--n", str(tasks), "--concurrency", "4", "--json", env=env, quiet=True)
    rs = json.loads(r.stdout)
    n = len(rs)
    errors = sorted({(t.get("error") or "")[:70] for t in rs if t.get("error")})
    viols = sorted({v for t in rs for v in (t.get("violations") or [])})
    return {"success": f"{sum(t['success'] for t in rs)}/{n}",
            "LLM calls per task": round(sum(t["llm_calls"] for t in rs) / n, 1),
            "retries per task": round(sum(t["retries"] for t in rs) / n, 2),
            "prompt tokens per task": round(sum(t["prompt_tokens"] for t in rs) / n),
            "output tokens per task": round(sum(t["completion_tokens"] for t in rs) / n),
            "cache hit": f"{sum(t['cached_tokens'] for t in rs) / max(1, sum(t['prompt_tokens'] for t in rs)):.0%}",
            "p50 task latency (s)": round(sorted(t["latency_s"] for t in rs)[n // 2], 2),
            "errors": errors[:3], "violations": viols[:6]}


HEALTHY = None
_SECRET = None


def mystery(n=None):
    global HEALTHY, _SECRET
    _SECRET = n or random.choice([1, 2, 8, 10])
    if HEALTHY is None:
        HEALTHY = symptoms()
    env = {k: v for k, v in drill_env(_SECRET).items() if k != "TTG_DRILL"}
    fault, proc = env.pop("MOCK_FAULTS", None), None
    env = {k: v for k, v in env.items() if k.startswith("TTG_")}   # agent-side settings only, on a laptop
    if fault:                                                     # this fault lives in the engine: a second mock
        proc = start_mock(PORT + 1, {"MOCK_FAULTS": fault})
        env["TTG_BASE_URL"] = f"http://localhost:{PORT + 1}/v1"
    try:
        sick = symptoms(env)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)
    print(f"{'':<24}{'healthy':<28}mystery")
    for k in HEALTHY:
        print(f"{k:<24}{str(HEALTHY[k])[:26]:<28}{sick[k]}")


def reveal():
    print(f"Drill {_SECRET}: read 'Drill {_SECRET}' in drills/README.md")


mystery()'''),
    ("code", "reveal()"),
    ("md", """## 6. What the GPU changes (facilitator demo)
The facilitator shows the same benchmark on a real GPU: vLLM vs SGLang vs TensorRT-LLM vs NIM, then prefix caching,
FP8 and speculation, as one waterfall of $ per 1,000 disruptions. If you have a GPU VM,
`docs/09-results-and-exec-readout.md` has the commands. The cell below shows the newest ledger in `results/ledger/`
if there is one."""),
    ("code", '''ledger = ROOT / "results" / "ledger" / "ledger.md"
if ledger.exists():
    text = ledger.read_text()
    display(Markdown(text))
    if "| mock" in text:
        print("Rows on 'mock' are mock runs: shape only, not GPU results.")
else:
    print("No ledger yet: python -m ttg.bench.ledger results/bench/*.json --out results/ledger")'''),
    ("md", """## 7. Clean up
Stops the mock if this notebook started it. Your results stay in `results/workshop/` (git-ignored)."""),
    ("code", '''if MOCK:
    MOCK.terminate()
    MOCK.wait(timeout=10)
print("mock stopped" if MOCK else "nothing to stop")'''),
]


def build():
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for i, (kind, src) in enumerate(CELLS):
        cell = nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        cell["id"] = f"ttg-{i:02d}"
        nb.cells.append(cell)
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Lab 05 workshop notebook")
    ap.add_argument("--out", default=str(HERE / "ttg-workshop.ipynb"))
    a = ap.parse_args(argv)
    import nbformat as nbf

    nb = build()
    nbf.validate(nb)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, a.out)
    print(f"{len(nb.cells)} cells -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
