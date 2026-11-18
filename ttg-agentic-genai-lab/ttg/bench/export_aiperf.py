"""Export the agent's real traffic shape for AIPerf: the shared system prompt + multi-turn sessions.

  python -m ttg.bench.export_aiperf --n 64 --out results/aiperf
  aiperf profile --model ttg-llm --url http://localhost:8000 --endpoint-type chat --streaming \
      --tokenizer meta-llama/Llama-3.1-8B-Instruct \
      --system-prompt-file results/aiperf/system_prompt.txt \
      --input-file results/aiperf/sessions.jsonl --custom-dataset-type multi_turn \
      --extra-inputs max_tokens:96 --concurrency 16 --benchmark-duration 120 --artifact-dir results/aiperf/c16
Each session is one disruption: turn 1 = the traveler, later turns = the tool observations the oracle's plan would
produce. AIPerf keeps the history, so the prompt grows turn by turn on top of the byte-identical system prompt -
exactly the shape where prefix caching pays. Responses come from the engine, so they are not the agent's own steps:
use this for engine-level TTFT/ITL/goodput (🔁 Labs 01-04), and agent_bench for task-level truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ttg import world
from ttg.agent.prompts import build_system_prompt, render_observation, render_scenario


def sessions(n: int, seed: int = 7) -> list:
    out = []
    for sc in world.make_scenarios(n, seed):
        turns = [{"text": render_scenario(sc), "delay": 0}]
        for step in world.expected_plan(sc):
            if "action" in step:
                result = world.execute_tool(step["action"], step["args"])
                turns.append({"text": render_observation(step["action"], result), "delay": 0})
        out.append({"session_id": sc["scenario_id"], "turns": turns})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--variant", default="full", choices=["full", "short"])
    ap.add_argument("--out", default="results/aiperf")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_prompt.txt").write_text(build_system_prompt(variant=a.variant), encoding="utf-8")
    ss = sessions(a.n, a.seed)
    with open(out / "sessions.jsonl", "w", encoding="utf-8") as fh:
        for s in ss:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    turns = sum(len(s["turns"]) for s in ss)
    print(f"wrote {len(ss)} sessions ({turns} turns, {turns / len(ss):.1f} per session) + system_prompt.txt -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
