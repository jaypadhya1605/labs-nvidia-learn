#!/usr/bin/env python3
"""AIPerf datasets for Plover (synthetic prior-auth requests; runs anywhere, standard library + the app package).

  priorauth-long.jsonl     base model + the ~700-token style guide (what Plover ships today)
  priorauth-adapter.jsonl  LoRA adapter + one-line instruction (M8) -> same output, far fewer prompt tokens
Realistic prompts matter (🔁 Lab 02): random text has the wrong ISL/OSL shape and the wrong cache behaviour.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from plover_api import prompts, synth  # noqa: E402


def flatten(messages):
    return "\n\n".join(m["content"] for m in messages)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--start", type=int, default=5000, help="case ids >= 5000 never overlap the LoRA training set")
    ap.add_argument("--output-length", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for mode, name in (("long-prompt", "priorauth-long.jsonl"), ("adapter", "priorauth-adapter.jsonl")):
        with open(out / name, "w") as f:
            for c in synth.cases(a.n, a.start):
                f.write(json.dumps({"text": flatten(prompts.build_messages(c["text"], mode)), "output_length": a.output_length}) + "\n")
    print(f"wrote {a.n} rows each to {out}/priorauth-long.jsonl and {out}/priorauth-adapter.jsonl")


if __name__ == "__main__":
    main()
