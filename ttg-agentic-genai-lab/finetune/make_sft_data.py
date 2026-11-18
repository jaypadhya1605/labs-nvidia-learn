#!/usr/bin/env python3
"""SFT data for the M6 "prompt baking" LoRA: teach the policy to the weights so the prompt can shrink.

Usage:
  python3 finetune/make_sft_data.py                       # train = seed 11 x 1,000 tasks, eval = seed 13 x 200, short prompt
  python3 finetune/make_sft_data.py --variant full        # keep the full 7,360-char system prompt (comparison run)
  python3 finetune/make_sft_data.py --n-train 200 --out data/sft-small     # a quick run
  TEACHER_MODEL=<build.nvidia.com model id> NVIDIA_API_KEY=... python3 finetune/make_sft_data.py --teacher
Writes <out>/train.jsonl, <out>/eval.jsonl, <out>/stats.json (default out: data/sft).

One example per agent step (prompt-completion, conversational - TRL trains on the completion only):
  {"prompt": [system, user(scenario), assistant(step 1), user(OBSERVATION ...), ...],
   "completion": [{"role": "assistant", "content": "<step k JSON>"}], "meta": {...}}
The conversation is the agent's canonical one (ttg.agent.prompts + ttg.agent.protocol), and every target is the
policy oracle's step in canonical compact JSON - byte-identical to what the mock engine answers. Synthetic data
only: seeds 11 (train) and 13 (eval) never overlap the golden set (seed 7).
--teacher (optional): a strong hosted model, given the FULL prompt, proposes each step; a step is kept only when
its action and arguments (or final fields) match the oracle - rejection sampling. The student then learns the
teacher's wording behind the SHORT prompt: context distillation. Synthetic prompts only on hosted endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ttg import world  # noqa: E402
from ttg.agent.prompts import build_system_prompt, render_observation, render_scenario  # noqa: E402
from ttg.agent.protocol import ProtocolError, format_action, format_final, parse_step  # noqa: E402

TRAIN_SEED, EVAL_SEED = 11, 13
FINAL_THOUGHT = "All actions done; write the guest message."


def _thoughts() -> dict:
    """The mock engine's short thoughts - one source of truth, so SFT targets == mock output byte for byte."""
    from ttg.mock.fake_llm import THOUGHTS
    return THOUGHTS


def oracle_steps(scenario: dict) -> list:
    """[(assistant_text, step, observation_text_or_None), ...] for the oracle plan of one scenario."""
    thoughts = _thoughts()
    out = []
    for step in world.expected_plan(scenario):
        if "final" in step:
            out.append((format_final(step["final"], FINAL_THOUGHT), step, None))
        else:
            text = format_action(step["action"], step["args"], thoughts.get(step["action"], "Next step."))
            obs = render_observation(step["action"], world.execute_tool(step["action"], step["args"]))
            out.append((text, step, obs))
    return out


def examples_for(scenario: dict, system_prompt: str, teacher=None, stats: Counter | None = None) -> list:
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": render_scenario(scenario)}]
    steps = oracle_steps(scenario)
    rows = []
    for k, (text, step, obs) in enumerate(steps):
        target = text
        if teacher is not None:
            proposal = teacher(scenario, msgs[1:])          # the teacher sees its own full system prompt
            if proposal is not None and accept_step(step, proposal):
                target = proposal
                if stats is not None:
                    stats["teacher_accepted"] += 1
            elif stats is not None:
                stats["teacher_rejected"] += 1
        kind = "final" if "final" in step else "action"
        rows.append({"prompt": [dict(m) for m in msgs], "completion": [{"role": "assistant", "content": target}],
                     "meta": {"scenario_id": scenario["scenario_id"], "step": k + 1, "n_steps": len(steps),
                              "kind": kind, "action": step.get("action", "final")}})
        # The conversation continues with the ORACLE step (teacher forcing), whatever the target text was.
        msgs.append({"role": "assistant", "content": text})
        if obs is not None:
            msgs.append({"role": "user", "content": obs})
    return rows


def _norm_args(args: dict) -> dict:
    return {k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in (args or {}).items()}


def _vouchers(final: dict) -> list:
    return sorted((v.get("kind"), round(float(v.get("amount_usd", 0)), 2)) for v in final.get("vouchers") or [])


def accept_step(oracle_step: dict, text: str) -> bool:
    """Rejection-sampling rule: the proposal must parse and agree with the oracle on everything that matters."""
    try:
        s = parse_step(text)
    except ProtocolError:
        return False
    if "final" in oracle_step:
        if s.kind != "final":
            return False
        want, got = oracle_step["final"], s.final or {}
        return (got.get("rebooked_flight") == want.get("rebooked_flight") and got.get("hotel_id") == want.get("hotel_id")
                and _vouchers(got) == _vouchers(want)
                and bool(got.get("needs_human_approval")) == bool(want.get("needs_human_approval"))
                and bool(str(got.get("message") or "").strip()))
    return s.kind == "action" and s.action == oracle_step["action"] and _norm_args(s.args) == _norm_args(
        oracle_step["args"])


def make_teacher(model: str, base_url: str, api_key: str, temperature: float = 0.6):
    """OpenAI-compatible hosted teacher (build.nvidia.com by default). Imported lazily; never used by the tests."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    system = build_system_prompt(variant="full")

    def propose(_scenario: dict, history: list):
        try:
            r = client.chat.completions.create(model=model, temperature=temperature, max_tokens=320,
                                               messages=[{"role": "system", "content": system}] + history)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - a failed call just falls back to the oracle step
            print(f"  teacher call failed ({type(e).__name__}); using the oracle step", file=sys.stderr)
            return None
    return propose


def build(n: int, seed: int, variant: str, teacher=None, stats: Counter | None = None) -> list:
    system_prompt = build_system_prompt(variant=variant)
    rows = []
    for sc in world.make_scenarios(n, seed=seed):
        rows.extend(examples_for(sc, system_prompt, teacher, stats))
    return rows


def describe(rows: list) -> dict:
    kinds = Counter(r["meta"]["kind"] for r in rows)
    actions = Counter(r["meta"]["action"] for r in rows)
    p_chars = [sum(len(m["content"]) for m in r["prompt"]) for r in rows]
    c_chars = [len(r["completion"][0]["content"]) for r in rows]
    tasks = len({r["meta"]["scenario_id"] for r in rows})
    return {"examples": len(rows), "tasks": tasks, "kinds": dict(kinds), "actions": dict(sorted(actions.items())),
            "mean_prompt_chars": round(sum(p_chars) / max(len(rows), 1), 1), "max_prompt_chars": max(p_chars or [0]),
            "mean_completion_chars": round(sum(c_chars) / max(len(rows), 1), 1),
            # ~4 characters per token for this JSON-heavy English text (the mock uses the same rule)
            "approx_train_tokens": int((sum(p_chars) + sum(c_chars)) / 4),
            "approx_max_example_tokens": int(max(p + c for p, c in zip(p_chars, c_chars)) / 4) if rows else 0}


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["short", "full"], default="short", help="system prompt in the data")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    ap.add_argument("--out", default=str(REPO / "data" / "sft"))
    ap.add_argument("--teacher", action="store_true", help="rejection-sample a hosted teacher for TRAIN targets")
    ap.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", ""))
    ap.add_argument("--teacher-base-url", default=os.environ.get("TEACHER_BASE_URL",
                                                                 "https://integrate.api.nvidia.com/v1"))
    a = ap.parse_args(argv)
    if 7 in (a.train_seed, a.eval_seed):
        ap.error("seed 7 is the golden set - never train or tune on it")

    stats: Counter = Counter()
    teacher = None
    if a.teacher:
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not a.teacher_model or not key:
            ap.error("--teacher needs TEACHER_MODEL (or --teacher-model) and NVIDIA_API_KEY in the environment")
        teacher = make_teacher(a.teacher_model, a.teacher_base_url, key)

    out = Path(a.out)
    train = build(a.n_train, a.train_seed, a.variant, teacher, stats)
    evals = build(a.n_eval, a.eval_seed, a.variant)          # eval targets are always the oracle
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", evals)
    full_len, short_len = len(build_system_prompt(variant="full")), len(build_system_prompt(variant="short"))
    summary = {"variant": a.variant, "system_prompt_chars": {"full": full_len, "short": short_len},
               "train": {"seed": a.train_seed, **describe(train)}, "eval": {"seed": a.eval_seed, **describe(evals)},
               "teacher": {"model": a.teacher_model, **dict(stats)} if a.teacher else None}
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    t = summary["train"]
    print(f"train: {t['examples']:,} examples from {t['tasks']:,} tasks ({t['kinds']}) -> {out / 'train.jsonl'}")
    print(f"eval:  {summary['eval']['examples']:,} examples from {summary['eval']['tasks']:,} tasks -> {out / 'eval.jsonl'}")
    print(f"prompt: '{a.variant}' system prompt ({full_len if a.variant == 'full' else short_len:,} chars); mean "
          f"example ~{t['mean_prompt_chars'] / 4:,.0f} prompt tokens, ~{t['approx_train_tokens'] / 1e6:.1f} M training "
          f"tokens per epoch, longest example ~{t['approx_max_example_tokens']:,} tokens (set --max-length above it)")
    if a.teacher:
        print(f"teacher: accepted {stats['teacher_accepted']:,}, rejected {stats['teacher_rejected']:,} (oracle used)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
