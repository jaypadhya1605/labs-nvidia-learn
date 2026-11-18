#!/usr/bin/env python3
"""Knowledge sessions for the JR2027588 capstone. Standard library only; runs on Windows, WSL, macOS, Linux.

  python quiz/quiz.py                    status: every phase, your score, what is due today
  python quiz/quiz.py --phase P1         one phase: basic -> intermediate -> advanced
  python quiz/quiz.py --phase P1 --level advanced
  python quiz/quiz.py --review           spaced repetition: everything due today (misses come back)
  python quiz/quiz.py --mock 12          interview-style mix across phases you've started (harder weighted)
  python quiz/quiz.py --stats            scores per phase and level, weakest topics first
  python quiz/quiz.py --export QUIZ-BANK.md   printable bank with answers
  python quiz/quiz.py --selftest         checks the bank and the grader (no input needed)

Grading: multiple choice and numbers are graded automatically. Open questions show the key points; you
grade yourself honestly: y = hit every key point, p = partial, n = missed.
Progress is saved to quiz/progress.json next to this file (git-ignored).
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bank import PHASES, QUESTIONS  # noqa: E402

PROGRESS = os.path.join(HERE, "progress.json")
LEVELS = ["basic", "intermediate", "advanced"]
INTERVALS = [1, 3, 7, 14, 30]  # days until the next review, by box
PASS_PCT = 80

try:  # Windows consoles default to cp1252; keep output safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ----------------------------------------------------------------------------- progress
def today():
    return dt.date.today()


def load():
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as f:
            return json.load(f)
    return {"items": {}, "sessions": []}


def save(p):
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=1)


def record(p, q, score):
    it = p["items"].setdefault(q["id"], {"box": 0, "history": []})
    it["history"].append({"date": today().isoformat(), "score": score})
    if score >= 1:
        it["box"] = min(it["box"] + 1, len(INTERVALS) - 1)
    elif score <= 0:
        it["box"] = 0
    days = INTERVALS[it["box"]] if score >= 1 else 1
    it["due"] = (today() + dt.timedelta(days=days)).isoformat()


def is_due(p, q):
    it = p["items"].get(q["id"])
    return bool(it) and it.get("due", "9999") <= today().isoformat()


def last_score(p, q):
    it = p["items"].get(q["id"])
    return it["history"][-1]["score"] if it and it["history"] else None


# ----------------------------------------------------------------------------- grading
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", re.I)


def parse_number(text):
    m = NUM_RE.search(text.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def grade_mcq(q, reply):
    r = reply.strip().lower()
    if not r:
        return 0.0
    if r[0] in "abcdefgh" and (len(r) == 1 or not r[1].isalpha()):
        idx = ord(r[0]) - ord("a")
    elif r.isdigit():
        idx = int(r) - 1
    else:
        return 0.0
    return 1.0 if idx == q["answer"] else 0.0


def grade_num(q, reply):
    v = parse_number(reply)
    if v is None:
        return 0.0
    a = q["answer"]
    return 1.0 if abs(v - a) <= abs(a) * q["tol"] else 0.0


def grade_open(reply):
    r = reply.strip().lower()[:1]
    return {"y": 1.0, "p": 0.5}.get(r, 0.0)


# ----------------------------------------------------------------------------- asking
def ask(q, n, total, p):
    print("\n" + "-" * 78)
    print(f"[{n}/{total}] {q['id']} - {PHASES[q['phase']]} - {q['level'].upper()}")
    print(q["prompt"])
    if q["kind"] == "mcq":
        for i, o in enumerate(q["options"]):
            print(f"   {chr(97 + i)}) {o}")
        reply = input("Your answer (letter, or s = skip, q = quit): ")
    elif q["kind"] == "num":
        reply = input(f"Your answer in {q['unit']} (s = skip, q = quit): ")
    else:
        print("   (Answer out loud or type it - aim for under 90 seconds.)")
        reply = input("Your answer, then Enter (s = skip, q = quit): ")
    if reply.strip().lower() == "q":
        return None
    if reply.strip().lower() == "s":
        print("Skipped.")
        return "skip"
    if q["kind"] == "mcq":
        score = grade_mcq(q, reply)
        print(("[CORRECT] " if score else "[WRONG] ") + f"Answer: {chr(97 + q['answer'])}) "
              f"{q['options'][q['answer']]}")
    elif q["kind"] == "num":
        score = grade_num(q, reply)
        print(("[CORRECT] " if score else "[WRONG] ") + f"Answer: {q['answer']:g} {q['unit']} "
              f"(+-{q['tol'] * 100:.0f}% accepted)")
    else:
        print("Model answer:\n  " + q["answer"])
        print("Key points:\n" + "\n".join(f"  - {k}" for k in q["keys"]))
        score = grade_open(input("Grade yourself - y (all key points) / p (partial) / n: "))
    if q["kind"] != "open":
        print("Why: " + q["why"])
    print(f"Source: nvidia-labs/{q['source']}")
    record(p, q, score)
    save(p)
    return score


def run(questions, title, p):
    if not questions:
        print("Nothing to ask here.")
        return
    print(f"\n=== {title}: {len(questions)} questions ===")
    results = []
    for i, q in enumerate(questions, 1):
        s = ask(q, i, len(questions), p)
        if s is None:
            break
        if s != "skip":
            results.append((q, s))
    summarize(results, title, p)


def summarize(results, title, p):
    if not results:
        return
    print("\n" + "=" * 78)
    pct = 100 * sum(s for _, s in results) / len(results)
    print(f"{title}: {pct:.0f}% ({sum(s for _, s in results):g}/{len(results)})")
    for lv in LEVELS:
        lr = [s for q, s in results if q["level"] == lv]
        if lr:
            print(f"  {lv:<13} {100 * sum(lr) / len(lr):5.0f}%  ({len(lr)} q)")
    misses = [q for q, s in results if s < 1]
    if misses:
        print("Re-read before your next session (they come back tomorrow in --review):")
        for q in misses:
            print(f"  - {q['id']}: nvidia-labs/{q['source']}")
    verdict = "PASSED" if pct >= PASS_PCT else f"below {PASS_PCT}% - redo the Learn section, then retry"
    print(f"Verdict: {verdict}")
    p["sessions"].append({"date": today().isoformat(), "title": title, "pct": round(pct),
                          "n": len(results)})
    save(p)


# ----------------------------------------------------------------------------- views
def by_phase(ph, level=None):
    qs = [q for q in QUESTIONS if q["phase"] == ph and (level in (None, "all") or q["level"] == level)]
    return sorted(qs, key=lambda q: LEVELS.index(q["level"]))


def status(p):
    print("Phase  Title                                                     Done  Score  Due")
    for ph, title in PHASES.items():
        qs = by_phase(ph)
        done = [q for q in qs if last_score(p, q) is not None]
        sc = (100 * sum(last_score(p, q) for q in done) / len(done)) if done else None
        due = sum(1 for q in qs if is_due(p, q))
        print(f"{ph:<6} {title[:56]:<57} {len(done):>2}/{len(qs):<2} "
              f"{(f'{sc:.0f}%' if sc is not None else '-'):>5}  {due or ''}")
    nxt = next((ph for ph in PHASES if any(last_score(p, q) is None for q in by_phase(ph))), None)
    total_due = sum(1 for q in QUESTIONS if is_due(p, q))
    print(f"\nDue for review today: {total_due}   ->  python quiz/quiz.py --review")
    if nxt:
        print(f"Next new phase: {nxt}  ->  python quiz/quiz.py --phase {nxt}")


def stats(p):
    rows = []
    for ph in PHASES:
        for lv in LEVELS:
            qs = by_phase(ph, lv)
            sc = [last_score(p, q) for q in qs if last_score(p, q) is not None]
            if sc:
                rows.append((100 * sum(sc) / len(sc), ph, lv, len(sc)))
    if not rows:
        print("No answers yet. Start with: python quiz/quiz.py --phase P0")
        return
    print("Weakest first (latest attempt per question):")
    for pct, ph, lv, n in sorted(rows):
        print(f"  {pct:5.0f}%  {ph:<4} {lv:<13} ({n} q)  {PHASES[ph]}")
    print("\nSessions:")
    for s in p["sessions"][-10:]:
        print(f"  {s['date']}  {s['pct']:>3}%  {s['n']:>2} q  {s['title']}")


def mock(p, n):
    started = {q["phase"] for q in QUESTIONS if last_score(p, q) is not None} or set(PHASES)
    pool = [q for q in QUESTIONS if q["phase"] in started and q["phase"] != "P0"]
    weights = [{"basic": 1, "intermediate": 2, "advanced": 3}[q["level"]] for q in pool]
    picked, seen = [], set()
    while len(picked) < min(n, len(pool)):
        q = random.choices(pool, weights)[0]
        if q["id"] not in seen:
            seen.add(q["id"])
            picked.append(q)
    print("Mock interview: answer like you would on the call. Open questions: say it out loud, 90 s max.")
    run(picked, f"Mock ({len(picked)})", p)


def export(path):
    out = ["# JR2027588 capstone - question bank with answers", "",
           "Generated by `python quiz/quiz.py --export`. Use the quiz runner for practice; read this only "
           "after a session.", ""]
    for ph, title in PHASES.items():
        out += [f"## {ph} - {title}", ""]
        for q in by_phase(ph):
            out.append(f"**{q['id']} ({q['level']})** {q['prompt']}")
            out.append("")
            if q["kind"] == "mcq":
                out += [f"- {chr(97 + i)}) {o}" for i, o in enumerate(q["options"])]
                out += ["", f"Answer: **{chr(97 + q['answer'])}**. {q['why']}"]
            elif q["kind"] == "num":
                out.append(f"Answer: **{q['answer']:g} {q['unit']}**. {q['why']}")
            else:
                out.append(f"Model answer: {q['answer']}")
                out.append("")
                out.append("Key points: " + "; ".join(q["keys"]))
            out += ["", f"Source: `{q['source']}`", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path} ({len(QUESTIONS)} questions)")


def selftest():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for q in QUESTIONS:
        assert q["phase"] in PHASES, q["id"]
        assert q["level"] in LEVELS, q["id"]
        assert q["kind"] in ("mcq", "num", "open"), q["id"]
        assert q["source"], q["id"]
        if q["kind"] == "mcq":
            assert 0 <= q["answer"] < len(q["options"]), q["id"]
            assert grade_mcq(q, chr(97 + q["answer"])) == 1.0
            assert grade_mcq(q, str(q["answer"] + 1)) == 1.0
            assert grade_mcq(q, chr(97 + (q["answer"] + 1) % len(q["options"]))) == 0.0
        if q["kind"] == "num":
            assert grade_num(q, f"{q['answer']} {q['unit']}") == 1.0, q["id"]
            assert grade_num(q, str(q["answer"] * 2 + 1)) == 0.0, q["id"]
        if q["kind"] == "open":
            assert q["keys"], q["id"]
    for ph in PHASES:
        assert by_phase(ph), f"no questions for {ph}"
    assert parse_number("1,935 GB/s") == 1935.0 and parse_number("~8.3ms") == 8.3
    counts = {lv: sum(q["level"] == lv for q in QUESTIONS) for lv in LEVELS}
    print(f"selftest OK: {len(QUESTIONS)} questions across {len(PHASES)} phases {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", help="P0 .. P13")
    ap.add_argument("--level", choices=LEVELS + ["all"], default="all")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--mock", type=int, metavar="N")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--export", metavar="FILE")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.export:
        return export(a.export)
    p = load()
    try:
        if a.phase:
            ph = a.phase.upper()
            if ph not in PHASES:
                sys.exit(f"Unknown phase {a.phase}. Use one of: {', '.join(PHASES)}")
            run(by_phase(ph, a.level), f"{ph} {PHASES[ph]} ({a.level})", p)
        elif a.review:
            due = [q for q in QUESTIONS if is_due(p, q)]
            random.shuffle(due)
            run(due, "Review (due today)", p)
        elif a.mock:
            mock(p, a.mock)
        elif a.stats:
            stats(p)
        else:
            status(p)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Progress is saved.")


if __name__ == "__main__":
    main()
