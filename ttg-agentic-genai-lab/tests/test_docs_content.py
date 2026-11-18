"""Content checks for the FORK C area: modules end with Checkpoint + Say it, the interview drill keeps its verbatim
questions, the customer brief's numbers match the golden set, and nothing real leaks into fictional material."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULE_PAGES = ["docs/00-recall.md", "docs/01-customer-brief.md", "docs/02-architecture-and-adrs.md",
                "docs/09-results-and-exec-readout.md", "docs/10-interview-drill.md", "drills/README.md",
                "workshop/README.md", "sales/README.md", "collateral/README.md", "sales/mutual-action-plan.md"]
MINE = sorted({*REPO.glob("docs/0[0-2]-*.md"), *REPO.glob("docs/09-*.md"), *REPO.glob("docs/10-*.md"),
               *REPO.glob("workshop/*.md"), *REPO.glob("collateral/*.md"), *REPO.glob("sales/*.md"),
               *REPO.glob("drills/*"), REPO / "workshop" / "build_notebook.py"})
# Real organizations from the author's own work must never appear in the fictional lab material.
BANNED = ["Molina", "CommonSpirit", "UPMC", "Providence", "General Motors", "@gmail.com"]


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", MODULE_PAGES)
def test_module_ends_with_checkpoint_and_say_it(rel):
    tail = read(rel)[-1500:]
    assert "**Checkpoint:**" in tail and "**Say it:**" in tail, f"{rel} must end with Checkpoint + Say it"


def test_interview_drill_has_the_verbatim_questions():
    text = read("docs/10-interview-drill.md")
    assert '"Describe the structure of the NVL72 as best as you can."' in text
    assert '"What was your main goal when optimizing an ML model?"' in text
    assert "JR2019035" in text and "Auto customers" in text and "Hospitality and travel" in text
    assert "replace them with YOUR measured numbers" in text
    numbered = re.findall(r"^\*\*(\d+)\. ", text, flags=re.M)
    assert len(numbered) >= 25 and numbered == [str(i) for i in range(1, len(numbered) + 1)]
    assert text.count("- Proof:") >= len(numbered)


def test_customer_brief_matches_the_golden_set():
    world = pytest.importorskip("ttg.world")
    scs = world.make_scenarios(200, seed=7)
    outs = [world.expected_outcome(s) for s in scs]
    tools = [len(o["plan"]) for o in outs]
    status = Counter(s["status"] for s in scs)
    tiers = Counter(s["tier"] for s in scs)
    expected = {
        "Cancelled / delayed": f"{status['cancelled']} / {status['delayed']}",
        "Controllable cause (crew, mechanical)": str(sum(bool(s["controllable"]) for s in scs)),
        "Overnight": str(sum(bool(o["overnight"]) for o in outs)),
        "Hotel bookings owed": str(sum(bool(o["hotel"]) for o in outs)),
        "Receipts uploaded": str(sum(bool(s["receipt_ref"]) for s in scs)),
        "Need human approval (a voucher > $300)": str(sum(bool(o["final"]["needs_human_approval"]) for o in outs)),
        "Tiers: basic / silver / gold / platinum": " / ".join(str(tiers[t]) for t in
                                                             ("basic", "silver", "gold", "platinum")),
        "Tool calls per task": f"{min(tools)} to {max(tools)}",
    }
    text = read("docs/01-customer-brief.md")
    for label, value in expected.items():
        assert f"| {label} | {value} |" in text, f"docs/01 row '{label}' should read {value}"


def test_capacity_plan_uses_the_golden_set_eu_share():
    world = pytest.importorskip("ttg.world")
    eu = sum(bool(s["eu_resident"]) for s in world.make_scenarios(200, seed=7))
    assert f"{eu} EU residents out of 200" in read("docs/09-results-and-exec-readout.md")


def test_charter_has_ten_criteria_each_with_a_tool():
    rows = [ln for ln in read("sales/poc-charter.md").splitlines() if re.match(r"^\| \d+ \|", ln)]
    assert [ln.split("|")[1].strip() for ln in rows] == [str(i) for i in range(1, 11)]
    assert all(ln.rstrip("|").split("|")[-1].strip() for ln in rows)


def test_mutual_action_plan_dates_are_real_and_in_order():
    dates = re.findall(r"^\| (Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\d{4}-\d{2}-\d{2}) \|", read("sales/mutual-action-plan.md"),
                       flags=re.M)
    assert len(dates) >= 10
    parsed = [dt.date.fromisoformat(d) for _, d in dates]
    assert parsed == sorted(parsed)
    for day, d in dates:
        assert dt.date.fromisoformat(d).strftime("%a") == day, f"{d} is not a {day}"


@pytest.mark.parametrize("path", MINE, ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_real_names_or_contact_details(path):
    if path.is_dir() or path.suffix not in (".md", ".py", ".sh", ".jinja"):
        pytest.skip("not text")
    text = path.read_text(encoding="utf-8")
    found = [b for b in BANNED if b in text]
    assert not found, f"{path.relative_to(REPO)} mentions {found}"
