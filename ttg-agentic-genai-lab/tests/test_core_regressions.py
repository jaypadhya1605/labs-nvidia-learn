"""Regression tests for defects found in the independent review of the core (each test names the failure it pins)."""
import asyncio
import json
import math

import pytest

from ttg import world
from ttg.agent.config import Settings
from ttg.agent.graph import Agent
from ttg.agent.llm import LLMResult
from ttg.agent.protocol import format_action, format_final
from ttg.agent.tools import ToolBox, _to_dict

GOLDEN = world.make_scenarios(200, seed=7)


# ---------------------------------------------------------------- world + scorer
def test_cheapest_hotel_is_never_a_tie():
    """A tie at the cheapest rate made the oracle's id tie-break an unwritten rule (7 golden travelers)."""
    for airport in world.AIRPORTS:
        rates = [h["nightly_rate_usd"] for h in world._hotels(airport)]
        assert len(rates) == len(set(rates)), (airport, rates)
    for sc in GOLDEN:
        hotel = world.expected_outcome(sc)["hotel"]
        if hotel:
            avail = [h for h in world._hotels(sc["origin"]) if h["rooms_available"] >= math.ceil(sc["party_size"] / 2)]
            assert [h["hotel_id"] for h in avail if h["nightly_rate_usd"] == hotel["nightly_rate_usd"]] == \
                [hotel["hotel_id"]]


def _oracle_actions(sc):
    return [{"action": st["action"], "args": dict(st["args"]),
             "result": world.execute_tool(st["action"], st["args"])}
            for st in world.expected_plan(sc) if "action" in st]


def test_scorer_checks_hotel_nights():
    sc = next(s for s in GOLDEN if world.expected_outcome(s)["hotel"])
    acts = _oracle_actions(sc)
    for a in acts:
        if a["action"] == "book_hotel":
            a["args"]["nights"] = 3
            a["result"] = world.execute_tool("book_hotel", a["args"])
    ok, v = world.score(sc, acts, world.expected_outcome(sc)["final"])
    assert not ok and "wrong_nights" in v


def test_failed_call_with_a_bad_traveler_id_is_not_a_violation():
    """The protocol tells the model to fix its arguments; a corrected retry must not fail the task."""
    sc = next(s for s in GOLDEN if world.expected_outcome(s)["vouchers"])
    acts = _oracle_actions(sc)
    i = next(i for i, a in enumerate(acts) if a["action"] == "issue_voucher")
    bad = dict(acts[i], args={**acts[i]["args"], "traveler_id": sc["traveler_id"] + "X"})
    bad["result"] = world.execute_tool("issue_voucher", bad["args"])
    assert "error" in bad["result"]
    assert world.score(sc, acts[:i] + [bad] + acts[i:], world.expected_outcome(sc)["final"]) == (True, [])
    sloppy = [dict(a, args={**a["args"], "traveler_id": " " + a["args"]["traveler_id"].lower()})
              if "traveler_id" in a["args"] else a for a in acts]
    assert world.score(sc, sloppy, world.expected_outcome(sc)["final"]) == (True, [])


@pytest.mark.parametrize("tool,args", [
    ("rebooking_options", {"flight_no": "TN100", "party_size": float("nan")}),
    ("rebooking_options", {"flight_no": "TN100", "party_size": float("inf")}),
    ("rebooking_options", {"flight_no": "TN100", "party_size": 10 ** 400}),
    ("rebooking_options", {"flight_no": "TN100", "party_size": True}),
    ("issue_voucher", {"traveler_id": "T12345", "kind": "meal", "amount_usd": float("nan")}),
    ("issue_voucher", {"traveler_id": "T12345", "kind": "meal", "amount_usd": float("inf")}),
])
def test_tools_return_errors_for_non_finite_numbers(tool, args):
    assert "error" in world.execute_tool(tool, args)


def test_mcp_layer_errors_become_error_observations():
    assert _to_dict([{"type": "text", "text": "Error executing tool book_hotel: 1 validation error"}])["error"]


# ---------------------------------------------------------------- the agent never takes the batch down
class ScriptedLLM:
    """Replays fixed replies; after the script, repeats the last one."""

    def __init__(self, replies):
        self.replies, self.i = list(replies), 0

    async def complete(self, messages):
        text = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return LLMResult(text=text, ttft_s=0.01, latency_s=0.02, prompt_tokens=100, completion_tokens=10,
                         cached_tokens=None, finish_reason="stop")


def _agent(llm, **kw):
    s = Settings(mcp_mode="inproc", **kw)
    return Agent(s, llm, ToolBox(s))


@pytest.mark.parametrize("reply", [
    '{"thought": "x", "action": ["flight_status"], "args": {}}',
    '{"thought": "x", "final": {"message": "m", "rebooked_flight": "TN1", "vouchers": 30}}',
    '{"thought": "x", "action": "rebooking_options", "args": {"flight_no": "TN1", "party_size": NaN}}',
])
def test_hostile_replies_cost_retries_not_the_batch(reply):
    sc = GOLDEN[2]
    r = asyncio.run(_agent(ScriptedLLM([reply])).run(sc))
    assert not r["success"] and r["error"].startswith("protocol") and r["retries"] == Settings().max_retries + 1


def test_many_retries_never_hit_the_graph_recursion_limit():
    r = asyncio.run(_agent(ScriptedLLM(["not json"]), max_retries=100).run(GOLDEN[0]))
    assert r["error"].startswith("protocol") and r["retries"] == 101 and r["llm_calls"] == 101


def test_a_looping_agent_with_retries_still_stops_at_max_steps():
    sc = GOLDEN[0]
    loop = format_action("flight_status", {"flight_no": sc["flight_no"]}, "again")
    r = asyncio.run(_agent(ScriptedLLM(["bad", "bad", loop]), max_steps=5).run(sc))
    assert r["steps"] == 5 and r["retries"] == 2 and r["error"].startswith("max_steps")


def test_trace_file_name_never_comes_from_input(tmp_path):
    sc = {**world.public_view(GOLDEN[0]), "scenario_id": "../../escape"}
    final = format_final(world.expected_outcome(GOLDEN[0])["final"], "done")
    r = asyncio.run(_agent(ScriptedLLM([final]), trace_dir=str(tmp_path / "traces")).run(sc))
    assert (tmp_path / "traces") in __import__("pathlib").Path(r["trace"]).parents
    assert json.loads(open(r["trace"]).read())["scenario_id"] == "../../escape"


def test_unreported_cached_tokens_stay_unknown_per_task():
    sc = GOLDEN[0]
    final = format_final(world.expected_outcome(sc)["final"], "done")
    r = asyncio.run(_agent(ScriptedLLM([final])).run(sc))
    assert r["cached_tokens"] is None and r["prompt_tokens"] == 100


# ---------------------------------------------------------------- API input validation
@pytest.fixture
def client(agent_env):
    from fastapi.testclient import TestClient

    from ttg.agent import api
    with TestClient(api.app) as c:
        yield c


@pytest.mark.parametrize("patch", [{"scenario_id": "../../x"}, {"scenario_id": "/abs/path"},
                                   {"party_size": "two"}, {"party_size": 12}, {"flight_no": "XX1"},
                                   {"traveler_id": "12345"}, {"receipt_ref": "../etc/passwd"}])
def test_api_rejects_bad_scenarios_before_spending_gpu_time(client, patch):
    pub = {**world.public_view(GOLDEN[3]), **patch}
    r = client.post("/v1/disruptions", json={"scenario": pub})
    assert r.status_code == 422, r.text


def test_api_ignores_smuggled_ground_truth(client):
    pub = {**world.public_view(GOLDEN[3]), "tier": "platinum", "cause": "crew"}
    r = client.post("/v1/disruptions", json={"scenario": pub})
    assert r.status_code == 200 and r.json()["success"]


def test_bench_levels_are_sliced_by_position(agent_env, tmp_path):
    from ttg.bench import agent_bench
    out = tmp_path / "dup.json"
    assert agent_bench.main(["--label", "mock-dup", "--levels", "2,2", "--n", "4", "--warmup", "0",
                             "--out", str(out)]) == 0
    ids = [t["scenario_id"] for t in json.loads(out.read_text())["tasks"]]
    assert len(ids) == len(set(ids)) == 8
