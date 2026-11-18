"""The LangGraph agent end to end against the mock engine: success, metrics, retries, guards and the drills."""
import asyncio
import json

import pytest

from ttg import world
from ttg.agent import run as run_cli
from ttg.agent.config import Settings
from ttg.agent.graph import Agent
from ttg.agent.llm import LLMClient
from ttg.agent.tools import ToolBox

GOLDEN = world.make_scenarios(200, seed=7)


async def _run(scenarios, settings=None, concurrency=4):
    s = (settings or Settings()).validate()
    llm = LLMClient(s)
    async with ToolBox(s) as tb:
        agent = Agent(s, llm, tb)
        sem = asyncio.Semaphore(concurrency)

        async def one(sc):
            async with sem:
                return await agent.run(sc)
        out = await asyncio.gather(*(one(sc) for sc in scenarios))
    await llm.aclose()
    return out


def test_agent_solves_the_golden_slice(agent_env):
    res = asyncio.run(_run(GOLDEN[:24]))
    assert all(r["success"] for r in res), [(r["scenario_id"], r["violations"]) for r in res if not r["success"]]
    for r, sc in zip(res, GOLDEN[:24]):
        plan = world.expected_plan(sc)
        assert r["tool_calls"] == len(plan) - 1 and r["llm_calls"] == len(plan)
        assert r["prompt_tokens"] > r["cached_tokens"] > 0 and r["completion_tokens"] > 0
        assert r["latency_s"] > 0 and r["first_ttft_s"] is not None and r["retries"] == 0
    assert sum(r["cached_tokens"] for r in res) / sum(r["prompt_tokens"] for r in res) > 0.8


def test_trace_has_the_whole_conversation(agent_env):
    r = asyncio.run(_run(GOLDEN[4:5]))[0]
    t = json.load(open(r["trace"]))
    assert t["messages"][0]["role"] == "system" and len(t["calls"]) == r["llm_calls"]
    assert all(m["content"].startswith("OBSERVATION") for m in t["messages"][3::2][:-1])


def test_invalid_json_is_repaired_with_feedback(agent_env, monkeypatch):
    monkeypatch.setenv("TTG_MODEL", "ttg-llm+invalid_json_once")
    res = asyncio.run(_run(GOLDEN[:4]))
    assert all(r["success"] and r["retries"] == 1 for r in res)


def test_over_compensation_is_caught_by_the_scorer(agent_env, monkeypatch):
    monkeypatch.setenv("TTG_MODEL", "mock-wrong")
    owed = [s for s in GOLDEN if world.expected_outcome(s)["vouchers"]][:6]     # something to over-pay
    res = asyncio.run(_run(owed))
    assert not any(r["success"] for r in res)
    assert all(any(v.startswith("wrong_amount") for v in r["violations"]) for r in res)


def test_max_steps_guard_stops_a_looping_agent(agent_env, monkeypatch):
    monkeypatch.setenv("TTG_MODEL", "ttg-llm+no_final")
    monkeypatch.setenv("TTG_MAX_STEPS", "6")
    r = asyncio.run(_run(GOLDEN[:1]))[0]
    assert not r["success"] and r["steps"] == 6 and "error:max_steps" in " ".join(r["violations"])


def test_drill_cache_bust_zeroes_prefix_reuse(agent_env, monkeypatch):
    monkeypatch.setenv("TTG_BUST_CACHE", "1")
    res = asyncio.run(_run(GOLDEN[:6]))
    assert all(r["success"] for r in res) and sum(r["cached_tokens"] for r in res) == 0


def test_drill_verbose_thoughts(agent_env, monkeypatch):
    plain = asyncio.run(_run(GOLDEN[:1]))[0]
    monkeypatch.setenv("TTG_THOUGHTS", "verbose")
    cut = asyncio.run(_run(GOLDEN[:1]))[0]                       # default max_tokens 256: the JSON is cut off
    assert not cut["success"] and cut["error"].startswith("protocol")
    monkeypatch.setenv("TTG_MAX_TOKENS", "1024")
    long = asyncio.run(_run(GOLDEN[:1]))[0]                      # room to finish: correct, but 5x+ the tokens
    assert long["success"] and long["completion_tokens"] > 5 * plain["completion_tokens"]


def test_engine_down_is_an_error_not_a_crash(agent_env, monkeypatch):
    monkeypatch.setenv("TTG_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("TTG_TIMEOUT_S", "2")
    r = asyncio.run(_run(GOLDEN[:1]))[0]
    assert not r["success"] and r["error"].startswith("llm:")


def test_end_to_end_over_stdio_mcp(agent_env, monkeypatch):
    monkeypatch.setenv("TTG_MCP_MODE", "stdio")
    res = asyncio.run(_run(GOLDEN[10:14]))
    assert all(r["success"] for r in res)


def test_cli_prints_a_summary(agent_env, capsys):
    assert run_cli.main(["--n", "3"]) == 0
    out = capsys.readouterr().out
    assert "3/3 policy-compliant" in out


def test_settings_validation():
    with pytest.raises(AssertionError):
        Settings(prompt_variant="tiny").validate()
    assert Settings(api_key="nvapi-secret").public()["api_key"] == "set"
