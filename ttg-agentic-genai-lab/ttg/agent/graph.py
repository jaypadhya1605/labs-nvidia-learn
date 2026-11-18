"""The Disruption Concierge as a LangGraph state machine.

    START -> think --(action)--> act --> think ... --(final)--> END
               |--(bad JSON: feedback, retry <= TTG_MAX_RETRIES)--^
Each `think` is ONE LLM call on the append-only conversation; each `act` is ONE MCP tool call. The task result
carries what the bake-off and the ledger need: per-task latency, LLM calls, tokens (prompt / cached / completion),
retries, tool time, and the policy-compliance score from ttg.world.score.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ttg import world
from ttg.agent.config import Settings
from ttg.agent.llm import LLMClient
from ttg.agent.prompts import build_system_prompt, initial_messages, render_observation
from ttg.agent.protocol import ProtocolError, parse_step
from ttg.agent.tools import ToolBox


class TaskState(TypedDict, total=False):
    scenario: dict
    messages: list
    actions: list
    final: dict | None
    pending: dict | None
    steps: int
    retries: int
    calls: list
    tool_time_s: float
    error: str | None
    done: bool


class Agent:
    def __init__(self, settings: Settings, llm: LLMClient, tools: ToolBox):
        self.s, self.llm, self.tools = settings.validate(), llm, tools
        self.system_prompt = build_system_prompt(world.TOOL_SPECS, settings.prompt_variant, settings.thoughts)
        g = StateGraph(TaskState)
        g.add_node("think", self.think)
        g.add_node("act", self.act)
        g.add_edge(START, "think")
        g.add_conditional_edges("think", self._after_think, {"act": "act", "think": "think", "end": END})
        g.add_conditional_edges("act", self._after_act, {"think": "think", "end": END})
        self.graph = g.compile()

    # ------------------------------------------------------------------ nodes
    async def think(self, state: TaskState) -> dict:
        messages_in = state["messages"]
        if self.s.bust_cache:      # drill 1: a fresh timestamp at the top of EVERY request kills prefix reuse
            messages_in = [{"role": "system", "content": build_system_prompt(
                world.TOOL_SPECS, self.s.prompt_variant, self.s.thoughts, bust_cache=True)}] + messages_in[1:]
        try:
            res = await self.llm.complete(messages_in)
        except Exception as e:                                   # engine down, timeout, 4xx/5xx
            return {"error": f"llm: {type(e).__name__}: {str(e)[:200]}", "done": True}
        calls = state["calls"] + [res.as_dict()]
        messages = state["messages"] + [{"role": "assistant", "content": res.text.strip() or "{}"}]
        try:
            step = parse_step(res.text)
        except ProtocolError as e:
            retries = state["retries"] + 1
            if retries > self.s.max_retries:
                return {"calls": calls, "messages": messages, "retries": retries, "error": f"protocol: {e}",
                        "done": True}
            return {"calls": calls, "retries": retries, "pending": None,
                    "messages": messages + [{"role": "user", "content": f"ERROR: {e}"}]}
        if step.kind == "final":
            return {"calls": calls, "messages": messages, "final": step.final, "pending": None, "done": True}
        return {"calls": calls, "messages": messages, "pending": {"action": step.action, "args": step.args}}

    async def act(self, state: TaskState) -> dict:
        p = state["pending"]
        t0 = time.perf_counter()
        try:
            result = await self.tools.call(p["action"], p["args"])
        except Exception as e:                                   # a tool bug is an observation, not a crash
            result = {"error": f"tool {p['action']} failed: {type(e).__name__}: {str(e)[:200]}"}
        dt = time.perf_counter() - t0
        actions = state["actions"] + [{"action": p["action"], "args": p["args"], "result": result,
                                       "ms": round(dt * 1000, 2)}]
        out = {"actions": actions, "steps": state["steps"] + 1, "pending": None,
               "tool_time_s": state["tool_time_s"] + dt,
               "messages": state["messages"] + [{"role": "user", "content": render_observation(p["action"], result)}]}
        if out["steps"] >= self.s.max_steps:
            out.update(error=f"max_steps ({self.s.max_steps}) reached without a final answer", done=True)
        return out

    @staticmethod
    def _after_think(state: TaskState) -> str:
        if state.get("done"):
            return "end"
        return "act" if state.get("pending") else "think"

    @staticmethod
    def _after_act(state: TaskState) -> str:
        return "end" if state.get("done") else "think"

    # ------------------------------------------------------------------ one task
    def recursion_limit(self) -> int:
        """Upper bound on graph super-steps: every tool step is think + act, every retry is one more think
        (retries are counted per task), plus the final think. A task can never hit LangGraph's limit first."""
        return 2 * self.s.max_steps + self.s.max_retries + 10

    async def run(self, scenario: dict) -> dict:
        """Run ONE task to completion. Never raises: anything unexpected becomes result["error"] (the batch goes on)."""
        state: TaskState = {"scenario": scenario, "messages": initial_messages(scenario, self.system_prompt),
                            "actions": [],
                            "final": None, "pending": None, "steps": 0, "retries": 0, "calls": [],
                            "tool_time_s": 0.0, "error": None, "done": False}
        t0 = time.perf_counter()
        out = dict(state)
        try:
            async for snap in self.graph.astream(state, config={"recursion_limit": self.recursion_limit()},
                                                 stream_mode="values"):
                out = snap                                     # keep the latest state: partial results survive
        except Exception as e:                                 # GraphRecursionError, a bug, a cancelled tool call
            out = {**out, "error": f"agent: {type(e).__name__}: {str(e)[:200]}"}
        latency = time.perf_counter() - t0
        try:
            success, violations = world.score(scenario, out["actions"], out.get("final"))
        except Exception as e:                                 # unscorable scenario (e.g. bad fields from an API)
            success, violations = False, [f"error:score:{type(e).__name__}"]
        if out.get("error"):
            success, violations = False, violations + [f"error:{out['error'].split(':')[0]}"]
        calls = out["calls"]

        def total(key):
            vals = [c[key] for c in calls if c.get(key) is not None]
            return sum(vals) if vals else None

        ttft = calls[0].get("ttft_s") if calls else None
        result = {
            "scenario_id": scenario.get("scenario_id"), "success": success, "violations": violations,
            "steps": out["steps"], "llm_calls": len(calls), "tool_calls": len(out["actions"]),
            "retries": out["retries"], "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"), "cached_tokens": total("cached_tokens"),
            "latency_s": round(latency, 4), "first_ttft_s": round(ttft, 4) if ttft is not None else None,
            "llm_time_s": round(sum(c.get("latency_s") or 0.0 for c in calls), 4),
            "tool_time_s": round(out["tool_time_s"], 4), "final": out.get("final"), "error": out.get("error"),
        }
        if self.s.trace_dir:
            Path(self.s.trace_dir).mkdir(parents=True, exist_ok=True)
            trace = {**result, "settings": self.s.public(), "calls": calls, "actions": out["actions"],
                     "messages": out["messages"]}
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(scenario.get("scenario_id") or "task"))[:64]
            path = Path(self.s.trace_dir) / f"{safe}-{uuid.uuid4().hex[:6]}.json"      # never a path from input
            path.write_text(json.dumps(trace, indent=1, ensure_ascii=False), encoding="utf-8")
            result["trace"] = str(path)
        return result
