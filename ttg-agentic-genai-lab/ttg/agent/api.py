"""The agent as a service (M9): what the Helm chart deploys next to the engine and the MCP servers.

  uvicorn ttg.agent.api:app --port 8080
  curl -s localhost:8080/v1/disruptions -H 'content-type: application/json' -d '{"scenario_id": "S0003"}'
POST /v1/disruptions  {"scenario_id": "S0003"} (golden set) or {"scenario": {public fields}} -> task result
GET  /healthz         liveness + which engine/model/MCP mode
GET  /metrics         Prometheus text: tasks, successes, violations, latency histogram, LLM calls, tokens
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ttg import world
from ttg.agent.config import Settings
from ttg.agent.graph import Agent
from ttg.agent.llm import LLMClient
from ttg.agent.tools import ToolBox

BUCKETS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 60]
STATE: dict = {}
M = {"tasks": 0, "success": 0, "errors": 0, "llm_calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
     "completion_tokens": 0, "latency_sum": 0.0, "buckets": [0] * (len(BUCKETS) + 1)}
GOLDEN = {s["scenario_id"]: s for s in world.make_scenarios(200, seed=7)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings().validate()
    llm = LLMClient(settings)
    async with ToolBox(settings) as tools:
        STATE.update(settings=settings, agent=Agent(settings, llm, tools))
        yield
    await llm.aclose()


app = FastAPI(title="Tern Disruption Concierge", lifespan=lifespan)


class ScenarioIn(BaseModel):
    """The public facts of one disruption - validated before any GPU time is spent (bad input = 422, not a late 500).
    Unknown fields are ignored: ground truth never enters the agent through the API."""
    scenario_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    traveler_id: str = Field(pattern=r"^T\d{5}$")
    flight_no: str = Field(pattern=r"^TN\d{3}$")
    party_size: int = Field(ge=1, le=6)                 # every cancelled flight has a rebooking option for 6
    guest_name: str = Field(default="", max_length=120)
    receipt_ref: str | None = Field(default=None, pattern=r"^R-[A-Za-z0-9-]{1,40}$")
    locale: str = Field(default="en-US", max_length=16)
    guest_message: str = Field(default="", max_length=4000)


class DisruptionIn(BaseModel):
    scenario_id: str | None = Field(default=None, max_length=64)
    scenario: ScenarioIn | None = None


@app.post("/v1/disruptions")
async def disruption(req: DisruptionIn):
    if req.scenario:
        sc = req.scenario.model_dump()
    elif req.scenario_id in GOLDEN:
        sc = GOLDEN[req.scenario_id]
    else:
        raise HTTPException(404, "unknown scenario_id (golden set is S0001-S0200)")
    r = await STATE["agent"].run(sc)
    M["tasks"] += 1
    M["success"] += int(r["success"])
    M["errors"] += int(bool(r.get("error")))
    for k in ("llm_calls", "prompt_tokens", "cached_tokens", "completion_tokens"):
        M[k] += r.get(k) or 0
    M["latency_sum"] += r["latency_s"]
    M["buckets"][next((i for i, b in enumerate(BUCKETS) if r["latency_s"] <= b), len(BUCKETS))] += 1
    return r


@app.get("/healthz")
async def healthz():
    s: Settings = STATE["settings"]
    return {"status": "ok", "model": s.model, "base_url": s.base_url, "mcp_mode": s.mcp_mode,
            "prompt_variant": s.prompt_variant}


@app.get("/metrics")
async def metrics():
    lines = []
    for name, key in [("ttg_tasks_total", "tasks"), ("ttg_task_success_total", "success"),
                      ("ttg_task_errors_total", "errors"), ("ttg_llm_calls_total", "llm_calls"),
                      ("ttg_prompt_tokens_total", "prompt_tokens"), ("ttg_cached_tokens_total", "cached_tokens"),
                      ("ttg_completion_tokens_total", "completion_tokens")]:
        lines += [f"# TYPE {name} counter", f"{name} {M[key]}"]
    lines.append("# TYPE ttg_task_latency_seconds histogram")
    acc = 0
    for b, c in zip(BUCKETS + ["+Inf"], M["buckets"]):
        acc += c
        lines.append(f'ttg_task_latency_seconds_bucket{{le="{b}"}} {acc}')
    lines += [f"ttg_task_latency_seconds_sum {M['latency_sum']:.4f}", f"ttg_task_latency_seconds_count {M['tasks']}"]
    return PlainTextResponse("\n".join(lines) + "\n")
