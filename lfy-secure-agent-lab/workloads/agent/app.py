"""SOC agent API: POST /triage {"alert": {...}, "mode": "defended|undefended"} + Prometheus metrics."""
import os

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

import agent
import soc_tools

BUNDLE = os.environ.get("PLAYBOOK_DIR", "/work/playbooks")
COMMUNITY = os.environ.get("COMMUNITY_DIR", "/community")
PB_DEFENDED = soc_tools.Playbooks([BUNDLE])
PB_ALL = soc_tools.Playbooks([BUNDLE, COMMUNITY])

TRIAGES = Counter("agent_triage_total", "Triage runs", ["mode", "classification"])
LAT = Histogram("agent_triage_latency_seconds", "Triage latency", ["mode"], buckets=(1, 2, 5, 10, 20, 40, 60, 120, 240))
TOOLS = Counter("agent_tool_calls_total", "Tool calls", ["tool", "mode"])
DENIALS = Counter("agent_policy_denials_total", "Tool policy denials", ["mode"])
INJ = Counter("agent_injection_detected_total", "Prompt injection detected in telemetry", ["mode"])
LLM = Counter("agent_llm_calls_total", "LLM calls", ["mode"])

app = FastAPI(title="Lanternfly SOC agent")


class TriageRequest(BaseModel):
    alert: dict
    mode: str = "defended"


@app.post("/triage")
def triage(req: TriageRequest):
    mode = "undefended" if req.mode == "undefended" else "defended"
    out = agent.triage(req.alert, mode, PB_DEFENDED, PB_ALL)
    TRIAGES.labels(mode, str(out["verdict"].get("classification"))).inc()
    LAT.labels(mode).observe(out["latency_s"])
    LLM.labels(mode).inc(out["llm_calls"])
    for t in out["tool_calls"]:
        TOOLS.labels(str(t), mode).inc()
    if out["policy_denials"]:
        DENIALS.labels(mode).inc(out["policy_denials"])
    if out["injection_detected"]:
        INJ.labels(mode).inc()
    return out


@app.get("/healthz")
def healthz():
    return {"ok": True, "playbooks_defended": len(PB_DEFENDED.docs), "playbooks_all": len(PB_ALL.docs)}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
