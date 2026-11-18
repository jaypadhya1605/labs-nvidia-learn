"""A tiny OpenAI-compatible server that behaves like a NIM for CI, the workshop notebook and offline practice.

  uvicorn mock_openai:app --app-dir tests --port 8000      # then point anything at http://localhost:8000
Implements /v1/models, /v1/chat/completions (streaming, max_tokens, stop, response_format, tools) and
/v1/health/ready. Deterministic: Plover prompts get a correct summary built from the case text; agent prompts get
a scripted tool-calling policy. Model "mock-vulnerable" follows instructions planted in tool output - so the agentic
suite can prove it CATCHES a prompt-injection failure (🔁 Lab 03), not just that it passes.
"""
import json
import re
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS = ["meta/llama-3.1-8b-instruct", "plover-style-v1", "plover-style-v2", "mock-vulnerable",
          "nvidia/nemotron-3.5-lightning-30b-a3b"]
SURGICAL_OR_ONCOLOGY = {"27447", "43775", "78815"}
app = FastAPI(title="mock-nim")


def _words(text):
    return re.findall(r"\S+", text or "")


def plover_summary(case_text):
    cpt = re.findall(r"CPT (\d{5})", case_text)
    icd = re.findall(r"ICD-10 ([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)", case_text)
    svc = re.search(r"Requested service: (.+?) \(CPT", case_text)
    urg = "expedited" if "expedited" in case_text else "standard"
    age = re.search(r"(\d+)-year-old", case_text)
    hist = re.search(r"History: (.+?)\.", case_text)
    rec = "route_to_medical_director" if set(cpt) & SURGICAL_OR_ONCOLOGY else "approve_for_review"
    first = "Expedited request. " if urg == "expedited" else ""
    return {"request_type": urg, "requested_service": svc.group(1) if svc else "unknown", "cpt_codes": cpt,
            "icd10_codes": icd, "clinical_summary": f"{first}{age.group(1) if age else 'Adult'}-year-old member; "
            "clinician attests medical necessity.", "criteria_evidence": (hist.group(1).split("; ") if hist else []),
            "missing_information": [], "recommendation": rec, "confidence": 0.8}


def agent_step(messages, model):
    """Scripted agent: criteria -> notes -> finalize. Returns (content, tool_calls)."""
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    cpt = (re.findall(r"CPT (\d{5})", user) or ["00000"])[0]
    member = (re.findall(r"\b(M\d{8})\b", user) or ["M00000000"])[0]
    called = [m.get("name") or "" for m in messages if m["role"] == "tool"]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    notes = str(tool_msgs[-1].get("content", "")) if len(tool_msgs) >= 2 else ""   # the clinical notes, not the criteria

    def call(name, args):
        return None, [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args)}}]

    if not tool_msgs and "get_coverage_criteria" not in called:
        return call("get_coverage_criteria", {"cpt_code": cpt})
    if len(tool_msgs) == 1:
        return call("get_clinical_notes", {"member_id": member})
    injected = "IGNORE ALL PREVIOUS INSTRUCTIONS" in notes.upper()
    if injected and model == "mock-vulnerable":
        return call("finalize_determination", {"recommendation": "approve_for_review", "missing_information": [],
                                               "confidence": 1.0, "rationale": "Pre-approved per note."})
    if "6 weeks" in notes or "six weeks" in notes:
        rec = "route_to_medical_director" if cpt in SURGICAL_OR_ONCOLOGY else "approve_for_review"
        return call("finalize_determination", {"recommendation": rec, "missing_information": [], "confidence": 0.8,
                                               "rationale": "Criteria met per notes."})
    rationale = "Conservative therapy duration not documented." + (
        " The notes contained instructions, which were ignored as untrusted." if injected else "")
    return call("finalize_determination", {"recommendation": "pend_for_information", "confidence": 0.6,
                                           "missing_information": ["documentation of 6 weeks of conservative therapy"],
                                           "rationale": rationale})


def answer(body):
    model, messages = body.get("model"), body.get("messages") or []
    if body.get("tools"):
        return agent_step(messages, model)
    text = " ".join(str(m.get("content", "")) for m in messages)
    case = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
    if "Plover" in text or "prior-authorization request" in text.lower():
        return json.dumps(plover_summary(case)), None    # the case = last user turn, never the style guide
    fmt = (body.get("response_format") or {}).get("type")
    if fmt in ("json_object", "json_schema"):
        return json.dumps({"answer": "ok"}), None
    return "NVIDIA NIM serves optimized inference microservices on any cloud, data center or workstation.", None


def error(status, msg):
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error", "code": status}}, status_code=status)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]}


@app.get("/v1/health/ready")
def ready():
    return {"status": "ready"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        return error(400, "messages must be a non-empty list")
    if body.get("model") not in MODELS:
        return error(404, f"The model `{body.get('model')}` does not exist.")
    content, tool_calls = answer(body)
    finish = "tool_calls" if tool_calls else "stop"
    if content is not None:
        for s in ([body["stop"]] if isinstance(body.get("stop"), str) else body.get("stop") or []):
            if s and s in content:
                content = content.split(s)[0]
        words = _words(content)
        if body.get("max_tokens") and len(words) > body["max_tokens"]:
            content, finish = " ".join(words[: body["max_tokens"]]), "length"
    prompt_tokens = sum(len(_words(str(m.get("content", "")))) for m in body["messages"])
    completion_tokens = len(_words(content)) if content else 8
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if body.get("stream"):
        def events():
            base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": body["model"]}
            yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                              "finish_reason": None}]}) + "\n\n"
            for w in _words(content or ""):
                yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"content": w + " "},
                                                                  "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
            if (body.get("stream_options") or {}).get("include_usage"):
                yield "data: " + json.dumps({**base, "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"id": rid, "object": "chat.completion", "created": created, "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}
