"""Plover API unit tests: routing, prompts, schema/faithfulness, passthrough - no GPU, no network."""
import importlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient


def good_summary(case):
    return {"request_type": case["urgency"], "requested_service": case["service"], "cpt_codes": case["cpt"],
            "icd10_codes": case["icd10"], "clinical_summary": "58-year-old with radiculopathy.",
            "criteria_evidence": ["six weeks of PT"], "missing_information": [],
            "recommendation": "approve_for_review", "confidence": 0.8}


def make_app(monkeypatch, handler, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import plover_api.main as main
    main = importlib.reload(main)
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(main, "http_client", lambda: httpx.AsyncClient(transport=transport, timeout=5))
    return main, TestClient(main.app)


def test_routing_is_sticky_and_weighted():
    from plover_api.routing import choose_variant
    assert choose_variant("abc", 0) == "stable"
    assert choose_variant("abc", 100) == "canary"
    assert choose_variant("same-id", 30) == choose_variant("same-id", 30)
    share = sum(choose_variant(f"r{i}", 20) == "canary" for i in range(5000)) / 5000
    assert 0.17 < share < 0.23


def test_prompt_modes_differ_in_size():
    from plover_api.prompts import build_messages
    long = build_messages("case", "long-prompt")
    short = build_messages("case", "adapter")
    assert long[0]["role"] == "system" and len(long) == 2
    assert len(short) == 1 and len(json.dumps(short)) < len(json.dumps(long)) / 5


def test_schema_and_faithfulness():
    from plover_api import schema, synth
    case = synth.make_case(3)
    obj = good_summary(case)
    assert schema.validate(obj) == []
    assert schema.unfaithful_codes(obj, case["text"]) == []
    bad = dict(obj, cpt_codes=["99999"], recommendation="maybe")
    assert any("recommendation" in e for e in schema.validate(bad))
    assert schema.unfaithful_codes(bad, case["text"]) == ["99999"]
    assert schema.extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert schema.extract_json("no json here") is None


def test_summarize_stable_long_prompt(monkeypatch):
    from plover_api import synth
    case = synth.make_case(7)
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["body"] = body
        content = json.dumps(good_summary(case))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 812}})

    _, client = make_app(monkeypatch, handler, BASE_MODEL="meta/llama-3.1-8b-instruct", STABLE_MODEL="", CANARY_MODEL="", CANARY_WEIGHT=0)
    r = client.post("/v1/prior-auth/summarize", json={"case_text": case["text"], "request_id": "x1"})
    assert r.status_code == 200
    out = r.json()
    assert out["variant"] == "stable" and out["schema_valid"] is True
    assert seen["body"]["model"] == "meta/llama-3.1-8b-instruct"
    assert seen["body"]["messages"][0]["role"] == "system"            # long style guide
    assert "plover_api_schema_valid_total" in client.get("/metrics").text


def test_summarize_canary_adapter_short_prompt(monkeypatch):
    from plover_api import synth
    case = synth.make_case(8)
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["body"] = body
        content = json.dumps(dict(good_summary(case), icd10_codes=["Z99.9"]))   # hallucinated code
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 140}})

    _, client = make_app(monkeypatch, handler, BASE_MODEL="meta/llama-3.1-8b-instruct",
                         STABLE_MODEL="plover-style-v1", CANARY_MODEL="plover-style-v2", CANARY_WEIGHT=100)
    out = client.post("/v1/prior-auth/summarize", json={"case_text": case["text"]}).json()
    assert out["variant"] == "canary" and out["model"] == "plover-style-v2"
    assert len(seen["body"]["messages"]) == 1                         # adapter mode: no style guide
    assert out["schema_valid"] is False and any("Z99.9" in e for e in out["errors"])


def test_upstream_failure_is_502(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)
    _, client = make_app(monkeypatch, handler)
    r = client.post("/v1/prior-auth/summarize", json={"case_text": "x"})
    assert r.status_code == 502


def test_chat_passthrough_sets_default_model(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    _, client = make_app(monkeypatch, handler, BASE_MODEL="meta/llama-3.1-8b-instruct")
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200 and seen["body"]["model"] == "meta/llama-3.1-8b-instruct"


def test_synth_is_deterministic():
    from plover_api import synth
    assert synth.make_case(42) == synth.make_case(42)
    assert len({c["id"] for c in synth.cases(50)}) == 50


def test_extra_body_is_merged_into_summarize(monkeypatch):
    """Add-on: Nemotron 3.5 needs per-request knobs (reasoning off) - configured by env, not code."""
    from plover_api import synth
    case = synth.make_case(11)
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(good_summary(case))}}], "usage": {}})

    _, client = make_app(monkeypatch, handler, LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}',
                         CANARY_MODEL="", CANARY_WEIGHT=0)
    r = client.post("/v1/prior-auth/summarize", json={"case_text": case["text"], "request_id": "x"})
    assert r.status_code == 200 and r.json()["schema_valid"]
    assert seen["chat_template_kwargs"] == {"enable_thinking": False} and seen["temperature"] == 0.0
