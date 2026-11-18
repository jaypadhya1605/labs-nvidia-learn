"""OpenAI-API conformance for ANY endpoint you put in front of a customer (NIM on 5 platforms, trtllm-serve,
managed endpoints). Each test is a promise the Plover API or a customer SDK relies on.
  pytest tests/conformance                                   # mock (CI)
  pytest tests/conformance --endpoint http://localhost:8000  # kubectl port-forward svc/plover-llm 8000
"""
import json

import pytest

from plover_api import prompts, schema, synth

pytestmark = pytest.mark.conformance


def chat(llm, model, extra_body, **kw):
    body = {"model": model, "messages": [{"role": "user", "content": "In one sentence, what is NVIDIA NIM?"}],
            "max_tokens": 64, "temperature": 0.0, **extra_body, **kw}
    return llm.post("/v1/chat/completions", json=body)


def test_models_endpoint_lists_the_model(llm, model):
    r = llm.get("/v1/models")
    assert r.status_code == 200
    assert model in [m["id"] for m in r.json()["data"]]


def test_basic_chat_shape_and_usage(llm, model, extra_body):
    r = chat(llm, model, extra_body)
    assert r.status_code == 200, r.text
    d = r.json()
    msg = d["choices"][0]["message"]
    assert msg["role"] == "assistant" and (msg.get("content") or "").strip()
    assert d["choices"][0]["finish_reason"] in ("stop", "length")
    assert d["usage"]["prompt_tokens"] > 0 and d["usage"]["completion_tokens"] > 0


def test_max_tokens_is_enforced(llm, model, extra_body):
    d = chat(llm, model, extra_body, max_tokens=5, messages=[{"role": "user", "content": "Count from 1 to 100, comma separated."}]).json()
    assert d["usage"]["completion_tokens"] <= 5
    assert d["choices"][0]["finish_reason"] == "length"


def test_stop_sequence(llm, model, extra_body):
    d = chat(llm, model, extra_body, stop=["NIM"], max_tokens=64,
             messages=[{"role": "user", "content": "Repeat exactly: NVIDIA NIM serves models."}]).json()
    assert "NIM" not in (d["choices"][0]["message"].get("content") or "")


def test_streaming_sse(llm, model, extra_body):
    body = {"model": model, "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 32,
            "stream": True, "stream_options": {"include_usage": True}, **extra_body}
    chunks, text, done = [], "", False
    with llm.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                done = True
                break
            c = json.loads(payload)
            chunks.append(c)
            for ch in c.get("choices", []):
                text += (ch.get("delta") or {}).get("content") or ""
    assert done, "stream must end with data: [DONE] (AIPerf and SDKs rely on it)"
    assert text.strip(), "streamed deltas must carry content"
    assert any(c.get("usage") for c in chunks), "stream_options.include_usage -> a final usage chunk (AIPerf uses it)"


def test_unknown_model_is_a_clean_4xx(llm, extra_body):
    r = chat(llm, "no-such-model", extra_body)
    assert 400 <= r.status_code < 500
    assert "error" in r.json() or "detail" in r.json() or "message" in r.json()


def test_json_object_mode(llm, model, extra_body):
    d = chat(llm, model, extra_body, response_format={"type": "json_object"}, max_tokens=128,
             messages=[{"role": "user", "content": 'Return a JSON object with key "answer" set to "ok".'}]).json()
    assert isinstance(json.loads(d["choices"][0]["message"]["content"]), dict)


def test_guided_json_schema_for_plover(llm, model, extra_body):
    """Structured output is how Plover turns 'mostly valid JSON' into 'always valid JSON'."""
    case = synth.make_case(42)
    js = {"type": "object", "required": sorted(schema.REQUIRED), "properties": {
        "request_type": {"enum": ["standard", "expedited"]}, "requested_service": {"type": "string"},
        "cpt_codes": {"type": "array", "items": {"type": "string"}}, "icd10_codes": {"type": "array", "items": {"type": "string"}},
        "clinical_summary": {"type": "string"}, "criteria_evidence": {"type": "array", "items": {"type": "string"}},
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"enum": sorted(schema.RECOMMENDATIONS)}, "confidence": {"type": "number"}}}
    d = chat(llm, model, extra_body, messages=prompts.build_messages(case["text"], "long-prompt"), max_tokens=512,
             response_format={"type": "json_schema", "json_schema": {"name": "plover_summary", "schema": js}}).json()
    obj = json.loads(d["choices"][0]["message"]["content"])
    assert schema.validate(obj) == []
    assert schema.unfaithful_codes(obj, case["text"]) == [], "guided decoding fixes the SHAPE, not hallucinated codes"


def test_lora_adapter_is_a_model(request, llm, extra_body):
    adapter = request.config.getoption("--adapter")
    if not adapter:
        pytest.skip("pass --adapter plover-style-v1 to check a LoRA adapter")
    assert adapter in [m["id"] for m in llm.get("/v1/models").json()["data"]], "NIM lists loaded adapters as models"
    case = synth.make_case(5)
    d = chat(llm, adapter, extra_body, messages=prompts.build_messages(case["text"], "adapter"), max_tokens=400).json()
    assert schema.validate(schema.extract_json(d["choices"][0]["message"]["content"])) == []
