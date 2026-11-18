"""M8 VLM tooling: exact image-token math, receipt rendering, JSON parsing, extract_fields over a mocked HTTP endpoint.

Run: python3 -m pytest -q tests/test_vlm.py   (offline: no GPU, no VLM, no torch; Pillow + httpx from requirements)
"""
import base64
import inspect
import io
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vlm import extract_receipt as ex  # noqa: E402
from vlm import vlm_tokens as vt  # noqa: E402

TRUTH = {"receipt_ref": "R-S0003", "merchant": "Runway Noodle Bar", "date": "2026-01-14", "category": "meal",
         "items": [{"desc": "Veggie bowl", "usd": 14.25}, {"desc": "Salad", "usd": 11.25},
                   {"desc": "Chicken sandwich", "usd": 13.5}, {"desc": "Coffee", "usd": 4.1}],
         "tax_usd": 2.59, "total_usd": 45.69, "party_size": 1, "reimbursement_usd": 45.69}


# ------------------------------------------------------------------------------------------------ Qwen2.5-VL tokens
@pytest.mark.parametrize("w,h,tokens", [
    (3840, 2160, 10_549),          # 4K frame: 3836 x 2156 -> 137 x 77
    (3024, 4032, 15_552),          # 12 MP phone photo fits under the checkpoint max_pixels
    (640, 662, 552),               # the Tern receipt render: 644 x 672 -> 23 x 24
    (1008, 1344, 1_728),
    (1000, 200, 252),
    (560, 560, 400),
])
def test_qwen_tokens_checkpoint_max_pixels(w, h, tokens):
    q = vt.qwen_image_tokens(w, h)
    assert q["tokens"] == tokens
    assert q["prompt_tokens"] == tokens + 2                          # <|vision_start|> ... <|vision_end|>
    assert q["vit_patches"] == 4 * tokens                            # 2 x 2 patch merge
    assert q["resized_w"] % 28 == 0 and q["resized_h"] % 28 == 0


def test_qwen_max_pixels_cap_class_default():
    q = vt.qwen_image_tokens(3024, 4032, max_pixels=vt.QWEN_CLASS_MAX_PIXELS)
    assert (q["resized_w"], q["resized_h"]) == (840, 1148)
    assert q["tokens"] == 1_230 <= vt.QWEN_CLASS_MAX_PIXELS // 784


def test_qwen_min_pixels_and_python_rounding():
    assert vt.qwen_smart_resize(20, 20) == (56, 56)                  # upscaled to min_pixels 3,136
    assert vt.qwen_smart_resize(42, 42) == (56, 56)                  # 1.5 -> 2 (half to even)
    assert vt.qwen_smart_resize(70, 70) == (56, 56)                  # 2.5 -> 2 (half to even), not 84
    with pytest.raises(ValueError):
        vt.qwen_smart_resize(10, 2100)


def test_qwen_invariants_over_many_sizes():
    for w in range(30, 5000, 397):
        for h in range(30, 5000, 433):
            if max(w, h) / min(w, h) > 200:
                continue
            q = vt.qwen_image_tokens(w, h)
            px = q["resized_w"] * q["resized_h"]
            assert vt.QWEN_MIN_PIXELS <= px <= vt.QWEN_MAX_PIXELS
            assert q["tokens"] * 784 == px


def test_preprocessor_values_match_the_bundled_config():
    cfg = json.loads((REPO / "archmath" / "configs" / "qwen2.5-vl-7b.json").read_text())
    assert cfg["_preprocessor"]["min_pixels"] == vt.QWEN_MIN_PIXELS
    assert cfg["_preprocessor"]["max_pixels"] == vt.QWEN_MAX_PIXELS
    assert cfg["vision_config"]["patch_size"] * cfg["vision_config"]["spatial_merge_size"] == vt.QWEN_FACTOR


# ------------------------------------------------------------------------------------------------ Llama 3.2 Vision
@pytest.mark.parametrize("w,h,canvas,tiles", [
    (640, 662, (1120, 1120), 4),       # small upscale wins: 2 x 2 tiles for a receipt barely bigger than one tile
    (433, 448, (560, 560), 1),
    (560, 560, (560, 560), 1),
    (1170, 2532, (560, 1680), 3),      # tall screenshot: the largest DOWNscale, then the least area
    (3024, 4032, (1120, 1120), 4),
    (1000, 200, (1120, 560), 2),       # ties at the same scale -> smallest canvas
])
def test_mllama_canvas(w, h, canvas, tiles):
    t = vt.mllama_image_tokens(w, h)
    assert (t["canvas_w"], t["canvas_h"]) == canvas
    assert t["tiles"] == tiles
    assert t["vision_tokens"] == tiles * 1601
    assert t["text_tokens"] == 1


def test_mllama_constants_match_the_bundled_config():
    cfg = json.loads((REPO / "archmath" / "configs" / "llama-3.2-11b-vision.json").read_text())
    v = cfg["vision_config"]
    assert vt.MLLAMA_TOKENS_PER_TILE == (v["image_size"] // v["patch_size"]) ** 2 + 1 == 1601
    assert vt.MLLAMA_MAX_TILES == v["max_num_tiles"]
    assert {tuple(x) for x in v["supported_aspect_ratios"]} == set(vt.mllama_aspect_ratios(4))
    assert len(cfg["text_config"]["cross_attention_layers"]) == 8


def test_kv_costs_and_generic_tiles():
    c = vt.cost(3840, 2160)
    assert c["qwen"]["image_kv_mib"] == pytest.approx(10_549 * 56 / 1024, rel=1e-6)    # 56 KiB per token
    assert c["mllama"]["xattn_kv_mib"] == pytest.approx(4 * 1601 * 32 / 1024, rel=1e-6)  # 32 KiB per vision token
    assert c["qwen"]["ttft_floor_ms"] > vt.cost(1008, 1344)["qwen"]["ttft_floor_ms"] * 5
    assert vt.tiled_tokens(1000, 800) == {"tiles": 7, "tokens": 1792}
    assert vt.tiled_tokens(300, 300) == {"tiles": 1, "tokens": 256}


def test_vlm_tokens_cli(capsys):
    assert vt.main(["--table", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == len(vt.EXAMPLES)
    assert vt.main(["--size", "3024x4032", "--max-pixels", "1003520"]) == 0
    assert "1,230" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ receipts
def test_render_receipt_is_deterministic_and_has_no_card_digits():
    pytest.importorskip("PIL")
    from vlm import make_receipts as mr
    a = mr.render_receipt(TRUTH)
    b = mr.render_receipt(TRUTH)
    assert a.mode == "L" and a.width == mr.WIDTH
    assert a.tobytes() == b.tobytes()
    text = "\n".join(t for t, _ in mr.receipt_lines(TRUTH))
    assert "45.69" in text and "TOTAL USD" in text and "RUNWAY NOODLE BAR" in text
    assert not re.search(r"\d{12,19}", text.replace(" ", ""))                # PCI: no card numbers on the image
    noisy = mr.render_receipt(TRUTH, noise=0.8)
    assert noisy.size != a.size                                              # rotated with expand
    assert mr.render_receipt(TRUTH, scale=2).width == 2 * mr.WIDTH


def test_receipt_lines_fit_the_measured_width():
    pytest.importorskip("PIL")
    from vlm import make_receipts as mr
    lines = mr.receipt_lines(TRUTH, cols=40)
    assert all(len(t) <= 40 for t, st in lines if st in ("body", "total"))
    assert any(t.endswith("45.69") and t.startswith("TOTAL USD") for t, _ in lines)


def test_generate_from_the_world(tmp_path):
    pytest.importorskip("PIL")
    world = pytest.importorskip("ttg.world")
    from vlm import make_receipts as mr
    rows = mr.generate(n=20, seed=7, out=str(tmp_path))
    expected = [s for s in world.make_scenarios(20, seed=7) if s.get("receipt")]
    assert len(rows) == len(expected) > 0
    lines = (tmp_path / "labels.jsonl").read_text().splitlines()
    assert len(lines) == len(rows)
    first = json.loads(lines[0])
    assert (tmp_path / f"{first['receipt_ref']}.png").exists()
    assert first["reimbursement_usd"] == round(min(first["total_usd"], 50.0 * first["party_size"]), 2)
    assert all(r["bytes"] < 60_000 for r in rows)                            # small enough to send inline


# ------------------------------------------------------------------------------------------------ JSON parsing
@pytest.mark.parametrize("reply", [
    '{"merchant": "A", "total_usd": 1.5}',
    '```json\n{"merchant": "A", "total_usd": 1.5}\n```',
    'Here is the receipt:\n{"merchant": "A", "total_usd": 1.5}\nLet me know if you need more.',
    '{"merchant": "A", "total_usd": 1.5,}',
    "{'merchant': 'A', 'total_usd': 1.5}",
    '<think>the total is 1.5</think>{"merchant": "A", "total_usd": 1.5}',
])
def test_parse_json_reply_tolerates_wrappers(reply):
    obj = ex.parse_json_reply(reply)
    assert obj["merchant"] == "A" and obj["total_usd"] == 1.5


def test_parse_json_reply_braces_inside_strings_and_failures():
    assert ex.parse_json_reply('{"merchant": "Cafe {Gate 4}", "items": []} trailing')["merchant"] == "Cafe {Gate 4}"
    for bad in ("", "no json here", '{"merchant": "A", "total_usd": 1.5'):
        with pytest.raises(ValueError):
            ex.parse_json_reply(bad)


def test_normalize_types_dates_categories():
    out = ex.normalize({"merchant": " Metro Cab Co. ", "date": "01/14/2026", "category": "Cab ride",
                        "items": [{"name": "Taxi fare", "price": "$37.04"}, {"description": "Tip", "amount": 5.56}],
                        "total": "USD 42.60"})
    assert list(out) == list(ex.FIELDS)
    assert out == {"merchant": "Metro Cab Co.", "date": "2026-01-14", "total_usd": 42.6, "category": "taxi",
                   "items": [{"desc": "Taxi fare", "usd": 37.04}, {"desc": "Tip", "usd": 5.56}]}
    assert ex.normalize({"date": "14 Jan 2026", "category": "Restaurant"})["date"] == "2026-01-14"
    assert ex.normalize({"category": "Restaurant"})["category"] == "meal"
    assert ex.normalize({"category": "parking"})["category"] == "other"
    assert ex.normalize({"total_usd": "1,234.50"})["total_usd"] == 1234.5


def test_score_one_and_the_reimbursement_cap():
    perfect = {k: TRUTH[k] for k in ex.FIELDS}
    s = ex.score_one(perfect, TRUTH)
    assert all(s.values())
    capped = dict(TRUTH, total_usd=72.75, reimbursement_usd=50.0)
    wrong_total = dict(perfect, total_usd=71.75)                 # misread, but above the $50 cap for one traveler
    s = ex.score_one(wrong_total, capped)
    assert not s["total_usd"] and s["reimbursement"] and not s["all_fields"]


# ------------------------------------------------------------------------------------------------ extract_fields over HTTP
httpx = pytest.importorskip("httpx")


def png_file(tmp_path, w=64, h=48):
    pil = pytest.importorskip("PIL.Image")
    p = tmp_path / "R-S0003.png"
    pil.new("L", (w, h), 255).save(p)
    return str(p)


def reply(content, prompt_tokens=700, completion_tokens=60):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}],
                                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


def test_extract_fields_contract_signature():
    sig = inspect.signature(ex.extract_fields)
    params = list(sig.parameters.values())
    assert [p.name for p in params[:5]] == ["image_path", "base_url", "model", "api_key", "timeout_s"]
    assert sig.parameters["api_key"].default == "not-used" and sig.parameters["timeout_s"].default == 60
    assert all(p.kind == p.KEYWORD_ONLY for p in params[5:])


def test_extract_fields_sends_a_data_url_and_parses_the_reply(tmp_path):
    path = png_file(tmp_path)
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = body
        return reply("```json\n" + json.dumps({k: TRUTH[k] for k in ex.FIELDS}) + "\n```")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = ex.extract_fields(path, "http://vlm.test/v1", "ttg-llm", "test-key", 5, http_client=client)
    assert out == {k: TRUTH[k] for k in ex.FIELDS}
    assert seen["url"] == "http://vlm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "ttg-llm" and body["temperature"] == 0
    part = body["messages"][1]["content"][0]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_fix_it_retry_then_success(tmp_path):
    path = png_file(tmp_path)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return reply("The total is 45.69 dollars.")
        return reply(json.dumps({k: TRUTH[k] for k in ex.FIELDS}))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = ex.extract_receipt(path, "http://vlm.test/v1", "m", http_client=client)
    assert out["retries"] == 1 and out["fields"]["total_usd"] == 45.69
    assert out["usage"]["prompt_tokens"] == 1400                              # both calls counted
    assert "not one valid JSON object" in calls[1]["messages"][-1]["content"]


def test_errors_surface(tmp_path):
    path = png_file(tmp_path)
    c500 = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ex.extract_fields(path, "http://vlm.test/v1", "m", http_client=c500)
    junk = httpx.Client(transport=httpx.MockTransport(lambda r: reply("no idea")))
    with pytest.raises(ValueError, match="could not parse"):
        ex.extract_fields(path, "http://vlm.test/v1", "m", http_client=junk)


def test_img_tag_mode_and_structured_outputs(tmp_path):
    path = png_file(tmp_path)
    body = ex.build_request(path, "meta/llama-3.2-11b-vision-instruct", image_mode="img_tag", structured="json_schema")
    assert isinstance(body["messages"][1]["content"], str)
    assert '<img src="data:image/png;base64,' in body["messages"][1]["content"]
    assert body["response_format"]["type"] == "json_schema"
    assert ex.chat_url("https://h.test/v1/chat/completions") == "https://h.test/v1/chat/completions"
    nonce = ex.build_request(path, "m", nonce="abc123")["messages"][0]["content"]
    assert nonce.startswith("Request abc123. ")


def test_run_eval_and_summary(tmp_path):
    pytest.importorskip("ttg.world")
    from vlm import make_receipts as mr
    rows = mr.generate(n=3, seed=7, out=str(tmp_path))           # R-S0001 + R-S0003 in the golden set
    assert {r["receipt_ref"] for r in rows} >= {"R-S0001", "R-S0003"}
    truth3 = next(r for r in rows if r["receipt_ref"] == "R-S0003")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: reply(json.dumps({k: truth3[k] for k in ex.FIELDS}))))
    results = ex.run_eval(str(tmp_path), rows, "http://vlm.test/v1", "m", "k", http_client=client)
    s = ex.summarize(results)
    assert s["n"] == len(rows) and s["errors"] == 0
    assert s["accuracy"]["all_fields"] == pytest.approx(1 / len(rows))
    assert s["mean_prompt_tokens"] == 700


# ------------------------------------------------------------------------------------------------ the sweep
def test_stream_once_reads_sse():
    from vlm import vlm_sweep as sw
    chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}}]},
              {"choices": [{"delta": {"content": '{"total_usd": '}}]},
              {"choices": [{"delta": {"content": "45.69}"}}]},
              {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 9}}]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    client = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})))
    ttft, lat, text, usage = sw.stream_once(client, "http://vlm.test/v1/chat/completions", {}, {}, 5)
    assert text == '{"total_usd": 45.69}' and usage["prompt_tokens"] == 900
    assert ttft is not None and ttft <= lat


def test_sweep_token_check_against_a_simulated_qwen_server():
    pytest.importorskip("ttg.world")
    pil = pytest.importorskip("PIL.Image")
    from vlm import vlm_sweep as sw

    def handler(request):
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        n = 120                                                    # text tokens of the prompt
        if isinstance(content, list):
            url = next(p["image_url"]["url"] for p in content if p["type"] == "image_url")
            img = pil.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            n += vt.qwen_image_tokens(*img.size)["prompt_tokens"]  # what vLLM's processor would expand
        answer = json.dumps({"merchant": "Runway Noodle Bar", "date": "2026-01-14", "category": "meal",
                             "items": [], "total_usd": 45.69})
        if body.get("stream"):
            sse = (f"data: {json.dumps({'choices': [{'delta': {'content': answer}}]})}\n\n"
                   f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': n, 'completion_tokens': 20}})}\n\n"
                   "data: [DONE]\n\n")
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return reply(answer, prompt_tokens=n)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rows = sw.sweep("http://vlm.test/v1", "m", "k", [448, 896], 2, "R-S0003", 3.0, 0.3, False, d, 5,
                        vt.QWEN_MAX_PIXELS, http_client=client)
    assert [r["measured_image_tokens"] for r in rows] == [r["predicted_image_tokens"] for r in rows]
    assert rows[0]["predicted_image_tokens"] < rows[1]["predicted_image_tokens"]
    assert all(r["total_ok_rate"] == 1.0 for r in rows)


def test_sweep_dry_run(capsys):
    pytest.importorskip("ttg.world")
    pytest.importorskip("PIL")
    from vlm import vlm_sweep as sw
    assert sw.main(["--dry-run", "--sizes", "448,896"]) == 0
    out = capsys.readouterr().out
    assert "| 448 |" in out and "| 896 |" in out


# ------------------------------------------------------------------------------------------------ agent pass comparison
def test_agent_receipts_compare():
    from vlm import agent_receipts as ar
    oracle = [{"scenario_id": "S0005", "success": True, "violations": [], "latency_s": 2.0, "tool_time_s": 0.1},
              {"scenario_id": "S0006", "success": True, "violations": [], "latency_s": 3.0, "tool_time_s": 0.1}]
    vlm = [{"scenario_id": "S0005", "success": True, "violations": [], "latency_s": 2.5, "tool_time_s": 0.6},
           {"scenario_id": "S0006", "success": False, "latency_s": 3.5, "tool_time_s": 0.7,
            "violations": ["wrong_amount:reimbursement:42.0!=42.6"]}]
    c = ar.compare(oracle, vlm)
    assert c["oracle"]["success_rate"] == 1.0 and c["vlm"]["success"] == 1
    assert c["vlm"]["reimbursement_violations"] == 1
    assert [f["scenario_id"] for f in c["lost_by_vlm"]] == ["S0006"]


def test_receipt_scenarios_are_the_read_receipt_path():
    world = pytest.importorskip("ttg.world")
    from vlm import agent_receipts as ar
    scs = ar.receipt_scenarios()
    assert len(scs) == 35
    for s in scs[:5]:
        assert any(step.get("action") == "read_receipt" for step in world.expected_plan(s))


def test_vlm_tools_run_as_plain_files_and_take_an_api_key():
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for script, args in (("vlm/vlm_tokens.py", ["--size", "640x662"]), ("vlm/extract_receipt.py", ["--help"]),
                         ("vlm/vlm_sweep.py", ["--help"]), ("vlm/make_receipts.py", ["--help"]),
                         ("vlm/agent_receipts.py", ["--help"])):
        out = subprocess.run([sys.executable, script, *args], cwd=REPO, env=env, capture_output=True, text=True)
        assert out.returncode == 0, (script, out.stderr[-500:])
        if script.endswith(("extract_receipt.py", "vlm_sweep.py")):
            assert "--api-key API_KEY" in out.stdout and "--api-key-env" in out.stdout
