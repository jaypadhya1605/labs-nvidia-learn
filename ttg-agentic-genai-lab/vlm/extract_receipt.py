#!/usr/bin/env python3
"""M8 - read a receipt image with a VLM behind any OpenAI-compatible endpoint, and score it against the truth.

extract_fields() is the function the MCP guest server calls for read_receipt when TTG_VLM_BASE_URL is set
(ttg/mcp/guest_server.py). It returns exactly {"merchant", "date", "total_usd", "category", "items"}.
The model may wrap JSON in prose or code fences; the parser finds the object, and one fix-it retry handles the rest
(🔁 M3: the agent's JSON action protocol does the same).

Usage (from the repo root):
  python3 -m vlm.make_receipts                                            # data/receipts/*.png + labels.jsonl
  engines/launch.sh vllm vlm                                              # Qwen2.5-VL-7B on vLLM as "ttg-llm"
  python3 -m vlm.extract_receipt --dir data/receipts --base-url http://localhost:8000/v1 --model ttg-llm \\
      --label vllm-qwen25vl --concurrency 8
  python3 -m vlm.extract_receipt --dir data/receipts --n 10 --base-url https://integrate.api.nvidia.com/v1 \\
      --model meta/llama-3.2-11b-vision-instruct --api-key-env NVIDIA_API_KEY --label hosted-llama32v
      (or --api-key "$NVIDIA_API_KEY")
  python3 -m vlm.extract_receipt --image data/receipts/R-S0003.png --dry-run     # show the request, send nothing
Env defaults: TTG_VLM_BASE_URL, TTG_VLM_MODEL, TTG_VLM_API_KEY (keys are read by NAME and never printed).
Hosted endpoints: synthetic receipts only (these are). Check the model page for the exact base URL and image format;
--image-mode img_tag sends <img src="data:..."> inside the text for endpoints without image_url parts.
"""
import argparse
import ast
import base64
import json
import os
import re
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

FIELDS = ("merchant", "date", "total_usd", "category", "items")
SYSTEM = (
    "You read receipt images for Tern Travel Group reimbursements. Reply with ONE JSON object and nothing else:\n"
    '{"merchant": string, "date": "YYYY-MM-DD", "category": "meal" | "taxi" | "other", '
    '"items": [{"desc": string, "usd": number}], "total_usd": number}\n'
    "Rules: copy amounts exactly as printed. total_usd is the TOTAL line (it includes tax and tip). items are the "
    "purchased lines only - not SUBTOTAL, TAX or TOTAL. category: meal = food or drink, taxi = a cab or car ride. "
    "If a value is not printed, use null. Never invent values."
)
USER = "Extract the receipt fields as JSON."
SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": "string"},
        "date": {"type": "string"},
        "category": {"type": "string", "enum": ["meal", "taxi", "other"]},
        "items": {"type": "array", "items": {"type": "object", "properties": {"desc": {"type": "string"},
                                                                            "usd": {"type": "number"}},
                                             "required": ["desc", "usd"]}},
        "total_usd": {"type": "number"},
    },
    "required": ["merchant", "date", "category", "items", "total_usd"],
}
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d", "%d.%m.%Y",
                "%m-%d-%Y", "%d-%b-%Y")
CATEGORY_WORDS = {"meal": ("meal", "food", "restaurant", "dining", "drink", "cafe", "coffee", "lunch", "dinner",
                           "breakfast", "snack", "grocer"),
                  "taxi": ("taxi", "cab", "ride", "car", "transport", "uber", "lyft", "shuttle", "limo")}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


# ----------------------------------------------------------------------------------------------- request
def image_data_url(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        return f"data:{MIME.get(ext, 'image/png')};base64,{base64.b64encode(f.read()).decode()}"


def chat_url(base_url):
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def build_messages(data_url, image_mode="image_url", nonce="", system=SYSTEM, user=USER):
    sys_text = (f"Request {nonce}. " if nonce else "") + system
    if image_mode == "img_tag":                       # text-only clients: the image rides inside the string
        content = f'{user} <img src="{data_url}" />'
    elif image_mode == "none":                        # same prompt without the image (token-count baseline)
        content = user
    else:
        content = [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": user}]
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": content}]


def build_request(image_path, model, max_tokens=400, structured="none", image_mode="image_url", nonce="",
                  stream=False):
    data_url = image_data_url(image_path) if image_mode != "none" else ""
    body = {"model": model, "messages": build_messages(data_url, image_mode, nonce), "temperature": 0,
            "max_tokens": max_tokens}
    if structured == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif structured == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "receipt", "schema": SCHEMA}}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


# ----------------------------------------------------------------------------------------------- parsing
def parse_json_reply(text):
    """The first JSON object in a model reply: tolerates code fences, prose around it, <think> blocks,
    trailing commas and Python-style quotes. Raises ValueError with a short reason."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty reply")
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, end, in_str, esc = 0, -1, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("unterminated JSON object (max_tokens too small?)")
    blob = t[start:end + 1]
    for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            obj = json.loads(attempt)
            break
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        try:
            obj = ast.literal_eval(re.sub(r"\bnull\b", "None", re.sub(r"\btrue\b", "True",
                                                                        re.sub(r"\bfalse\b", "False", blob))))
        except (ValueError, SyntaxError):
            raise ValueError("reply contains braces but not valid JSON") from None
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def to_float(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(v))
    return round(float(m.group(0).replace(",", "")), 2) if m else None


def to_date(v):
    if not v:
        return None
    s = str(v).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return m.group(0) if m else s


def to_category(v):
    s = str(v or "").strip().lower()
    for cat, words in CATEGORY_WORDS.items():
        if s == cat or any(w in s for w in words):
            return cat
    return "other"


def normalize(obj):
    """Model JSON -> exactly the five fields, typed: strings, ISO date, floats rounded to cents."""
    items = []
    for it in obj.get("items") or []:
        if not isinstance(it, dict):
            continue
        desc = next((it[k] for k in ("desc", "description", "name", "item") if it.get(k)), "")
        usd = next((to_float(it[k]) for k in ("usd", "price", "amount", "total") if to_float(it.get(k)) is not None),
                   None)
        if desc or usd is not None:
            items.append({"desc": str(desc).strip(), "usd": usd})
    merchant = obj.get("merchant") or obj.get("store") or obj.get("vendor")
    total = obj.get("total_usd", obj.get("total"))
    return {"merchant": str(merchant).strip() if merchant else None, "date": to_date(obj.get("date")),
            "total_usd": to_float(total), "category": to_category(obj.get("category")), "items": items}


# ----------------------------------------------------------------------------------------------- calling
def _client(http_client, timeout_s):
    if http_client is not None:
        return http_client, False
    import httpx                                       # lazy: pure helpers above import nothing heavy
    return httpx.Client(timeout=timeout_s), True


def extract_receipt(image_path, base_url, model, api_key="not-used", timeout_s=60.0, *, http_client=None,
                    structured="none", image_mode="image_url", max_tokens=400, retries=1, nonce=""):
    """One receipt -> {"fields", "raw", "usage", "latency_s", "retries"}. Raises on HTTP errors or unparseable
    output after `retries` fix-it attempts."""
    client, owned = _client(http_client, timeout_s)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = build_request(image_path, model, max_tokens, structured, image_mode, nonce)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.perf_counter()
    try:
        for attempt in range(retries + 1):
            r = client.post(chat_url(base_url), json=body, headers=headers, timeout=timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} from {chat_url(base_url)}: {r.text[:300]}")
            data = r.json()
            for k in usage:
                usage[k] += int((data.get("usage") or {}).get(k) or 0)
            raw = data["choices"][0]["message"].get("content") or ""
            try:
                fields = normalize(parse_json_reply(raw))
                return {"fields": fields, "raw": raw, "usage": usage, "latency_s": time.perf_counter() - t0,
                        "retries": attempt}
            except ValueError as e:
                if attempt == retries:
                    raise ValueError(f"could not parse the VLM reply ({e}): {raw[:200]!r}") from None
                body["messages"] = body["messages"] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That was not one valid JSON object ({e}). Reply with the JSON "
                                                f"object only, keys: {', '.join(FIELDS)}."}]
    finally:
        if owned:
            client.close()


def extract_fields(image_path: str, base_url: str, model: str, api_key: str = "not-used", timeout_s: float = 60,
                   *, http_client=None, structured: str = "none", image_mode: str = "image_url") -> dict:
    """Read one receipt image -> {"merchant", "date", "total_usd", "category", "items"} (the read_receipt contract)."""
    return extract_receipt(image_path, base_url, model, api_key, timeout_s, http_client=http_client,
                           structured=structured, image_mode=image_mode)["fields"]


# ----------------------------------------------------------------------------------------------- scoring
def _norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def score_one(pred, truth):
    total = pred.get("total_usd")
    items = pred.get("items") or []
    item_sum = round(sum(i["usd"] or 0 for i in items), 2)
    truth_sum = round(sum(i["usd"] for i in truth["items"]), 2)
    pay = round(min(total, 50.0 * truth["party_size"]), 2) if total is not None else None
    s = {"merchant": _norm_name(pred.get("merchant")) == _norm_name(truth["merchant"]),
         "date": pred.get("date") == truth["date"],
         "category": pred.get("category") == truth["category"],
         "total_usd": total is not None and abs(total - truth["total_usd"]) <= 0.011,
         "items": len(items) == len(truth["items"]) and abs(item_sum - truth_sum) <= 0.011,
         "reimbursement": pay is not None and abs(pay - truth["reimbursement_usd"]) <= 0.011}
    s["all_fields"] = all(s[k] for k in ("merchant", "date", "category", "total_usd", "items"))
    return s


def pct(values, q):
    v = sorted(values)
    if not v:
        return 0.0
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return v[k]


def load_labels(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(directory, labels, base_url, model, api_key, n=0, concurrency=4, structured="none",
             image_mode="image_url", timeout_s=60.0, http_client=None):
    rows = labels[:n] if n else labels
    client, owned = _client(http_client, timeout_s)

    def one(truth):
        path = os.path.join(directory, truth.get("image") or f"{truth['receipt_ref']}.png")
        try:
            out = extract_receipt(path, base_url, model, api_key, timeout_s, http_client=client,
                                  structured=structured, image_mode=image_mode, nonce=uuid.uuid4().hex[:8])
            return {"receipt_ref": truth["receipt_ref"], "pred": out["fields"], "usage": out["usage"],
                    "latency_s": out["latency_s"], "retries": out["retries"], "score": score_one(out["fields"], truth),
                    "error": None}
        except Exception as e:                                      # noqa: BLE001 - score the failure, keep going
            return {"receipt_ref": truth["receipt_ref"], "pred": None, "usage": {}, "latency_s": None, "retries": 0,
                    "score": {k: False for k in ("merchant", "date", "category", "total_usd", "items",
                                                 "reimbursement", "all_fields")},
                    "error": f"{type(e).__name__}: {e}"[:300]}
    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = list(pool.map(one, rows))
    finally:
        if owned:
            client.close()
    return results


def summarize(results):
    n = len(results)
    ok = [r for r in results if not r["error"]]
    lat = [r["latency_s"] for r in ok]
    acc = {k: sum(r["score"][k] for r in results) / n if n else 0.0
           for k in ("merchant", "date", "category", "total_usd", "items", "all_fields", "reimbursement")}
    return {"n": n, "errors": n - len(ok), "accuracy": acc,
            "p50_latency_s": statistics.median(lat) if lat else None, "p95_latency_s": pct(lat, 0.95) if lat else None,
            "mean_prompt_tokens": statistics.mean(r["usage"].get("prompt_tokens", 0) for r in ok) if ok else None,
            "mean_completion_tokens": statistics.mean(r["usage"].get("completion_tokens", 0) for r in ok) if ok else None,
            "json_retries": sum(r["retries"] for r in results)}


def print_summary(label, s, results):
    a = s["accuracy"]
    print(f"## {label}: {s['n']} receipts, {s['errors']} errors, {s['json_retries']} JSON fix-it retries\n")
    print("| merchant | date | category | total_usd | items | all five | reimbursement paid right | p50 s | p95 s | "
          "prompt tok | output tok |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    def f(v, fmt):
        return fmt.format(v) if v is not None else "-"
    print(f"| {a['merchant']:.0%} | {a['date']:.0%} | {a['category']:.0%} | {a['total_usd']:.0%} | {a['items']:.0%} | "
          f"{a['all_fields']:.0%} | {a['reimbursement']:.0%} | {f(s['p50_latency_s'], '{:.2f}')} | "
          f"{f(s['p95_latency_s'], '{:.2f}')} | {f(s['mean_prompt_tokens'], '{:,.0f}')} | "
          f"{f(s['mean_completion_tokens'], '{:,.0f}')} |")
    misses = [r for r in results if not r["score"]["reimbursement"]][:5]
    if misses:
        print("\nFirst reimbursement misses (what the agent would have paid wrong):")
        for r in misses:
            got = r["pred"]["total_usd"] if r["pred"] else None
            print(f"- {r['receipt_ref']}: read total {got} {('- ' + r['error']) if r['error'] else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/receipts")
    ap.add_argument("--labels", default="", help="default: <dir>/labels.jsonl")
    ap.add_argument("--image", default="", help="one image: print the extracted fields")
    ap.add_argument("--base-url", default=os.environ.get("TTG_VLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("TTG_VLM_MODEL", "ttg-llm"))
    ap.add_argument("--api-key-env", default="TTG_VLM_API_KEY", help="NAME of the env var holding the key")
    ap.add_argument("--api-key", default=None,
                    help='the key itself, e.g. "$NVIDIA_API_KEY" (prefer --api-key-env: keys typed out land in history)')
    ap.add_argument("--n", type=int, default=0, help="first n receipts (0 = all)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--structured", default="none", choices=["none", "json_object", "json_schema"])
    ap.add_argument("--image-mode", default="image_url", choices=["image_url", "img_tag"])
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--label", default="", help="results/vlm/extract-<label>.json")
    ap.add_argument("--dry-run", action="store_true", help="print the request for --image (base64 shortened)")
    a = ap.parse_args(argv)
    api_key = a.api_key or os.environ.get(a.api_key_env, "not-used")
    if a.dry_run:
        img = a.image or os.path.join(a.dir, "R-S0003.png")
        body = build_request(img, a.model, structured=a.structured, image_mode=a.image_mode)
        text = json.dumps(body, indent=2)
        print(re.sub(r"base64,[A-Za-z0-9+/=]{40,}", lambda m: f"base64,<{len(m.group(0)) - 7:,} chars>", text))
        print(f"POST {chat_url(a.base_url)}  (Authorization: Bearer ${a.api_key_env})")
        return 0
    if a.image:
        print(json.dumps(extract_fields(a.image, a.base_url, a.model, api_key, a.timeout_s,
                                        structured=a.structured, image_mode=a.image_mode), indent=2))
        return 0
    labels = load_labels(a.labels or os.path.join(a.dir, "labels.jsonl"))
    results = run_eval(a.dir, labels, a.base_url, a.model, api_key, a.n, a.concurrency, a.structured,
                       a.image_mode, a.timeout_s)
    s = summarize(results)
    label = a.label or f"{a.model.replace('/', '_')}"
    print_summary(label, s, results)
    os.makedirs("results/vlm", exist_ok=True)
    out = f"results/vlm/extract-{label}.json"
    with open(out, "w") as f:
        json.dump({"label": label, "model": a.model, "base_url": a.base_url, "structured": a.structured,
                   "summary": s, "results": results}, f, indent=2)
    print(f"\n-> {out}")
    return 0 if s["errors"] < s["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
