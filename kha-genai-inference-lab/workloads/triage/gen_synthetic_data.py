#!/usr/bin/env python3
"""Generate synthetic prior-authorization request notes with KNOWN urgency labels, using the in-cluster NIM.

Synthetic only: prompts ask for fictional details; no PHI is used or produced.
The label comes from the generation prompt, and words that would leak the label are filtered out,
so the classifier has to learn clinical signals rather than keywords. Standard library only.
"""
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = os.environ.get("LLM_URL", "http://priorauth-llm.kha-um.svc:8000/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "meta/llama-3.1-8b-instruct")
N = int(os.environ.get("NUM_SAMPLES", "1200"))
CONC = int(os.environ.get("CONCURRENCY", "16"))
OUT = Path(os.environ.get("OUT_PATH", "/models/triage/data/synthetic.jsonl"))
URGENT_SHARE = 0.4

SERVICES = [
    "lumbar spine MRI", "outpatient chemotherapy regimen", "home oxygen therapy", "total knee arthroplasty",
    "continuous glucose monitor", "inpatient rehabilitation admission", "biologic infusion for rheumatoid arthritis",
    "cardiac catheterization", "PET-CT for cancer staging", "bariatric surgery", "sleep study", "power wheelchair",
    "specialty medication for multiple sclerosis", "spinal cord stimulator trial", "home health nursing visits",
    "genetic testing panel", "transcatheter aortic valve replacement", "residential behavioral health treatment",
]
URGENT_CUE = (
    "The clinical situation is time-sensitive: waiting the normal review period could seriously jeopardize the "
    "patient's life, health, or ability to regain maximum function (for example new neurological deficits, rapidly "
    "progressing disease, an imminent hospital discharge that depends on the service, or interruption of active treatment)."
)
STANDARD_CUE = (
    "The clinical situation is stable: the request is appropriate, but there is no immediate risk if the review "
    "follows the normal timeline (for example chronic stable symptoms or planned follow-up care)."
)
LEAK_WORDS = ("urgent", "expedite", "emergenc", "routine", "elective", "standard review", "standard timeline")


def build_prompt(label: str, service: str, age: int, sex: str) -> str:
    cue = URGENT_CUE if label == "URGENT" else STANDARD_CUE
    return (
        f"Write a realistic 3-6 sentence prior authorization request note from a provider's office to a health plan "
        f"for a {service}. Patient: fictional {age}-year-old {sex}. {cue} Include relevant history, findings, and "
        "what was tried before. Do NOT use the words urgent, expedited, emergency, routine, or elective. "
        "Use no real names, dates, or identifiers. Return only the note text."
    )


def chat(prompt: str, retries: int = 3) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 220,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)["choices"][0]["message"]["content"].strip()
        except Exception:  # noqa: BLE001 - retry any transport error
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def one(i: int):
    rng = random.Random(1000 + i)
    label = "URGENT" if rng.random() < URGENT_SHARE else "STANDARD"
    text = chat(build_prompt(label, rng.choice(SERVICES), rng.randint(19, 88), rng.choice(["woman", "man"])))
    if len(text) < 40 or any(w in text.lower() for w in LEAK_WORDS):
        return None  # drop label leakage / empty generations
    return {"id": i, "label": label, "text": text}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done_marker = OUT.parent / ".done"
    if done_marker.exists():
        print(f"data already generated: {OUT}")
        return
    kept = dropped = 0
    t0 = time.time()
    with OUT.open("w") as f, ThreadPoolExecutor(CONC) as pool:
        futures = [pool.submit(one, i) for i in range(N)]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"request failed: {exc}", file=sys.stderr)
                dropped += 1
                continue
            if row is None:
                dropped += 1
                continue
            f.write(json.dumps(row) + "\n")
            kept += 1
            if k % 100 == 0:
                print(f"{k}/{N} kept={kept} dropped={dropped} elapsed={time.time() - t0:.0f}s", flush=True)
    done_marker.write_text(str(kept))
    print(f"done: kept={kept} dropped={dropped} -> {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
