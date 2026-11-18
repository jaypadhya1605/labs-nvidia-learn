#!/usr/bin/env python3
"""Synthetic urgency-triage notes (standard library only, deterministic). Lab 02 asked the LLM to write them;
Lab 04 generates them locally so the engine module runs on any cluster without a running NIM.
Label URGENT = expedited-review cues (risk to life/function, rapid progression, imminent discharge) without the
literal word "expedited", so the classifier has to learn the clinical language, not a keyword. ~8% label noise.
"""
import json
import os
import random
from pathlib import Path

SERVICES = ["MRI lumbar spine", "knee arthroplasty", "continuous glucose monitor", "sleep study", "stress echo",
            "PET-CT", "physical therapy evaluation", "sleeve gastrectomy", "MRI brain", "cardiac catheterization"]
URGENT_CUES = ["new focal weakness since yesterday", "rapidly worsening symptoms over 48 hours",
               "suspected malignancy with planned surgery next week", "discharge pending this result tomorrow",
               "progressive neurologic deficit", "chest pain at rest overnight", "risk of permanent loss of function",
               "oncology staging needed before chemotherapy starts Monday"]
ROUTINE_CUES = ["stable symptoms for several months", "routine follow-up requested", "no change since last visit",
                "elective procedure, scheduling flexible", "annual reassessment", "symptoms well controlled on medication",
                "patient prefers to schedule after the holidays", "chronic, non-progressive pain"]
FILLER = ["Conservative therapy documented.", "Prior imaging available.", "Clinician attests medical necessity.",
          "Comorbid hypertension.", "BMI 31.", "Allergic to penicillin.", "Lives alone.", "Insurance verified."]


def note(rng, urgent):
    cues = rng.sample(URGENT_CUES if urgent else ROUTINE_CUES, 2)
    return (f"{rng.randint(19, 88)}-year-old, request for {rng.choice(SERVICES)}. {cues[0].capitalize()}; {cues[1]}. "
            + " ".join(rng.sample(FILLER, 3)))


def main():
    out = Path(os.environ.get("OUT_PATH", "/work/triage/data/synthetic.jsonl"))
    n = int(os.environ.get("NUM_SAMPLES", "2400"))
    rng = random.Random(7)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for _ in range(n):
            urgent = rng.random() < 0.3
            label = urgent if rng.random() > 0.08 else not urgent
            f.write(json.dumps({"text": note(rng, urgent), "label": "URGENT" if label else "STANDARD"}) + "\n")
    (out.parent / ".done").touch()
    print(f"wrote {n} notes to {out}")


if __name__ == "__main__":
    main()
