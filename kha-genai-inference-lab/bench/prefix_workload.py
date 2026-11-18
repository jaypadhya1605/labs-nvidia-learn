#!/usr/bin/env python3
"""Synthetic prefix-heavy prior-auth traffic for prefix caching (M7) and KV-aware routing (M10).

Each request = one medical-policy document (shared prefix, ~3k tokens) + a unique case summary + a question.
Policy popularity follows a Zipf distribution, like real UM traffic where a few policies dominate.
Writes AIPerf single_turn JSONL: {"text": ..., "output_length": N}. Standard library only.
Example (inside the bench toolbox):
  python3 /code/prefix_workload.py --out /results/prefix-llama.jsonl --policies 48 --requests 1200
"""
import argparse
import json
import random

SERVICES = ["lumbar spine MRI", "total knee arthroplasty", "home oxygen therapy", "continuous glucose monitoring",
            "outpatient chemotherapy", "inpatient rehabilitation", "biologic infusion therapy", "cardiac catheterization",
            "PET-CT imaging", "bariatric surgery", "polysomnography", "power mobility devices", "spinal cord stimulation",
            "genetic testing", "home health services", "transcatheter valve replacement"]
CONDITIONS = ["radiculopathy", "osteoarthritis", "chronic hypoxemia", "type 1 diabetes", "metastatic disease",
              "stroke with residual deficits", "rheumatoid arthritis", "unstable angina", "lymphoma", "obesity class III",
              "obstructive sleep apnea", "spinal stenosis", "neuropathic pain", "hereditary cancer syndrome"]
FINDINGS = ["progressive motor weakness", "failed six weeks of physical therapy", "oxygen saturation below 88 percent",
            "recurrent severe hypoglycemia", "new lesions on prior imaging", "inability to perform activities of daily living",
            "inadequate response to two conventional agents", "positive stress test", "documented weight-related comorbidities",
            "apnea-hypopnea index above 15", "neurogenic claudication", "pain refractory to medication management"]
TESTS = ["imaging", "laboratory", "functional assessment", "specialist consultation", "pulmonary function", "echocardiogram"]
TEMPLATES = [
    "Coverage for {svc} requires documentation of {n} weeks of conservative management including {f1} and {f2}.",
    "{svc} is considered medically necessary when the member has {cond} with {f1}.",
    "The request must include {test} results obtained within {d} days of the date of service.",
    "Repeat {svc} within {m} months requires evidence of a significant change in clinical status such as {f2}.",
    "Exclusions apply when {cond} is managed adequately without {svc} or when {f1} is not documented.",
    "The treating clinician must attest that alternatives were considered and that {f2} persists despite treatment.",
    "Site-of-service review applies: {svc} in a hospital outpatient setting requires justification over a freestanding facility.",
    "Continued authorization beyond {m} months requires a progress note demonstrating measurable functional improvement.",
]


def sentence(rng, svc):
    return rng.choice(TEMPLATES).format(svc=svc, cond=rng.choice(CONDITIONS), f1=rng.choice(FINDINGS),
                                        f2=rng.choice(FINDINGS), test=rng.choice(TESTS), n=rng.choice([4, 6, 8, 12]),
                                        d=rng.choice([30, 60, 90]), m=rng.choice([3, 6, 12]))


def policy(idx, words):
    rng = random.Random(10_000 + idx)
    svc = SERVICES[idx % len(SERVICES)]
    parts = [f"KESTREL HEALTH ALLIANCE MEDICAL POLICY MP-{idx:03d}: {svc.upper()} (synthetic, for testing only)."]
    n = len(parts[0].split())
    while n < words:
        s = sentence(rng, svc)
        parts.append(s)
        n += len(s.split())
    return svc, " ".join(parts)


def case(rng, svc, words):
    parts = [f"Case summary: fictional {rng.randint(19, 88)}-year-old member requesting {svc}."]
    n = len(parts[0].split())
    while n < words:
        s = (f"History notes {rng.choice(CONDITIONS)} with {rng.choice(FINDINGS)}; "
             f"{rng.choice(TESTS)} completed {rng.randint(2, 120)} days ago showing {rng.choice(FINDINGS)}.")
        parts.append(s)
        n += len(s.split())
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", type=int, default=48)
    ap.add_argument("--requests", type=int, default=1200)
    ap.add_argument("--policy-words", type=int, default=2300, help="~1.3 tokens per word -> ~3k-token prefix")
    ap.add_argument("--case-words", type=int, default=300)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--osl", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    policies = [policy(i, a.policy_words) for i in range(a.policies)]
    weights = [1.0 / (i + 1) ** a.zipf for i in range(a.policies)]
    counts = [0] * a.policies
    with open(a.out, "w") as f:
        for _ in range(a.requests):
            i = rng.choices(range(a.policies), weights=weights)[0]
            counts[i] += 1
            svc, text = policies[i]
            prompt = (f"{text}\n\n{case(rng, svc, a.case_words)}\n\nQuestion: Does this request meet the policy "
                      "criteria above? Answer MEETS or DOES NOT MEET, then give a three-sentence rationale citing the criteria.")
            f.write(json.dumps({"text": prompt, "output_length": a.osl}) + "\n")
    top = sorted(counts, reverse=True)
    est_words = a.policy_words + a.case_words + 30
    print(f"wrote {a.requests} requests to {a.out}; ~{int(est_words * 1.3)} tokens each; "
          f"top-5 policies carry {sum(top[:5]) / a.requests:.0%} of traffic")


if __name__ == "__main__":
    main()
