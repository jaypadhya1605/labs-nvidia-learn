"""Deterministic synthetic prior-authorization cases with ground truth (standard library only).

Everything is fictional. Each case carries the codes that appear in its text, so an evaluator can check that a
summary never invents a CPT or ICD-10 code (faithfulness) - the failure a payer cares about most.
"""
import random

SERVICES = [
    ("lumbar spine MRI without contrast", "72148", "M54.16", "lumbar radiculopathy"),
    ("total knee arthroplasty", "27447", "M17.11", "primary osteoarthritis, right knee"),
    ("continuous glucose monitor", "95250", "E10.65", "type 1 diabetes with hyperglycemia"),
    ("polysomnography", "95810", "G47.33", "obstructive sleep apnea"),
    ("cardiac stress echocardiogram", "93351", "I20.9", "angina pectoris"),
    ("PET-CT for staging", "78815", "C34.90", "malignant neoplasm of lung"),
    ("outpatient physical therapy evaluation", "97163", "S83.511A", "ACL sprain, right knee"),
    ("bariatric surgery (sleeve gastrectomy)", "43775", "E66.01", "morbid obesity due to excess calories"),
]
HISTORY = ["six weeks of physical therapy without improvement", "NSAIDs and activity modification for 8 weeks",
           "two prior steroid injections with partial relief", "documented failure of conservative management",
           "medication adherence confirmed at the last two visits", "progressive symptoms over 3 months"]
FINDINGS = ["positive straight-leg raise on exam", "Kellgren-Lawrence grade 4 on x-ray", "HbA1c 9.4%",
            "apnea-hypopnea index estimated above 20 on home screening", "new exertional chest pressure",
            "suspicious 2.3 cm nodule on CT", "positive Lachman test", "BMI 43 with hypertension"]
URGENCY = ["standard", "standard", "standard", "expedited"]


def make_case(i: int) -> dict:
    rng = random.Random(10_000 + i)
    service, cpt, icd, dx = rng.choice(SERVICES)
    age = rng.randint(19, 84)
    hist = rng.sample(HISTORY, 2)
    find = rng.sample(FINDINGS, 1)[0]
    urgency = rng.choice(URGENCY)
    member = f"M{rng.randint(10_000_000, 99_999_999)}"
    text = (f"Prior authorization request {i:05d} ({urgency}). Member {member}, {age}-year-old. "
            f"Requested service: {service} (CPT {cpt}). Diagnosis: {dx} (ICD-10 {icd}). "
            f"History: {hist[0]}; {hist[1]}. Exam/results: {find}. "
            f"Ordering clinician attests medical necessity and requests review.")
    return {"id": f"PA-{i:05d}", "text": text, "cpt": [cpt], "icd10": [icd], "age": age,
            "service": service, "urgency": urgency, "member": member}


def cases(n: int, start: int = 0) -> list:
    return [make_case(i) for i in range(start, start + n)]
