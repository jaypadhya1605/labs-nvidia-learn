"""The contract of /v1/prior-auth/summarize - validated on every response (schema + faithfulness)."""
import json
import re

REQUIRED = {
    "request_type": str, "requested_service": str, "cpt_codes": list, "icd10_codes": list,
    "clinical_summary": str, "criteria_evidence": list, "missing_information": list,
    "recommendation": str, "confidence": (int, float),
}
RECOMMENDATIONS = {"approve_for_review", "pend_for_information", "route_to_medical_director"}
CPT_RE = re.compile(r"\b\d{5}\b")
ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")


def extract_json(text: str):
    """Models often wrap JSON in prose or code fences - take the outermost {...} block."""
    if text is None:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate(obj) -> list:
    """Return a list of human-readable errors; empty means valid."""
    if not isinstance(obj, dict):
        return ["response is not a JSON object"]
    errors = []
    for key, typ in REQUIRED.items():
        if key not in obj:
            errors.append(f"missing field: {key}")
        elif not isinstance(obj[key], typ) or isinstance(obj[key], bool) and typ != bool:
            errors.append(f"wrong type for {key}")
    if obj.get("recommendation") not in RECOMMENDATIONS:
        errors.append("recommendation must be one of " + ", ".join(sorted(RECOMMENDATIONS)))
    conf = obj.get("confidence")
    if isinstance(conf, (int, float)) and not 0 <= conf <= 1:
        errors.append("confidence must be between 0 and 1")
    return errors


def unfaithful_codes(obj, source_text: str) -> list:
    """Codes in the summary that do not appear in the source request (hallucinated codes)."""
    if not isinstance(obj, dict):
        return []
    src = set(CPT_RE.findall(source_text)) | set(ICD_RE.findall(source_text))
    out = [str(c) for c in (obj.get("cpt_codes") or []) + (obj.get("icd10_codes") or [])]
    return [c for c in out if c not in src]
