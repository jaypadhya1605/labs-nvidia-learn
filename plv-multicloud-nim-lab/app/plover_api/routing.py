"""Sticky, deterministic canary routing: the same request_id always lands on the same variant."""
import hashlib


def choose_variant(request_id: str, canary_weight: int) -> str:
    if canary_weight <= 0:
        return "stable"
    if canary_weight >= 100:
        return "canary"
    bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 100
    return "canary" if bucket < canary_weight else "stable"
