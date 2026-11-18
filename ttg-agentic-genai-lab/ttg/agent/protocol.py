"""The JSON action protocol: one JSON object per LLM turn, parsed and validated against the MCP tool schemas.

Why not native tool calling? Tool-call parsers differ per engine and per model (vLLM llama3_json, SGLang llama3,
trtllm-serve --tool_parser, NIM built-in). For a FAIR engine bake-off every engine must see byte-identical prompts
and produce the same kind of output, so the agent speaks plain JSON and validates it itself. Native tool calling
is an M4 stretch.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from ttg.world import TOOL_SPECS, VOUCHER_KINDS

FIX = ('Reply with exactly ONE JSON object: {"thought": "...", "action": "<tool>", "args": {...}} or '
       '{"thought": "...", "final": {"message": "...", "rebooked_flight": "...", "hotel_id": null, "vouchers": [], '
       '"needs_human_approval": false}}.')


class ProtocolError(ValueError):
    """The model's output could not be used. str(e) is written for the MODEL (it is sent back as feedback)."""


@dataclass
class Step:
    kind: str                      # "action" | "final"
    thought: str = ""
    action: str | None = None
    args: dict = field(default_factory=dict)
    final: dict | None = None


def format_action(action: str, args: dict, thought: str = "") -> str:
    return json.dumps({"thought": thought, "action": action, "args": args}, ensure_ascii=False)


def format_final(final: dict, thought: str = "") -> str:
    return json.dumps({"thought": thought, "final": final}, ensure_ascii=False)


def _first_object(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0]
    s = s.replace("<|python_tag|>", "").strip()
    start = s.find("{")
    if start < 0:
        raise ProtocolError("No JSON object found. " + FIX)
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    raise ProtocolError("The JSON object is not closed (output cut off or malformed). " + FIX)


def _number(value, name: str) -> float:
    if isinstance(value, bool) or isinstance(value, (dict, list)) or value is None:
        raise ProtocolError(f"Argument {name} must be a number. " + FIX)
    try:
        f = float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError, OverflowError):
        raise ProtocolError(f"Argument {name} must be a number. " + FIX) from None
    if not math.isfinite(f) or abs(f) > 1e9:
        raise ProtocolError(f"Argument {name} must be a finite number below one billion. " + FIX)
    return f


def _coerce(value, schema: dict, name: str):
    """Coerce one argument to its schema type. Anything unusable becomes a ProtocolError (never another exception)."""
    t = schema.get("type")
    if t == "integer":
        f = _number(value, name)
        if f != int(f):
            raise ProtocolError(f"Argument {name} must be a whole number. " + FIX)
        return int(f)
    if t == "number":
        return _number(value, name)
    if t == "string":
        if isinstance(value, (dict, list)) or value is None:
            raise ProtocolError(f"Argument {name} must be a string. " + FIX)
        return value if isinstance(value, str) else str(value)
    return value


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ProtocolError("final.needs_human_approval must be true or false. " + FIX)


def _final(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ProtocolError("final must be an object. " + FIX)
    missing = [k for k in ("message", "rebooked_flight") if not obj.get(k)]
    if missing:
        raise ProtocolError(f"final is missing {', '.join(missing)}. " + FIX)
    for k in ("message", "rebooked_flight"):
        if isinstance(obj[k], (dict, list)):
            raise ProtocolError(f"final.{k} must be a string. " + FIX)
    hotel = obj.get("hotel_id")
    if isinstance(hotel, str) and hotel.strip().lower() in ("", "null", "none"):
        hotel = None
    if hotel is not None and not isinstance(hotel, str):
        raise ProtocolError("final.hotel_id must be a string or null. " + FIX)
    raw = obj.get("vouchers")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ProtocolError("final.vouchers must be a list (use [] for none). " + FIX)
    vouchers = []
    for v in raw:
        if not isinstance(v, dict) or "kind" not in v or "amount_usd" not in v:
            raise ProtocolError('Each voucher needs {"kind", "amount_usd"}. ' + FIX)
        kind = str(v["kind"]).lower()
        if kind not in VOUCHER_KINDS:
            raise ProtocolError(f"Voucher kind must be one of {', '.join(VOUCHER_KINDS)}. " + FIX)
        vouchers.append({"kind": kind, "amount_usd": _coerce(v["amount_usd"], {"type": "number"}, "amount_usd")})
    return {"message": str(obj["message"]), "rebooked_flight": str(obj["rebooked_flight"]).strip().upper(),
            "hotel_id": hotel.strip().upper() if hotel else None, "vouchers": vouchers,
            "needs_human_approval": _bool(obj.get("needs_human_approval", False))}


def parse_step(text: str, tool_specs=TOOL_SPECS) -> Step:
    """Parse one model turn. Raises ProtocolError (and only ProtocolError) with feedback the model can act on."""
    try:
        return _parse_step(text, tool_specs)
    except ProtocolError:
        raise
    except (TypeError, ValueError, OverflowError, AttributeError, KeyError, RecursionError) as e:
        # A shape we did not anticipate must cost one retry, never the whole batch.
        raise ProtocolError(f"Could not use the reply ({type(e).__name__}). " + FIX) from None


def _parse_step(text: str, tool_specs) -> Step:
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("Empty reply. " + FIX)
    try:
        obj = json.loads(_first_object(text))
    except json.JSONDecodeError as e:
        raise ProtocolError(f"Invalid JSON ({e.msg} at char {e.pos}). " + FIX) from None
    if not isinstance(obj, dict):
        raise ProtocolError("The reply must be a JSON object. " + FIX)
    thought = obj.get("thought", "")
    thought = thought if isinstance(thought, str) else json.dumps(thought, ensure_ascii=False)
    if "final" in obj and obj["final"] is not None:
        return Step(kind="final", thought=thought, final=_final(obj["final"]))
    name = obj.get("action")
    args = obj.get("args", obj.get("arguments", {}))
    if isinstance(name, dict):                     # OpenAI function-call habit: {"action": {"name", "arguments"}}
        args = name.get("arguments", name.get("args", args))
        name = name.get("name")
    if not name:
        raise ProtocolError('No "action" and no "final". ' + FIX)
    if not isinstance(name, str):
        raise ProtocolError('"action" must be one tool name (a string). ' + FIX)
    specs = {t["name"]: t for t in tool_specs}
    if name not in specs:
        raise ProtocolError(f"Unknown tool {name!r}. Tools: {', '.join(specs)}. " + FIX)
    args = args or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            raise ProtocolError("args must be a JSON object. " + FIX) from None
    if not isinstance(args, dict):
        raise ProtocolError("args must be a JSON object. " + FIX)
    schema = specs[name]["input_schema"]
    missing = [r for r in schema.get("required", []) if r not in args]
    if missing:
        raise ProtocolError(f"{name} needs {', '.join(missing)}. " + FIX)
    props = schema.get("properties", {})
    clean = {k: _coerce(v, props[k], k) for k, v in args.items() if k in props}
    return Step(kind="action", thought=thought, action=name, args=clean)


FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"}, "rebooked_flight": {"type": "string"},
        "hotel_id": {"type": ["string", "null"]},
        "vouchers": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": list(VOUCHER_KINDS)}, "amount_usd": {"type": "number"}},
            "required": ["kind", "amount_usd"]}},
        "needs_human_approval": {"type": "boolean"}},
    "required": ["message", "rebooked_flight", "hotel_id", "vouchers", "needs_human_approval"],
}
# Constrained decoding (TTG_STRUCTURED=json_schema): every reply is EITHER an action step OR a final step, so the
# engine cannot emit {"thought": "..."} alone. Argument VALUES are still checked by parse_step (a wrong id can retry).
STEP_JSON_SCHEMA = {
    "anyOf": [
        {"type": "object",
         "properties": {"thought": {"type": "string"},
                        "action": {"type": "string", "enum": [t["name"] for t in TOOL_SPECS]},
                        "args": {"type": "object"}},
         "required": ["thought", "action", "args"]},
        {"type": "object",
         "properties": {"thought": {"type": "string"}, "final": FINAL_SCHEMA},
         "required": ["thought", "final"]},
    ],
}


def response_format(mode: str):
    """OpenAI response_format for TTG_STRUCTURED (constrained decoding: xgrammar in vLLM/SGLang/TensorRT-LLM)."""
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": STEP_JSON_SCHEMA}}
    return None
