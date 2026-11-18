"""The synthetic Tern Travel Group world: flights, travelers, hotels, receipts, the Tern Care policy oracle and scoring.

Everything is a pure function of an identifier (flight number, traveler id, receipt ref), so the MCP servers are
stateless, the mock engine can rebuild the right answer from the prompt alone, and every run is reproducible.

  python -m ttg.world --n 200 --seed 7 --out data/scenarios.jsonl   # golden set (+ expected plans)
  python -m ttg.world --n 200 --seed 7 --stats                       # what the golden set contains

Synthetic data only: the company, people, policy and amounts are invented for the lab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

OPS_DATE = "2026-01-14"                      # "Winter Storm Plover" day
REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "data" / "policies" / "tern-care-v3.md"

AIRPORTS = {
    "DTW": "Detroit", "ORD": "Chicago O'Hare", "BOS": "Boston", "DEN": "Denver", "SEA": "Seattle",
    "MSP": "Minneapolis", "JFK": "New York JFK", "ATL": "Atlanta", "DUB": "Dublin", "AMS": "Amsterdam",
}
ROUTES = [("DTW", "BOS"), ("DTW", "ORD"), ("DTW", "DEN"), ("DTW", "AMS"), ("ORD", "SEA"), ("ORD", "DUB"),
          ("BOS", "DUB"), ("BOS", "ATL"), ("DEN", "SEA"), ("DEN", "MSP"), ("MSP", "JFK"), ("JFK", "AMS"),
          ("ATL", "DTW"), ("SEA", "MSP"), ("JFK", "DEN"), ("AMS", "DTW"), ("DUB", "BOS"), ("ORD", "ATL")]
CAUSES = [("weather", 0.40), ("air_traffic_control", 0.15), ("crew", 0.20), ("mechanical", 0.25)]
CONTROLLABLE = {"crew", "mechanical"}
TIERS = [("basic", 0.45), ("silver", 0.25), ("gold", 0.20), ("platinum", 0.10)]
PARTY = [(1, 0.55), (2, 0.25), (3, 0.12), (4, 0.08)]
FIRST = ["Avery", "Jordan", "Priya", "Mateo", "Sofia", "Liam", "Aisha", "Noah", "Elena", "Kenji", "Maya", "Lucas",
         "Amara", "Finn", "Leila", "Oskar", "Nora", "Ravi", "Chloe", "Diego"]
LAST = ["Okafor", "Lindqvist", "Patel", "Moreau", "Haddad", "Kowalski", "Brennan", "Tanaka", "Fischer", "Alvarez",
        "Nguyen", "Dubois", "Rossi", "Mensah", "Larsen", "Costa"]
VOUCHER_KINDS = ("meal", "credit", "reimbursement")
APPROVAL_LIMIT_USD = 300.0
FLIGHT_RE = re.compile(r"^TN\d{3}$")
TRAVELER_RE = re.compile(r"^T\d{5}$")
RECEIPT_RE = re.compile(r"^R-[A-Za-z0-9-]+$")


def _rng(key: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def _pick(rng: random.Random, weighted):
    r, acc = rng.random(), 0.0
    for value, w in weighted:
        acc += w
        if r < acc:
            return value
    return weighted[-1][0]


def _code(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:6].upper()


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def _fmt(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M")


def _overnight(dep: datetime) -> bool:
    return dep.strftime("%Y-%m-%d") > OPS_DATE or dep.hour >= 23


# ---------------------------------------------------------------- world records (pure functions of an id)
@lru_cache(maxsize=4096)
def _flight(flight_no: str) -> dict:
    rng = _rng("flight|" + flight_no)
    origin, dest = rng.choice(ROUTES)
    sched = _dt(f"{OPS_DATE}T{rng.randint(6, 21):02d}:{rng.choice([0, 10, 15, 25, 30, 40, 45, 55]):02d}")
    status = "cancelled" if rng.random() < 0.45 else "delayed"
    cause = _pick(rng, CAUSES)
    delay = rng.choice([45, 90, 150, 185, 210, 270, 330, 420]) if status == "delayed" else 0
    new_dep = sched + timedelta(minutes=delay) if status == "delayed" else None
    alts = []
    if status == "cancelled":
        own = int(flight_no[2:])
        nums = [n for n in rng.sample(range(100, 1000), 6) if n != own][:4]
        t1 = sched + timedelta(minutes=rng.choice([90, 120, 180]))
        t2 = t1 + timedelta(minutes=rng.choice([150, 240]))
        next_day = (_dt(f"{OPS_DATE}T00:00") + timedelta(days=1)).strftime("%Y-%m-%d")
        times = [t1, t2, _dt(f"{next_day}T07:05"), _dt(f"{next_day}T09:40")]
        seats = [rng.randint(0, 5), rng.randint(0, 6), rng.randint(1, 8), rng.randint(6, 9)]
        for n, t, s in sorted(zip(nums, times, seats), key=lambda x: x[1]):
            alts.append({"flight_no": f"TN{n}", "departure": _fmt(t), "seats_available": s,
                         "overnight": _overnight(t)})
    return {"flight_no": flight_no, "origin": origin, "destination": dest, "scheduled_departure": _fmt(sched),
            "status": status, "cause": cause, "controllable": cause in CONTROLLABLE, "delay_minutes": delay,
            "new_departure": _fmt(new_dep) if new_dep else None,
            "overnight": _overnight(new_dep) if new_dep else None, "_alternatives": alts}


@lru_cache(maxsize=8192)
def _traveler(traveler_id: str) -> dict:
    rng = _rng("traveler|" + traveler_id)
    eu = rng.random() < 0.2
    return {"traveler_id": traveler_id, "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "tier": _pick(rng, TIERS),
            "eu_resident": eu, "contact_channel": rng.choice(["sms", "email", "app"]),
            "member_since": rng.randint(2009, 2025),
            "locale": rng.choice(["de-DE", "fr-FR", "nl-NL", "en-IE"]) if eu else rng.choice(["en-US", "en-US", "en-GB"])}


@lru_cache(maxsize=512)
def _hotels(airport: str) -> list:
    rng = _rng("hotels|" + airport)
    city = AIRPORTS.get(airport, airport)
    names = [f"Tern Hotel {city} Airport", f"Tern Express {city}", f"Tern Suites {city}"]
    out = []
    for i, name in enumerate(names, start=1):
        rate = rng.choice([129, 139, 149, 159, 169, 189, 209, 229, 249])
        rooms = rng.randint(0, 12) if i < 3 else rng.randint(4, 12)     # the third always has rooms
        out.append({"hotel_id": f"H-{airport}-{i}", "name": name, "nightly_rate_usd": rate,
                    "distressed_rate_usd": round(rate * 0.6), "rooms_available": rooms,
                    "shuttle": rng.random() < 0.7})
    seen = set()                       # "the cheapest hotel" must be unambiguous: no two hotels share a rate
    for h in out:
        while h["nightly_rate_usd"] in seen:
            h["nightly_rate_usd"] += 10
        h["distressed_rate_usd"] = round(h["nightly_rate_usd"] * 0.6)
        seen.add(h["nightly_rate_usd"])
    rng.shuffle(out)                                                    # not sorted by price on purpose
    return out


MERCHANTS = {"meal": ["Gate 12 Kitchen", "Runway Noodle Bar", "Concourse Deli", "Skyline Grill", "Terminal Taqueria"],
             "taxi": ["Metro Cab Co.", "Airport Express Taxi", "BlueLine Cars"]}
MEAL_ITEMS = [("Chicken sandwich", 13.5), ("Veggie bowl", 14.25), ("Kids meal", 8.75), ("Coffee", 4.1),
              ("Soup of the day", 7.95), ("Pasta", 16.5), ("Salad", 11.25), ("Water", 3.5), ("Burger", 15.95)]


@lru_cache(maxsize=4096)
def _receipt(ref: str) -> dict:
    rng = _rng("receipt|" + ref)
    category = "meal" if rng.random() < 0.7 else "taxi"
    merchant = rng.choice(MERCHANTS[category])
    if category == "meal":
        items = [{"desc": d, "usd": p} for d, p in rng.sample(MEAL_ITEMS, rng.randint(1, 4))]
        sub = round(sum(i["usd"] for i in items), 2)
        tax = round(sub * 0.06, 2)
    else:
        fare = round(rng.uniform(18, 65), 2)
        items = [{"desc": "Taxi fare", "usd": fare}, {"desc": "Tip", "usd": round(fare * 0.15, 2)}]
        sub, tax = round(sum(i["usd"] for i in items), 2), 0.0
    return {"receipt_ref": ref, "merchant": merchant, "date": OPS_DATE, "category": category, "items": items,
            "tax_usd": tax, "total_usd": round(sub + tax, 2)}


@lru_cache(maxsize=1)
def _policy_sections() -> list:
    text = POLICY_PATH.read_text(encoding="utf-8")
    out = []
    for block in text.split("\n## ")[1:]:
        title, _, body = block.partition("\n")
        out.append({"section": title.strip(), "text": " ".join(body.split())})
    return out


def _tokens(s: str) -> list:
    return re.findall(r"[a-z0-9]+", s.lower())


# ---------------------------------------------------------------- the 9 tools
TOOL_SPECS = [
    {"name": "flight_status",
     "description": "Status of a Tern Air flight on the disruption day: status (cancelled|delayed), cause, controllable "
                    "(true for crew or mechanical), delay_minutes, new_departure, overnight, origin, destination.",
     "input_schema": {"type": "object", "properties": {"flight_no": {"type": "string", "description": "e.g. TN482"}},
                      "required": ["flight_no"]}},
    {"name": "rebooking_options",
     "description": "Alternative flights on the same route for a cancelled flight, earliest first, with seats_available "
                    "and an overnight flag.",
     "input_schema": {"type": "object", "properties": {"flight_no": {"type": "string"},
                                                       "party_size": {"type": "integer", "minimum": 1, "maximum": 9}},
                      "required": ["flight_no", "party_size"]}},
    {"name": "book_flight",
     "description": "Book the whole party on an alternative flight. Returns a record locator.",
     "input_schema": {"type": "object", "properties": {"traveler_id": {"type": "string"}, "flight_no": {"type": "string"},
                                                       "party_size": {"type": "integer", "minimum": 1, "maximum": 9}},
                      "required": ["traveler_id", "flight_no", "party_size"]}},
    {"name": "hotel_availability",
     "description": "Tern Hotels at an airport tonight: hotel_id, name, nightly_rate_usd, distressed_rate_usd, "
                    "rooms_available (not sorted).",
     "input_schema": {"type": "object", "properties": {"airport": {"type": "string", "description": "IATA code"},
                                                       "nights": {"type": "integer", "minimum": 1, "maximum": 3},
                                                       "rooms": {"type": "integer", "minimum": 1, "maximum": 5}},
                      "required": ["airport", "nights", "rooms"]}},
    {"name": "book_hotel",
     "description": "Book a complimentary Tern Hotel stay for the party.",
     "input_schema": {"type": "object", "properties": {"traveler_id": {"type": "string"}, "hotel_id": {"type": "string"},
                                                       "nights": {"type": "integer", "minimum": 1, "maximum": 3},
                                                       "rooms": {"type": "integer", "minimum": 1, "maximum": 5}},
                      "required": ["traveler_id", "hotel_id", "nights", "rooms"]}},
    {"name": "loyalty_profile",
     "description": "Loyalty tier (basic|silver|gold|platinum), name, eu_resident and contact channel for a traveler.",
     "input_schema": {"type": "object", "properties": {"traveler_id": {"type": "string"}}, "required": ["traveler_id"]}},
    {"name": "policy_search",
     "description": "Search the Tern Care Commitments policy. Returns the k most relevant sections.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "k": {"type": "integer", "minimum": 1, "maximum": 5}},
                      "required": ["query"]}},
    {"name": "issue_voucher",
     "description": "Issue ONE voucher for the whole party. kind is meal, credit or reimbursement; amount_usd is the "
                    "party total. Vouchers above $300 return PENDING_APPROVAL.",
     "input_schema": {"type": "object", "properties": {"traveler_id": {"type": "string"},
                                                       "kind": {"type": "string", "enum": list(VOUCHER_KINDS)},
                                                       "amount_usd": {"type": "number", "exclusiveMinimum": 0}},
                      "required": ["traveler_id", "kind", "amount_usd"]}},
    {"name": "read_receipt",
     "description": "Read a receipt image the guest uploaded: merchant, date, category, items, tax_usd and total_usd.",
     "input_schema": {"type": "object", "properties": {"receipt_ref": {"type": "string"}}, "required": ["receipt_ref"]}},
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


class ToolError(ValueError):
    """Bad arguments: returned to the agent as {"error": ...} so it can correct itself."""


def _int(v, name, lo, hi):
    if isinstance(v, bool):
        raise ToolError(f"{name} must be an integer")
    try:
        f = float(v)
        i = int(f)
    except (TypeError, ValueError, OverflowError):              # "two", None, NaN, Infinity, 1e999
        raise ToolError(f"{name} must be an integer") from None
    if f != i or not lo <= i <= hi:
        raise ToolError(f"{name} must be an integer from {lo} to {hi}")
    return i


def _flight_no(v):
    s = str(v or "").strip().upper()
    if not FLIGHT_RE.match(s):
        raise ToolError("flight_no must look like TN482")
    return s


def _traveler_id(v):
    s = str(v or "").strip().upper()
    if not TRAVELER_RE.match(s):
        raise ToolError("traveler_id must look like T10023")
    return s


def flight_status(flight_no):
    f = dict(_flight(_flight_no(flight_no)))
    f.pop("_alternatives")
    return f


def rebooking_options(flight_no, party_size):
    f = _flight(_flight_no(flight_no))
    p = _int(party_size, "party_size", 1, 9)
    if f["status"] != "cancelled":
        return {"flight_no": f["flight_no"], "status": f["status"], "options": [],
                "note": "Flight is delayed, not cancelled: the party keeps the original flight."}
    return {"flight_no": f["flight_no"], "party_size": p, "options": [dict(o) for o in f["_alternatives"]]}


def book_flight(traveler_id, flight_no, party_size):
    t, fl, p = _traveler_id(traveler_id), _flight_no(flight_no), _int(party_size, "party_size", 1, 9)
    return {"status": "CONFIRMED", "record_locator": _code("pnr", t, fl, p), "traveler_id": t, "flight_no": fl,
            "party_size": p}


def hotel_availability(airport, nights, rooms):
    a = str(airport or "").strip().upper()
    if a not in AIRPORTS:
        raise ToolError(f"unknown airport {a!r}; use the origin IATA code from flight_status")
    return {"airport": a, "nights": _int(nights, "nights", 1, 3), "rooms_requested": _int(rooms, "rooms", 1, 5),
            "hotels": [dict(h) for h in _hotels(a)]}


def book_hotel(traveler_id, hotel_id, nights, rooms):
    t = _traveler_id(traveler_id)
    hid = str(hotel_id or "").strip().upper()
    m = re.match(r"^H-([A-Z]{3})-[1-3]$", hid)
    if not m or m.group(1) not in AIRPORTS:
        raise ToolError("hotel_id must come from hotel_availability, e.g. H-DTW-2")
    n, r = _int(nights, "nights", 1, 3), _int(rooms, "rooms", 1, 5)
    hotel = next(h for h in _hotels(m.group(1)) if h["hotel_id"] == hid)
    if hotel["rooms_available"] < r:
        return {"status": "SOLD_OUT", "hotel_id": hid, "rooms_available": hotel["rooms_available"]}
    return {"status": "CONFIRMED", "confirmation": _code("hotel", t, hid, n, r), "hotel_id": hid,
            "name": hotel["name"], "nights": n, "rooms": r}


def loyalty_profile(traveler_id):
    return dict(_traveler(_traveler_id(traveler_id)))


def policy_search(query, k=3):
    k = _int(k, "k", 1, 5)
    secs = _policy_sections()
    q = set(_tokens(str(query or "")))
    df = Counter(t for s in secs for t in set(_tokens(s["section"] + " " + s["text"])))
    n = len(secs)

    def score(s):
        toks = _tokens(s["section"] + " " + s["text"])
        tf = Counter(toks)
        return sum((tf[t] / (tf[t] + 1.2)) * math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q if t in tf)

    ranked = sorted(secs, key=lambda s: (-score(s), s["section"]))
    return {"query": str(query), "results": [dict(s) for s in ranked[:k]]}


def issue_voucher(traveler_id, kind, amount_usd):
    t = _traveler_id(traveler_id)
    kd = str(kind or "").strip().lower()
    if kd not in VOUCHER_KINDS:
        raise ToolError(f"kind must be one of {', '.join(VOUCHER_KINDS)}")
    try:
        amt = round(float(amount_usd), 2)
    except (TypeError, ValueError, OverflowError):
        raise ToolError("amount_usd must be a number") from None
    if not math.isfinite(amt) or amt <= 0:
        raise ToolError("amount_usd must be a positive, finite number")
    status = "PENDING_APPROVAL" if amt > APPROVAL_LIMIT_USD else "ISSUED"
    return {"status": status, "voucher_id": "V-" + _code("voucher", t, kd, amt), "traveler_id": t, "kind": kd,
            "amount_usd": amt}


def read_receipt(receipt_ref):
    ref = str(receipt_ref or "").strip()
    if not RECEIPT_RE.match(ref):
        raise ToolError("receipt_ref must look like R-S0003")
    return dict(_receipt(ref))


TOOLS = {"flight_status": flight_status, "rebooking_options": rebooking_options, "book_flight": book_flight,
         "hotel_availability": hotel_availability, "book_hotel": book_hotel, "loyalty_profile": loyalty_profile,
         "policy_search": policy_search, "issue_voucher": issue_voucher, "read_receipt": read_receipt}


def execute_tool(name: str, args: dict) -> dict:
    """Run a tool. Bad arguments come back as {"error": ...} (the agent sees and fixes them), never as exceptions."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}; tools: {', '.join(TOOL_NAMES)}"}
    args = dict(args or {})
    spec = next(t for t in TOOL_SPECS if t["name"] == name)
    allowed = set(spec["input_schema"]["properties"])
    missing = [r for r in spec["input_schema"]["required"] if r not in args]
    if missing:
        return {"error": f"missing argument(s): {', '.join(missing)}"}
    try:
        return fn(**{k: v for k, v in args.items() if k in allowed})
    except ToolError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- scenarios
def public_view(scenario: dict) -> dict:
    """What the agent is told up front. Status, cause, tier and amounts must come from tools."""
    return {k: scenario[k] for k in ("scenario_id", "traveler_id", "guest_name", "flight_no", "party_size",
                                    "receipt_ref", "locale", "guest_message")}


def make_scenarios(n: int, seed: int = 7) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        sid = f"S{i + 1:04d}" if seed == 7 else f"S{seed}-{i + 1:04d}"
        flight_no = f"TN{rng.randint(100, 999)}"
        traveler_id = f"T{rng.randint(10000, 99999)}"
        party = _pick(rng, PARTY)
        receipt_ref = f"R-{sid}" if rng.random() < 0.3 else None
        f, t = _flight(flight_no), _traveler(traveler_id)
        receipt = _receipt(receipt_ref) if receipt_ref else None
        what = "was cancelled" if f["status"] == "cancelled" else "is delayed"
        party_txt = "I'm traveling alone" if party == 1 else f"there are {party} of us on the booking"
        extra = ""
        if receipt:
            extra = (" I already paid for food at the airport, the receipt is uploaded." if receipt["category"] == "meal"
                     else " I paid for a taxi to the airport, the receipt is uploaded.")
        msg = rng.choice([
            f"Hi, this is {t['name']}. My flight {flight_no} to {AIRPORTS[f['destination']]} {what} and {party_txt}.{extra} What now?",
            f"Flight {flight_no} {what}. {party_txt.capitalize()}.{extra} Please help us get home.",
            f"Hello Tern team, {flight_no} {what}, {party_txt}.{extra} What are our options tonight?",
        ])
        out.append({
            "scenario_id": sid, "traveler_id": traveler_id, "guest_name": t["name"], "flight_no": flight_no,
            "party_size": party, "receipt_ref": receipt_ref, "locale": t["locale"], "guest_message": msg,
            # hidden truth (never sent to the agent; used by docs, analysis and the M8 receipt renderer)
            "origin": f["origin"], "destination": f["destination"], "scheduled_departure": f["scheduled_departure"],
            "status": f["status"], "cause": f["cause"], "controllable": f["controllable"],
            "delay_minutes": f["delay_minutes"], "tier": t["tier"], "eu_resident": t["eu_resident"],
            "receipt": receipt,
        })
    return out


# ---------------------------------------------------------------- the policy oracle
def _call(action, **args):
    return {"action": action, "args": args}


def expected_outcome(public: dict) -> dict:
    """Apply Tern Care Commitments v3 to the public facts + tool results. Returns plan, bookings and vouchers."""
    fno, tid, party = public["flight_no"], public["traveler_id"], int(public["party_size"])
    fs, lp = flight_status(fno), loyalty_profile(tid)
    tier, ctrl = lp["tier"], fs["controllable"]
    plan = [_call("flight_status", flight_no=fno), _call("loyalty_profile", traveler_id=tid),
            _call("policy_search", query=f"{fs['status']} {fs['cause'].replace('_', ' ')} {tier} meal hotel credit")]
    hotel = None
    if fs["status"] == "cancelled":
        opts = rebooking_options(fno, party)["options"]
        alt = next(o for o in opts if o["seats_available"] >= party)
        plan += [_call("rebooking_options", flight_no=fno, party_size=party),
                 _call("book_flight", traveler_id=tid, flight_no=alt["flight_no"], party_size=party)]
        rebooked, overnight, long_disruption, new_time = alt["flight_no"], alt["overnight"], True, alt["departure"]
    else:
        rebooked, overnight = fno, bool(fs["overnight"])
        long_disruption, new_time = fs["delay_minutes"] >= 180, fs["new_departure"]
    if overnight:
        rooms = math.ceil(party / 2)
        plan.append(_call("hotel_availability", airport=fs["origin"], nights=1, rooms=rooms))
        if ctrl or tier == "platinum":
            avail = [h for h in _hotels(fs["origin"]) if h["rooms_available"] >= rooms]
            hotel = min(avail, key=lambda h: (h["nightly_rate_usd"], h["hotel_id"]))
            plan.append(_call("book_hotel", traveler_id=tid, hotel_id=hotel["hotel_id"], nights=1, rooms=rooms))
    receipt = None
    if public.get("receipt_ref") and ctrl:
        receipt = read_receipt(public["receipt_ref"])
        plan.append(_call("read_receipt", receipt_ref=public["receipt_ref"]))
    gold = tier in ("gold", "platinum")
    vouchers = []
    if long_disruption and (ctrl or tier == "platinum"):
        vouchers.append({"kind": "meal", "amount_usd": float((25 if (ctrl and gold) else 15) * party)})
    if ctrl:
        base = 200 if fs["status"] == "cancelled" else (100 if fs["delay_minutes"] >= 180 else 0)
        if base:
            vouchers.append({"kind": "credit", "amount_usd": float(base * (1.5 if gold else 1.0) * party)})
    if receipt:
        vouchers.append({"kind": "reimbursement", "amount_usd": round(min(receipt["total_usd"], 50.0 * party), 2)})
    for v in vouchers:
        plan.append(_call("issue_voucher", traveler_id=tid, kind=v["kind"], amount_usd=v["amount_usd"]))
    approval = any(v["amount_usd"] > APPROVAL_LIMIT_USD for v in vouchers)
    parts = [f"Hi {lp['name'].split()[0]}, we are sorry about {fno}."]
    parts.append(f"You are confirmed on {rebooked} departing {new_time.replace('T', ' at ')}." if rebooked != fno
                 else f"Your flight {fno} still operates, now departing {new_time.replace('T', ' at ')}.")
    if hotel:
        parts.append(f"Tonight you are booked at {hotel['name']} ({math.ceil(party / 2)} room(s)), on us.")
    elif overnight:
        h = min(_hotels(fs["origin"]), key=lambda x: x["distressed_rate_usd"])
        parts.append(f"If you need a room, {h['name']} offers a distressed-passenger rate of ${h['distressed_rate_usd']}.")
    for v in vouchers:
        parts.append(f"A {v['kind']} voucher of ${v['amount_usd']:.2f} is on its way.")
    if approval:
        parts.append("One voucher needs a quick supervisor approval; you will get it within the hour.")
    final = {"message": " ".join(parts), "rebooked_flight": rebooked, "hotel_id": hotel["hotel_id"] if hotel else None,
             "vouchers": vouchers, "needs_human_approval": approval}
    return {"plan": plan, "final": final, "hotel": hotel, "vouchers": vouchers, "rebooked_flight": rebooked,
            "overnight": overnight, "controllable": ctrl, "tier": tier}


def expected_plan(scenario: dict) -> list:
    out = expected_outcome(scenario)
    return out["plan"] + [{"final": out["final"]}]


# ---------------------------------------------------------------- scoring
def score(scenario: dict, actions: list, final: dict | None) -> tuple:
    """Policy-compliance check of what the agent DID (bookings, vouchers, final answer). Returns (success, violations)."""
    exp = expected_outcome(scenario)
    tid = scenario["traveler_id"]
    v = []

    def calls(name):
        return [a for a in actions if a.get("action") == name and not (a.get("result") or {}).get("error")]

    flights = calls("book_flight")
    want_flight = exp["rebooked_flight"] if exp["rebooked_flight"] != scenario["flight_no"] else None
    if want_flight and not flights:
        v.append("missing_rebooking")
    if not want_flight and flights:
        v.append("unexpected_rebooking")
    if len(flights) > 1:
        v.append("duplicate_booking:flight")
    for a in flights:
        args = a.get("args") or {}
        if want_flight and str(args.get("flight_no", "")).upper() != want_flight:
            v.append(f"wrong_rebooking:{args.get('flight_no')}")
        if str(args.get("party_size")) != str(scenario["party_size"]):
            v.append("wrong_party_size")
    hotels = [a for a in calls("book_hotel") if (a.get("result") or {}).get("status") == "CONFIRMED"
              or not a.get("result")]
    want_hotel = exp["hotel"]["hotel_id"] if exp["hotel"] else None
    if want_hotel and not hotels:
        v.append("missing_hotel")
    if not want_hotel and hotels:
        v.append("unexpected_hotel")
    if len(hotels) > 1:
        v.append("duplicate_booking:hotel")
    for a in hotels:
        args = a.get("args") or {}
        if want_hotel and str(args.get("hotel_id", "")).strip().upper() != want_hotel:
            v.append(f"wrong_hotel:{args.get('hotel_id')}")
        if want_hotel and str(args.get("rooms")) != str(math.ceil(scenario["party_size"] / 2)):
            v.append("wrong_rooms")
        if want_hotel and str(args.get("nights")) != "1":
            v.append("wrong_nights")
    want_v = {x["kind"]: x["amount_usd"] for x in exp["vouchers"]}
    seen = Counter()
    for a in calls("issue_voucher"):
        args = a.get("args") or {}
        kind = str(args.get("kind", "")).lower()
        seen[kind] += 1
        try:
            amt = float(args.get("amount_usd"))
        except (TypeError, ValueError):
            amt = float("nan")
        if kind not in want_v:
            v.append(f"unexpected_voucher:{kind}")
        elif not abs(amt - want_v[kind]) <= 0.01:
            v.append(f"wrong_amount:{kind}:{args.get('amount_usd')}!={want_v[kind]}")
    for kind in want_v:
        if seen[kind] == 0:
            v.append(f"missing_voucher:{kind}")
        elif seen[kind] > 1:
            v.append(f"duplicate_voucher:{kind}")
    for name in ("book_flight", "book_hotel", "issue_voucher"):      # successful calls only, like every check above
        if any(str((a.get("args") or {}).get("traveler_id", tid)).strip().upper() != tid for a in calls(name)):
            v.append("wrong_traveler")
            break
    if not final:
        v.append("no_final")
    else:
        if str(final.get("rebooked_flight", "")).upper() != exp["rebooked_flight"]:
            v.append("final_flight_mismatch")
        if bool(final.get("needs_human_approval")) != exp["final"]["needs_human_approval"]:
            v.append("approval_flag_wrong")
        if exp["rebooked_flight"].lower() not in str(final.get("message", "")).lower():
            v.append("message_missing_flight")
    return (not v, v)


def _stats(scs):
    c = Counter()
    for s in scs:
        e = expected_outcome(s)
        c[f"status={s['status']}"] += 1
        c[f"controllable={s['controllable']}"] += 1
        c[f"tier={s['tier']}"] += 1
        c[f"overnight={e['overnight']}"] += 1
        c[f"hotel_booked={bool(e['hotel'])}"] += 1
        c[f"receipt={bool(s['receipt_ref'])}"] += 1
        c[f"needs_approval={e['final']['needs_human_approval']}"] += 1
        c[f"tool_calls={len(e['plan'])}"] += 1
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", help="write JSONL: scenario + expected plan per line")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args(argv)
    scs = make_scenarios(a.n, a.seed)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            for s in scs:
                fh.write(json.dumps({**s, "expected_plan": expected_plan(s)}) + "\n")
        print(f"wrote {len(scs)} scenarios -> {a.out}")
    if a.stats or not a.out:
        for k, v in sorted(_stats(scs).items()):
            print(f"{k:28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
