"""The synthetic world, the Tern Care oracle and the scorer - the ground truth everything else is measured against."""
import math

import pytest

from ttg import world

GOLDEN = world.make_scenarios(200, seed=7)


def _perfect(sc):
    plan = world.expected_plan(sc)
    acts = [{"action": p["action"], "args": p["args"], "result": world.execute_tool(p["action"], p["args"])}
            for p in plan if "action" in p]
    return acts, plan[-1]["final"]


def test_scenarios_are_deterministic_and_well_formed():
    assert world.make_scenarios(5, 7) == world.make_scenarios(5, 7)
    assert world.make_scenarios(3, 11)[0]["scenario_id"] == "S11-0001"
    ids = [s["scenario_id"] for s in GOLDEN]
    assert ids[0] == "S0001" and ids[-1] == "S0200" and len(set(ids)) == 200
    for s in GOLDEN:
        assert world.FLIGHT_RE.match(s["flight_no"]) and world.TRAVELER_RE.match(s["traveler_id"])
        assert 1 <= s["party_size"] <= 4
        assert (s["receipt"] is None) == (s["receipt_ref"] is None)


def test_golden_set_mix_is_what_the_docs_say():
    c = world._stats(GOLDEN)
    assert c["status=cancelled"] == 106 and c["status=delayed"] == 94
    assert c["controllable=True"] == 100 and c["overnight=True"] == 35
    assert c["hotel_booked=True"] == 23 and c["receipt=True"] == 66 and c["needs_approval=True"] == 26


def test_oracle_plan_is_policy_compliant_for_every_golden_scenario():
    for sc in GOLDEN:
        acts, final = _perfect(sc)
        ok, v = world.score(sc, acts, final)
        assert ok, (sc["scenario_id"], v)


@pytest.mark.parametrize("sc", GOLDEN, ids=lambda s: s["scenario_id"])
def test_policy_invariants(sc):
    out = world.expected_outcome(sc)
    v = {x["kind"]: x["amount_usd"] for x in out["vouchers"]}
    party, gold = sc["party_size"], sc["tier"] in ("gold", "platinum")
    long_disruption = sc["status"] == "cancelled" or sc["delay_minutes"] >= 180
    if not sc["controllable"]:
        assert "credit" not in v and "reimbursement" not in v
        if sc["tier"] != "platinum":
            assert v == {} and out["hotel"] is None
    if sc["controllable"] and sc["status"] == "cancelled":
        assert v["credit"] == 200 * party * (1.5 if gold else 1.0)
    if sc["status"] == "delayed":
        assert out["rebooked_flight"] == sc["flight_no"]
        if sc["delay_minutes"] < 180:
            assert "meal" not in v and "credit" not in v
    if "meal" in v:
        assert long_disruption and v["meal"] in (15 * party, 25 * party)
    if "reimbursement" in v:
        assert sc["controllable"] and v["reimbursement"] <= 50 * party + 1e-9
    if out["hotel"]:
        assert out["overnight"] and (sc["controllable"] or sc["tier"] == "platinum")
        rooms = math.ceil(party / 2)
        cheaper = [h for h in world._hotels(sc["origin"]) if h["rooms_available"] >= rooms
                   and h["nightly_rate_usd"] < out["hotel"]["nightly_rate_usd"]]
        assert not cheaper
    assert out["final"]["needs_human_approval"] == any(x > 300 for x in v.values())


def test_rebooking_picks_earliest_option_with_enough_seats():
    for sc in GOLDEN:
        if sc["status"] != "cancelled":
            continue
        opts = world.rebooking_options(sc["flight_no"], sc["party_size"])["options"]
        assert [o["departure"] for o in opts] == sorted(o["departure"] for o in opts)
        chosen = world.expected_outcome(sc)["rebooked_flight"]
        first_fit = next(o for o in opts if o["seats_available"] >= sc["party_size"])
        assert chosen == first_fit["flight_no"]


def test_scorer_catches_each_kind_of_mistake():
    sc = next(s for s in GOLDEN if s["controllable"] and s["status"] == "cancelled" and s["receipt_ref"])
    acts, final = _perfect(sc)
    voucher = next(a for a in acts if a["action"] == "issue_voucher")

    def bad(mutate_acts=None, mutate_final=None):
        a2 = [dict(x, args=dict(x["args"])) for x in acts]
        f2 = dict(final)
        if mutate_acts:
            a2 = mutate_acts(a2)
        if mutate_final:
            f2 = mutate_final(f2)
        return world.score(sc, a2, f2)[1]

    assert any(x.startswith("wrong_amount") for x in bad(lambda a: [dict(x, args={**x["args"], "amount_usd":
                                                                                  x["args"]["amount_usd"] * 2})
                                                                    if x["action"] == "issue_voucher" else x for x in a]))
    assert any(x.startswith("missing_voucher") for x in bad(lambda a: [x for x in a if x is not a[acts.index(voucher)]]))
    assert any(x.startswith("duplicate_voucher") for x in bad(lambda a: a + [a[acts.index(voucher)]]))
    assert "missing_rebooking" in bad(lambda a: [x for x in a if x["action"] != "book_flight"])
    assert any(x.startswith("wrong_rebooking") for x in bad(lambda a: [dict(x, args={**x["args"], "flight_no": "TN999"})
                                                                       if x["action"] == "book_flight" else x for x in a]))
    assert "approval_flag_wrong" in bad(mutate_final=lambda f: {**f, "needs_human_approval": not f["needs_human_approval"]})
    assert "message_missing_flight" in bad(mutate_final=lambda f: {**f, "message": "We are sorry."})
    assert "no_final" in world.score(sc, acts, None)[1]
    assert "wrong_traveler" in bad(lambda a: [dict(x, args={**x["args"], "traveler_id": "T00001"})
                                              if x["action"] == "issue_voucher" else x for x in a])


def test_tools_validate_arguments_and_never_raise():
    assert "error" in world.execute_tool("flight_status", {"flight_no": "AA100"})
    assert "error" in world.execute_tool("nope", {})
    assert "missing" in world.execute_tool("book_flight", {"traveler_id": "T12345"})["error"]
    assert "error" in world.execute_tool("issue_voucher", {"traveler_id": "T12345", "kind": "cash", "amount_usd": 5})
    assert "error" in world.execute_tool("hotel_availability", {"airport": "XXX", "nights": 1, "rooms": 1})
    assert world.execute_tool("issue_voucher", {"traveler_id": "T12345", "kind": "credit",
                                                "amount_usd": 300.01})["status"] == "PENDING_APPROVAL"
    assert world.execute_tool("issue_voucher", {"traveler_id": "T12345", "kind": "credit",
                                                "amount_usd": 300})["status"] == "ISSUED"
    sold_out = next(h for a in world.AIRPORTS for h in world._hotels(a) if h["rooms_available"] == 0)
    r = world.execute_tool("book_hotel", {"traveler_id": "T12345", "hotel_id": sold_out["hotel_id"], "nights": 1,
                                          "rooms": 1})
    assert r["status"] == "SOLD_OUT"


def test_tool_specs_match_implementations():
    assert sorted(world.TOOL_NAMES) == sorted(world.TOOLS)
    for t in world.TOOL_SPECS:
        props = t["input_schema"]["properties"]
        assert set(t["input_schema"]["required"]) <= set(props)
        assert len(t["description"]) > 30


def test_policy_search_finds_the_right_section():
    r = world.policy_search("hotel overnight rooms cheapest", k=2)["results"]
    assert r[0]["section"].startswith("5. Hotels")
    assert world.policy_search("receipt reimbursement taxi", k=1)["results"][0]["section"].startswith("7. Receipts")


def test_hidden_truth_is_not_in_the_public_view():
    pub = world.public_view(GOLDEN[0])
    for hidden in ("status", "cause", "controllable", "tier", "delay_minutes", "receipt", "origin"):
        assert hidden not in pub
