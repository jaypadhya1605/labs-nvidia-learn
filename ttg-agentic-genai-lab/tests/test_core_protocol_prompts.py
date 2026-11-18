"""Cache-stable prompts and the JSON action protocol."""
import json

import pytest

from ttg import world
from ttg.agent import prompts
from ttg.agent.protocol import (STEP_JSON_SCHEMA, ProtocolError, format_action, format_final, parse_step,
                                response_format)

GOLDEN = world.make_scenarios(50, seed=7)


def test_system_prompt_is_byte_identical_and_complete():
    a, b = prompts.build_system_prompt(), prompts.build_system_prompt()
    assert a == b
    for t in world.TOOL_NAMES:
        assert t in a
    assert 6000 < len(a) < 9000
    short = prompts.build_system_prompt(variant="short")
    assert len(short) < 800 and "digest" not in short and "## Tools" not in short
    assert "great detail" in prompts.build_system_prompt(thoughts="verbose")


def test_bust_cache_changes_the_very_first_bytes():
    busted = prompts.build_system_prompt(bust_cache=True)
    assert busted.startswith("Current time: ") and busted.endswith(prompts.build_system_prompt())


def test_scenario_message_hides_ground_truth():
    msg = prompts.render_scenario(GOLDEN[0])
    payload = json.loads(msg.split(": ", 1)[1])
    assert payload["scenario_id"] == "S0001"
    assert "cause" not in payload and "tier" not in payload and "status" not in payload


def test_observations_are_compact_and_sorted():
    obs = prompts.render_observation("flight_status", {"b": 1, "a": 2})
    assert obs == 'OBSERVATION flight_status: {"a":2,"b":1}'


def test_every_oracle_step_round_trips():
    for sc in GOLDEN:
        for st in world.expected_plan(sc):
            if "action" in st:
                s = parse_step(format_action(st["action"], st["args"], "why"))
                assert (s.kind, s.action, s.args) == ("action", st["action"], st["args"])
            else:
                s = parse_step(format_final(st["final"], "done"))
                assert s.kind == "final" and s.final["rebooked_flight"] == st["final"]["rebooked_flight"]


@pytest.mark.parametrize("text", [
    '```json\n{"thought": "x", "action": "flight_status", "args": {"flight_no": "TN100"}}\n```',
    '<|python_tag|>{"thought": "x", "action": "flight_status", "args": {"flight_no": "TN100"}}',
    'Here you go: {"thought": "a {brace} inside", "action": "flight_status", "args": {"flight_no": "TN100"}} done',
    '{"thought": "x", "action": "flight_status", "arguments": "{\\"flight_no\\": \\"TN100\\"}"}',
])
def test_parse_tolerates_common_model_habits(text):
    s = parse_step(text)
    assert s.action == "flight_status" and s.args == {"flight_no": "TN100"}


def test_parse_coerces_types_and_normalizes_final():
    s = parse_step('{"thought":"t","action":"book_flight","args":{"traveler_id":"T12345","flight_no":"TN1","party_size":"2"}}')
    assert s.args["party_size"] == 2
    s = parse_step('{"thought":"t","action":"issue_voucher","args":{"traveler_id":"T12345","kind":"meal","amount_usd":"$1,030"}}')
    assert s.args["amount_usd"] == 1030.0
    f = parse_step('{"final":{"message":"hi","rebooked_flight":"tn123","hotel_id":"null","vouchers":'
                   '[{"kind":"MEAL","amount_usd":"30"}],"needs_human_approval":"true"}}').final
    assert f == {"message": "hi", "rebooked_flight": "TN123", "hotel_id": None,
                 "vouchers": [{"kind": "meal", "amount_usd": 30.0}], "needs_human_approval": True}


@pytest.mark.parametrize("text,needle", [
    ("", "Empty"), ("I will check the flight.", "No JSON"), ('{"thought": "x", "action": "fly"}', "Unknown tool"),
    ('{"thought": "x", "action": "book_flight", "args": {"traveler_id": "T12345"}}', "needs"),
    ('{"thought": "x", "action": "flight_status", "args": {"flight_no": "TN1"', "not closed"),
    ('{"thought": "x"}', 'No "action"'), ('{"final": {"message": "hi"}}', "missing"),
    ('{"final": {"message": "hi", "rebooked_flight": "TN1", "vouchers": [{"kind": "cash", "amount_usd": 1}]}}', "kind"),
    ('{"thought": "x", "action": "book_flight", "args": {"traveler_id": "T1", "flight_no": "TN1", "party_size": 1.5}}',
     "whole"),
])
def test_parse_errors_carry_fix_it_feedback(text, needle):
    with pytest.raises(ProtocolError) as e:
        parse_step(text)
    assert needle in str(e.value) and "Reply with exactly ONE JSON object" in str(e.value)


def test_structured_output_modes():
    assert response_format("none") is None
    assert response_format("json_object") == {"type": "json_object"}
    rf = response_format("json_schema")
    assert rf["json_schema"]["schema"] is STEP_JSON_SCHEMA
    action, final = STEP_JSON_SCHEMA["anyOf"]
    assert set(action["properties"]["action"]["enum"]) == set(world.TOOL_NAMES)
    assert action["required"] == ["thought", "action", "args"] and final["required"] == ["thought", "final"]


@pytest.mark.parametrize("text", [
    '{"thought": "x", "action": {"name": "flight_status", "arguments": {"flight_no": "TN100"}}}',
])
def test_openai_style_action_object_is_accepted(text):
    s = parse_step(text)
    assert (s.action, s.args) == ("flight_status", {"flight_no": "TN100"})


@pytest.mark.parametrize("text", [
    '{"thought": "x", "action": ["flight_status"], "args": {}}',
    '{"thought": "x", "action": 7, "args": {}}',
    '{"thought": "x", "action": "rebooking_options", "args": {"flight_no": "TN1", "party_size": NaN}}',
    '{"thought": "x", "action": "rebooking_options", "args": {"flight_no": "TN1", "party_size": Infinity}}',
    '{"thought": "x", "action": "rebooking_options", "args": {"flight_no": "TN1", "party_size": 1e999}}',
    '{"thought": "x", "action": "rebooking_options", "args": {"flight_no": "TN1", "party_size": ' + "9" * 400 + '}}',
    '{"thought": "x", "action": "rebooking_options", "args": {"flight_no": "TN1", "party_size": true}}',
    '{"thought": "x", "action": "issue_voucher", "args": {"traveler_id": "T1", "kind": "meal", "amount_usd": NaN}}',
    '{"thought": "x", "action": "flight_status", "args": {"flight_no": {"id": "TN1"}}}',
    '{"thought": "x", "final": {"message": "m", "rebooked_flight": "TN1", "vouchers": 30}}',
    '{"thought": "x", "final": {"message": "m", "rebooked_flight": "TN1", "vouchers": true}}',
    '{"thought": "x", "final": {"message": "m", "rebooked_flight": "TN1", "hotel_id": 5, "vouchers": []}}',
    '{"thought": "x", "final": {"message": {"a": 1}, "rebooked_flight": "TN1", "vouchers": []}}',
    '{"thought": "x", "final": {"message": "m", "rebooked_flight": "TN1", "vouchers": [{"kind": "meal", "amount_usd": "lots"}]}}',
    "[" * 5000 + "]" * 5000,
    '{"a": ' + "[" * 5000 + "]" * 5000 + "}",
])
def test_hostile_shapes_raise_protocol_error_only(text):
    with pytest.raises(ProtocolError) as e:
        parse_step(text)
    assert "Reply with exactly ONE JSON object" in str(e.value)


def test_fuzzed_mutations_never_escape_as_other_exceptions():
    import random
    rnd = random.Random(5)
    seeds = [format_action(st["action"], st["args"], "t") if "action" in st else format_final(st["final"], "t")
             for sc in GOLDEN[:20] for st in world.expected_plan(sc)]
    junk = ['NaN', 'Infinity', '-1e999', 'true', 'null', '[]', '{}', '"x"', '30', '{"name": 1}', '[1, 2]']
    for _ in range(4000):
        s = rnd.choice(seeds)
        i = rnd.randrange(len(s))
        s = s[:i] + rnd.choice(junk + [s[i:i + 5][::-1], "", '"', "}", "{"]) + s[i + rnd.randrange(1, 6):]
        try:
            parse_step(s)
        except ProtocolError:
            pass
