"""The MCP servers over the two real transports the lab uses: stdio (laptop) and streamable HTTP (Kubernetes)."""
import asyncio
import os
import socket
import subprocess
import sys
import time

import pytest

from ttg import world
from ttg.agent.config import Settings
from ttg.agent.tools import ToolBox, _to_dict

REPO = world.REPO


def _calls():
    sc = world.make_scenarios(1, 7)[0]
    return [("flight_status", {"flight_no": sc["flight_no"]}), ("loyalty_profile", {"traveler_id": sc["traveler_id"]}),
            ("hotel_availability", {"airport": "DTW", "nights": 1, "rooms": 2}),
            ("issue_voucher", {"traveler_id": sc["traveler_id"], "kind": "credit", "amount_usd": 450}),
            ("read_receipt", {"receipt_ref": "R-S0001"}), ("policy_search", {"query": "hotel", "k": 2})]


async def _exercise(settings):
    async with ToolBox(settings) as tb:
        adv = {t["name"]: t for t in tb.advertised()}
        for spec in world.TOOL_SPECS:
            assert adv[spec["name"]]["description"] == spec["description"]
            assert adv[spec["name"]]["required"] == sorted(spec["input_schema"]["required"])
        for name, args in _calls():
            assert await tb.call(name, args) == world.execute_tool(name, args), name
        bad = await tb.call("flight_status", {"flight_no": "XX1"})
        assert "error" in bad
        many = await asyncio.gather(*(tb.call("flight_status", {"flight_no": f"TN{100 + i}"}) for i in range(40)))
        assert all(m["flight_no"] == f"TN{100 + i}" for i, m in enumerate(many))


def test_stdio_mcp_matches_the_world(monkeypatch):
    monkeypatch.setenv("TTG_MCP_MODE", "stdio")
    asyncio.run(_exercise(Settings()))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_servers():
    ports, procs = {"ops": _free_port(), "guest": _free_port()}, []
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for name, port in ports.items():
        procs.append(subprocess.Popen([sys.executable, "-m", f"ttg.mcp.{name}_server", "--transport", "streamable-http",
                                       "--host", "127.0.0.1", "--port", str(port)], cwd=REPO, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    deadline = time.time() + 20
    for port in ports.values():
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
    yield ports
    for p in procs:
        p.terminate()
        p.wait(timeout=10)


def test_streamable_http_mcp_matches_the_world(http_servers, monkeypatch):
    monkeypatch.setenv("TTG_MCP_MODE", "http")
    monkeypatch.setenv("TTG_MCP_OPS_URL", f"http://127.0.0.1:{http_servers['ops']}/mcp")
    monkeypatch.setenv("TTG_MCP_GUEST_URL", f"http://127.0.0.1:{http_servers['guest']}/mcp")
    asyncio.run(_exercise(Settings()))


def test_inproc_mode_skips_mcp():
    tb = ToolBox(Settings(mcp_mode="inproc"))
    assert asyncio.run(tb.call("flight_status", {"flight_no": "TN100"})) == world.flight_status("TN100")


def test_tool_result_decoding():
    assert _to_dict([{"type": "text", "text": '{"a": 1}'}]) == {"a": 1}
    assert _to_dict(('{"a": 2}', None)) == {"a": 2}
    assert _to_dict("not json") == {"result": "not json"}
    assert _to_dict({"x": 1}) == {"x": 1}
