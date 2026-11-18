"""The same SOC tools exposed over MCP (streamable HTTP) for NeMo Agent Toolkit. Policy lives here too - the tool
server is the security boundary, whatever agent framework calls it. Always runs with defended policy."""
import json
import os

from mcp.server.fastmcp import FastMCP

import soc_tools

PLAYBOOKS = soc_tools.Playbooks([os.environ.get("PLAYBOOK_DIR", "/work/playbooks")])
mcp = FastMCP("lanternfly-soc-tools", host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "9901")))


def _safe(fn, *args):
    try:
        return fn(*args)
    except soc_tools.PolicyDenied as exc:
        return json.dumps({"policy_denied": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)[:300]})


@mcp.tool()
def run_kql(query: str) -> str:
    """Run a READ-ONLY KQL query over SigninLogs, DeviceProcessEvents, DeviceNetworkEvents, ThreatIntel, SecurityAlert."""
    return _safe(soc_tools.run_kql, query)


@mcp.tool()
def lookup_ip(ip: str) -> str:
    """Threat-intel reputation for one IPv4 address."""
    return _safe(soc_tools.lookup_ip, ip)


@mcp.tool()
def search_playbooks(question: str) -> str:
    """Retrieve the most relevant signed SOC playbooks."""
    return _safe(PLAYBOOKS.search, question)


@mcp.tool()
def propose_action(action: str, target: str, reason: str) -> str:
    """Propose containment (disable_account, isolate_host, block_ip, revoke_sessions). Always requires human approval."""
    return _safe(lambda: json.dumps(soc_tools.propose_action(action, target, reason, defended=True)))


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
