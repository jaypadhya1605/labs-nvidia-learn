"""Shared plumbing for the two FastMCP servers: transport selection (stdio for local runs, streamable HTTP for k8s).

Host-header check (DNS-rebinding protection). FastMCP turns it on by default and accepts only localhost Host headers,
so behind a Kubernetes Service ("Host: ttg-mcp-ops:8101") every call would get 421 Misdirected Request.
TTG_MCP_ALLOWED_HOSTS adds names to the allow-list and keeps the protection ON for everyone else:
  TTG_MCP_ALLOWED_HOSTS="ttg-mcp-ops:*,ttg-mcp-ops.ttg.svc.cluster.local:*"    (the Helm chart sets this)
  TTG_MCP_ALLOWED_HOSTS="*"                                                     turns the check off (lab only)
"""
from __future__ import annotations

import argparse
import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
WILDCARD_BINDS = ("0.0.0.0", "::", "")


def transport_security(bind_host: str, allowed: str | None) -> TransportSecuritySettings:
    """Allow-list = localhost + the bind address (when it is a specific one) + TTG_MCP_ALLOWED_HOSTS."""
    extra = [h.strip() for h in (allowed or "").split(",") if h.strip()]
    if extra == ["*"]:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    hosts = list(LOCAL_HOSTS)
    if bind_host not in WILDCARD_BINDS and bind_host not in ("127.0.0.1", "localhost", "::1"):
        hosts.append(f"{bind_host}:*")
    hosts += [h for h in extra if h not in hosts]
    origins = list(LOCAL_ORIGINS) + [f"http://{h}" for h in hosts[len(LOCAL_HOSTS):]]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                     allowed_origins=origins)


def run(mcp: FastMCP, default_port: int, argv=None) -> None:
    ap = argparse.ArgumentParser(description=f"MCP server {mcp.name}")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=default_port)
    a = ap.parse_args(argv)
    if a.transport != "stdio":
        allowed = os.environ.get("TTG_MCP_ALLOWED_HOSTS")
        mcp.settings.host, mcp.settings.port = a.host, a.port
        mcp.settings.transport_security = transport_security(a.host, allowed)
        if not allowed and a.host in WILDCARD_BINDS:
            print(f"{mcp.name}: listening on {a.host}:{a.port}; only localhost Host headers are accepted. "
                  "Behind a Service or in Docker, set TTG_MCP_ALLOWED_HOSTS=<name>:* (or 421 errors follow).",
                  file=sys.stderr)
    mcp.run(transport=a.transport)
