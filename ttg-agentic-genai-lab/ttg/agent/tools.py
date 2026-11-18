"""Tool access for the agent: MCP sessions (stdio or streamable HTTP) through langchain-mcp-adapters, or in-process.

  TTG_MCP_MODE=stdio   the agent spawns both MCP servers as subprocesses (laptop default)
  TTG_MCP_MODE=http    TTG_MCP_OPS_URL / TTG_MCP_GUEST_URL (Kubernetes; long-running benchmark)
  TTG_MCP_MODE=inproc  call ttg.world directly - unit tests only; skips the MCP hop on purpose
Sessions are opened once and shared by every concurrent task (MCP multiplexes requests over one session).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

from ttg import world
from ttg.agent.config import Settings


def _to_dict(raw) -> dict:
    if isinstance(raw, tuple):                       # (content, artifact) when response_format=content_and_artifact
        raw = raw[0]
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in raw)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.startswith("Error executing tool"):   # MCP-layer error (e.g. schema validation)
        return {"error": raw}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {"result": out}
    except (TypeError, json.JSONDecodeError):
        return {"result": str(raw)}


def connections(s: Settings) -> dict:
    if s.mcp_mode == "http":
        return {"ops": {"url": s.mcp_ops_url, "transport": "streamable_http"},
                "guest": {"url": s.mcp_guest_url, "transport": "streamable_http"}}
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(world.REPO), env.get("PYTHONPATH", "")] if p)
    return {name: {"command": sys.executable, "args": ["-m", f"ttg.mcp.{name}_server"], "transport": "stdio",
                   "env": env, "cwd": str(world.REPO)} for name in ("ops", "guest")}


class ToolBox:
    def __init__(self, settings: Settings):
        self.s = settings
        self.tools: dict = {}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "ToolBox":
        if self.s.mcp_mode == "inproc":
            return self
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
        client = MultiServerMCPClient(connections(self.s))
        self._stack = AsyncExitStack()
        try:
            for name in ("ops", "guest"):
                session = await self._stack.enter_async_context(client.session(name))
                for tool in await load_mcp_tools(session):
                    self.tools[tool.name] = tool
        except BaseException:
            await self._stack.aclose()
            raise
        missing = sorted(set(world.TOOL_NAMES) - set(self.tools))
        if missing:
            await self._stack.aclose()
            raise RuntimeError(f"MCP servers did not advertise: {', '.join(missing)}")
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    def advertised(self) -> list:
        """What the MCP servers say about themselves (a test compares this with world.TOOL_SPECS)."""
        out = []
        for name, t in self.tools.items():
            schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
            out.append({"name": name, "description": t.description, "required": sorted(schema.get("required", []))})
        return out

    async def call(self, name: str, args: dict) -> dict:
        if self.s.mcp_mode == "inproc":
            return world.execute_tool(name, args)
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            raw = await asyncio.wait_for(tool.ainvoke(args), timeout=self.s.tool_timeout_s)
        except asyncio.TimeoutError:
            return {"error": f"tool {name} timed out after {self.s.tool_timeout_s:g}s"}
        except Exception as e:                          # surfaced to the model as an observation
            return {"error": f"{type(e).__name__}: {e}"}
        return _to_dict(raw)
