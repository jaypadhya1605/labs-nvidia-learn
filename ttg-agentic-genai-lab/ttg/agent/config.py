"""Agent settings from environment variables (TTG_*). One place, so the bench, API, drills and CI agree."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


@dataclass
class Settings:
    base_url: str = field(default_factory=lambda: _env("TTG_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: _env("TTG_API_KEY", "not-used"))
    model: str = field(default_factory=lambda: _env("TTG_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    prompt_variant: str = field(default_factory=lambda: _env("TTG_PROMPT_VARIANT", "full"))       # full | short
    thoughts: str = field(default_factory=lambda: _env("TTG_THOUGHTS", "brief"))                  # brief | verbose
    bust_cache: bool = field(default_factory=lambda: _env("TTG_BUST_CACHE", "0") == "1")
    structured: str = field(default_factory=lambda: _env("TTG_STRUCTURED", "none"))   # none | json_object | json_schema
    max_steps: int = field(default_factory=lambda: int(_env("TTG_MAX_STEPS", "14")))
    max_tokens: int = field(default_factory=lambda: int(_env("TTG_MAX_TOKENS", "256")))
    temperature: float = field(default_factory=lambda: float(_env("TTG_TEMPERATURE", "0")))
    timeout_s: float = field(default_factory=lambda: float(_env("TTG_TIMEOUT_S", "60")))
    tool_timeout_s: float = field(default_factory=lambda: float(_env("TTG_TOOL_TIMEOUT_S", "10")))
    max_retries: int = field(default_factory=lambda: int(_env("TTG_MAX_RETRIES", "2")))
    mcp_mode: str = field(default_factory=lambda: _env("TTG_MCP_MODE", "stdio"))          # stdio | http | inproc
    mcp_ops_url: str = field(default_factory=lambda: _env("TTG_MCP_OPS_URL", "http://localhost:8101/mcp"))
    mcp_guest_url: str = field(default_factory=lambda: _env("TTG_MCP_GUEST_URL", "http://localhost:8102/mcp"))
    trace_dir: str = field(default_factory=lambda: _env("TTG_TRACE_DIR", ""))

    def validate(self) -> "Settings":
        assert self.prompt_variant in ("full", "short"), "TTG_PROMPT_VARIANT must be full or short"
        assert self.thoughts in ("brief", "verbose"), "TTG_THOUGHTS must be brief or verbose"
        assert self.structured in ("none", "json_object", "json_schema"), "TTG_STRUCTURED: none|json_object|json_schema"
        assert self.mcp_mode in ("stdio", "http", "inproc"), "TTG_MCP_MODE must be stdio, http or inproc"
        assert self.max_steps >= 3 and self.max_tokens >= 32
        return self

    def public(self) -> dict:
        d = asdict(self)
        d["api_key"] = "set" if self.api_key not in ("", "not-used") else "not-used"
        return d
