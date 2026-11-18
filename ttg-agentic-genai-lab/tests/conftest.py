"""Shared fixtures (MAIN owns this file; other suites add their own fixtures inside their test files).

mock_engine: the lab's mock OpenAI-compatible engine as a real HTTP server on a free port (MOCK_SPEED=0, so it
answers instantly). Used by the agent, API and bench tests - everything runs offline, no GPU, no API keys.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(module: str, port: int, env: dict) -> subprocess.Popen:
    full = {**os.environ, "PYTHONPATH": str(REPO), **env}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", module, "--port", str(port), "--log-level", "warning"],
                            cwd=REPO, env=full, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"{module} did not start: {proc.stderr.read().decode()[-2000:]}")


@pytest.fixture(scope="session")
def mock_engine():
    port = _free_port()
    proc = _start("ttg.mock.fake_llm:app", port, {"MOCK_SPEED": "0"})
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def agent_env(mock_engine, monkeypatch, tmp_path):
    """TTG_* settings pointing at the mock engine; MCP in-process by default (fast); tests that need the real
    MCP hop set TTG_MCP_MODE=stdio themselves."""
    monkeypatch.setenv("TTG_BASE_URL", f"{mock_engine}/v1")
    monkeypatch.setenv("TTG_MODEL", "ttg-llm")
    monkeypatch.setenv("TTG_MCP_MODE", "inproc")
    monkeypatch.setenv("TTG_TRACE_DIR", str(tmp_path / "traces"))
    for k in ("TTG_BUST_CACHE", "TTG_THOUGHTS", "TTG_PROMPT_VARIANT", "TTG_STRUCTURED", "TTG_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    httpx.post(f"{mock_engine}/reset", timeout=5)
    return mock_engine
