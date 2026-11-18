"""The in-cluster MCP hop, without a cluster: start each MCP server with the command and env the Helm chart renders,
then call it with the Host header the agent sends inside Kubernetes ("ttg-mcp-ops:8101").

Why: FastMCP turns DNS-rebinding protection on and accepts only localhost Host headers by default, so a server that
works on the laptop (and in the CI eval gate, which uses localhost) answers 421 Misdirected Request behind a Service.
The chart passes TTG_MCP_ALLOWED_HOSTS; ttg/mcp/_common.py must turn it into TransportSecuritySettings.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
HELM = shutil.which("helm")
pytestmark = pytest.mark.skipif(not HELM, reason="helm is not installed")
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "host-probe", "version": "0"}}}
HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rendered_mcp(name: str, ns: str = "ttg") -> dict:
    out = subprocess.run([HELM, "template", "ttg", "deploy/helm/ttg", "-n", ns, "-f", "deploy/values/kind.yaml"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    for d in yaml.safe_load_all(out):
        if d and d["kind"] == "Deployment" and d["metadata"]["name"] == name:
            return d["spec"]["template"]["spec"]["containers"][0]
    raise AssertionError(f"{name} not rendered")


@pytest.mark.parametrize("name, port", [("ttg-mcp-ops", 8101), ("ttg-mcp-guest", 8102)])
def test_mcp_server_accepts_its_kubernetes_service_name(name, port, tmp_path):
    pytest.importorskip("mcp")
    c = _rendered_mcp(name)
    cmd = list(c["command"])
    assert cmd[0] == "python" and cmd[cmd.index("--port") + 1] == str(port)
    local = _free_port()
    cmd[0], cmd[cmd.index("--port") + 1] = sys.executable, str(local)        # same command, a free local port
    env = {**os.environ, "PYTHONPATH": str(REPO), **{e["name"]: e["value"] for e in c.get("env", []) if "value" in e}}
    log = (tmp_path / f"{name}.log").open("w")
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{local}/mcp"
        deadline = time.time() + 30
        while True:
            try:
                httpx.get(url, timeout=1)
                break
            except httpx.HTTPError:
                if time.time() > deadline or proc.poll() is not None:
                    pytest.fail(f"{name} did not start:\n{(tmp_path / f'{name}.log').read_text()[-2000:]}")
                time.sleep(0.2)
        assert httpx.post(url, json=INIT, headers={**HEADERS, "host": f"localhost:{local}"}, timeout=10).status_code == 200
        in_cluster = httpx.post(url, json=INIT, headers={**HEADERS, "host": f"{name}:{port}"}, timeout=10)
        assert in_cluster.status_code == 200, f"Host {name}:{port} -> {in_cluster.status_code} {in_cluster.text[:200]}"
        fqdn = httpx.post(url, json=INIT, headers={**HEADERS, "host": f"{name}.ttg.svc.cluster.local:{port}"}, timeout=10)
        assert fqdn.status_code == 200
        stranger = httpx.post(url, json=INIT, headers={**HEADERS, "host": f"evil.example:{port}"}, timeout=10)
        assert stranger.status_code == 421                                     # protection stays on for everyone else
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
