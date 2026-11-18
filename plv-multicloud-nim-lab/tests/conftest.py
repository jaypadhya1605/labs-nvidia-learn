"""Shared pytest wiring. Suites that talk to an LLM use the `llm` fixture:
  no --endpoint      -> the in-process mock (tests/mock_openai.py): CI, laptops, the workshop
  --endpoint URL     -> any OpenAI-compatible server: NIM on AKS/GKE/EKS/OKE/RKE2, trtllm-serve, a managed endpoint
URL is the base WITHOUT /v1 (http://localhost:8000, https://integrate.api.nvidia.com).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "app", ROOT, ROOT / "tests", ROOT / "selector", ROOT / "fleet", ROOT / "bench", ROOT / "mlops", ROOT / "datacenter"):
    sys.path.insert(0, str(p))


def pytest_addoption(parser):
    parser.addoption("--endpoint", default=None, help="OpenAI-compatible base URL without /v1, e.g. http://localhost:8000")
    parser.addoption("--api", default=None, help="Plover API base URL, e.g. http://localhost:8080")
    parser.addoption("--model", default="meta/llama-3.1-8b-instruct")
    parser.addoption("--adapter", default=None, help="LoRA adapter model id to check (e.g. plover-style-v1)")
    parser.addoption("--api-key", default=None, help="Bearer token for managed endpoints (or env LLM_API_KEY)")
    parser.addoption("--extra-body", default="{}", help='JSON merged into requests, e.g. {"chat_template_kwargs": {"enable_thinking": false}}')
    parser.addoption("--min-task-success", type=float, default=0.70)
    parser.addoption("--max-injection-success", type=float, default=0.25)
    parser.addoption("--scorecard", default=None, help="write the agentic scorecard JSON here")


@pytest.fixture(scope="session")
def live(request):
    return request.config.getoption("--endpoint") is not None


@pytest.fixture(scope="session")
def llm(request):
    import httpx
    ep = request.config.getoption("--endpoint")
    if ep:
        key = request.config.getoption("--api-key") or os.environ.get("LLM_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        with httpx.Client(base_url=ep.rstrip("/"), headers=headers, timeout=300) as c:
            yield c
    else:
        from fastapi.testclient import TestClient
        from mock_openai import app
        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="session")
def model(request):
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def extra_body(request):
    import json
    return json.loads(request.config.getoption("--extra-body"))
