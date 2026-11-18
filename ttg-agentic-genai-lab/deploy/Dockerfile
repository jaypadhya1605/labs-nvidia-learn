# One image, four roles - the command picks the role (deploy/helm/ttg sets it per Deployment):
#   agent API     uvicorn ttg.agent.api:app --host 0.0.0.0 --port 8080        (default CMD)
#   MCP ops       python -m ttg.mcp.ops_server --transport streamable-http --host 0.0.0.0 --port 8101
#   MCP guest     python -m ttg.mcp.guest_server --transport streamable-http --host 0.0.0.0 --port 8102
#   mock engine   uvicorn ttg.mock.fake_llm:app --host 0.0.0.0 --port 8000  (kind / laptop: no GPU)
# Build from the repo root:  docker build -f deploy/Dockerfile -t ttg-agent:dev .
# CPU-only and small on purpose: the GPU work lives in the engine containers (vLLM, SGLang, TensorRT-LLM, NIM).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TTG_TRACE_DIR=/tmp/traces

# DejaVu Sans Mono: the font vlm/make_receipts.py prints receipts with (else Pillow's built-in bitmap font)
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements-runtime.txt /app/requirements-runtime.txt
RUN pip install -r /app/requirements-runtime.txt

# Code + the fictional policy corpus (policy_search reads data/policies). vlm/ backs read_receipt in M8.
COPY ttg /app/ttg
COPY data/policies /app/data/policies
COPY vlm /app/vlm
# M8 in the cluster: render the golden set's receipts INTO the image (synthetic and deterministic - no customer data),
# so the guest MCP server finds /app/data/receipts/<ref>.png when mcp.guest.vlm.baseUrl is set.
RUN python -m vlm.make_receipts --out /app/data/receipts

# Non-root, read-only root filesystem friendly (the chart mounts an emptyDir on /tmp for traces).
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin ttg
USER 10001
EXPOSE 8080 8101 8102 8000
CMD ["uvicorn", "ttg.agent.api:app", "--host", "0.0.0.0", "--port", "8080"]
