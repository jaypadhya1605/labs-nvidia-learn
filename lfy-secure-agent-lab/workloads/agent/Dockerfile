# SOC agent + MCP tool server image. Build context = repo root (scripts/build-images.sh).
FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" prometheus-client "mcp>=1.9" "model-signing>=1.0"
WORKDIR /app
COPY workloads/agent/*.py scripts/verify_bundle.sh /app/
ENV HOME=/tmp
USER 1000
EXPOSE 8080 9901
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
