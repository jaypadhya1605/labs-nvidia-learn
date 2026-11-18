# Guarded gateway image. Build context = repo root (scripts/build-images.sh). Pinned to the versions verified for Lab 03.
FROM python:3.12-slim
RUN pip install --no-cache-dir "nemoguardrails[nvidia]==0.24.1" presidio-analyzer presidio-anonymizer \
      fastapi "uvicorn[standard]" prometheus-client httpx "model-signing>=1.0" \
 && python -m spacy download en_core_web_lg
WORKDIR /app
COPY workloads/gateway/app.py scripts/verify_bundle.sh /app/
ENV NVIDIA_API_KEY=not-used-self-hosted-nims HOME=/tmp
USER 1000
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
