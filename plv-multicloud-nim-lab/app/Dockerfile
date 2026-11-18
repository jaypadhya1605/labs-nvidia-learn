# Plover Assist API image - built and pushed to GHCR by .github/workflows/release.yml (cloud-neutral registry).
FROM python:3.12-slim
WORKDIR /srv
COPY app/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt
COPY app/plover_api /srv/plover_api
USER 1000
EXPOSE 8080
CMD ["uvicorn", "plover_api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
