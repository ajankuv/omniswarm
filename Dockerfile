# OmniSwarm — slim, multi-stage image.
# Final image: python:3.12-slim + 4 runtime deps + the package. No build tools,
# no caches, no tests/docs. Plain `uvicorn` (not [standard]) to keep it small.

# ---- builder: install runtime deps into an isolated venv ----
FROM python:3.12-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn>=0.29" \
    "httpx>=0.27" \
    "pydantic>=2.6"

# ---- runtime: copy the venv + the package onto a clean base ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    OMNISWARM_DB_PATH=/data/omniswarm.db \
    OMNISWARM_RUNTIME=/data/omniswarm.runtime.json
RUN useradd -m -u 10001 omni && mkdir -p /data && chown omni:omni /data
COPY --from=builder /opt/venv /opt/venv
COPY omniswarm /app/omniswarm
WORKDIR /app
USER omni
EXPOSE 8100
# /healthz always returns 200 (it reports OmniRoute reachability in the body),
# so it is a good liveness check that the app itself is up.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/healthz', timeout=2).status==200 else 1)"]
CMD ["uvicorn", "omniswarm.app:app", "--host", "0.0.0.0", "--port", "8100"]
