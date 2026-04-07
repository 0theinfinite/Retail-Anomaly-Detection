# ── build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY retail_anomaly/ retail_anomaly/
COPY configs/        configs/

RUN pip install --upgrade pip \
 && pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="retail-anomaly" \
      org.opencontainers.image.description="Retail store scan-quality anomaly detection" \
      org.opencontainers.image.source="https://github.com/garyhuang/Retail-Anomaly-Detection"

# Non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Install wheel
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
 && rm /tmp/*.whl

# Copy config (models are expected to be mounted or fetched at runtime)
COPY configs/ configs/

USER app

EXPOSE 8000

# Uvicorn with 2 workers; increase in production
CMD ["uvicorn", "retail_anomaly.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
