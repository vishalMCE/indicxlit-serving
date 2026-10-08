# ==============================================================================
# Multi-Stage Production Dockerfile for IndicXlit Serving Engine
# Lightweight, sub-150MB image without PyTorch, CUDA, or Fairseq dependencies.
# ==============================================================================

FROM python:3.10-slim AS builder

WORKDIR /build

# Install build essentials for wheels if required
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY server/requirements.txt .

# Install dependencies into wheels directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ------------------------------------------------------------------------------
# Final Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS runner

WORKDIR /app

# Create non-root user for security compliance
RUN useradd -u 10001 -m -s /bin/bash appuser

# Install minimal runtime libraries (curl for healthcheck)
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code and converted 12MB CTranslate2 INT8 model
COPY --chown=appuser:appuser benchmarks /app/benchmarks
COPY --chown=appuser:appuser server /app/server
COPY --chown=appuser:appuser engine.py /app/engine.py
COPY --chown=appuser:appuser models/ct2_indicxlit_int8 /app/models/ct2_indicxlit_int8

# Set permissions
USER appuser

# Expose HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
