FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Create non-root user
RUN useradd -m quantswarm && chown -R quantswarm:quantswarm /app
USER quantswarm

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run dashboard API
# Override with: docker run quantswarm python scripts/run_paper.py
CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
