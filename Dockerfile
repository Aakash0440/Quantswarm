FROM python:3.11-slim@sha256:9358444059ed78e2975ada2c189f1c1a3144a5dab6f35bff8c981afb38946634

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# heavy first so Railway caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && find /usr/local/lib/python3.11 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

COPY . .

RUN useradd -m quantswarm && chown -R quantswarm:quantswarm /app
USER quantswarm