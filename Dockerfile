FROM python:3.11-slim@sha256:9358444059ed78e2975ada2c189f1c1a3144a5dab6f35bff8c981afb38946634

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: heavy deps (torch, transformers) — cached unless requirements-heavy.txt changes
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Layer 2: medium deps
COPY requirements-medium.txt .
RUN pip install --no-cache-dir -r requirements-medium.txt

# Layer 3: light deps (change most often)
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt \
    && pip cache purge

COPY . .

RUN useradd -m quantswarm && chown -R quantswarm:quantswarm /app
USER quantswarm