# syntax=docker/dockerfile:1.6
# Two-stage build:
#   1. node — compile the Vite/React frontend → web/dist
#   2. python — install backend deps, copy models + built frontend, run uvicorn

# ─────────────────────────────────────────────────────────────────────
# Stage 1 — build the React frontend
# ─────────────────────────────────────────────────────────────────────
FROM node:20-slim AS web-build
WORKDIR /web

COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Python backend + serve the built frontend
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS app

# System deps for pyrodigal / xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires the container run as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project (everything needed at runtime)
COPY --chown=user:user api/ ./api/
COPY --chown=user:user src/ ./src/
COPY --chown=user:user scripts/recommend.py ./scripts/recommend.py
COPY --chown=user:user models/ ./models/
COPY --chown=user:user artifacts/uncultured_predictions.parquet ./artifacts/uncultured_predictions.parquet
COPY --chown=user:user artifacts/hybrid_predictions.parquet ./artifacts/hybrid_predictions.parquet
COPY --chown=user:user data/media_metadata.parquet ./data/media_metadata.parquet
COPY --chown=user:user data/media_recipes.parquet ./data/media_recipes.parquet
COPY --chown=user:user pyproject.toml README.md ./

# Built frontend from stage 1
COPY --from=web-build --chown=user:user /web/dist ./web/dist

# Make sure microbe_model is importable
RUN pip install --no-cache-dir -e .

# HF Spaces expects 7860; allow override
ENV PORT=7860
USER user

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
