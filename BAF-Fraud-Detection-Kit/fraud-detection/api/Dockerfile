# fraud-detection/api/Dockerfile
#
# Build context must be fraud-detection/ (not api/), because this image needs
# both src/ (the ML pipeline) and api/ (this service):
#   docker build -f api/Dockerfile -t fraud-api:latest .
#
# Minimal base image, non-root user, no secrets baked in (all secrets come
# from environment variables / a mounted .env at run time -- see
# api/.env.example), only the API port exposed.

FROM python:3.12-slim AS base

# Native deps needed to build lightgbm/xgboost/scikit-learn/shap wheels on
# slim images; removed from the final layer's apt cache to keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ML pipeline deps first (src/prediction.py, src/models.py etc. are imported
# directly by api/model_service.py), then the API's own deps.
COPY requirements.txt ./requirements.txt
COPY api/requirements.txt ./api-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r api-requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml ./config.yaml

# Runtime directories the app writes to (SQLite file, trained model
# artifacts if mounted/copied in separately -- see docker-compose.yml).
RUN mkdir -p /app/models /app/data/processed /app/data/predictions /app/reports \
    && addgroup --system fraudapi && adduser --system --ingroup fraudapi fraudapi \
    && chown -R fraudapi:fraudapi /app

USER fraudapi

EXPOSE 8000

# No secrets are set here. JWT_SECRET, DATABASE_URL, BOOTSTRAP_ADMIN_* etc.
# must be supplied at `docker run` / compose time via environment variables
# or an env_file -- never COPY a real .env into the image.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
