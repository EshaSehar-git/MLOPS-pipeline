# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python API, serving the built frontend ----
FROM python:3.11-slim

# Tesseract OCR system dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# The trained model (models/layoutlmv3-sroie/, ~500MB) is intentionally NOT
# baked into this image — mount it at runtime instead (see docker-compose.yml,
# which already does this via a volume). Without it, the API automatically
# falls back to the rule-based extractor rather than failing to start.

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
