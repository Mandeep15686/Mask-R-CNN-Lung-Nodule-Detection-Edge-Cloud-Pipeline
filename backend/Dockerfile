# FIXES APPLIED
# =============
# FIX-11  curl missing from image:
#   BEFORE: Compose health check used `curl -f http://localhost:8000/health`
#           but curl was not installed in the image.
#           The health check always failed, so the frontend and edge services
#           (which depend on `service_healthy`) never started.
#   AFTER:  curl added to the apt-get install list.

FROM python:3.11-slim

WORKDIR /app

# Runtime libraries for scipy and image handling + curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend + training package
COPY backend/ /app/backend/
COPY training/ /app/training/

WORKDIR /app/backend

ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:////data/lung_nodule.db
ENV MODEL_PATH=/app/experiments/best_model.pth

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
